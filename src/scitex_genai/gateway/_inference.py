"""Relay Anthropic ``/v1/messages`` to a pool of local inference upstreams.

Ported from ``anthropic_system_hoist_proxy.py`` — 369 lines that lived outside
any package on compute-04, untested, bound to loopback by a literal, and the
fleet's ONLY path to the local model. The upstreams it fronts are
Anthropic-compatible ``/v1/messages`` servers (vLLM's, behind SSH forwards),
so there is no protocol translation here: the one transformation is the
system hoist below, and everything else — path, query string, headers, body,
upstream status and body — is forwarded and returned verbatim.

WHY THE HOIST EXISTS (measured 2026-08-15, scitex-compute-04, by scitex-hub)
----------------------------------------------------------------------------
Claude Code >= v2.1.x sends its Agent-tool listing ("Available agent types for
the Agent tool: ...") as an EXTRA message with ``role: "system"`` inside
``messages[]``, in addition to the correct top-level ``system`` field. It posts
this to ``/v1/messages?beta=true``. Anthropic's beta endpoint accepts that.
vLLM's Anthropic-compatible endpoint implements the stricter public schema and
rejects it::

    400  {'loc': ('body','messages',1,'role'),
          'msg': "Input should be 'user' or 'assistant'"}

So every agent pointed at the local endpoint 400s on its FIRST turn. That is
the whole "the local model channel is broken" outage of 2026-08-13.

WHAT THE HOIST DOES: exactly one transformation — move every ``role: system``
entry out of ``messages[]`` and append its content blocks to the top-level
``system``. Order among the hoisted blocks is preserved, and the top-level
system (if any) stays first. A body that is not JSON is forwarded untouched
rather than dropped. Upstream status codes and bodies come back verbatim — no
fallback, no masking.

WHY A POOL, AND WHY STICKY (measured 2026-08-15 on a 4-GPU fleet)
------------------------------------------------------------------
Every Claude Code agent reaches vLLM through this relay (Claude Code cannot
speak raw vLLM), and the first relay had ONE upstream, so nine agents landed on
one card while three identical H100s idled: ``18773 running=3``, the other
three ``running=0 waiting=0``. Never saturation — routing.

vLLM keeps a PER-ENGINE PREFIX CACHE and these conversations run to 262144
tokens. Round-robining each REQUEST would send every turn of one conversation
to a different card, missing the cache every time. So an upstream is chosen
ONCE per conversation and reused for its later turns: spread ACROSS agents,
locality WITHIN one. Measured on compute-04 2026-08-18, the same 279,710-token
prompt: cold 60.60 s, warm 1.54 s — a miss costs ~40x a hit, so STICKINESS
OUTRANKS LOAD. Ranking load first looked correct at idle and, under load,
migrated conversations, each migration adding ~60 s, which raised in-flight
counts and drove further migration. :class:`~._pool.StickyPool` encodes
exactly that ordering, which is why this module builds on it instead of
carrying its own scheduler.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import time
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from ._admission import AdmissionController, CacheResidency
from ._errors import (
    HomeMemberReloading,
    InferenceAdmissionError,
    NoAccountAvailable,
    UpstreamReloading,
    UpstreamUnreachable,
)
from ._health import (
    DEFAULT_HEALTH_PROBE_TIMEOUT_S,
    UpstreamReachability,
    probe_upstream,
    timed_out_reachability,
)
from ._pool import StickyPool

#: The fleet's systemd drop-ins set these; the names are kept so they keep
#: working unchanged. Comma-separated base URLs, seconds, and a truthy flag.
UPSTREAM_ENV = "HOIST_UPSTREAM"
TIMEOUT_ENV = "HOIST_TIMEOUT_S"
PREFIX_TELEMETRY_ENV = "HOIST_PREFIX_TELEMETRY"
DEFAULT_TIMEOUT_S = 600.0
DEFAULT_CAPACITY_PER_UPSTREAM = 8
DEFAULT_MAX_QUEUE_SIZE = 128
DEFAULT_TOKEN_CAPACITY_PER_UPSTREAM: int | None = None
DEFAULT_HEALTH_CACHE_TTL_S = 1.0
DEFAULT_HEALTH_FAILURE_THRESHOLD = 2
DEFAULT_CONTINUATION_QOS_ENABLED = False
DEFAULT_CONTINUATION_QOS_MAX_RETRIES = 1
DEFAULT_CONTINUATION_QOS_MIN_PREEMPT_TOKENS = 0

#: Bounded so a long-lived gateway cannot grow without limit; conversations
#: are few (one per agent) and eviction only costs a prefix-cache miss, never
#: correctness.
MAX_ROUTES = 512

#: How long an upstream that produced NO HTTP response stays out of rotation.
#: A dead member of a fan-out pool is not resilience: measured 2026-08-28,
#: one request in three came back 502 for days, and killed a working agent
#: mid-task each time. Cooling the member lets the same request try the next
#: one; a later request probes it again.
UNREACHABLE_COOLDOWN_S = 30.0
#: How long a conversation stays pinned to a home upstream that just went out
#: of rotation before it may be re-placed on another. Above the ~90 s a vLLM
#: engine takes to reload after a crash (measured 2026-09-05, Qwen pair), so a
#: request that killed one replica is not handed to the other by the harness's
#: immediate retry. A home that is STILL out after this is treated as gone.
HOME_FAILOVER_AFTER_S = 150.0
#: How long one request will WAIT, inside the relay, for its reloading home
#: upstream before the caller is told to retry. Measured 2026-09-05: Codex
#: treats a 503 as the end of its turn rather than backing off, so the 503 +
#: Retry-After of #45 protected the surviving replica and killed the agent's
#: task. Waiting here keeps the client's request open across the ~90 s vLLM
#: reload; past this bound the 503 still goes out.
WAIT_FOR_HOME_S = 170.0
#: One sleep slice while waiting; the pool's retry hint is capped to this so a
#: home that comes back early is used early.
WAIT_SLICE_S = 5.0

#: Never forwarded. The script dropped the first three; ``transfer-encoding``
#: joins them because the server has already de-chunked the body it hands us.
_SESSION_ID_HEADERS = ("x-scitex-session-id", "session_id", "x-session-id")
_HOP_BY_HOP = frozenset(
    {"host", "content-length", "connection", "transfer-encoding"}
).union(_SESSION_ID_HEADERS)
_SESSION_KEY_DOMAIN = b"scitex-genai-session-affinity\0"

# First pass (7 conversations) showed ALL agents identical at 1k and ALL
# distinct at 4k, so the entire divergence happens in that band. These
# checkpoints bracket it finely; the coarse ones are kept so the two passes
# stay comparable.
_CHECKPOINTS = (1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096, 16384)


def parse_upstreams(value: str) -> list[str]:
    """``HOIST_UPSTREAM`` format: comma-separated, whitespace-tolerant, no empties."""
    return [url.strip() for url in value.split(",") if url.strip()]


def telemetry_enabled(value: str) -> bool:
    """The script's truthiness: anything but empty / ``0`` / ``false`` is on."""
    return value.lower() not in ("", "0", "false")


def estimate_input_tokens(body: bytes | None) -> int:
    """Estimate input tokens without coupling the gateway to a model tokenizer.

    Four UTF-8 bytes per token is the same deliberately simple approximation
    exposed by ``/v1/messages/count_tokens``.  Admission is therefore a
    configurable safety budget, not a claim that the estimate is exact.
    """
    if not body:
        return 0
    return max(1, (len(body) + 3) // 4)


def request_session_key(headers: Mapping[str, str]) -> str:
    """Return a bounded, opaque key for caller-declared conversation identity.

    Header names are matched case-insensitively even for a plain ``dict``;
    Starlette's request header mapping already provides that behavior.
    ``X-SciTeX-Session-ID`` is the gateway-owned contract; the older generic
    spellings remain accepted for existing Codex and Anthropic clients.

    The raw value is trimmed and immediately digested. That bounds the pool
    key and keeps user/session identifiers out of request-journal prefixes.
    """
    for expected in _SESSION_ID_HEADERS:
        for name, value in headers.items():
            if name.lower() != expected or not isinstance(value, str):
                continue
            normalized = value.strip()
            if normalized:
                return hashlib.sha256(
                    _SESSION_KEY_DOMAIN + normalized.encode("utf-8")
                ).hexdigest()
    return ""


def continuation_qos_session_key(headers: Mapping[str, str]) -> str:
    """QoS identity from the gateway-owned canonical header only."""
    canonical = {
        name: value
        for name, value in headers.items()
        if name.lower() == "x-scitex-session-id"
    }
    return request_session_key(canonical)


def accepts_session_id(path: str) -> bool:
    """Whether the pinned SGLang protocol model propagates top-level sessions.

    OpenAI Chat Completions and Responses carry ``session_id`` into the
    scheduler. The native Anthropic adapter currently neither declares nor
    propagates it, so adding the unknown field there would only be discarded.
    """
    route = path.split("?", 1)[0].rstrip("/")
    return route in {"/v1/chat/completions", "/v1/responses"}


def inject_request_id(body: bytes | None, request_id: str) -> tuple[bytes | None, bool]:
    """Inject an SGLang abort handle only into a valid JSON object body."""
    if not body or not request_id:
        return body, False
    try:
        payload = json.loads(body)
    except (TypeError, ValueError):
        return body, False
    if not isinstance(payload, dict):
        return body, False
    payload["rid"] = request_id
    return json.dumps(payload).encode(), True


def as_blocks(content: Any) -> list[Any]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content)


def hoist_system(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Return ``(payload, n_hoisted)``. Pure; no I/O."""
    messages = payload.get("messages") or []
    hoisted = [m for m in messages if m.get("role") == "system"]
    if not hoisted:
        return payload, 0
    blocks = as_blocks(payload.get("system"))
    for message in hoisted:
        blocks.extend(as_blocks(message.get("content")))
    payload["system"] = blocks
    payload["messages"] = [m for m in messages if m.get("role") != "system"]
    return payload, len(hoisted)


_PREAMBLE_ROLES = frozenset({"system", "developer"})


def conversation_key(payload: Any) -> str | None:
    """Stable id for a conversation, or None when there is nothing to key on.

    Uses the preamble plus the FIRST real turn — both fixed for the life of a
    conversation while the turn list grows, so the same agent keeps hashing to
    the same value. Three body shapes share one derivation (2026-09-05, for
    Codex over the OpenAI protocol):

    * Anthropic Messages: ``system`` + ``messages`` (the hoist has already
      moved any in-band system message up, so ``messages[0]`` is a user
      turn and the key is byte-identical to the pre-2026-09-05 one — no
      fleet-wide cache flush on deploy).
    * OpenAI chat completions: ``messages`` whose FIRST entry is usually a
      ``system`` / ``developer`` message shared by every session of the same
      agent in the same cwd; keying on it would collide distinct
      conversations onto one replica, so it is skipped and the first real
      turn is used.
    * OpenAI Responses: ``instructions`` + ``input`` (``input`` may be a
      bare string — the schema allows it).

    Stickiness outranks load here: a miss re-pays a full prefill (~40x a
    hit on this fleet), so a body that yields no key is placed round-robin
    and a body that yields the wrong key is worse than none.
    """
    if not isinstance(payload, dict):
        return None
    preamble = (
        payload.get("system") if "system" in payload else payload.get("instructions")
    )
    items = payload.get("messages")
    if items is None:
        items = payload.get("input")
    if isinstance(items, str):
        items = [items]
    if not items:
        return None
    first = next(
        (
            item
            for item in items
            if not (isinstance(item, dict) and item.get("role") in _PREAMBLE_ROLES)
        ),
        None,
    )
    if first is None:
        return None
    seed = json.dumps([preamble, first], sort_keys=True, default=str)
    return hashlib.sha256(seed.encode()).hexdigest()


_OPENAI_PREAMBLE_ROLE = "developer"
_PREAMBLE_ROLES_OPENAI = frozenset({"system", "developer"})


def _item_text(item: Any) -> str:
    """The text of a chat message / Responses input item, whatever its shape."""
    content = item.get("content") if isinstance(item, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        return "\n".join(p for p in parts if p)
    return ""


def repair_tool_call_arguments(payload: Any) -> tuple[Any, int]:
    """Make every tool-call ``arguments`` string parse as JSON before relaying.

    Measured 2026-09-05 17:58Z (handyman-01, Codex CLI -> this gateway ->
    vLLM 0.28.0 with ``--tool-call-parser qwen3_xml``): the model emitted one
    function_call whose ``arguments`` was cut at 97 characters with a leaked
    ``</parameter`` tag. Codex stores the item and re-sends the whole
    conversation on every turn, and vLLM's Responses endpoint json-decodes
    every ``function_call.arguments`` on input, so from then on EVERY request
    of that conversation answered ``400 Expecting value: line 1 column 98``
    and the session was dead for good. One bad tool call must not do that.

    A string that is not JSON is replaced by a JSON object that carries it,
    ``{"_invalid_arguments": "<original>"}`` — the model still sees what it
    said, the upstream accepts the item, and the turn continues. Both the
    Responses shape (``input[].type == "function_call"``) and the chat shape
    (``messages[].tool_calls[].function``) are covered. Pure; returns
    ``(payload, repaired_count)``. Never touches anything that already parses.
    """
    if not isinstance(payload, dict):
        return payload, 0
    repaired = 0

    def _fix(args: Any) -> Any:
        nonlocal repaired
        if not isinstance(args, str):
            return args
        try:
            json.loads(args)
        except ValueError:
            repaired += 1
            return json.dumps({"_invalid_arguments": args})
        return args

    items = payload.get("input")
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and item.get("type") == "function_call":
                if "arguments" in item:
                    item["arguments"] = _fix(item["arguments"])
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            calls = message.get("tool_calls") if isinstance(message, dict) else None
            if not isinstance(calls, list):
                continue
            for call in calls:
                function = call.get("function") if isinstance(call, dict) else None
                if isinstance(function, dict) and "arguments" in function:
                    function["arguments"] = _fix(function["arguments"])
    return payload, repaired


def adapt_openai_roles(payload: Any) -> tuple[Any, bool]:
    """Leave the upstream exactly ONE system preamble, at the front.

    vLLM 0.22.0's Responses and chat endpoints refuse Codex's request twice
    over — measured on the first live codex turns through this gateway,
    2026-09-05 09:20 and 09:37 UTC:

    * ``{"error": {"message": "Unexpected message role."}}`` for the
      ``developer`` role Codex uses for its instructions;
    * ``{"error": {"message": "System message must be at the beginning."}}``
      once that role is renamed, because Codex ALSO sends top-level
      ``instructions`` (which vLLM turns into the first system message) and
      a second system item then follows it.

    So, for a Responses body, every ``developer`` / ``system`` input item is
    folded into ``instructions`` (appended, in order) and dropped from
    ``input``; a bare-string ``input`` is left alone. For a chat body, the
    preamble messages are merged into a single ``system`` message placed
    first. Pure; returns ``(payload, changed)``.
    """
    if not isinstance(payload, dict):
        return payload, False
    changed = False

    items = payload.get("input")
    if isinstance(items, list):
        preamble = [
            i
            for i in items
            if isinstance(i, dict) and i.get("role") in _PREAMBLE_ROLES_OPENAI
        ]
        if preamble:
            texts = [t for t in (_item_text(i) for i in preamble) if t]
            existing = payload.get("instructions")
            head = [existing] if isinstance(existing, str) and existing else []
            payload["instructions"] = "\n\n".join(head + texts)
            payload["input"] = [i for i in items if i not in preamble]
            changed = True

    messages = payload.get("messages")
    if isinstance(messages, list):
        preamble = [
            m
            for m in messages
            if isinstance(m, dict) and m.get("role") in _PREAMBLE_ROLES_OPENAI
        ]
        rest = [m for m in messages if m not in preamble]
        if preamble and (
            len(preamble) > 1
            or preamble[0].get("role") != "system"
            or messages[0] is not preamble[0]
        ):
            texts = [t for t in (_item_text(m) for m in preamble) if t]
            payload["messages"] = [
                {"role": "system", "content": "\n\n".join(texts)}
            ] + rest
            changed = True

    return payload, changed


def hoists_on(path: str) -> bool:
    """True only for the Anthropic Messages route.

    ``hoist_system`` rewrites in-band ``role: system`` messages into the
    top-level ``system`` field that the Anthropic shape requires. On the
    OpenAI routes that rewrite is DESTRUCTIVE: measured 2026-09-05 against
    the live replica, a chat/completions body run through the hoist came
    out with its system message deleted from ``messages[]`` and parked
    under a top-level ``system`` key that vLLM accepts and discards —
    HTTP 200, token count byte-identical to sending no system prompt at
    all. An agent served that way silently loses its instructions.
    """
    return path.split("?", 1)[0] == "/v1/messages"


def prefix_report(payload: dict[str, Any], key: str | None) -> str | None:
    """Size-only description of the emitted system prompt. No content, ever.

    Added 2026-08-16 to answer one question for scitex-hpc: how much of the
    emitted system prompt is SHARED between agents? vLLM's prefix cache can
    only reuse a common PREFIX, and KV retention is the term that limits
    concurrency, so eight agents holding eight nearly-distinct ~59k prefixes
    instead of one shared one is an ~8x difference in retained KV. The agent
    spec is the wrong artifact for that question — it describes what was
    declared, not what is sent — and this relay is the one place the SENT
    prompt passes through. Comparing the truncated-sha256 checkpoints across
    requests brackets the divergence point; the total says what fraction is
    shared.
    """
    system = payload.get("system")
    if system is None:
        return None
    raw = json.dumps(system, sort_keys=True, default=str).encode()
    parts = [f"conv={(key or 'none')[:12]}", f"system_bytes={len(raw)}"]
    for cut in _CHECKPOINTS:
        if len(raw) >= cut:
            parts.append(
                f"p{cut // 1024}k={hashlib.sha256(raw[:cut]).hexdigest()[:12]}"
            )
    parts.append(f"full={hashlib.sha256(raw).hexdigest()[:12]}")
    return " ".join(parts)


@dataclass
class InferenceUpstream:
    """One Anthropic-compatible inference server.

    Its alias IS its base URL: that is the name the fleet's drop-ins, the boot
    line and every refusal use, and there is nothing else to call it.
    """

    alias: str
    in_flight: int = 0
    last_used_at: float = 0.0
    capacity: int = DEFAULT_CAPACITY_PER_UPSTREAM
    queued: int = 0
    token_capacity: int | None = DEFAULT_TOKEN_CAPACITY_PER_UPSTREAM
    input_tokens_in_flight: int = 0
    input_tokens_queued: int = 0
    cooldown_until: float = 0.0
    #: When this upstream last went out of rotation (None = healthy).
    cooling_since: float | None = None
    #: Incremented under the pool lock for race-safe health reconciliation.
    cooldown_generation: int = 0

    @property
    def base_url(self) -> str:
        return self.alias

    @property
    def usage_score(self) -> float:
        """No quota notion: the pool assumes interchangeable upstreams."""
        return 0.0

    @property
    def scheduling_load(self) -> int:
        return self.in_flight + self.queued

    def status(self, *, closing: bool = False) -> dict[str, Any]:
        status = {
            "url": self.alias,
            "active": not closing and self.cooldown_until <= time.time(),
            "in_flight": self.in_flight,
            "queued": self.queued,
            "capacity": self.capacity,
        }
        if self.token_capacity is not None:
            status.update(
                input_tokens_in_flight=self.input_tokens_in_flight,
                input_tokens_queued=self.input_tokens_queued,
                token_capacity=self.token_capacity,
            )
        return status


@dataclass(frozen=True)
class _PoolTicket:
    priority: bool = False


class InferenceUpstreamPool(StickyPool[InferenceUpstream]):
    """Sticky per-conversation pool with bounded per-upstream admission.

    First placement remains least-loaded round robin across interchangeable
    upstreams. Once placed, a conversation waits for that same member's
    capacity so admission never trades away prefix-cache locality.
    """

    empty_message = "No inference upstreams are configured"
    duplicate_message = "Inference upstream URLs must be unique"
    cooling_message = "All inference upstreams are cooling down"
    ineligible_message = "Inference upstream selector returned an ineligible upstream"

    def __init__(
        self,
        upstreams: list[InferenceUpstream],
        *,
        choose: Callable[[list[InferenceUpstream]], InferenceUpstream] | None = None,
        max_sessions: int | None = MAX_ROUTES,
        capacity_per_upstream: int = DEFAULT_CAPACITY_PER_UPSTREAM,
        max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
        token_capacity_per_upstream: int | None = DEFAULT_TOKEN_CAPACITY_PER_UPSTREAM,
    ) -> None:
        if capacity_per_upstream < 1:
            raise ValueError("capacity_per_upstream must be >= 1")
        if max_queue_size < 0:
            raise ValueError("max_queue_size must be >= 0")
        if token_capacity_per_upstream is not None and token_capacity_per_upstream < 1:
            raise ValueError("token_capacity_per_upstream must be >= 1")
        for upstream in upstreams:
            upstream.capacity = capacity_per_upstream
            upstream.token_capacity = token_capacity_per_upstream
        self.max_queue_size = max_queue_size
        self._next_placement = 0
        super().__init__(
            upstreams,
            choose=choose or self._round_robin,
            max_sessions=max_sessions,
            failover_after_s=HOME_FAILOVER_AFTER_S,
        )
        self._admission = asyncio.Condition(self._lock)
        self._waiters: dict[str, deque[_PoolTicket]] = {
            upstream.alias: deque() for upstream in self.upstreams
        }
        self._active_sessions: set[str] = set()
        self._closing = False

    @property
    def upstreams(self) -> list[InferenceUpstream]:
        return self.members

    @classmethod
    def from_urls(
        cls,
        urls: str | list[str],
        *,
        capacity_per_upstream: int = DEFAULT_CAPACITY_PER_UPSTREAM,
        max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
        token_capacity_per_upstream: int | None = DEFAULT_TOKEN_CAPACITY_PER_UPSTREAM,
    ) -> "InferenceUpstreamPool":
        """Build from the ``HOIST_UPSTREAM`` string or an already-split list."""
        if isinstance(urls, str):
            urls = parse_upstreams(urls)
        return cls(
            [InferenceUpstream(alias=url) for url in urls],
            capacity_per_upstream=capacity_per_upstream,
            max_queue_size=max_queue_size,
            token_capacity_per_upstream=token_capacity_per_upstream,
        )

    async def route_alias(
        self, session_id: str, *, exclude: set[str] | None = None
    ) -> str:
        """Resolve and pin a session's target without consuming capacity."""
        async with self._admission:
            if self._closing:
                raise InferenceAdmissionError("Inference gateway is shutting down")
            return self._select_locked(
                session_id, exclude or set(), now=time.time()
            ).alias

    async def acquire(
        self,
        session_id: str = "",
        *,
        exclude: set[str] | None = None,
        input_tokens: int = 0,
        priority: bool = False,
    ) -> InferenceUpstream:
        """Place first for cache locality, then wait for that member's capacity."""
        if input_tokens < 0:
            raise ValueError("input_tokens must be >= 0")
        async with self._admission:
            if self._closing:
                raise InferenceAdmissionError("Inference gateway is shutting down")
            selected = self._select_locked(
                session_id, exclude or set(), now=time.time()
            )
            if (
                selected.token_capacity is not None
                and input_tokens > selected.token_capacity
            ):
                raise InferenceAdmissionError(
                    "Estimated request input exceeds this upstream's token capacity "
                    f"({input_tokens}/{selected.token_capacity})"
                )
            if self._fits(selected, input_tokens, session_id) and not selected.queued:
                selected.in_flight += 1
                selected.input_tokens_in_flight += input_tokens
                if session_id:
                    self._active_sessions.add(session_id)
                return selected
            total_queued = sum(upstream.queued for upstream in self.upstreams)
            if total_queued >= self.max_queue_size:
                raise InferenceAdmissionError(
                    f"Inference queue is full ({total_queued}/{self.max_queue_size})"
                )
            ticket = _PoolTicket(priority=priority)
            waiters = self._waiters[selected.alias]
            # A proven continuation may take the next slot. This is deliberately
            # not a general priority API: callers opt in explicitly, and the
            # gateway only does so for a stable session with a prior 2xx reply.
            if priority:
                # FIFO within the continuation class, ahead of ordinary work.
                index = next(
                    (
                        position
                        for position, queued_ticket in enumerate(waiters)
                        if not queued_ticket.priority
                    ),
                    len(waiters),
                )
                waiters.insert(index, ticket)
            else:
                waiters.append(ticket)
            selected.queued += 1
            selected.input_tokens_queued += input_tokens
            queued = True
            try:
                while True:
                    if self._closing:
                        raise InferenceAdmissionError(
                            "Inference gateway is shutting down"
                        )
                    now = time.time()
                    if (
                        waiters[0] is ticket
                        and self._fits(selected, input_tokens, session_id)
                        and selected.cooldown_until <= now
                    ):
                        waiters.popleft()
                        selected.queued -= 1
                        selected.input_tokens_queued -= input_tokens
                        queued = False
                        selected.in_flight += 1
                        selected.input_tokens_in_flight += input_tokens
                        if session_id:
                            self._active_sessions.add(session_id)
                        return selected
                    cooldown_s = max(0.0, selected.cooldown_until - now)
                    try:
                        if cooldown_s:
                            await asyncio.wait_for(
                                self._admission.wait(), timeout=cooldown_s
                            )
                        else:
                            await self._admission.wait()
                    except TimeoutError:
                        pass
            finally:
                if queued:
                    waiters.remove(ticket)
                    selected.queued -= 1
                    selected.input_tokens_queued -= input_tokens
                    self._admission.notify_all()

    def _fits(
        self,
        member: InferenceUpstream,
        input_tokens: int,
        session_id: str = "",
    ) -> bool:
        token_capacity = member.token_capacity
        return (
            session_id not in self._active_sessions
            and member.in_flight < member.capacity
            and (
                token_capacity is None
                or member.input_tokens_in_flight + input_tokens <= token_capacity
            )
        )

    async def release(
        self,
        member: InferenceUpstream,
        *,
        input_tokens: int = 0,
        session_id: str = "",
    ) -> None:
        async with self._admission:
            member.in_flight = max(0, member.in_flight - 1)
            member.input_tokens_in_flight = max(
                0, member.input_tokens_in_flight - input_tokens
            )
            if session_id:
                self._active_sessions.discard(session_id)
            self._admission.notify_all()

    async def cool_down(self, member: InferenceUpstream, seconds: float) -> None:
        await super().cool_down(member, seconds)
        async with self._admission:
            self._admission.notify_all()

    def _cooldown_changed(self, member: InferenceUpstream) -> None:
        """Record the failure generation while ``StickyPool`` holds its lock."""
        member.cooldown_generation += 1

    async def cooldown_snapshot(self) -> list[tuple[InferenceUpstream, int]]:
        """Capture the generation each reachability probe is about to test."""
        async with self._lock:
            return [
                (upstream, upstream.cooldown_generation)
                for upstream in self.upstreams
            ]

    async def reconcile_recovered(
        self,
        snapshot: list[tuple[InferenceUpstream, int]],
        observations: list[UpstreamReachability],
    ) -> None:
        """Clear only a stale cooldown proven recovered by a current probe."""
        async with self._admission:
            changed = False
            for (upstream, generation), observed in zip(
                snapshot, observations, strict=True
            ):
                if (
                    observed.reachable
                    and upstream.cooldown_generation == generation
                    and upstream.cooldown_until > time.time()
                ):
                    upstream.cooldown_until = 0.0
                    upstream.cooling_since = None
                    changed = True
            if changed:
                self._admission.notify_all()

    async def close(self) -> None:
        """Reject new work and wake every queued request during shutdown."""
        async with self._admission:
            self._closing = True
            self._admission.notify_all()

    async def resume(self) -> None:
        """Reopen admission after an operator-aborted drain."""
        async with self._admission:
            self._closing = False
            self._admission.notify_all()

    @property
    def draining(self) -> bool:
        return self._closing

    def status(self) -> list[dict[str, Any]]:
        return [upstream.status(closing=self._closing) for upstream in self.upstreams]

    def _round_robin(self, candidates: list[InferenceUpstream]) -> InferenceUpstream:
        chosen = candidates[self._next_placement % len(candidates)]
        self._next_placement += 1
        return chosen


def announce(host: str, port: int, pool: InferenceUpstreamPool) -> str:
    """The boot line naming the resolved routing.

    The bug the relay fixed was invisible precisely because nothing ever said
    which upstream was in use: the unit reported active/running while every
    agent funnelled to one card. One line at start makes the next such
    misconfiguration a five-second read of the journal.
    """
    urls = ", ".join(upstream.alias for upstream in pool.upstreams)
    return (
        f"scitex-genai-gateway: listening {host}:{port} -> "
        f"{len(pool.upstreams)} inference upstream(s): {urls}  [sticky per conversation]"
    )


@dataclass
class RelayedResponse:
    """An upstream reply on its way back: status and content-type verbatim."""

    status_code: int
    content_type: str
    body: AsyncIterator[bytes]
    feedback_headers: dict[str, str] = field(default_factory=dict)


class _ClientDisconnected(Exception):
    """Internal control flow for an observed downstream close before headers."""


async def _empty_body() -> AsyncIterator[bytes]:
    """A valid empty streaming body for a request whose client is already gone."""
    if False:
        yield b""


@dataclass
class _FirstTurnAttempt:
    """A replay-safe first turn currently waiting for response headers."""

    upstream_alias: str
    preempt: asyncio.Event = field(default_factory=asyncio.Event)
    resume_after: asyncio.Future[None] | None = None
    released: asyncio.Future[None] | None = None


@dataclass
class _ContinuationHandoff:
    resume_first_turn: asyncio.Future[None]
    victim_released: asyncio.Future[None]


class ContinuationQoS:
    """Ephemeral session-success classification and cooperative preemption.

    This is intentionally not called cache residency: a successful request is
    evidence of conversation history only. Engine cache state can disappear at
    any time and is neither queried nor inferred here.
    """

    def __init__(
        self, *, enabled: bool = False, max_retries: int = 1, min_preempt_tokens: int = 0
    ) -> None:
        if max_retries < 0:
            raise ValueError("continuation_qos_max_retries must be >= 0")
        if min_preempt_tokens < 0:
            raise ValueError("continuation_qos_min_preempt_tokens must be >= 0")
        self.enabled = enabled
        self.max_retries = max_retries
        self.min_preempt_tokens = min_preempt_tokens
        self._successful: OrderedDict[str, None] = OrderedDict()
        self._first_turns: dict[str, deque[_FirstTurnAttempt]] = {}
        self._counters = {
            "first_turn": 0,
            "continuation": 0,
            "unclassified": 0,
            "preemptions_requested": 0,
            "first_turns_preempted": 0,
            "first_turn_retries": 0,
            "retry_budget_exhausted": 0,
            "abort_failures": 0,
            "unconfirmed_cleanup_holds": 0,
            "cleanup_reapers_active": 0,
            "cleanup_reaper_recoveries": 0,
            "cleanup_reaper_attempts": 0,
            "cleanup_reaper_cancellations": 0,
        }

    def classify(self, explicit_session: str) -> str:
        if not explicit_session:
            kind = "unclassified"
        elif explicit_session in self._successful:
            kind = "continuation"
        else:
            kind = "first-turn"
        self._counters[kind.replace("-", "_")] += 1
        return kind

    def mark_successful(self, explicit_session: str) -> None:
        if not explicit_session:
            return
        self._successful.pop(explicit_session, None)
        self._successful[explicit_session] = None
        while len(self._successful) > MAX_ROUTES:
            self._successful.popitem(last=False)

    def register_first_turn(self, upstream_alias: str) -> _FirstTurnAttempt:
        attempt = _FirstTurnAttempt(upstream_alias)
        self._first_turns.setdefault(upstream_alias, deque()).append(attempt)
        return attempt

    def unregister_first_turn(self, attempt: _FirstTurnAttempt) -> None:
        attempts = self._first_turns.get(attempt.upstream_alias)
        if attempts is None:
            return
        try:
            attempts.remove(attempt)
        except ValueError:
            pass
        if not attempts:
            self._first_turns.pop(attempt.upstream_alias, None)

    def request_preemption(self, upstream_alias: str) -> _ContinuationHandoff | None:
        """Signal one replay-safe first turn and return its continuation barrier."""
        attempts = self._first_turns.get(upstream_alias, ())
        victim = next((item for item in attempts if not item.preempt.is_set()), None)
        if victim is None:
            return None
        barrier = asyncio.get_running_loop().create_future()
        released = asyncio.get_running_loop().create_future()
        victim.resume_after = barrier
        victim.released = released
        victim.preempt.set()
        self._counters["preemptions_requested"] += 1
        return _ContinuationHandoff(barrier, released)

    def preempted(self) -> None:
        self._counters["first_turns_preempted"] += 1

    def retried(self) -> None:
        self._counters["first_turn_retries"] += 1

    def exhausted(self) -> None:
        self._counters["retry_budget_exhausted"] += 1

    def abort_failed(self) -> None:
        self._counters["abort_failures"] += 1

    def cleanup_held(self) -> None:
        self._counters["unconfirmed_cleanup_holds"] += 1
        self._counters["cleanup_reapers_active"] += 1

    def cleanup_recovered(self) -> None:
        self._counters["cleanup_reapers_active"] = max(
            0, self._counters["cleanup_reapers_active"] - 1
        )
        self._counters["cleanup_reaper_recoveries"] += 1

    def cleanup_reaper_attempted(self) -> None:
        self._counters["cleanup_reaper_attempts"] += 1

    def cleanup_reaper_cancelled(self) -> None:
        self._counters["cleanup_reapers_active"] = max(
            0, self._counters["cleanup_reapers_active"] - 1
        )
        self._counters["cleanup_reaper_cancellations"] += 1

    @staticmethod
    def finish_continuation(handoff: _ContinuationHandoff | None) -> None:
        if handoff is not None and not handoff.resume_first_turn.done():
            handoff.resume_first_turn.set_result(None)

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": "enabled" if self.enabled else "disabled",
            "max_retries": self.max_retries,
            "min_preempt_tokens": self.min_preempt_tokens,
            "known_successful_sessions": len(self._successful),
            "replay_safe_first_turns": sum(map(len, self._first_turns.values())),
            **self._counters,
        }


class InferenceBackend:
    """Hoist, key, pick an upstream, and relay the exchange verbatim."""

    provider = "inference-upstream"
    active_health_probe = True
    health_strategy = "local_control_plane"

    def __init__(
        self,
        pool: InferenceUpstreamPool,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        telemetry_sink: Callable[[str], None] | None = None,
        wait_for_home_s: float = WAIT_FOR_HOME_S,
        journal: Callable[[str], None] | None = None,
        health_probe_timeout_s: float = DEFAULT_HEALTH_PROBE_TIMEOUT_S,
        health_probe: Callable[[str, float], Awaitable[UpstreamReachability]]
        | None = None,
        health_cache_ttl_s: float = DEFAULT_HEALTH_CACHE_TTL_S,
        health_failure_threshold: int = DEFAULT_HEALTH_FAILURE_THRESHOLD,
        continuation_qos_enabled: bool = DEFAULT_CONTINUATION_QOS_ENABLED,
        continuation_qos_max_retries: int = DEFAULT_CONTINUATION_QOS_MAX_RETRIES,
        continuation_qos_min_preempt_tokens: int = (
            DEFAULT_CONTINUATION_QOS_MIN_PREEMPT_TOKENS
        ),
    ) -> None:
        self.pool = pool
        self.timeout_s = timeout_s
        self.wait_for_home_s = wait_for_home_s
        # The request journal ([relay] lines) is separate from the opt-in
        # prefix telemetry: it carries no payload and it is the only record
        # of which request went to which upstream, so the CLI always wires
        # it. Falls back to the telemetry sink so tests capture both.
        self.journal = journal
        # ``None`` means the prefix telemetry is off. The library never picks
        # an output on its own; the CLI hands in stdout when the env asks.
        self.telemetry_sink = telemetry_sink
        if health_probe_timeout_s <= 0:
            raise ValueError("health_probe_timeout_s must be > 0")
        if health_cache_ttl_s < 0:
            raise ValueError("health_cache_ttl_s must be >= 0")
        if health_failure_threshold < 1:
            raise ValueError("health_failure_threshold must be >= 1")
        self.health_probe_timeout_s = health_probe_timeout_s
        self._health_probe = health_probe
        self.health_cache_ttl_s = health_cache_ttl_s
        self.health_failure_threshold = health_failure_threshold
        self._health_probe_lock = asyncio.Lock()
        self._health_probe_task: asyncio.Task[list[UpstreamReachability]] | None = None
        self._health_cache: tuple[float, list[UpstreamReachability]] | None = None
        self._health_failures = {
            upstream.alias: 0 for upstream in self.pool.upstreams
        }
        # SAC/Hermes currently supplies stable identity but no authoritative
        # pre-admission cache-residency result. Record UNKNOWN observations;
        # do not activate cache-priority scheduling from prompt size or history.
        self.cache_admission = AdmissionController()
        self.continuation_qos = ContinuationQoS(
            enabled=continuation_qos_enabled,
            max_retries=continuation_qos_max_retries,
            min_preempt_tokens=continuation_qos_min_preempt_tokens,
        )
        self._cleanup_reapers: set[asyncio.Task[None]] = set()

    async def probe_upstreams(self) -> list[UpstreamReachability]:
        """Return one coalesced, briefly cached local-control-plane observation."""

        async with self._health_probe_lock:
            now = time.monotonic()
            if self._health_cache is not None and self._health_cache[0] > now:
                return self._health_cache[1]
            if self._health_probe_task is None:
                self._health_probe_task = asyncio.create_task(self._probe_upstreams_fresh())
            task = self._health_probe_task
        try:
            observations = await asyncio.shield(task)
        finally:
            async with self._health_probe_lock:
                if self._health_probe_task is task and task.done():
                    if not task.cancelled() and task.exception() is None:
                        self._health_cache = (
                            time.monotonic() + self.health_cache_ttl_s,
                            task.result(),
                        )
                    self._health_probe_task = None
        return observations

    async def _probe_upstreams_fresh(self) -> list[UpstreamReachability]:
        """Run one bounded probe generation and reconcile authoritative success."""

        async def run(url: str) -> UpstreamReachability:
            started = time.monotonic()

            async def request_observation() -> UpstreamReachability:
                if self._health_probe is not None:
                    return await self._health_probe(url, self.health_probe_timeout_s)
                return await probe_upstream(
                    url, timeout_s=self.health_probe_timeout_s
                )

            try:
                return await asyncio.wait_for(
                    request_observation(), timeout=self.health_probe_timeout_s
                )
            except asyncio.TimeoutError:
                return timed_out_reachability(started, monotonic=time.monotonic)

        snapshot = await self.pool.cooldown_snapshot()
        raw_observations = list(
            await asyncio.gather(
                *(run(upstream.base_url) for upstream in self.pool.upstreams)
            )
        )
        await self.pool.reconcile_recovered(snapshot, raw_observations)
        observations = []
        for upstream, observed in zip(
            self.pool.upstreams, raw_observations, strict=True
        ):
            failures = 0 if observed.reachable else self._health_failures[upstream.alias] + 1
            self._health_failures[upstream.alias] = failures
            ready = observed.reachable or failures < self.health_failure_threshold
            observations.append(
                replace(
                    observed,
                    ready=ready,
                    consecutive_failures=failures,
                )
            )
        return observations

    def _note(self, line: str) -> None:
        """One journal line. Never affects the request path (see prepare)."""
        sink = self.journal or self.telemetry_sink
        if sink is None:
            return
        try:
            sink(line)
        except Exception:  # noqa: BLE001
            pass

    def prepare(
        self,
        body: bytes | None,
        *,
        hoist: bool = True,
        affinity_key: str = "",
        inject_session_id: bool = True,
    ) -> tuple[bytes | None, str | None]:
        """Derive the sticky key, hoisting the body only where the shape asks.

        Pure apart from the sink. ``hoist`` is the route's verdict (see
        :func:`hoists_on`): the Anthropic Messages route hoists, the OpenAI
        routes forward the bytes untouched. A caller-declared ``affinity_key``
        wins over the body heuristic; the heuristic remains the compatibility
        path for clients that cannot attach an identity header. The key has
        already been normalized by :func:`request_session_key`, so no raw
        request identity reaches telemetry or the journal.
        """
        key = affinity_key or None
        if body:
            try:
                payload = json.loads(body)
                hoisted = 0
                if hoist:
                    payload, hoisted = hoist_system(payload)
                else:
                    payload, adapted = adapt_openai_roles(payload)
                    payload, repaired = repair_tool_call_arguments(payload)
                    if repaired:
                        self._note(
                            f"[relay] repaired {repaired} tool-call argument "
                            "string(s) that were not JSON (kept under "
                            "_invalid_arguments)"
                        )
                    hoisted = int(adapted) + repaired
                if key is None:
                    key = conversation_key(payload)
                # SGLang's session-aware radix cache consumes a TOP-LEVEL
                # ``session_id``.  Only an explicit caller identity is strong
                # enough to label cache ownership: the body heuristic remains
                # useful for replica affinity, but is not an application
                # session contract.  Replace any caller-provided body value
                # with the same bounded opaque digest used for stickiness, so
                # raw identities never cross the gateway boundary.
                session_injected = False
                if inject_session_id and affinity_key and isinstance(payload, dict):
                    payload["session_id"] = affinity_key
                    session_injected = True
                if self.telemetry_sink is not None:
                    # Telemetry must NEVER affect the request path. A broad
                    # except is deliberate: any failure here is a lost
                    # measurement, and a dropped agent request is an outage.
                    try:
                        report = prefix_report(payload, key)
                        if report:
                            self.telemetry_sink(f"[prefix] {report}")
                    except Exception:  # noqa: BLE001
                        pass
                if hoisted or session_injected:
                    body = json.dumps(payload).encode()
            except (ValueError, AttributeError, TypeError):
                pass  # not JSON we understand - forward untouched, never drop
        return body, key

    async def relay(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None,
        headers: Mapping[str, str],
        upstream_path: str | None = None,
        client_disconnected: Callable[[], Awaitable[bool]] | None = None,
    ) -> RelayedResponse:
        """Forward one request; return the upstream's reply as it streams in.

        Only a transport-level failure (no HTTP response at all) rotates to
        the next upstream. An HTTP response of ANY status is the answer and is
        returned verbatim, exactly as the script did.
        """
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "Inference relay requires scitex-genai[gateway]"
            ) from exc

        explicit_session = request_session_key(headers)
        qos_session = continuation_qos_session_key(headers)
        body, session = self.prepare(
            body,
            hoist=hoists_on(path),
            affinity_key=explicit_session,
            inject_session_id=accepts_session_id(path),
        )
        routing_session = session or ""
        qos_kind = (
            self.continuation_qos.classify(qos_session)
            if self.continuation_qos.enabled
            else "disabled"
        )
        self.cache_admission.observe(CacheResidency.UNKNOWN)
        feedback_headers = {
            "x-scitex-admission-mode": "observe-only",
            "x-scitex-cache-residency": CacheResidency.UNKNOWN.value,
            "x-scitex-session-key": (session or "")[:12] or "none",
        }
        input_tokens = estimate_input_tokens(body)
        forwarded = {
            name: value
            for name, value in headers.items()
            if name.lower() not in _HOP_BY_HOP
        }
        attempted: set[str] = set()
        failures: list[str] = []
        waited = 0.0
        replay_count = 0
        continuation_handoff: _ContinuationHandoff | None = None
        handed_to_stream = False
        try:
            if qos_kind == "continuation":
                target_alias = await self.pool.route_alias(routing_session)
                continuation_handoff = self.continuation_qos.request_preemption(
                    target_alias
                )
                if continuation_handoff is not None:
                    # Do not overlap at the engine: capacity=2 means gateway
                    # admission alone cannot tell that the continuation is
                    # queued behind a long prefill inside SGLang.
                    await continuation_handoff.victim_released
            while len(attempted) < len(self.pool.upstreams):
                try:
                    upstream = await self._acquire_while_connected(
                        routing_session,
                        exclude=attempted,
                        input_tokens=input_tokens,
                        priority=qos_kind == "continuation",
                        client_disconnected=client_disconnected,
                    )
                except _ClientDisconnected:
                    self._note(
                        f"[relay] conv={routing_session[:8] or '-'} <- queue "
                        "client_disconnected_before_admission"
                    )
                    return RelayedResponse(
                        status_code=499,
                        content_type="application/json",
                        body=_empty_body(),
                        feedback_headers=feedback_headers,
                    )
                except HomeMemberReloading as exc:
                    if waited < self.wait_for_home_s:
                        # Wait it out here rather than hand the caller a 503: the
                        # home is reloading, the request stays open, and the
                        # upstream is tried again as soon as its cooldown lapses.
                        # Time passed, so an upstream that gave no response is a
                        # candidate again.
                        slice_s = min(max(exc.retry_after_s, 0.1), WAIT_SLICE_S)
                        self._note(
                            f"[relay] conv={routing_session[:8] or '-'} waiting {slice_s:.0f}s "
                            f"for its home (waited {waited:.0f}s of "
                            f"{self.wait_for_home_s:.0f}s): {exc}"
                        )
                        await asyncio.sleep(slice_s)
                        waited += slice_s
                        attempted.clear()
                        continue
                    self._note(
                        f"[relay] conv={routing_session[:8] or '-'} held: {exc} "
                        f"(retry after {exc.retry_after_s:.0f}s; waited {waited:.0f}s)"
                    )
                    raise UpstreamReloading(
                        f"{exc}. Retry this conversation after "
                        f"{exc.retry_after_s:.0f}s; it stays pinned to its home "
                        f"upstream while that upstream reloads.",
                        retry_after_s=exc.retry_after_s,
                    ) from exc
                except NoAccountAvailable as exc:
                    failures.append(str(exc))
                    break
                attempted.add(upstream.alias)
                started = time.monotonic()
                self._note(
                    f"[relay] conv={routing_session[:8] or '-'} -> {upstream.alias} "
                    f"{method} {path} bytes={len(body or b'')} "
                    f"estimated_input_tokens={input_tokens} "
                    f"admitted_input_tokens={upstream.input_tokens_in_flight}"
                )
                dispatch_request_id = (
                    secrets.token_urlsafe(24)
                    if self.continuation_qos.enabled and accepts_session_id(path)
                    else ""
                )
                dispatch_body, rid_confirmed = inject_request_id(
                    body, dispatch_request_id
                )
                client = httpx.AsyncClient(timeout=self.timeout_s)
                slot_owned = True

                async def release_slot() -> None:
                    nonlocal slot_owned
                    if not slot_owned:
                        return
                    # Transfer ownership before awaiting. If this task is
                    # cancelled during shield, the one release keeps running
                    # and no cleanup path can decrement a later request.
                    slot_owned = False
                    task = asyncio.create_task(
                        self.pool.release(
                            upstream,
                            input_tokens=input_tokens,
                            session_id=routing_session,
                        )
                    )
                    await asyncio.shield(task)

                send_task: asyncio.Task[Any] | None = None
                preempt_task: asyncio.Task[Any] | None = None
                disconnect_task: asyncio.Task[None] | None = None
                disconnect_stop: asyncio.Event | None = None
                first_chunk_task: asyncio.Task[bytes] | None = None
                stream: AsyncIterator[bytes] | None = None
                first_attempt = None
                if (
                    qos_kind == "first-turn"
                    and rid_confirmed
                    and input_tokens >= self.continuation_qos.min_preempt_tokens
                    and replay_count < self.continuation_qos.max_retries
                ):
                    first_attempt = self.continuation_qos.register_first_turn(
                        upstream.alias
                    )
                try:
                    request = client.build_request(
                        method,
                        upstream.base_url + (upstream_path or path),
                        content=dispatch_body,
                        headers=forwarded,
                    )
                    if self.continuation_qos.enabled:
                        send_task = asyncio.create_task(
                            client.send(request, stream=True)
                        )
                    if rid_confirmed and client_disconnected is not None:
                        disconnect_stop = asyncio.Event()
                        disconnect_task = asyncio.create_task(
                            self._wait_for_client_disconnect(
                                client_disconnected, stop=disconnect_stop
                            )
                        )
                    if first_attempt is not None:
                        preempt_task = asyncio.create_task(first_attempt.preempt.wait())
                    waiters = tuple(
                        task
                        for task in (send_task, preempt_task, disconnect_task)
                        if task is not None
                    )
                    if waiters:
                        done, _ = await asyncio.wait(
                            waiters,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        # Upstream headers end preemption eligibility, but do
                        # not prove that a streaming body exists. Keep the
                        # downstream monitor alive until the first body byte.
                        if send_task is not None and send_task in done:
                            for task in (preempt_task,):
                                if task is not None:
                                    task.cancel()
                            await asyncio.gather(
                                *(task for task in (preempt_task,) if task is not None),
                                return_exceptions=True,
                            )
                            if (
                                first_attempt is not None
                                and first_attempt.preempt.is_set()
                                and first_attempt.released is not None
                                and not first_attempt.released.done()
                            ):
                                first_attempt.released.set_result(None)
                            response = await send_task
                        elif disconnect_task is not None and disconnect_task in done:
                            raise _ClientDisconnected
                        else:
                            abort_ok = await self._abort_request(
                                upstream, dispatch_request_id, headers=forwarded
                            )
                            if not abort_ok:
                                if (
                                    first_attempt.released is not None
                                    and not first_attempt.released.done()
                                ):
                                    first_attempt.released.set_exception(
                                        InferenceAdmissionError(
                                            "Continuation QoS could not confirm upstream abort"
                                        )
                                    )
                                response = await send_task
                            else:
                                send_task.cancel()
                                await asyncio.gather(send_task, return_exceptions=True)
                                self.continuation_qos.preempted()
                                self._note(
                                    f"[relay] conv={routing_session[:8] or '-'} <- "
                                    f"{upstream.alias} cooperatively_preempted_before_response"
                                )
                                await asyncio.shield(client.aclose())
                                await release_slot()
                                if (
                                    first_attempt.released is not None
                                    and not first_attempt.released.done()
                                ):
                                    first_attempt.released.set_result(None)
                                barrier = first_attempt.resume_after
                                self.continuation_qos.unregister_first_turn(
                                    first_attempt
                                )
                                first_attempt = None
                                if barrier is not None:
                                    await barrier
                                replay_count += 1
                                self.continuation_qos.retried()
                                if replay_count >= self.continuation_qos.max_retries:
                                    self.continuation_qos.exhausted()
                                attempted.clear()
                                continue
                    else:
                        response = await client.send(request, stream=True)

                    # Do not commit an HTTP 200 to the caller merely because
                    # the upstream sent response headers. SGLang has been
                    # observed returning headers and then orphaning the body
                    # stream. Prime one body chunk while the ASGI disconnect
                    # monitor can still abort the request deterministically.
                    stream = response.aiter_bytes()
                    first_chunk_task = asyncio.create_task(anext(stream))
                    if disconnect_task is not None:
                        done, _ = await asyncio.wait(
                            (first_chunk_task, disconnect_task),
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if first_chunk_task not in done:
                            raise _ClientDisconnected
                        disconnect_stop.set()
                        await asyncio.gather(disconnect_task, return_exceptions=True)
                    try:
                        first_chunk = await first_chunk_task
                    except StopAsyncIteration:
                        first_chunk = None
                except (_ClientDisconnected, asyncio.CancelledError) as exc:
                    # Cancellation before a response body exists must not leak a
                    # capacity slot; streaming cancellation is handled by _drain.
                    observed_disconnect = isinstance(exc, _ClientDisconnected)
                    outcome = (
                        "client_disconnected_before_response"
                        if observed_disconnect
                        else "relay_cancelled_before_response"
                    )
                    self._note(
                        f"[relay] conv={routing_session[:8] or '-'} <- {upstream.alias} "
                        f"{outcome} "
                        f"after {time.monotonic() - started:.1f}s"
                    )
                    abort_ok = False
                    safe_to_release = not self.continuation_qos.enabled
                    if rid_confirmed:
                        abort_ok = await asyncio.shield(
                            self._abort_request(
                                upstream, dispatch_request_id, headers=forwarded
                            )
                        )
                    if abort_ok and send_task is not None and not send_task.done():
                        send_task.cancel()
                        safe_to_release = True
                    elif abort_ok:
                        safe_to_release = True
                    elif send_task is not None:
                        # Without a confirmed engine abort, retain the gateway
                        # slot until the upstream really finishes.
                        try:
                            cancelled_response = (
                                send_task.result()
                                if send_task.done()
                                else await asyncio.shield(send_task)
                            )
                            cancelled_stream = (
                                stream or cancelled_response.aiter_bytes()
                            )
                            if first_chunk_task is not None:
                                try:
                                    await asyncio.shield(first_chunk_task)
                                except StopAsyncIteration:
                                    pass
                            async for _ in cancelled_stream:
                                pass
                            await cancelled_response.aclose()
                            safe_to_release = True
                        except BaseException:  # task/transport state is unknown
                            safe_to_release = False
                    if preempt_task is not None and not preempt_task.done():
                        preempt_task.cancel()
                    if disconnect_task is not None and not disconnect_task.done():
                        disconnect_stop.set()
                        disconnect_task.cancel()
                    if (
                        abort_ok
                        and first_chunk_task is not None
                        and not first_chunk_task.done()
                    ):
                        first_chunk_task.cancel()
                    pending = [
                        task
                        for task in (
                            send_task,
                            preempt_task,
                            disconnect_task,
                            first_chunk_task,
                        )
                        if task is not None
                    ]
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)
                    await asyncio.shield(client.aclose())
                    if safe_to_release:
                        await release_slot()
                    elif rid_confirmed:
                        slot_owned = False  # ownership transfers to the reaper
                        self._schedule_cleanup_reaper(
                            upstream,
                            dispatch_request_id,
                            headers=forwarded,
                            input_tokens=input_tokens,
                            session_id=routing_session,
                        )
                    if (
                        first_attempt is not None
                        and first_attempt.preempt.is_set()
                        and first_attempt.released is not None
                        and not first_attempt.released.done()
                    ):
                        first_attempt.released.set_exception(
                            InferenceAdmissionError(
                                "Preempted first-turn client disconnected"
                            )
                        )
                    if observed_disconnect:
                        return RelayedResponse(
                            status_code=499,
                            content_type="application/json",
                            body=_empty_body(),
                            feedback_headers=feedback_headers,
                        )
                    raise
                except httpx.TransportError as exc:
                    await client.aclose()
                    safe_to_release = not rid_confirmed
                    if rid_confirmed:
                        safe_to_release = await self._abort_request(
                            upstream, dispatch_request_id, headers=forwarded
                        )
                    if safe_to_release:
                        await release_slot()
                    else:
                        slot_owned = False  # ownership transfers to the reaper
                        self._schedule_cleanup_reaper(
                            upstream,
                            dispatch_request_id,
                            headers=forwarded,
                            input_tokens=input_tokens,
                            session_id=routing_session,
                        )
                        raise InferenceAdmissionError(
                            "Upstream request state is unknown after transport "
                            "failure; capacity remains reserved because abort "
                            "could not be confirmed"
                        ) from exc
                    await self.pool.cool_down(upstream, UNREACHABLE_COOLDOWN_S)
                    self._note(
                        f"[relay] conv={routing_session[:8] or '-'} <- {upstream.alias} "
                        f"no response ({exc.__class__.__name__}) after "
                        f"{time.monotonic() - started:.1f}s; out of rotation for "
                        f"{UNREACHABLE_COOLDOWN_S:.0f}s"
                    )
                    failures.append(
                        f"{upstream.alias} ({exc.__class__.__name__}: {exc})"
                    )
                    continue
                finally:
                    if disconnect_task is not None:
                        if not disconnect_task.done():
                            disconnect_stop.set()
                            disconnect_task.cancel()
                        await asyncio.gather(disconnect_task, return_exceptions=True)
                    if first_attempt is not None:
                        if (
                            first_attempt.preempt.is_set()
                            and first_attempt.released is not None
                            and not first_attempt.released.done()
                        ):
                            first_attempt.released.set_exception(
                                InferenceAdmissionError(
                                    "First-turn handoff did not complete"
                                )
                            )
                        self.continuation_qos.unregister_first_turn(first_attempt)
                handed_to_stream = True
                return RelayedResponse(
                    status_code=response.status_code,
                    content_type=response.headers.get(
                        "content-type", "application/json"
                    ),
                    feedback_headers=feedback_headers,
                    body=self._drain(
                        client,
                        response,
                        upstream,
                        stream=stream,
                        first_chunk=first_chunk,
                        tag=f"conv={routing_session[:8] or '-'} <- {upstream.alias} "
                        f"status={response.status_code}",
                        started=started,
                        input_tokens=input_tokens,
                        session_id=routing_session,
                        continuation_handoff=continuation_handoff,
                        request_id=dispatch_request_id if rid_confirmed else "",
                        request_headers=forwarded,
                        successful_session=(
                            qos_session if 200 <= response.status_code < 300 else ""
                        ),
                    ),
                )
            raise UpstreamUnreachable(self._refusal(failures))
        finally:
            if not handed_to_stream:
                self.continuation_qos.finish_continuation(continuation_handoff)

    @staticmethod
    async def _wait_for_client_disconnect(
        client_disconnected: Callable[[], Awaitable[bool]],
        *,
        stop: asyncio.Event | None = None,
    ) -> None:
        """Poll Starlette's non-blocking ASGI disconnect observation.

        The request body has already been consumed before this starts, so the
        receive channel is now used only for ``http.disconnect``.  A failed
        observation must not kill a healthy inference request; it is retried
        until headers arrive and the monitor is cancelled.
        """
        while stop is None or not stop.is_set():
            try:
                if await client_disconnected():
                    return
            except Exception:  # noqa: BLE001
                pass
            if stop is None:
                await asyncio.sleep(0.05)
            else:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=0.05)
                except TimeoutError:
                    pass

    async def _acquire_while_connected(
        self,
        session_id: str,
        *,
        exclude: set[str],
        input_tokens: int,
        priority: bool,
        client_disconnected: Callable[[], Awaitable[bool]] | None,
    ) -> InferenceUpstream:
        """Remove a queued admission ticket as soon as its caller disappears."""
        acquire = asyncio.create_task(
            self.pool.acquire(
                session_id,
                exclude=exclude,
                input_tokens=input_tokens,
                priority=priority,
            )
        )
        if client_disconnected is None:
            return await acquire
        stop = asyncio.Event()
        disconnected = asyncio.create_task(
            self._wait_for_client_disconnect(client_disconnected, stop=stop)
        )
        try:
            done, _ = await asyncio.wait(
                (acquire, disconnected), return_when=asyncio.FIRST_COMPLETED
            )
            if acquire in done:
                return await acquire
            acquire.cancel()
            await asyncio.gather(acquire, return_exceptions=True)
            raise _ClientDisconnected
        finally:
            if not disconnected.done():
                stop.set()
                disconnected.cancel()
            await asyncio.gather(disconnected, return_exceptions=True)

    async def _abort_request(
        self,
        upstream: InferenceUpstream,
        request_id: str,
        *,
        headers: Mapping[str, str],
    ) -> bool:
        """Ask pinned SGLang to remove a request before closing its transport."""
        if not request_id:
            return False
        try:
            import httpx

            async with httpx.AsyncClient(timeout=min(self.timeout_s, 5.0)) as client:
                response = await client.post(
                    upstream.base_url + "/abort_request",
                    json={"rid": request_id},
                    headers=headers,
                )
            if 200 <= response.status_code < 300:
                return True
            detail = f"HTTP {response.status_code}"
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            detail = exc.__class__.__name__
        self.continuation_qos.abort_failed()
        self._note(
            f"[relay] rid abort failed on {upstream.alias}: {detail}; "
            "continuation not dispatched"
        )
        return False

    def _schedule_cleanup_reaper(
        self,
        upstream: InferenceUpstream,
        request_id: str,
        *,
        headers: Mapping[str, str],
        input_tokens: int,
        session_id: str,
    ) -> None:
        """Retain admission and retry abort until engine cleanup is confirmed."""
        self.continuation_qos.cleanup_held()

        async def reap() -> None:
            delay_s = 0.1
            try:
                while True:
                    await asyncio.sleep(delay_s)
                    self.continuation_qos.cleanup_reaper_attempted()
                    if await self._abort_request(upstream, request_id, headers=headers):
                        release = asyncio.create_task(
                            self.pool.release(
                                upstream,
                                input_tokens=input_tokens,
                                session_id=session_id,
                            )
                        )
                        await asyncio.shield(release)
                        self.continuation_qos.cleanup_recovered()
                        return
                    delay_s = min(5.0, delay_s * 2)
            except asyncio.CancelledError:
                self.continuation_qos.cleanup_reaper_cancelled()
                raise

        task = asyncio.create_task(reap())
        self._cleanup_reapers.add(task)
        task.add_done_callback(self._cleanup_reapers.discard)

    def _refusal(self, failures: list[str]) -> str:
        urls = ", ".join(upstream.alias for upstream in self.pool.upstreams)
        return (
            "No inference upstream answered: "
            + "; ".join(failures)
            + f". Configured inference upstreams ({len(self.pool.upstreams)}): {urls}."
            + " An upstream that produced no response is out of rotation for"
            f" {UNREACHABLE_COOLDOWN_S:.0f} s."
        )

    async def _drain(
        self,
        client: Any,
        response: Any,
        upstream: InferenceUpstream,
        *,
        stream: AsyncIterator[bytes] | None = None,
        first_chunk: bytes | None = None,
        tag: str = "",
        started: float | None = None,
        input_tokens: int = 0,
        session_id: str = "",
        continuation_handoff: _ContinuationHandoff | None = None,
        successful_session: str = "",
        request_id: str = "",
        request_headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[bytes]:
        sent = 0
        outcome = "complete"
        stream = stream or response.aiter_bytes()
        next_chunk: asyncio.Task[bytes] | None = None
        release_capacity = True

        async def drain_to_eof() -> None:
            nonlocal next_chunk
            while True:
                if next_chunk is None:
                    next_chunk = asyncio.create_task(anext(stream))
                try:
                    await asyncio.shield(next_chunk)
                except StopAsyncIteration:
                    next_chunk = None
                    return
                next_chunk = None

        try:
            if first_chunk is not None:
                sent += len(first_chunk)
                yield first_chunk
            while True:
                next_chunk = asyncio.create_task(anext(stream))
                try:
                    chunk = await asyncio.shield(next_chunk)
                except StopAsyncIteration:
                    next_chunk = None
                    break
                next_chunk = None
                sent += len(chunk)
                yield chunk
            self.continuation_qos.mark_successful(successful_session)
        except (asyncio.CancelledError, GeneratorExit):
            outcome = "client_disconnected"
            aborted = False
            if request_id:
                aborted = await asyncio.shield(
                    self._abort_request(
                        upstream,
                        request_id,
                        headers=request_headers or {},
                    )
                )
            if not aborted:
                # The pinned engine cannot be trusted to notice transport
                # closure. Keep capacity until clean EOF when explicit abort
                # could not be confirmed.
                release_capacity = False
                await drain_to_eof()
                release_capacity = True
            raise
        except BaseException:
            outcome = "stream_error"
            if request_id:
                release_capacity = await asyncio.shield(
                    self._abort_request(
                        upstream,
                        request_id,
                        headers=request_headers or {},
                    )
                )
            raise
        finally:
            # ``shield`` deliberately leaves the upstream read running when
            # its caller is cancelled.  Settle that task on *every* exit,
            # including the race where it has already completed with
            # StopAsyncIteration before the cancellation branch observes it.
            # Merely skipping cancellation for a done task is insufficient:
            # its result still has to be retrieved to avoid an unhandled task
            # exception after the response generator is collected.
            if next_chunk is not None:
                if not next_chunk.done():
                    next_chunk.cancel()
                await asyncio.gather(next_chunk, return_exceptions=True)
                next_chunk = None
            if tag:
                took = time.monotonic() - started if started is not None else 0.0
                self._note(f"[relay] {tag} outcome={outcome} bytes={sent} {took:.1f}s")
            # Close transport on every path. Capacity is released only after
            # clean EOF or confirmed abort; ambiguous engine state transfers
            # ownership to the tracked background abort reaper.
            await asyncio.shield(
                self._finish(
                    client,
                    response,
                    upstream,
                    input_tokens=input_tokens,
                    session_id=session_id,
                    release_capacity=release_capacity,
                    request_id=request_id,
                    request_headers=request_headers or {},
                )
            )
            self.continuation_qos.finish_continuation(continuation_handoff)

    async def _finish(
        self,
        client: Any,
        response: Any,
        upstream: InferenceUpstream,
        *,
        input_tokens: int = 0,
        session_id: str = "",
        release_capacity: bool = True,
        request_id: str = "",
        request_headers: Mapping[str, str] | None = None,
    ) -> None:
        try:
            await response.aclose()
            await client.aclose()
        finally:
            if release_capacity:
                await self.pool.release(
                    upstream,
                    input_tokens=input_tokens,
                    session_id=session_id,
                )
            else:
                self._schedule_cleanup_reaper(
                    upstream,
                    request_id,
                    headers=request_headers or {},
                    input_tokens=input_tokens,
                    session_id=session_id,
                )

    async def close(self) -> None:
        """Stop admission and wake requests waiting for capacity."""
        tasks = tuple(self._cleanup_reapers)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self.pool.close()
