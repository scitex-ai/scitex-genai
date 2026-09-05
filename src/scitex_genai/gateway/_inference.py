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
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ._errors import (
    HomeMemberReloading,
    NoAccountAvailable,
    UpstreamReloading,
    UpstreamUnreachable,
)
from ._pool import StickyPool

#: The fleet's systemd drop-ins set these; the names are kept so they keep
#: working unchanged. Comma-separated base URLs, seconds, and a truthy flag.
UPSTREAM_ENV = "HOIST_UPSTREAM"
TIMEOUT_ENV = "HOIST_TIMEOUT_S"
PREFIX_TELEMETRY_ENV = "HOIST_PREFIX_TELEMETRY"
DEFAULT_TIMEOUT_S = 600.0

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

#: Never forwarded. The script dropped the first three; ``transfer-encoding``
#: joins them because the server has already de-chunked the body it hands us.
_HOP_BY_HOP = frozenset({"host", "content-length", "connection", "transfer-encoding"})

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
    cooldown_until: float = 0.0
    #: When this upstream last went out of rotation (None = healthy).
    cooling_since: float | None = None

    @property
    def base_url(self) -> str:
        return self.alias

    @property
    def usage_score(self) -> float:
        """No quota notion: the pool assumes interchangeable upstreams."""
        return 0.0


class InferenceUpstreamPool(StickyPool[InferenceUpstream]):
    """Sticky per-conversation pool of interchangeable inference upstreams.

    The scheduling is :class:`~._pool.StickyPool`'s. What this adds is the
    script's FIRST-PLACEMENT policy: a conversation with an unseen key is
    assigned by ROUND ROBIN. That is the right policy and not a legacy
    accident — at idle every in-flight count is 0, so a least-loaded rule
    resolving on an index tiebreak would send every new conversation to the
    first upstream. In-flight then only breaks ties among non-sticky
    upstreams, which in practice is the key-less path.

    THE POOL ASSUMES UPSTREAMS ARE INTERCHANGEABLE. Measured 2026-08-16: an
    A100 ~8.7x slower than the H100s drained its queue and therefore kept
    LOOKING least-loaded, carrying more traffic than any H100. Exclude the odd
    member from the configuration; do not tune the policy.
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
    ) -> None:
        self._next_placement = 0
        super().__init__(
            upstreams,
            choose=choose or self._round_robin,
            max_sessions=max_sessions,
            failover_after_s=HOME_FAILOVER_AFTER_S,
        )

    @property
    def upstreams(self) -> list[InferenceUpstream]:
        return self.members

    @classmethod
    def from_urls(cls, urls: str | list[str]) -> "InferenceUpstreamPool":
        """Build from the ``HOIST_UPSTREAM`` string or an already-split list."""
        if isinstance(urls, str):
            urls = parse_upstreams(urls)
        return cls([InferenceUpstream(alias=url) for url in urls])

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


class InferenceBackend:
    """Hoist, key, pick an upstream, and relay the exchange verbatim."""

    provider = "inference-upstream"

    def __init__(
        self,
        pool: InferenceUpstreamPool,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        telemetry_sink: Callable[[str], None] | None = None,
    ) -> None:
        self.pool = pool
        self.timeout_s = timeout_s
        # ``None`` means the prefix telemetry is off. The library never picks
        # an output on its own; the CLI hands in stdout when the env asks.
        self.telemetry_sink = telemetry_sink

    def prepare(
        self, body: bytes | None, *, hoist: bool = True
    ) -> tuple[bytes | None, str]:
        """Derive the sticky key, hoisting the body only where the shape asks.

        Pure apart from the sink. ``hoist`` is the route's verdict (see
        :func:`hoists_on`): the Anthropic Messages route hoists, the OpenAI
        routes forward the bytes untouched.
        """
        key = None
        if body:
            try:
                payload = json.loads(body)
                hoisted = 0
                if hoist:
                    payload, hoisted = hoist_system(payload)
                else:
                    payload, adapted = adapt_openai_roles(payload)
                    hoisted = int(adapted)
                key = conversation_key(payload)
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
                if hoisted:
                    body = json.dumps(payload).encode()
            except (ValueError, AttributeError, TypeError):
                pass  # not JSON we understand - forward untouched, never drop
        return body, key or ""

    async def relay(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None,
        headers: Mapping[str, str],
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

        body, session = self.prepare(body, hoist=hoists_on(path))
        forwarded = {
            name: value
            for name, value in headers.items()
            if name.lower() not in _HOP_BY_HOP
        }
        attempted: set[str] = set()
        failures: list[str] = []
        while len(attempted) < len(self.pool.upstreams):
            try:
                upstream = await self.pool.acquire(session, exclude=attempted)
            except HomeMemberReloading as exc:
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
            client = httpx.AsyncClient(timeout=self.timeout_s)
            try:
                request = client.build_request(
                    method, upstream.base_url + path, content=body, headers=forwarded
                )
                response = await client.send(request, stream=True)
            except httpx.TransportError as exc:
                await client.aclose()
                await self.pool.release(upstream)
                await self.pool.cool_down(upstream, UNREACHABLE_COOLDOWN_S)
                failures.append(f"{upstream.alias} ({exc.__class__.__name__}: {exc})")
                continue
            return RelayedResponse(
                status_code=response.status_code,
                content_type=response.headers.get("content-type", "application/json"),
                body=self._drain(client, response, upstream),
            )
        raise UpstreamUnreachable(self._refusal(failures))

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
        self, client: Any, response: Any, upstream: InferenceUpstream
    ) -> AsyncIterator[bytes]:
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        finally:
            # ALWAYS, on every path — including a client that disconnected
            # mid-stream, which cancels this generator. An unreleased counter
            # marks that upstream busy FOREVER, so the balancer would route
            # away from a card that is actually idle: a leak that degrades into
            # the exact pile-up this routing exists to prevent, and one that
            # gets worse with every failed request. Shielded because the
            # server's cancel scope re-cancels at every await, and the release
            # must finish even after the response is gone.
            await asyncio.shield(self._finish(client, response, upstream))

    async def _finish(
        self, client: Any, response: Any, upstream: InferenceUpstream
    ) -> None:
        try:
            await response.aclose()
            await client.aclose()
        finally:
            await self.pool.release(upstream)
