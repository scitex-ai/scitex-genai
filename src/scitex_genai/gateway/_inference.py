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
import math
import secrets
import time
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ._admission import (
    AdmissionController,
    CacheAdmissionSettings,
    CacheResidency,
    classify_cache_prediction,
)
from ._errors import (
    HomeMemberReloading,
    InferenceAdmissionError,
    InferenceMemberUnavailable,
    NoAccountAvailable,
    UpstreamReloading,
    UpstreamUnreachable,
)
from ._health import (
    DEFAULT_HEALTH_PROBE_TIMEOUT_S,
    UpstreamReachability,
    probe_upstream,
    public_upstream_url,
    timed_out_reachability,
)
from ._member_quiesce import (
    InferenceMemberQuiesceState,
    InferenceMemberQuiesceTimeout,
    InferenceMemberResumeError,
)
from ._pool import StickyPool
from ._prediction import AdmissionPrediction, AdmissionPredictionTelemetry
from ._request_observability import (
    AGENT_ID_HEADER,
    RequestLifecycleRegistry,
    RequestObservation,
    request_agent_label,
    request_session_label,
)
from ._session_state import GatewaySessionState
from ._sglang_metrics import SGLangSchedulerObservation, probe_sglang_metrics

#: The fleet's systemd drop-ins set these; the names are kept so they keep
#: working unchanged. Comma-separated base URLs, seconds, and a truthy flag.
TIMEOUT_ENV = "HOIST_TIMEOUT_S"
PREFIX_TELEMETRY_ENV = "HOIST_PREFIX_TELEMETRY"
DEFAULT_TIMEOUT_S = 600.0
DEFAULT_METADATA_TIMEOUT_S = 10.0
DEFAULT_CAPACITY_PER_UPSTREAM = 8
DEFAULT_MAX_QUEUE_SIZE = 128
DEFAULT_TOKEN_CAPACITY_PER_UPSTREAM: int | None = None
DEFAULT_HEALTH_CACHE_TTL_S = 1.0
DEFAULT_HEALTH_FAILURE_THRESHOLD = 2
DEFAULT_CONTINUATION_QOS_ENABLED = False
DEFAULT_CONTINUATION_QOS_MAX_RETRIES = 1
DEFAULT_CONTINUATION_QOS_MIN_PREEMPT_TOKENS = 0
DEFAULT_MAX_ADMISSION_BYPASSES = 4
DEFAULT_PRIORITY_AGING_S = 30.0
DEFAULT_CACHE_PREDICTION_MAX_AGE_S = 300.0
MAX_STATUS_TICKETS = 256

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
).union(_SESSION_ID_HEADERS, {AGENT_ID_HEADER})
_SESSION_KEY_DOMAIN = b"scitex-genai-session-affinity\0"
_REQUEST_PREFIX_DOMAIN = b"scitex-genai-request-prefix-v1\0"
_REQUEST_PREFIX_BYTES = 16_384

# First pass (7 conversations) showed ALL agents identical at 1k and ALL
# distinct at 4k, so the entire divergence happens in that band. These
# checkpoints bracket it finely; the coarse ones are kept so the two passes
# stay comparable.
_CHECKPOINTS = (1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096, 16384)


def parse_upstreams(value: str) -> list[str]:
    """Parse a programmatic comma-separated URL list for legacy/external pools."""
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


def request_prefix_fingerprint(body: bytes | None) -> str:
    """Return an opaque identifier for the first 16 KiB of the relayed body.

    The bounded prefix is where changing timestamps, agent identity, or tool
    schemas destroy radix-cache reuse.  Only a domain-separated digest leaves
    this process; prompt bytes and secrets never do.  The byte limit is part of
    the fingerprint version, so equal identifiers are comparable across runs.
    """
    if not body:
        return "none"
    prefix = body[:_REQUEST_PREFIX_BYTES]
    return hashlib.sha256(_REQUEST_PREFIX_DOMAIN + prefix).hexdigest()[:16]


def inject_cache_report_request(
    body: bytes | None, path: str, *, enabled: bool
) -> tuple[bytes | None, bool]:
    """Ask a configured SGLang OpenAI endpoint for per-tier cache details.

    This is deliberately opt-in: ``return_cached_tokens_details`` is an SGLang
    extension and must not leak to an arbitrary OpenAI-compatible provider.
    Anthropic Messages is left untouched because SGLang already reports
    ``cache_read_input_tokens`` there without an extension field.
    """
    route = path.split("?", 1)[0].rstrip("/")
    if not enabled or route not in {"/v1/chat/completions", "/v1/completions"}:
        return body, False
    try:
        payload = json.loads(body or b"")
    except (TypeError, ValueError):
        return body, False
    if not isinstance(payload, dict):
        return body, False
    changed = payload.get("return_cached_tokens_details") is not True
    payload["return_cached_tokens_details"] = True
    if payload.get("stream") is True:
        options = payload.get("stream_options")
        if not isinstance(options, dict):
            options = {}
            payload["stream_options"] = options
        if options.get("include_usage") is not True:
            options["include_usage"] = True
            changed = True
    return (json.dumps(payload).encode() if changed else body), True


@dataclass
class ResponseTokenReport:
    """Payload-free token/cache facts observed in an upstream response."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    cache_creation_tokens: int | None = None
    device_cached_tokens: int | None = None
    host_cached_tokens: int | None = None
    storage_cached_tokens: int | None = None
    storage_backend: str | None = None
    _buffer: bytearray = field(default_factory=bytearray, repr=False)
    _line_buffer: bytearray = field(default_factory=bytearray, repr=False)

    def feed(self, chunk: bytes) -> None:
        """Observe JSON or SSE incrementally without retaining large output."""
        if len(self._buffer) < 1_048_576:
            room = 1_048_576 - len(self._buffer)
            self._buffer.extend(chunk[:room])
        self._line_buffer.extend(chunk)
        while b"\n" in self._line_buffer:
            raw_line, _, remainder = self._line_buffer.partition(b"\n")
            self._line_buffer = bytearray(remainder)
            line = raw_line.strip()
            if line.startswith(b"data:"):
                candidate = line[5:].strip()
                if candidate and candidate != b"[DONE]":
                    self._parse(candidate)
        if len(self._line_buffer) > 1_048_576:
            self._line_buffer.clear()

    def finish(self) -> None:
        """Parse a non-streaming JSON response after EOF, if applicable."""
        self._parse(bytes(self._buffer).strip())

    def _parse(self, candidate: bytes) -> None:
        try:
            payload = json.loads(candidate)
        except (TypeError, ValueError):
            return
        self._observe(payload)

    def _observe(self, payload: Any) -> None:
        if isinstance(payload, list):
            for item in payload:
                self._observe(item)
            return
        if not isinstance(payload, dict):
            return
        usage = payload.get("usage")
        if isinstance(usage, dict):
            self.input_tokens = _integer(
                usage.get("prompt_tokens", usage.get("input_tokens")),
                self.input_tokens,
            )
            self.output_tokens = _integer(
                usage.get("completion_tokens", usage.get("output_tokens")),
                self.output_tokens,
            )
            self.cached_tokens = _integer(
                usage.get("cache_read_input_tokens"), self.cached_tokens
            )
            self.cache_creation_tokens = _integer(
                usage.get("cache_creation_input_tokens"), self.cache_creation_tokens
            )
            details = usage.get("prompt_tokens_details") or usage.get(
                "input_tokens_details"
            )
            if isinstance(details, dict):
                self.cached_tokens = _integer(
                    details.get("cached_tokens"), self.cached_tokens
                )
        extension = payload.get("sglext")
        if isinstance(extension, dict):
            details = extension.get("cached_tokens_details")
            if isinstance(details, dict):
                self.device_cached_tokens = _integer(
                    details.get("device"), self.device_cached_tokens
                )
                self.host_cached_tokens = _integer(
                    details.get("host"), self.host_cached_tokens
                )
                self.storage_cached_tokens = _integer(
                    details.get("storage"), self.storage_cached_tokens
                )
                backend = details.get("storage_backend")
                if isinstance(backend, str) and backend:
                    self.storage_backend = backend
        # Responses and Anthropic message_start nest their usage one level down.
        for key in ("response", "message"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                self._observe(nested)

    def fields(self) -> str:
        values: tuple[tuple[str, Any], ...] = (
            ("reported_input_tokens", self.input_tokens),
            ("reported_output_tokens", self.output_tokens),
            ("cached_tokens", self.cached_tokens),
            ("cache_creation_tokens", self.cache_creation_tokens),
            ("cache_device_tokens", self.device_cached_tokens),
            ("cache_host_tokens", self.host_cached_tokens),
            ("cache_storage_tokens", self.storage_cached_tokens),
            ("cache_storage_backend", self.storage_backend),
        )
        return " ".join(
            f"{name}={value}" for name, value in values if value is not None
        )

    def observed_cached_tokens(self) -> int | None:
        """Return reported cached tokens without inventing a missing report."""
        if self.cached_tokens is not None:
            return self.cached_tokens
        tiers = (
            self.device_cached_tokens,
            self.host_cached_tokens,
            self.storage_cached_tokens,
        )
        return (
            sum(value or 0 for value in tiers)
            if any(v is not None for v in tiers)
            else None
        )

    def observed_cache_tier(self) -> str:
        """Name the slowest tier that supplied at least one reported token."""
        for name, value in (
            ("storage", self.storage_cached_tokens),
            ("host", self.host_cached_tokens),
            ("device", self.device_cached_tokens),
        ):
            if value:
                return name
        return "none" if self.observed_cached_tokens() == 0 else "unknown"

    def cache_observation(self) -> dict[str, Any]:
        """Structured cache fields for one terminal lifecycle observation."""
        values = {
            "reported_input_tokens": self.input_tokens,
            "reported_output_tokens": self.output_tokens,
            "cached_tokens": self.observed_cached_tokens(),
            "cache_creation_tokens": self.cache_creation_tokens,
            "device_cached_tokens": self.device_cached_tokens,
            "host_cached_tokens": self.host_cached_tokens,
            "storage_cached_tokens": self.storage_cached_tokens,
            "storage_backend": self.storage_backend,
        }
        if any(
            value is not None
            for value in (
                self.cached_tokens,
                self.device_cached_tokens,
                self.host_cached_tokens,
                self.storage_cached_tokens,
            )
        ):
            values["cache_tier"] = self.observed_cache_tier()
        return {key: value for key, value in values.items() if value is not None}


@dataclass
class RelayMetrics:
    """Fixed-cardinality cumulative facts already observed by the relay."""

    observed_streams_total: int = 0
    ttft_s_sum: float = 0.0
    ttft_samples: int = 0
    reported_output_tokens_total: int = 0
    output_token_samples: int = 0
    cached_tokens_total: int = 0
    cache_creation_tokens_total: int = 0
    cache_device_tokens_total: int = 0
    cache_host_tokens_total: int = 0
    cache_storage_tokens_total: int = 0
    cache_token_reports: int = 0

    def observe(self, report: ResponseTokenReport, *, ttft_s: float | None) -> None:
        self.observed_streams_total += 1
        if ttft_s is not None:
            self.ttft_s_sum += max(0.0, ttft_s)
            self.ttft_samples += 1
        if report.output_tokens is not None:
            self.reported_output_tokens_total += max(0, report.output_tokens)
            self.output_token_samples += 1
        cache_values = (
            report.cached_tokens,
            report.cache_creation_tokens,
            report.device_cached_tokens,
            report.host_cached_tokens,
            report.storage_cached_tokens,
        )
        if any(value is not None for value in cache_values):
            self.cache_token_reports += 1
        for field_name, value in (
            ("cached_tokens_total", report.cached_tokens),
            ("cache_creation_tokens_total", report.cache_creation_tokens),
            ("cache_device_tokens_total", report.device_cached_tokens),
            ("cache_host_tokens_total", report.host_cached_tokens),
            ("cache_storage_tokens_total", report.storage_cached_tokens),
        ):
            if value is not None:
                setattr(self, field_name, getattr(self, field_name) + max(0, value))

    def snapshot(self) -> dict[str, int | float]:
        return {
            "observed_streams_total": self.observed_streams_total,
            "ttft_s_sum": self.ttft_s_sum,
            "ttft_samples": self.ttft_samples,
            "reported_output_tokens_total": self.reported_output_tokens_total,
            "output_token_samples": self.output_token_samples,
            "cached_tokens_total": self.cached_tokens_total,
            "cache_creation_tokens_total": self.cache_creation_tokens_total,
            "cache_device_tokens_total": self.cache_device_tokens_total,
            "cache_host_tokens_total": self.cache_host_tokens_total,
            "cache_storage_tokens_total": self.cache_storage_tokens_total,
            "cache_token_reports": self.cache_token_reports,
        }


def _integer(value: Any, previous: int | None) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else previous


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


def is_read_only_control_route(method: str, path: str) -> bool:
    """Identify model discovery without exempting future compute-bearing GETs."""
    route = path.split("?", 1)[0].rstrip("/")
    return method.upper() in {"GET", "HEAD"} and (
        route == "/v1/models" or route.startswith("/v1/models/")
    )


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

    ``alias`` is the stable non-secret routing identity; ``base_url`` is the
    transport address. Legacy programmatic pools may omit ``url``, making the
    URL its own alias, but deployment configuration is structured and named.
    """

    alias: str
    url: str | None = None
    in_flight: int = 0
    last_used_at: float = 0.0
    capacity: int = DEFAULT_CAPACITY_PER_UPSTREAM
    queued: int = 0
    token_capacity: int | None = DEFAULT_TOKEN_CAPACITY_PER_UPSTREAM
    input_tokens_in_flight: int = 0
    input_tokens_queued: int = 0
    cold_prefills_in_flight: int = 0
    cooldown_until: float = 0.0
    #: When this upstream last went out of rotation (None = healthy).
    cooling_since: float | None = None
    #: Incremented under the pool lock for race-safe health reconciliation.
    cooldown_generation: int = 0
    #: Operator cutover gate. Existing owned work drains; later work is held.
    quiesced: bool = False
    held: int = 0
    input_tokens_held: int = 0
    #: ``None`` before the first control-plane probe, then authoritative.
    reachable: bool | None = None
    unreachable_reason: str = "unobserved"
    reachability_generation: int = 0

    @property
    def base_url(self) -> str:
        return self.url or self.alias

    @property
    def usage_score(self) -> float:
        """Worst normalized request/token pressure across heterogeneous members."""
        request_pressure = (self.in_flight + self.queued) / self.capacity
        token_pressure = (
            (self.input_tokens_in_flight + self.input_tokens_queued)
            / self.token_capacity
            if self.token_capacity is not None
            else 0.0
        )
        return max(request_pressure, token_pressure)

    @property
    def scheduling_load(self) -> int:
        return self.in_flight + self.queued

    def status(self, *, closing: bool = False) -> dict[str, Any]:
        status = {
            "url": self.base_url,
            "active": (
                not closing
                and not self.quiesced
                and self.reachable is not False
                and self.cooldown_until <= time.time()
            ),
            "in_flight": self.in_flight,
            "queued": self.queued,
            "capacity": self.capacity,
        }
        if self.quiesced or self.held:
            status.update(held=self.held, quiesced=self.quiesced)
        if self.url is not None:
            status["label"] = self.alias
        if self.token_capacity is not None:
            status.update(
                input_tokens_in_flight=self.input_tokens_in_flight,
                input_tokens_queued=self.input_tokens_queued,
                token_capacity=self.token_capacity,
            )
            if self.quiesced or self.held:
                status["input_tokens_held"] = self.input_tokens_held
        return status


@dataclass
class _PoolTicket:
    priority: bool = False
    cache_priority: bool = False
    input_tokens: int = 0
    session_id: str = ""
    queued_at: float = 0.0
    bypasses: int = 0
    cold_prefill: bool = False
    admission_class: str = "unclassified"
    cache_classification: str = CacheResidency.UNKNOWN.value
    predicted_uncached_tokens: int | None = None
    prediction_expires_at: float | None = None
    block_reason: str = "none"

    def status(self, *, state: str, now: float, upstream: str) -> dict[str, Any]:
        """Return bounded, payload-free request metadata for operators."""
        label = request_session_label(self.session_id)
        return {
            "session_label": label,
            "upstream": public_upstream_url(upstream),
            "state": state,
            "input_tokens": self.input_tokens,
            "queue_age_s": (
                max(0.0, now - self.queued_at) if state in {"queued", "held"} else 0.0
            ),
            "priority": self.priority or self.cache_priority,
            "cache_priority": self.cache_priority,
            "admission_class": self.admission_class,
            "cache_classification": self.cache_classification,
            "predicted_uncached_tokens": self.predicted_uncached_tokens,
            "cold_prefill": self.cold_prefill,
            "bypasses": self.bypasses,
            "block_reason": self.block_reason,
        }


@dataclass(frozen=True)
class InferenceDrainState:
    """An admission-locked snapshot used to authorize a process cutover."""

    draining: bool
    in_flight: int
    queued: int

    @property
    def ready(self) -> bool:
        return not self.draining

    @property
    def empty(self) -> bool:
        return self.in_flight == 0 and self.queued == 0

    def as_dict(self) -> dict[str, bool | int]:
        return {
            "draining": self.draining,
            "ready": self.ready,
            "in_flight": self.in_flight,
            "queued": self.queued,
        }


class InferenceDrainTimeout(TimeoutError):
    """Admission is closed, but owned work did not finish before the deadline."""

    def __init__(self, state: InferenceDrainState) -> None:
        self.state = state
        super().__init__(
            "drain deadline expired with "
            f"in_flight={state.in_flight} queued={state.queued}; admission remains closed"
        )


class InferenceUpstreamPool(StickyPool[InferenceUpstream]):
    """Sticky per-conversation pool with bounded per-upstream admission.

    First placement uses the smallest token-capacity tier that can hold the
    request, then least-loaded round robin among interchangeable members in
    that tier. Once placed, a conversation waits for that same member's
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
        max_admission_bypasses: int = DEFAULT_MAX_ADMISSION_BYPASSES,
        priority_aging_s: float = DEFAULT_PRIORITY_AGING_S,
        cache_prediction_max_age_s: float = DEFAULT_CACHE_PREDICTION_MAX_AGE_S,
        cold_prefill_limit_per_upstream: int | None = None,
        cold_prefill_min_tokens: int | None = None,
        session_state: GatewaySessionState | None = None,
    ) -> None:
        if capacity_per_upstream < 1:
            raise ValueError("capacity_per_upstream must be >= 1")
        if max_queue_size < 0:
            raise ValueError("max_queue_size must be >= 0")
        if token_capacity_per_upstream is not None and token_capacity_per_upstream < 1:
            raise ValueError("token_capacity_per_upstream must be >= 1")
        if max_admission_bypasses < 0:
            raise ValueError("max_admission_bypasses must be >= 0")
        if priority_aging_s < 0:
            raise ValueError("priority_aging_s must be >= 0")
        if cache_prediction_max_age_s < 0:
            raise ValueError("cache_prediction_max_age_s must be >= 0")
        if (
            cold_prefill_limit_per_upstream is not None
            and cold_prefill_limit_per_upstream < 1
        ):
            raise ValueError("cold_prefill_limit_per_upstream must be >= 1")
        if (cold_prefill_limit_per_upstream is None) != (
            cold_prefill_min_tokens is None
        ):
            raise ValueError(
                "cold prefill limit and minimum tokens must be configured together"
            )
        if cold_prefill_min_tokens is not None and cold_prefill_min_tokens < 1:
            raise ValueError("cold_prefill_min_tokens must be >= 1")
        for upstream in upstreams:
            upstream.capacity = capacity_per_upstream
            if token_capacity_per_upstream is not None:
                upstream.token_capacity = token_capacity_per_upstream
        self.max_queue_size = max_queue_size
        self.max_admission_bypasses = max_admission_bypasses
        self.priority_aging_s = priority_aging_s
        self.cache_prediction_max_age_s = cache_prediction_max_age_s
        self.cold_prefill_limit_per_upstream = cold_prefill_limit_per_upstream
        self.cold_prefill_min_tokens = cold_prefill_min_tokens
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
        self._held_waiters: dict[str, deque[_PoolTicket]] = {
            upstream.alias: deque() for upstream in self.upstreams
        }
        self._running_tickets: dict[str, deque[_PoolTicket]] = {
            upstream.alias: deque() for upstream in self.upstreams
        }
        self._admissions_total = 0
        self._queued_total = 0
        self._queue_time_s_sum = 0.0
        self._queue_time_samples = 0
        self._blocked_total: dict[str, int] = {}
        self._held_total = 0
        self._member_quiesces_total = 0
        self._member_resumes_total = 0
        self._closing = False
        self.session_state = session_state

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
        max_admission_bypasses: int = DEFAULT_MAX_ADMISSION_BYPASSES,
        priority_aging_s: float = DEFAULT_PRIORITY_AGING_S,
        cache_prediction_max_age_s: float = DEFAULT_CACHE_PREDICTION_MAX_AGE_S,
        cold_prefill_limit_per_upstream: int | None = None,
        cold_prefill_min_tokens: int | None = None,
        session_state: GatewaySessionState | None = None,
    ) -> "InferenceUpstreamPool":
        """Build a programmatic legacy/external pool from transport URLs."""
        if isinstance(urls, str):
            urls = parse_upstreams(urls)
        return cls(
            [InferenceUpstream(alias=url) for url in urls],
            capacity_per_upstream=capacity_per_upstream,
            max_queue_size=max_queue_size,
            token_capacity_per_upstream=token_capacity_per_upstream,
            max_admission_bypasses=max_admission_bypasses,
            priority_aging_s=priority_aging_s,
            cache_prediction_max_age_s=cache_prediction_max_age_s,
            cold_prefill_limit_per_upstream=cold_prefill_limit_per_upstream,
            cold_prefill_min_tokens=cold_prefill_min_tokens,
            session_state=session_state,
        )

    @classmethod
    def from_specs(
        cls,
        specs: list[Any] | tuple[Any, ...],
        *,
        capacity_per_upstream: int = DEFAULT_CAPACITY_PER_UPSTREAM,
        max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
        max_admission_bypasses: int = DEFAULT_MAX_ADMISSION_BYPASSES,
        priority_aging_s: float = DEFAULT_PRIORITY_AGING_S,
        cache_prediction_max_age_s: float = DEFAULT_CACHE_PREDICTION_MAX_AGE_S,
        cold_prefill_limit_per_upstream: int | None = None,
        cold_prefill_min_tokens: int | None = None,
        session_state: GatewaySessionState | None = None,
    ) -> "InferenceUpstreamPool":
        """Build named members from validated deployment settings."""
        return cls(
            [
                InferenceUpstream(
                    alias=spec.label,
                    url=spec.url,
                    token_capacity=spec.token_capacity,
                )
                for spec in specs
            ],
            capacity_per_upstream=capacity_per_upstream,
            max_queue_size=max_queue_size,
            token_capacity_per_upstream=None,
            max_admission_bypasses=max_admission_bypasses,
            priority_aging_s=priority_aging_s,
            cache_prediction_max_age_s=cache_prediction_max_age_s,
            cold_prefill_limit_per_upstream=cold_prefill_limit_per_upstream,
            cold_prefill_min_tokens=cold_prefill_min_tokens,
            session_state=session_state,
        )

    async def route_alias(
        self,
        session_id: str,
        *,
        exclude: set[str] | None = None,
        input_tokens: int = 0,
    ) -> str:
        """Resolve and pin a session's target without consuming capacity."""
        if input_tokens < 0:
            raise ValueError("input_tokens must be >= 0")
        async with self._admission:
            if self._closing:
                raise InferenceAdmissionError("Inference gateway is shutting down")
            self._restore_route_locked(session_id)
            selected = self._select_for_input_locked(
                session_id, set(exclude or ()), input_tokens=input_tokens
            )
            self._remember_route(session_id, selected.alias)
            return selected.alias

    def _restore_route_locked(self, session_id: str) -> None:
        if not session_id or session_id in self._sessions or self.session_state is None:
            return
        alias = self.session_state.route(
            session_id, {upstream.alias for upstream in self.upstreams}
        )
        if alias is not None:
            self._sessions[session_id] = alias

    def _remember_route(self, session_id: str, alias: str) -> None:
        if self.session_state is not None:
            self.session_state.remember_route(session_id, alias)

    def _select_for_input_locked(
        self, session_id: str, excluded: set[str], *, input_tokens: int
    ) -> InferenceUpstream:
        """Select only a member on which this request can ever be resident."""
        now = time.time()
        unreachable = {
            upstream.alias for upstream in self.upstreams if upstream.reachable is False
        }
        incapable = {
            upstream.alias
            for upstream in self.upstreams
            if upstream.token_capacity is not None
            and input_tokens > upstream.token_capacity
        }
        if len(incapable) == len(self.upstreams):
            maximum = max(
                (upstream.token_capacity or 0 for upstream in self.upstreams),
                default=0,
            )
            raise InferenceAdmissionError(
                "Estimated request input exceeds every configured upstream's "
                f"token capacity ({input_tokens}; maximum {maximum})"
            )
        sticky_alias = self._sessions.get(session_id) if session_id else None
        if sticky_alias in incapable:
            # Capacity is a hard correctness boundary. Repin only when the
            # existing member can never fit the grown session; the new member
            # starts cold because cache prediction is label-bound.
            self._sessions.pop(session_id, None)
            sticky_alias = None

        hard_excluded = excluded | incapable | unreachable
        if (
            sticky_alias is not None
            and sticky_alias not in hard_excluded
            and self._by_alias(sticky_alias) is not None
        ):
            sticky = self._by_alias(sticky_alias)
            assert sticky is not None
            if sticky.quiesced:
                # Quiescing is a routing fence, not a cache-eviction event.
                # Hold a capable sticky request on its exact home.
                sticky.last_used_at = now
                return sticky
            # A compatible gateway-owned route is authoritative cache affinity.
            # Let StickyPool preserve it (including its reload hold semantics)
            # before applying any first-placement preference.
            return self._select_locked(session_id, hard_excluded, now=now)
        if sticky_alias is not None:
            self._sessions.pop(session_id, None)

        quiesced = {member.alias for member in self.upstreams if member.quiesced}
        nonquiesced_capable = [
            upstream
            for upstream in self.upstreams
            if upstream.alias not in hard_excluded | quiesced
        ]
        available = [
            upstream
            for upstream in nonquiesced_capable
            if upstream.cooldown_until <= now
        ]
        if available:
            # ``None`` is an unbounded/unspecified capacity and therefore the
            # last-resort tier after every finite capable tier. Selection
            # inside the chosen tier remains normalized-load then round robin.
            tier_capacity = min(
                (upstream.token_capacity for upstream in available),
                key=lambda capacity: (capacity is None, capacity or 0),
            )
            outside_tier = {
                upstream.alias
                for upstream in available
                if upstream.token_capacity != tier_capacity
            }
            return self._select_locked(
                session_id,
                hard_excluded | quiesced | outside_tier,
                now=now,
            )

        if nonquiesced_capable:
            # Preserve ordinary cooldown/reload behavior when another member
            # is capable but temporarily unavailable.
            return self._select_locked(session_id, hard_excluded | quiesced, now=now)

        # No non-quiesced member can ever fit this request. Keep it losslessly
        # on a capable quiesced member instead of returning 503 or repinning it
        # to an undersized member.
        held_candidates = [
            member
            for member in self.upstreams
            if member.quiesced
            and member.alias not in excluded
            and member.alias not in incapable
        ]
        if held_candidates:
            selected = self._choose(held_candidates)
            selected.last_used_at = now
            if session_id:
                self._sessions[session_id] = selected.alias
            return selected
        return self._select_locked(session_id, hard_excluded, now=now)

    async def acquire(
        self,
        session_id: str = "",
        *,
        exclude: set[str] | None = None,
        input_tokens: int = 0,
        priority: bool = False,
        cache_priority: bool = False,
        cold_prefill: bool = False,
        admission_class: str = "unclassified",
        cache_classification: str = CacheResidency.UNKNOWN.value,
        predicted_uncached_tokens: int | None = None,
        selected_alias: str | None = None,
    ) -> InferenceUpstream:
        """Place first for cache locality, then wait for that member's capacity."""
        if input_tokens < 0:
            raise ValueError("input_tokens must be >= 0")
        if predicted_uncached_tokens is not None and predicted_uncached_tokens < 0:
            raise ValueError("predicted_uncached_tokens must be >= 0")
        async with self._admission:
            if self._closing:
                raise InferenceAdmissionError("Inference gateway is shutting down")
            self._restore_route_locked(session_id)
            excluded = exclude or set()
            if selected_alias is None:
                selected = self._select_for_input_locked(
                    session_id, set(excluded), input_tokens=input_tokens
                )
            else:
                selected = self._by_alias(selected_alias)
                if (
                    selected is None
                    or selected.alias in excluded
                    or selected.reachable is False
                    or (not selected.quiesced and selected.cooldown_until > time.time())
                ):
                    # Never apply a cache prediction to a different upstream.
                    raise InferenceAdmissionError(
                        "Inference route changed before predicted admission; retry"
                    )
            self._remember_route(session_id, selected.alias)
            if (
                selected.token_capacity is not None
                and input_tokens > selected.token_capacity
            ):
                raise InferenceAdmissionError(
                    "Estimated request input exceeds this upstream's token capacity "
                    f"({input_tokens}/{selected.token_capacity})"
                )
            if predicted_uncached_tokens is not None:
                cold_prefill = bool(
                    self.cold_prefill_min_tokens is not None
                    and predicted_uncached_tokens >= self.cold_prefill_min_tokens
                )
            ticket = _PoolTicket(
                priority=priority,
                cache_priority=cache_priority,
                input_tokens=input_tokens,
                session_id=session_id,
                cold_prefill=cold_prefill,
                admission_class=(
                    admission_class
                    if admission_class
                    in {"continuation", "first-turn", "unclassified", "disabled"}
                    else "unclassified"
                ),
                cache_classification=(
                    cache_classification
                    if cache_classification in {item.value for item in CacheResidency}
                    else CacheResidency.UNKNOWN.value
                ),
                predicted_uncached_tokens=predicted_uncached_tokens,
                prediction_expires_at=(
                    time.monotonic() + self.cache_prediction_max_age_s
                    if cache_classification == CacheResidency.HOT.value
                    else None
                ),
            )
            was_held = selected.quiesced
            if was_held:
                await self._hold_ticket_locked(selected, ticket)
                # Resume promotes the ticket into the ordinary queue under this
                # same lock. From here it obeys all existing capacity/QoS rules.
            if not was_held and (
                self._fits(selected, input_tokens, session_id, cold_prefill)
                and not selected.queued
            ):
                selected.in_flight += 1
                selected.input_tokens_in_flight += input_tokens
                selected.cold_prefills_in_flight += int(cold_prefill)
                self._running_tickets[selected.alias].append(ticket)
                self._admissions_total += 1
                return selected
            waiters = self._waiters[selected.alias]
            if not was_held:
                total_queued = sum(upstream.queued for upstream in self.upstreams)
                if total_queued >= self.max_queue_size:
                    raise InferenceAdmissionError(
                        f"Inference queue is full ({total_queued}/{self.max_queue_size})"
                    )
                ticket.queued_at = time.monotonic()
                ticket.block_reason = self._block_reason(selected, ticket)
                self._blocked_total[ticket.block_reason] = (
                    self._blocked_total.get(ticket.block_reason, 0) + 1
                )
                waiters.append(ticket)
                self._queued_total += 1
                selected.queued += 1
                selected.input_tokens_queued += input_tokens
            queued = True
            try:
                while True:
                    if self._closing:
                        raise InferenceAdmissionError(
                            "Inference gateway is shutting down"
                        )
                    if selected.reachable is False:
                        raise InferenceMemberUnavailable(
                            selected.alias, selected.unreachable_reason
                        )
                    now = time.time()
                    if (
                        self._next_admissible_ticket(selected) is ticket
                        and selected.cooldown_until <= now
                    ):
                        self._record_bypasses(selected, ticket)
                        waiters.remove(ticket)
                        selected.queued -= 1
                        selected.input_tokens_queued -= input_tokens
                        queued = False
                        selected.in_flight += 1
                        selected.input_tokens_in_flight += input_tokens
                        selected.cold_prefills_in_flight += int(ticket.cold_prefill)
                        self._running_tickets[selected.alias].append(ticket)
                        self._admissions_total += 1
                        self._queue_time_s_sum += max(
                            0.0, time.monotonic() - ticket.queued_at
                        )
                        self._queue_time_samples += 1
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
                if queued and ticket in waiters:
                    waiters.remove(ticket)
                    selected.queued -= 1
                    selected.input_tokens_queued -= input_tokens
                    self._admission.notify_all()

    async def _hold_ticket_locked(
        self, member: InferenceUpstream, ticket: _PoolTicket
    ) -> None:
        """Hold post-cutoff work without consuming the bounded active queue."""
        held = self._held_waiters[member.alias]
        ticket.queued_at = time.monotonic()
        ticket.block_reason = "member-quiesced"
        held.append(ticket)
        member.held += 1
        member.input_tokens_held += ticket.input_tokens
        self._held_total += 1
        try:
            while ticket in held:
                if self._closing:
                    raise InferenceAdmissionError("Inference gateway is shutting down")
                await self._admission.wait()
        finally:
            if ticket in held:
                held.remove(ticket)
                member.held -= 1
                member.input_tokens_held -= ticket.input_tokens
                self._admission.notify_all()

    def _next_admissible_ticket(self, member: InferenceUpstream) -> _PoolTicket | None:
        """Choose work that fits now without permitting indefinite bypass.

        The request count is a hard safety ceiling. Below it, the token budget
        is the dynamic limit: smaller requests may backfill unused capacity
        while a large request waits. A token-blocked ticket may be bypassed a
        bounded number of times; after that, admission drains until it fits.
        Ordinary work also ages into the priority class, preventing an endless
        stream of continuations from monopolizing the upstream.
        """
        waiters = self._waiters[member.alias]
        if (
            not waiters
            or member.reachable is False
            or member.in_flight >= member.capacity
        ):
            return None

        now = time.monotonic()
        for ticket in waiters:
            self._expire_cache_prediction(ticket, now)
        oldest = waiters[0]
        oldest_blocked = not self._fits(
            member,
            oldest.input_tokens,
            oldest.session_id,
            oldest.cold_prefill,
        )
        if oldest_blocked and oldest.bypasses >= self.max_admission_bypasses:
            return None

        fitting = [
            ticket
            for ticket in waiters
            if self._fits(
                member, ticket.input_tokens, ticket.session_id, ticket.cold_prefill
            )
        ]
        if not fitting:
            return None
        aged = next(
            (
                ticket
                for ticket in fitting
                if not (ticket.priority or ticket.cache_priority)
                and now - ticket.queued_at >= self.priority_aging_s
            ),
            None,
        )
        selected = aged or next(
            (ticket for ticket in fitting if ticket.priority or ticket.cache_priority),
            fitting[0],
        )
        return selected

    def _expire_cache_prediction(self, ticket: _PoolTicket, now: float) -> None:
        """Turn a queued, stale hot guess back into conservative unknown work."""
        if ticket.prediction_expires_at is None or now < ticket.prediction_expires_at:
            return
        ticket.prediction_expires_at = None
        ticket.cache_classification = CacheResidency.UNKNOWN.value
        ticket.cache_priority = False
        ticket.predicted_uncached_tokens = ticket.input_tokens
        ticket.cold_prefill = bool(
            self.cold_prefill_min_tokens is not None
            and ticket.input_tokens >= self.cold_prefill_min_tokens
        )

    def _record_bypasses(
        self, member: InferenceUpstream, selected: _PoolTicket
    ) -> None:
        """Charge one bypass only when the later ticket is actually admitted."""
        waiters = self._waiters[member.alias]
        selected_index = waiters.index(selected)
        for bypassed in list(waiters)[:selected_index]:
            if not self._session_has_reachable_active(
                bypassed.session_id
            ) and not self._fits(
                member,
                bypassed.input_tokens,
                bypassed.session_id,
                bypassed.cold_prefill,
            ):
                bypassed.bypasses += 1

    def _fits(
        self,
        member: InferenceUpstream,
        input_tokens: int,
        session_id: str = "",
        cold_prefill: bool = False,
    ) -> bool:
        token_capacity = member.token_capacity
        running = self._running_tickets[member.alias]
        return (
            not self._session_has_reachable_active(session_id)
            and member.reachable is not False
            and member.in_flight < member.capacity
            and (not cold_prefill or all(ticket.cold_prefill for ticket in running))
            and (
                not cold_prefill
                or self.cold_prefill_limit_per_upstream is None
                or member.cold_prefills_in_flight < self.cold_prefill_limit_per_upstream
            )
            and (
                token_capacity is None
                or member.input_tokens_in_flight + input_tokens <= token_capacity
            )
        )

    def _session_has_reachable_active(self, session_id: str) -> bool:
        """Serialize a session unless its prior owner is authoritatively fenced."""
        if not session_id:
            return False
        return any(
            member.reachable is not False
            and any(ticket.session_id == session_id for ticket in running)
            for member in self.upstreams
            for running in (self._running_tickets[member.alias],)
        )

    def _block_reason(self, member: InferenceUpstream, ticket: _PoolTicket) -> str:
        """Explain the first deterministic gateway condition blocking a ticket."""
        if self._session_has_reachable_active(ticket.session_id):
            return "session-serialization"
        if member.in_flight >= member.capacity:
            return "request-capacity"
        if ticket.cold_prefill and any(
            not running.cold_prefill for running in self._running_tickets[member.alias]
        ):
            return "hot-work-in-flight"
        if (
            ticket.cold_prefill
            and self.cold_prefill_limit_per_upstream is not None
            and member.cold_prefills_in_flight >= self.cold_prefill_limit_per_upstream
        ):
            return "cold-prefill-limit"
        if (
            member.token_capacity is not None
            and member.input_tokens_in_flight + ticket.input_tokens
            > member.token_capacity
        ):
            return "token-capacity"
        return "queue-order"

    async def release(
        self,
        member: InferenceUpstream,
        *,
        input_tokens: int = 0,
        session_id: str = "",
        cold_prefill: bool = False,
    ) -> None:
        async with self._admission:
            if member.in_flight <= 0:
                return
            running = self._running_tickets[member.alias]
            matched = next(
                (
                    ticket
                    for ticket in running
                    if ticket.session_id == session_id
                    and ticket.input_tokens == input_tokens
                ),
                None,
            )
            if matched is None and not session_id and running:
                matched = running[0]
            # Member + session + token count is unique while the member is
            # reachable because session serialization is enforced there. It
            # remains unique on a fenced member while a retry runs elsewhere.
            # A duplicate or late cleanup therefore cannot consume a newer
            # ticket from another member.
            if matched is None:
                return
            actual_cold_prefill = (
                matched.cold_prefill if matched is not None else cold_prefill
            )
            member.in_flight = max(0, member.in_flight - 1)
            member.input_tokens_in_flight = max(
                0, member.input_tokens_in_flight - input_tokens
            )
            if actual_cold_prefill:
                member.cold_prefills_in_flight = max(
                    0, member.cold_prefills_in_flight - 1
                )
            if matched is not None:
                running.remove(matched)
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
                (upstream, upstream.cooldown_generation) for upstream in self.upstreams
            ]

    def _set_reachability_locked(
        self, member: InferenceUpstream, *, reachable: bool, reason: str
    ) -> None:
        """Apply authoritative control-plane reachability under admission lock."""
        was_reachable = member.reachable
        member.reachable = reachable
        member.unreachable_reason = "responded" if reachable else reason
        if not reachable and was_reachable is not False:
            member.reachability_generation += 1
            stale = [
                session
                for session, alias in self._sessions.items()
                if alias == member.alias
            ]
            for session in stale:
                self._sessions.pop(session, None)

    async def mark_unreachable(self, member: InferenceUpstream, *, reason: str) -> None:
        """Fence a failed member and wake every ticket pinned to it."""
        async with self._admission:
            self._set_reachability_locked(member, reachable=False, reason=reason)
            self._admission.notify_all()

    async def wait_until_unreachable(self, alias: str, generation: int) -> str:
        """Wait until a request's member crosses an unreachable boundary."""
        async with self._admission:
            member = self._by_alias(alias)
            if member is None:
                raise KeyError(alias)
            # Close the acquire-to-watcher race: health may fence the member
            # after its ticket was admitted but before this coroutine starts.
            if member.reachable is False:
                return member.unreachable_reason
            while member.reachability_generation <= generation:
                await self._admission.wait()
            return member.unreachable_reason

    async def reconcile_reachability(
        self,
        snapshot: list[tuple[InferenceUpstream, int]],
        observations: list[UpstreamReachability],
    ) -> None:
        """Fence failures and clear stale cooldown only after observed recovery."""
        async with self._admission:
            changed = False
            for (upstream, generation), observed in zip(
                snapshot, observations, strict=True
            ):
                previous = upstream.reachable
                self._set_reachability_locked(
                    upstream,
                    reachable=observed.reachable,
                    reason=observed.reason,
                )
                changed = changed or previous is not observed.reachable
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

    def _drain_state_locked(self) -> InferenceDrainState:
        return InferenceDrainState(
            draining=self._closing,
            in_flight=sum(upstream.in_flight for upstream in self.upstreams),
            queued=sum(upstream.queued + upstream.held for upstream in self.upstreams),
        )

    async def drain_state(self) -> InferenceDrainState:
        """Read readiness and ownership counters under the admission lock."""
        async with self._admission:
            return self._drain_state_locked()

    async def observability_snapshot(self) -> dict[str, Any]:
        """Return one lock-consistent, bounded view of admission ownership."""
        async with self._admission:
            now = time.monotonic()
            admitted = [
                ticket.status(state="admitted", now=now, upstream=upstream.alias)
                for upstream in self.upstreams
                for ticket in self._running_tickets[upstream.alias]
            ]
            queued = [
                ticket.status(state="queued", now=now, upstream=upstream.alias)
                for upstream in self.upstreams
                for ticket in self._waiters[upstream.alias]
            ]
            held = [
                ticket.status(state="held", now=now, upstream=upstream.alias)
                for upstream in self.upstreams
                for ticket in self._held_waiters[upstream.alias]
            ]
            tickets = admitted + queued + held
            return {
                "admitted": len(admitted),
                "queued": len(queued),
                "held": len(held),
                "input_tokens_admitted": sum(
                    upstream.input_tokens_in_flight for upstream in self.upstreams
                ),
                "input_tokens_queued": sum(
                    upstream.input_tokens_queued for upstream in self.upstreams
                ),
                "input_tokens_held": sum(
                    upstream.input_tokens_held for upstream in self.upstreams
                ),
                "oldest_queue_age_s": max(
                    (ticket["queue_age_s"] for ticket in queued), default=0.0
                ),
                "tickets": tickets[:MAX_STATUS_TICKETS],
                "tickets_omitted": max(0, len(tickets) - MAX_STATUS_TICKETS),
                "cumulative": {
                    "admissions_total": self._admissions_total,
                    "queued_total": self._queued_total,
                    "queue_time_s_sum": self._queue_time_s_sum,
                    "queue_time_samples": self._queue_time_samples,
                    "blocked_total": dict(sorted(self._blocked_total.items())),
                    "held_total": self._held_total,
                    "member_quiesces_total": self._member_quiesces_total,
                    "member_resumes_total": self._member_resumes_total,
                },
            }

    def _member_state_locked(
        self, member: InferenceUpstream
    ) -> InferenceMemberQuiesceState:
        return InferenceMemberQuiesceState(
            alias=member.alias,
            quiesced=member.quiesced,
            in_flight=member.in_flight,
            queued=member.queued,
            held=member.held,
        )

    async def member_state(self, alias: str) -> InferenceMemberQuiesceState:
        async with self._admission:
            member = self._by_alias(alias)
            if member is None:
                raise KeyError(alias)
            return self._member_state_locked(member)

    async def begin_member_quiesce(
        self,
        alias: str,
        timeout_s: float,
        *,
        on_cutoff: Callable[[], None] | None = None,
    ) -> InferenceMemberQuiesceState:
        """Fence one member atomically and drain only its pre-cutoff work."""
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and > 0")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        async with self._admission:
            member = self._by_alias(alias)
            if member is None:
                raise KeyError(alias)
            if not member.quiesced:
                member.quiesced = True
                self._member_quiesces_total += 1
                if on_cutoff is not None:
                    on_cutoff()
                self._admission.notify_all()
            while True:
                state = self._member_state_locked(member)
                if state.empty:
                    return state
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise InferenceMemberQuiesceTimeout(state)
                try:
                    await asyncio.wait_for(self._admission.wait(), remaining)
                except TimeoutError:
                    state = self._member_state_locked(member)
                    if not state.empty:
                        raise InferenceMemberQuiesceTimeout(state) from None

    async def _resume_member_after_validation(
        self, alias: str
    ) -> InferenceMemberQuiesceState:
        """Promote held work after the backend validates a replacement engine."""
        async with self._admission:
            member = self._by_alias(alias)
            if member is None:
                raise KeyError(alias)
            if not member.quiesced:
                raise InferenceMemberResumeError(f"member {alias} is not quiesced")
            member.quiesced = False
            held = self._held_waiters[alias]
            waiters = self._waiters[alias]
            while held:
                ticket = held.popleft()
                member.held -= 1
                member.input_tokens_held -= ticket.input_tokens
                # A prediction made before or during replacement cannot describe
                # the new engine cache. Resume it conservatively as cold/unknown.
                ticket.cache_classification = CacheResidency.UNKNOWN.value
                ticket.predicted_uncached_tokens = ticket.input_tokens
                ticket.prediction_expires_at = None
                ticket.cold_prefill = bool(
                    self.cold_prefill_min_tokens is not None
                    and ticket.input_tokens >= self.cold_prefill_min_tokens
                )
                ticket.block_reason = "queue-order"
                waiters.append(ticket)
                member.queued += 1
                member.input_tokens_queued += ticket.input_tokens
                self._queued_total += 1
            self._member_resumes_total += 1
            self._admission.notify_all()
            return self._member_state_locked(member)

    async def begin_drain(self, timeout_s: float) -> InferenceDrainState:
        """Atomically close admission and wait for all owned work to settle.

        The same condition lock guards both ``_closing`` and every transition
        into or out of the in-flight/queued counters. Therefore a successful
        return is a durable cutover barrier: no request can acquire ownership
        between this zero observation and the caller's process restart.
        """
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and > 0")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        async with self._admission:
            self._closing = True
            self._admission.notify_all()
            while True:
                state = self._drain_state_locked()
                if state.empty:
                    return state
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise InferenceDrainTimeout(state)
                try:
                    await asyncio.wait_for(self._admission.wait(), remaining)
                except TimeoutError:
                    state = self._drain_state_locked()
                    if not state.empty:
                        raise InferenceDrainTimeout(state) from None

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
    urls = ", ".join(
        f"{upstream.alias}={upstream.base_url}" for upstream in pool.upstreams
    )
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


class _UpstreamBecameUnreachable(Exception):
    """Internal control flow for a fenced member before response commitment."""


async def _empty_body() -> AsyncIterator[bytes]:
    """A valid empty streaming body for a request whose client is already gone."""
    if False:
        yield b""


async def _buffered_body(content: bytes) -> AsyncIterator[bytes]:
    """Yield a bounded control-plane response after its transport is closed."""
    if content:
        yield content


@dataclass
class _ReplaySafeAttempt:
    """Replay-safe cold work that has not emitted its first body byte."""

    upstream_alias: str
    admission_class: str
    preempt: asyncio.Event = field(default_factory=asyncio.Event)
    resume_after: asyncio.Future[None] | None = None
    released: asyncio.Future[None] | None = None


@dataclass
class _ContinuationHandoff:
    resume_first_turn: asyncio.Future[None]
    victim_released: asyncio.Future[None]


def _replay_safe_for_preemption(
    *, admission_class: str, cold_prefill: bool, prior_cache_tier: str
) -> bool:
    """Whether unfinished work may yield to a latency-sensitive continuation."""
    return (
        admission_class == "first-turn"
        or cold_prefill
        or prior_cache_tier in {"host", "storage"}
    )


class ContinuationQoS:
    """Ephemeral session-success classification and cooperative preemption.

    This is intentionally not called cache residency: a successful request is
    evidence of conversation history only. Engine cache state can disappear at
    any time and is neither queried nor inferred here.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        max_retries: int = 1,
        min_preempt_tokens: int = 0,
        session_state: GatewaySessionState | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("continuation_qos_max_retries must be >= 0")
        if min_preempt_tokens < 0:
            raise ValueError("continuation_qos_min_preempt_tokens must be >= 0")
        self.enabled = enabled
        self.max_retries = max_retries
        self.min_preempt_tokens = min_preempt_tokens
        self._successful: OrderedDict[str, None] = OrderedDict()
        self._session_state = session_state
        self._replay_safe: dict[str, deque[_ReplaySafeAttempt]] = {}
        self._counters = {
            "first_turn": 0,
            "continuation": 0,
            "unclassified": 0,
            "preemptions_requested": 0,
            "first_turns_preempted": 0,
            "first_turn_retries": 0,
            "cold_continuations_preempted": 0,
            "cold_continuation_retries": 0,
            "retry_budget_exhausted": 0,
            "abort_failures": 0,
            "unconfirmed_cleanup_holds": 0,
            "cleanup_reapers_active": 0,
            "cleanup_reaper_recoveries": 0,
            "cleanup_reaper_attempts": 0,
            "cleanup_reaper_cancellations": 0,
            "non_addressable_cleanup_releases": 0,
        }

    def kind(self, explicit_session: str) -> str:
        if not explicit_session:
            return "unclassified"
        if (
            explicit_session not in self._successful
            and self._session_state is not None
            and self._session_state.successful(explicit_session)
        ):
            self._successful[explicit_session] = None
        return "continuation" if explicit_session in self._successful else "first-turn"

    def classify(self, explicit_session: str) -> str:
        kind = self.kind(explicit_session)
        self._counters[kind.replace("-", "_")] += 1
        return kind

    def mark_successful(self, explicit_session: str) -> None:
        if not explicit_session:
            return
        self._successful.pop(explicit_session, None)
        self._successful[explicit_session] = None
        if self._session_state is not None:
            self._session_state.remember_success(explicit_session)
        while len(self._successful) > MAX_ROUTES:
            self._successful.popitem(last=False)

    def register_replay_safe(
        self, upstream_alias: str, *, admission_class: str = "first-turn"
    ) -> _ReplaySafeAttempt:
        attempt = _ReplaySafeAttempt(upstream_alias, admission_class)
        self._replay_safe.setdefault(upstream_alias, deque()).append(attempt)
        return attempt

    def unregister_replay_safe(self, attempt: _ReplaySafeAttempt) -> None:
        attempts = self._replay_safe.get(attempt.upstream_alias)
        if attempts is None:
            return
        try:
            attempts.remove(attempt)
        except ValueError:
            pass
        if not attempts:
            self._replay_safe.pop(attempt.upstream_alias, None)

    def request_preemption(self, upstream_alias: str) -> _ContinuationHandoff | None:
        """Signal one replay-safe first turn and return its continuation barrier."""
        attempts = self._replay_safe.get(upstream_alias, ())
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

    def preempted(self, admission_class: str = "first-turn") -> None:
        if admission_class == "continuation":
            self._counters["cold_continuations_preempted"] += 1
        else:
            self._counters["first_turns_preempted"] += 1

    def retried(self, admission_class: str = "first-turn") -> None:
        if admission_class == "continuation":
            self._counters["cold_continuation_retries"] += 1
        else:
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

    def non_addressable_cleanup_released(self) -> None:
        self._counters["non_addressable_cleanup_releases"] += 1

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
            "replay_safe_attempts": sum(map(len, self._replay_safe.values())),
            "replay_safe_first_turns": sum(
                attempt.admission_class == "first-turn"
                for attempts in self._replay_safe.values()
                for attempt in attempts
            ),
            "replay_safe_cold_continuations": sum(
                attempt.admission_class == "continuation"
                for attempts in self._replay_safe.values()
                for attempt in attempts
            ),
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
        metadata_timeout_s: float = DEFAULT_METADATA_TIMEOUT_S,
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
        cache_report_enabled: bool = False,
        cache_admission_settings: CacheAdmissionSettings | None = None,
        scheduler_probe: Callable[[str, float], Awaitable[SGLangSchedulerObservation]]
        | None = None,
        admission_history_path: Path | str | None = None,
    ) -> None:
        self.pool = pool
        self.timeout_s = timeout_s
        if metadata_timeout_s <= 0:
            raise ValueError("metadata_timeout_s must be > 0")
        self.metadata_timeout_s = metadata_timeout_s
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
        self._scheduler_probe = scheduler_probe
        self._scheduler_probe_lock = asyncio.Lock()
        self._scheduler_probe_tasks: dict[
            str, asyncio.Task[SGLangSchedulerObservation]
        ] = {}
        self._health_failures = {upstream.alias: 0 for upstream in self.pool.upstreams}
        self.cache_admission_settings = (
            cache_admission_settings or CacheAdmissionSettings()
        )
        if self.cache_admission_settings.active:
            expected = self.cache_admission_settings
            if not cache_report_enabled:
                raise ValueError("active cache admission requires cache_report_enabled")
            actual_policy = (
                pool.cold_prefill_limit_per_upstream,
                pool.cold_prefill_min_tokens,
                pool.max_admission_bypasses,
                pool.priority_aging_s,
                pool.cache_prediction_max_age_s,
            )
            expected_policy = (
                expected.cold_prefill_limit_per_upstream,
                expected.hot_max_uncached_tokens,
                expected.max_hot_bypasses,
                expected.starvation_age_s,
                expected.evidence_max_age_s,
            )
            if actual_policy != expected_policy:
                raise ValueError(
                    "active cache admission pool policy does not match its "
                    "validated settings"
                )
        self.cache_admission = AdmissionController(
            enabled=self.cache_admission_settings.active,
            max_cold_wait_s=self.cache_admission_settings.starvation_age_s,
        )
        self.continuation_qos = ContinuationQoS(
            enabled=continuation_qos_enabled,
            max_retries=continuation_qos_max_retries,
            min_preempt_tokens=continuation_qos_min_preempt_tokens,
            session_state=pool.session_state,
        )
        self.cache_report_enabled = cache_report_enabled
        self.relay_metrics = RelayMetrics()
        self.request_lifecycle = RequestLifecycleRegistry()
        self.admission_predictions = AdmissionPredictionTelemetry(
            max_observation_age_s=self.cache_admission_settings.evidence_max_age_s,
            state_path=admission_history_path,
        )
        self._engine_generations: dict[str, str | None] = {
            upstream.alias: None for upstream in self.pool.upstreams
        }
        self._quiesce_generations: dict[str, str | None] = {}
        self._backend_scheduler: dict[str, Any] = {
            "state": "unobserved",
            "running": None,
            "queued": None,
            "token_usage": None,
            "reason": "no-authoritative-engine-generation-or-metrics-contract",
        }
        self._cleanup_reapers: set[asyncio.Task[None]] = set()

    def observe_backend_scheduler(
        self,
        *,
        upstream: str,
        engine_generation: str,
        running: int,
        queued: int,
        token_usage: float,
    ) -> None:
        """Accept one external engine observation without changing admission."""
        if upstream not in self._engine_generations:
            raise ValueError("backend observation names an unknown upstream")
        if not engine_generation:
            raise ValueError("engine_generation must be non-empty")
        if running < 0 or queued < 0 or not 0 <= token_usage <= 1:
            raise ValueError("backend scheduler metrics are outside valid bounds")
        generation = self.observe_engine_generation(
            upstream=upstream, engine_generation=engine_generation
        )
        self._backend_scheduler = {
            "state": "observed",
            "upstream": public_upstream_url(upstream),
            "engine_generation": generation,
            "running": running,
            "queued": queued,
            "token_usage": token_usage,
            "reason": "engine-metrics-observed",
        }

    def observe_engine_generation(
        self, *, upstream: str, engine_generation: str
    ) -> str:
        """Bind prediction history to one authoritative engine incarnation."""
        if upstream not in self._engine_generations:
            raise ValueError("engine generation names an unknown upstream")
        if not engine_generation or engine_generation == "unavailable":
            raise ValueError("engine_generation must be authoritative")
        generation = hashlib.sha256(
            b"scitex-genai-engine-generation-v1\0" + engine_generation.encode()
        ).hexdigest()[:16]
        if self._engine_generations[upstream] != generation:
            self._engine_generations[upstream] = generation
            self.admission_predictions.restore_state(self._engine_generations)
        return generation

    def _backend_observation_failed(self, upstream: str, reason: str) -> None:
        """Make history non-reusable whenever generation cannot be established."""
        self._engine_generations[upstream] = None
        self._backend_scheduler = {
            "state": "unobserved",
            "upstream": public_upstream_url(upstream),
            "engine_generation": "unavailable",
            "running": None,
            "queued": None,
            "token_usage": None,
            "reason": reason,
        }

    async def refresh_backend_scheduler(self, upstream: str) -> None:
        """Coalesce a generation-bearing SGLang metrics probe for one upstream."""
        async with self._scheduler_probe_lock:
            task = self._scheduler_probe_tasks.get(upstream)
            if task is None:
                probe = self._scheduler_probe or probe_sglang_metrics
                member = self.pool._by_alias(upstream)
                if member is None:
                    raise ValueError("scheduler probe names an unknown upstream")
                task = asyncio.create_task(
                    probe(member.base_url, self.health_probe_timeout_s)
                )
                self._scheduler_probe_tasks[upstream] = task
        try:
            observation = await asyncio.shield(task)
        except Exception as exc:  # metrics absence is fail-closed, never a relay outage
            self._backend_observation_failed(
                upstream, f"engine-metrics-{type(exc).__name__.lower()}"
            )
        else:
            self.observe_backend_scheduler(
                upstream=upstream,
                engine_generation=observation.engine_generation,
                running=observation.running,
                queued=observation.queued,
                token_usage=observation.token_usage,
            )
        finally:
            async with self._scheduler_probe_lock:
                if self._scheduler_probe_tasks.get(upstream) is task and task.done():
                    self._scheduler_probe_tasks.pop(upstream, None)

    async def observability_snapshot(self) -> dict[str, Any]:
        """Return the stable operator-facing status document."""
        await asyncio.gather(
            *(
                self.refresh_backend_scheduler(upstream.alias)
                for upstream in self.pool.upstreams
            )
        )
        admission = await self.pool.observability_snapshot()
        admission["cumulative"].update(self.relay_metrics.snapshot())
        queued_tickets = [
            ticket for ticket in admission["tickets"] if ticket["state"] == "queued"
        ]
        block_reason = (
            queued_tickets[0]["block_reason"]
            if queued_tickets
            else (
                "backend-scheduler-queue"
                if (self._backend_scheduler.get("queued") or 0) > 0
                else "none"
            )
        )
        prediction_snapshot = self.admission_predictions.snapshot()
        prediction_snapshot["authoritative_for_admission"] = (
            self.cache_admission_settings.active
        )
        return {
            "schema_version": 3,
            "provider": self.provider,
            "draining": self.pool.draining,
            "request_lifecycle": self.request_lifecycle.snapshot(),
            "admission": admission,
            "admission_prediction": {
                **prediction_snapshot,
                "gateway_backend_comparison": {
                    "gateway_admitted": admission["admitted"],
                    "gateway_queued": admission["queued"],
                    "backend": dict(self._backend_scheduler),
                    "block_reason": block_reason,
                },
            },
        }

    async def quiesce_member(
        self, alias: str, timeout_s: float
    ) -> InferenceMemberQuiesceState:
        """Fence one member and invalidate cache evidence at the exact cutoff."""
        state = await self.pool.member_state(alias)
        if not state.quiesced:
            # Capture the last authoritative incarnation before the admission
            # fence. Metrics failure is allowed: a later authoritative identity
            # is still safer than reopening without any identity at all.
            await self.refresh_backend_scheduler(alias)

        def cutoff() -> None:
            self._quiesce_generations[alias] = self._engine_generations[alias]
            self._backend_observation_failed(alias, "member-quiesced")

        return await self.pool.begin_member_quiesce(alias, timeout_s, on_cutoff=cutoff)

    async def resume_member(self, alias: str) -> InferenceMemberQuiesceState:
        """Require reachable health and a new authoritative engine generation."""
        state = await self.pool.member_state(alias)
        if not state.quiesced:
            raise InferenceMemberResumeError(f"member {alias} is not quiesced")

        observations = await self._probe_upstreams_fresh()
        index = next(
            index
            for index, member in enumerate(self.pool.upstreams)
            if member.alias == alias
        )
        observed = observations[index]
        if not observed.reachable:
            raise InferenceMemberResumeError(
                f"member {alias} did not pass fresh health validation: {observed.reason}"
            )

        await self.refresh_backend_scheduler(alias)
        generation = self._engine_generations[alias]
        prior = self._quiesce_generations.get(alias)
        if generation is None:
            raise InferenceMemberResumeError(
                f"member {alias} lacks an authoritative engine generation"
            )
        if prior is not None and generation == prior:
            raise InferenceMemberResumeError(
                f"member {alias} still reports its pre-quiesce engine generation"
            )
        resumed = await self.pool._resume_member_after_validation(alias)
        self._quiesce_generations.pop(alias, None)
        return resumed

    def request_health_snapshot(self) -> dict[str, Any]:
        """Return labeled active phases for the unauthenticated health API."""
        return self.request_lifecycle.health_snapshot()

    def _note_request(self, record: RequestObservation) -> None:
        """Journal one lifecycle transition using labels, never raw identity."""
        row = record.status(time.monotonic())
        fields = (
            f"request={row['request_label']} agent={row['agent_label']} "
            f"session={row['session_label']} phase={row['phase']} "
            f"queue_elapsed_s={row['queue_elapsed_s']:.3f} "
            f"estimated_input_tokens={row['estimated_input_tokens']} "
            "gateway_capacity_owned="
            f"{str(row['gateway_capacity_owned']).lower()}"
        )
        if row["admitted_input_tokens"] is not None:
            fields += f" admitted_input_tokens={row['admitted_input_tokens']}"
        if row["gateway_input_tokens_admitted"] is not None:
            fields += (
                f" gateway_input_tokens_admitted={row['gateway_input_tokens_admitted']}"
            )
        if row["predicted_uncached_tokens"] is not None:
            fields += f" predicted_uncached_tokens={row['predicted_uncached_tokens']}"
        if row["upstream"] is not None:
            fields += f" upstream={row['upstream']}"
        if row["outcome"] is not None:
            fields += f" outcome={row['outcome']}"
        cache = row.get("cache", {})
        for name, value in cache.items():
            fields += f" {name}={value}"
        self._note(f"[request] {fields}")

    async def probe_upstreams(self) -> list[UpstreamReachability]:
        """Return one coalesced, briefly cached local-control-plane observation."""

        async with self._health_probe_lock:
            now = time.monotonic()
            if self._health_cache is not None and self._health_cache[0] > now:
                return self._health_cache[1]
            if self._health_probe_task is None:
                self._health_probe_task = asyncio.create_task(
                    self._probe_upstreams_fresh()
                )
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
                return await probe_upstream(url, timeout_s=self.health_probe_timeout_s)

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
        await self.pool.reconcile_reachability(snapshot, raw_observations)
        observations = []
        for upstream, observed in zip(
            self.pool.upstreams, raw_observations, strict=True
        ):
            failures = (
                0 if observed.reachable else self._health_failures[upstream.alias] + 1
            )
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

    def _cache_residency(self, prediction: AdmissionPrediction) -> CacheResidency:
        """Turn feedback into an admission class or fail closed in active mode."""
        if self.cache_admission_settings.active:
            try:
                return classify_cache_prediction(
                    settings=self.cache_admission_settings,
                    engine_generation=prediction.engine_generation,
                    evidence=prediction.evidence,
                    prior_cache_tier=prediction.prior_cache_tier,
                    predicted_uncached_tokens=prediction.predicted_uncached_tokens,
                )
            except ValueError as exc:
                raise InferenceAdmissionError(str(exc)) from exc
        if (
            self.pool.cold_prefill_min_tokens is not None
            and prediction.evidence != "no-compatible-history"
        ):
            return (
                CacheResidency.HOT
                if prediction.predicted_uncached_tokens
                < self.pool.cold_prefill_min_tokens
                else CacheResidency.COLD
            )
        return CacheResidency.UNKNOWN

    def _is_cold_prefill(self, prediction: AdmissionPrediction) -> bool:
        """Apply the measured uncached-token boundary, never full prompt size."""
        threshold = (
            self.cache_admission_settings.hot_max_uncached_tokens
            if self.cache_admission_settings.active
            else (self.pool.cold_prefill_min_tokens or 0)
        )
        return bool(
            self.pool.cold_prefill_limit_per_upstream is not None
            and prediction.predicted_uncached_tokens >= max(1, threshold)
        )

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

        # Generation admission exists to protect accelerator KV capacity and
        # preserve conversation locality. Read-only discovery/control calls do
        # neither. Charging (for example) GET /v1/models as a zero-token,
        # non-cold ticket can block every cold prefill pinned to that member if
        # its transport hangs. Keep all read-only routes outside generation
        # admission and give the complete exchange a short absolute deadline.
        if is_read_only_control_route(method, path):
            return await self._relay_read_only_control(
                method,
                path,
                body=body,
                headers=headers,
                upstream_path=upstream_path,
            )

        explicit_session = request_session_key(headers)
        qos_session = continuation_qos_session_key(headers)
        body, session = self.prepare(
            body,
            hoist=hoists_on(path),
            affinity_key=explicit_session,
            inject_session_id=accepts_session_id(path),
        )
        input_tokens = estimate_input_tokens(body)
        prediction_body = body
        prefix_fingerprint = request_prefix_fingerprint(body)
        body, cache_report_requested = inject_cache_report_request(
            body, path, enabled=self.cache_report_enabled
        )
        routing_session = session or ""
        qos_kind = (
            self.continuation_qos.classify(qos_session)
            if self.continuation_qos.enabled
            else "disabled"
        )
        feedback_headers = {
            "x-scitex-admission-mode": self.cache_admission_settings.mode.replace(
                "_", "-"
            ),
            "x-scitex-cache-residency": CacheResidency.UNKNOWN.value,
            "x-scitex-session-key": (session or "")[:12] or "none",
        }
        request_started = time.monotonic()
        request_observation = self.request_lifecycle.start(
            agent_label=request_agent_label(headers),
            session_label=request_session_label(routing_session),
            estimated_input_tokens=input_tokens,
        )
        feedback_headers["x-scitex-request-label"] = request_observation.request_label
        feedback_headers["x-scitex-agent-label"] = request_observation.agent_label
        feedback_headers["x-scitex-session-label"] = request_observation.session_label
        self._note_request(request_observation)
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
            while len(attempted) < len(self.pool.upstreams):
                if request_observation.phase != "admission_queued":
                    self.request_lifecycle.admission_queued(request_observation)
                    self._note_request(request_observation)
                try:
                    predicted_alias = await self.pool.route_alias(
                        routing_session, exclude=attempted, input_tokens=input_tokens
                    )
                    await self.refresh_backend_scheduler(predicted_alias)
                    prediction = self.admission_predictions.predict(
                        session_id=routing_session,
                        upstream=predicted_alias,
                        engine_generation=self._engine_generations[predicted_alias],
                        body=prediction_body,
                        estimated_input_tokens=input_tokens,
                    )
                    cache_classification = self._cache_residency(prediction)
                    self.cache_admission.observe(cache_classification)
                    cold_prefill = self._is_cold_prefill(prediction)
                    replay_safe = _replay_safe_for_preemption(
                        admission_class=qos_kind,
                        cold_prefill=cold_prefill,
                        prior_cache_tier=prediction.prior_cache_tier,
                    )
                    latency_sensitive = qos_kind == "continuation" and not replay_safe
                    feedback_headers["x-scitex-admission-mode"] = (
                        self.cache_admission_settings.mode.replace("_", "-")
                    )
                    feedback_headers["x-scitex-cache-residency"] = (
                        cache_classification.value
                    )
                    if latency_sensitive and continuation_handoff is None:
                        continuation_handoff = self.continuation_qos.request_preemption(
                            predicted_alias
                        )
                        if continuation_handoff is not None:
                            # Do not overlap at the engine: capacity=2 means
                            # gateway admission alone cannot tell that this hot
                            # continuation is queued behind a long prefill.
                            await continuation_handoff.victim_released
                    upstream = await self._acquire_while_connected(
                        routing_session,
                        exclude=attempted,
                        input_tokens=input_tokens,
                        priority=latency_sensitive,
                        cache_priority=(cache_classification is CacheResidency.HOT),
                        cold_prefill=cold_prefill,
                        admission_class=qos_kind,
                        cache_classification=cache_classification.value,
                        predicted_uncached_tokens=prediction.predicted_uncached_tokens,
                        selected_alias=predicted_alias,
                        client_disconnected=client_disconnected,
                    )
                    current_generation = (
                        self._engine_generations[upstream.alias] or "unavailable"
                    )
                    if prediction.engine_generation != current_generation:
                        if self.cache_admission_settings.active:
                            await self.pool.release(
                                upstream,
                                input_tokens=input_tokens,
                                session_id=routing_session,
                                cold_prefill=cold_prefill,
                            )
                            raise InferenceAdmissionError(
                                "active cache admission engine generation changed "
                                "while queued; retry with fresh cache evidence"
                            )
                        # A member-quiesce hold may span an engine replacement.
                        # Never dispatch or record that request with cache evidence
                        # from the incarnation observed before it entered the hold.
                        prediction = self.admission_predictions.predict(
                            session_id=routing_session,
                            upstream=upstream.alias,
                            engine_generation=self._engine_generations[upstream.alias],
                            body=prediction_body,
                            estimated_input_tokens=input_tokens,
                        )
                        cache_classification = self._cache_residency(prediction)
                        cold_prefill = self._is_cold_prefill(prediction)
                        replay_safe = _replay_safe_for_preemption(
                            admission_class=qos_kind,
                            cold_prefill=cold_prefill,
                            prior_cache_tier=prediction.prior_cache_tier,
                        )
                        feedback_headers["x-scitex-cache-residency"] = (
                            cache_classification.value
                        )
                except _ClientDisconnected:
                    self.request_lifecycle.disconnected(
                        request_observation,
                        outcome="client_disconnected_before_admission",
                    )
                    self._note_request(request_observation)
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
                except InferenceMemberUnavailable as exc:
                    attempted.add(exc.alias)
                    failures.append(str(exc))
                    self._note(
                        f"[relay] conv={routing_session[:8] or '-'} rerouting "
                        f"before dispatch: {exc}"
                    )
                    continue
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
                self.request_lifecycle.upstream_inflight(
                    request_observation,
                    upstream=public_upstream_url(upstream.alias),
                    admitted_input_tokens=input_tokens,
                    gateway_input_tokens_admitted=upstream.input_tokens_in_flight,
                    predicted_uncached_tokens=prediction.predicted_uncached_tokens,
                    cache_classification=cache_classification.value,
                )
                self._note_request(request_observation)
                self._note(
                    f"[relay] conv={routing_session[:8] or '-'} -> {upstream.alias} "
                    f"{method} {path} bytes={len(body or b'')} "
                    f"estimated_input_tokens={input_tokens} "
                    f"predicted_uncached_tokens={prediction.predicted_uncached_tokens} "
                    f"admitted_input_tokens={upstream.input_tokens_in_flight} "
                    f"prefix_fingerprint={prefix_fingerprint} "
                    f"cache_report_requested={str(cache_report_requested).lower()} "
                    f"queue_s={started - request_started:.3f}"
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
                            cold_prefill=cold_prefill,
                        )
                    )
                    await asyncio.shield(task)
                    self.request_lifecycle.capacity_released(request_observation)

                send_task: asyncio.Task[Any] | None = None
                preempt_task: asyncio.Task[Any] | None = None
                disconnect_task: asyncio.Task[None] | None = None
                disconnect_stop: asyncio.Event | None = None
                unreachable_task: asyncio.Task[str] | None = None
                first_chunk_task: asyncio.Task[bytes] | None = None
                stream: AsyncIterator[bytes] | None = None
                replay_attempt = None
                if (
                    replay_safe
                    and rid_confirmed
                    and input_tokens >= self.continuation_qos.min_preempt_tokens
                    and replay_count < self.continuation_qos.max_retries
                ):
                    replay_attempt = self.continuation_qos.register_replay_safe(
                        upstream.alias, admission_class=qos_kind
                    )

                async def preempt_for_continuation() -> bool:
                    """Abort and release this replay-safe attempt before retry."""
                    nonlocal replay_attempt, preempt_task
                    assert replay_attempt is not None
                    abort_ok = await self._abort_request(
                        upstream, dispatch_request_id, headers=forwarded
                    )
                    if not abort_ok:
                        if (
                            replay_attempt.released is not None
                            and not replay_attempt.released.done()
                        ):
                            replay_attempt.released.set_exception(
                                InferenceAdmissionError(
                                    "Continuation QoS could not confirm upstream abort"
                                )
                            )
                        self.continuation_qos.unregister_replay_safe(replay_attempt)
                        replay_attempt = None
                        preempt_task = None
                        return False
                    for task in (send_task, first_chunk_task, unreachable_task):
                        if task is not None and not task.done():
                            task.cancel()
                    await asyncio.gather(
                        *(
                            task
                            for task in (send_task, first_chunk_task)
                            if task is not None
                        ),
                        return_exceptions=True,
                    )
                    admission_class = replay_attempt.admission_class
                    self.continuation_qos.preempted(admission_class)
                    self._note(
                        f"[relay] conv={routing_session[:8] or '-'} <- "
                        f"{upstream.alias} cooperatively_preempted_before_first_byte"
                    )
                    await asyncio.shield(client.aclose())
                    await release_slot()
                    if (
                        replay_attempt.released is not None
                        and not replay_attempt.released.done()
                    ):
                        replay_attempt.released.set_result(None)
                    barrier = replay_attempt.resume_after
                    self.continuation_qos.unregister_replay_safe(replay_attempt)
                    replay_attempt = None
                    if barrier is not None:
                        await barrier
                    self.continuation_qos.retried(admission_class)
                    return True

                try:
                    request = client.build_request(
                        method,
                        upstream.base_url + (upstream_path or path),
                        content=dispatch_body,
                        headers=forwarded,
                    )
                    # The reachability watcher requires the transport send to
                    # be an independently cancellable waiter regardless of
                    # whether continuation QoS is enabled.
                    send_task = asyncio.create_task(client.send(request, stream=True))
                    if rid_confirmed and client_disconnected is not None:
                        disconnect_stop = asyncio.Event()
                        disconnect_task = asyncio.create_task(
                            self._wait_for_client_disconnect(
                                client_disconnected, stop=disconnect_stop
                            )
                        )
                    unreachable_task = asyncio.create_task(
                        self.pool.wait_until_unreachable(
                            upstream.alias, upstream.reachability_generation
                        )
                    )
                    if replay_attempt is not None:
                        preempt_task = asyncio.create_task(
                            replay_attempt.preempt.wait()
                        )
                    waiters = tuple(
                        task
                        for task in (
                            send_task,
                            preempt_task,
                            disconnect_task,
                            unreachable_task,
                        )
                        if task is not None
                    )
                    if waiters:
                        done, _ = await asyncio.wait(
                            waiters,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if send_task is not None and send_task in done:
                            response = await send_task
                        elif disconnect_task is not None and disconnect_task in done:
                            raise _ClientDisconnected
                        elif unreachable_task is not None and unreachable_task in done:
                            raise _UpstreamBecameUnreachable(await unreachable_task)
                        else:
                            if await preempt_for_continuation():
                                replay_count += 1
                                if replay_count >= self.continuation_qos.max_retries:
                                    self.continuation_qos.exhausted()
                                attempted.clear()
                                continue
                            response = await send_task
                    else:
                        response = await client.send(request, stream=True)

                    # Do not commit an HTTP 200 to the caller merely because
                    # the upstream sent response headers. SGLang has been
                    # observed returning headers and then orphaning the body
                    # stream. Prime one body chunk while the ASGI disconnect
                    # monitor can still abort the request deterministically.
                    stream = response.aiter_bytes()
                    first_chunk_task = asyncio.create_task(anext(stream))
                    body_waiters = tuple(
                        task
                        for task in (
                            first_chunk_task,
                            preempt_task,
                            disconnect_task,
                            unreachable_task,
                        )
                        if task is not None
                    )
                    if len(body_waiters) > 1:
                        done, _ = await asyncio.wait(
                            body_waiters,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if first_chunk_task in done:
                            if preempt_task is not None and not preempt_task.done():
                                preempt_task.cancel()
                                await asyncio.gather(
                                    preempt_task, return_exceptions=True
                                )
                        elif disconnect_task is not None and disconnect_task in done:
                            raise _ClientDisconnected
                        elif unreachable_task is not None and unreachable_task in done:
                            raise _UpstreamBecameUnreachable(await unreachable_task)
                        else:
                            if await preempt_for_continuation():
                                replay_count += 1
                                if replay_count >= self.continuation_qos.max_retries:
                                    self.continuation_qos.exhausted()
                                attempted.clear()
                                continue
                            remaining = tuple(
                                task
                                for task in (
                                    first_chunk_task,
                                    disconnect_task,
                                    unreachable_task,
                                )
                                if task is not None
                            )
                            done, _ = await asyncio.wait(
                                remaining, return_when=asyncio.FIRST_COMPLETED
                            )
                            if first_chunk_task not in done:
                                if (
                                    unreachable_task is not None
                                    and unreachable_task in done
                                ):
                                    raise _UpstreamBecameUnreachable(
                                        await unreachable_task
                                    )
                                raise _ClientDisconnected
                        if disconnect_task is not None and first_chunk_task in done:
                            disconnect_stop.set()
                            await asyncio.gather(
                                disconnect_task, return_exceptions=True
                            )
                    try:
                        first_chunk = await first_chunk_task
                    except StopAsyncIteration:
                        first_chunk = None
                    ttft_s = time.monotonic() - started
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
                    # A request without a gateway-owned engine id is not
                    # addressable after its transport is cancelled. Never
                    # shield its send task indefinitely: closing that
                    # transport is the terminal cleanup event. Addressable
                    # generation work instead transfers its slot to the abort
                    # reaper when the immediate abort cannot be confirmed.
                    safe_to_release = not rid_confirmed
                    if rid_confirmed:
                        abort_ok = await asyncio.shield(
                            self._abort_request(
                                upstream, dispatch_request_id, headers=forwarded
                            )
                        )
                    if abort_ok:
                        safe_to_release = True
                    for task in (send_task, first_chunk_task):
                        if task is not None and not task.done():
                            task.cancel()
                    if preempt_task is not None and not preempt_task.done():
                        preempt_task.cancel()
                    if disconnect_task is not None and not disconnect_task.done():
                        disconnect_stop.set()
                        disconnect_task.cancel()
                    if unreachable_task is not None and not unreachable_task.done():
                        unreachable_task.cancel()
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
                            unreachable_task,
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
                            cold_prefill=cold_prefill,
                            request_observation=request_observation,
                        )
                    if (
                        replay_attempt is not None
                        and replay_attempt.preempt.is_set()
                        and replay_attempt.released is not None
                        and not replay_attempt.released.done()
                    ):
                        replay_attempt.released.set_exception(
                            InferenceAdmissionError(
                                "Preempted replay-safe client disconnected"
                            )
                        )
                    if observed_disconnect:
                        self.request_lifecycle.disconnected(
                            request_observation,
                            outcome=outcome,
                        )
                        self._note_request(request_observation)
                        return RelayedResponse(
                            status_code=499,
                            content_type="application/json",
                            body=_empty_body(),
                            feedback_headers=feedback_headers,
                        )
                    raise
                except _UpstreamBecameUnreachable as exc:
                    for task in (
                        send_task,
                        first_chunk_task,
                        preempt_task,
                        disconnect_task,
                    ):
                        if task is not None and not task.done():
                            task.cancel()
                    await asyncio.gather(
                        *(
                            task
                            for task in (
                                send_task,
                                first_chunk_task,
                                preempt_task,
                                disconnect_task,
                            )
                            if task is not None
                        ),
                        return_exceptions=True,
                    )
                    await asyncio.shield(client.aclose())
                    safe_to_release = not rid_confirmed
                    if rid_confirmed:
                        safe_to_release = await self._abort_request(
                            upstream, dispatch_request_id, headers=forwarded
                        )
                    if safe_to_release:
                        await release_slot()
                    else:
                        slot_owned = False
                        self._schedule_cleanup_reaper(
                            upstream,
                            dispatch_request_id,
                            headers=forwarded,
                            input_tokens=input_tokens,
                            session_id=routing_session,
                            cold_prefill=cold_prefill,
                            request_observation=request_observation,
                        )
                    self._note(
                        f"[relay] conv={routing_session[:8] or '-'} <- "
                        f"{upstream.alias} failed fast after authoritative "
                        f"unreachable observation: {exc}"
                    )
                    raise InferenceAdmissionError(
                        f"Inference member {upstream.alias} became unreachable "
                        "before a response; retry may use another reachable member"
                    ) from exc
                except httpx.TransportError as exc:
                    await client.aclose()
                    await self.pool.mark_unreachable(
                        upstream, reason=exc.__class__.__name__
                    )
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
                            cold_prefill=cold_prefill,
                            request_observation=request_observation,
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
                    if unreachable_task is not None and not unreachable_task.done():
                        unreachable_task.cancel()
                    if unreachable_task is not None:
                        await asyncio.gather(unreachable_task, return_exceptions=True)
                    if replay_attempt is not None:
                        if (
                            replay_attempt.preempt.is_set()
                            and replay_attempt.released is not None
                            and not replay_attempt.released.done()
                        ):
                            replay_attempt.released.set_exception(
                                InferenceAdmissionError(
                                    "Replay-safe handoff did not complete"
                                )
                            )
                        self.continuation_qos.unregister_replay_safe(replay_attempt)
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
                        request_started=request_started,
                        ttft_s=ttft_s,
                        input_tokens=input_tokens,
                        session_id=routing_session,
                        continuation_handoff=continuation_handoff,
                        request_id=dispatch_request_id if rid_confirmed else "",
                        request_headers=forwarded,
                        successful_session=(
                            qos_session if 200 <= response.status_code < 300 else ""
                        ),
                        cold_prefill=cold_prefill,
                        admission_prediction=prediction,
                        request_observation=request_observation,
                    ),
                )
            raise UpstreamUnreachable(self._refusal(failures))
        except asyncio.CancelledError:
            self.request_lifecycle.disconnected(
                request_observation, outcome="relay_cancelled"
            )
            self._note_request(request_observation)
            raise
        except Exception as exc:
            self.request_lifecycle.completed(
                request_observation,
                outcome=f"error:{type(exc).__name__}",
            )
            self._note_request(request_observation)
            raise
        finally:
            if not handed_to_stream:
                self.continuation_qos.finish_continuation(continuation_handoff)

    async def _relay_read_only_control(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None,
        headers: Mapping[str, str],
        upstream_path: str | None,
    ) -> RelayedResponse:
        """Relay metadata without sticky placement or generation admission."""
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "Inference relay requires scitex-genai[gateway]"
            ) from exc

        forwarded = {
            name: value
            for name, value in headers.items()
            if name.lower() not in _HOP_BY_HOP
        }
        failures: list[str] = []
        now = time.time()
        deadline = asyncio.get_running_loop().time() + self.metadata_timeout_s
        # Preserve configured order. Metadata must not mutate or consult the
        # generation scheduler's usage/sticky state; fall through only when a
        # member is cooling or fails its own bounded exchange.
        candidates = [
            upstream
            for upstream in self.pool.upstreams
            if upstream.cooldown_until <= now
        ]
        if not candidates:
            raise UpstreamUnreachable(self.pool.cooling_message)

        for upstream in candidates:
            started = time.monotonic()
            try:
                async with asyncio.timeout_at(deadline):
                    async with httpx.AsyncClient(
                        timeout=self.metadata_timeout_s
                    ) as client:
                        response = await client.request(
                            method,
                            upstream.base_url + (upstream_path or path),
                            content=body,
                            headers=forwarded,
                        )
                        content = await response.aread()
            except (TimeoutError, httpx.TransportError) as exc:
                failures.append(f"{upstream.alias} ({type(exc).__name__})")
                self._note(
                    f"[relay-control] -> {upstream.alias} {method} {path} "
                    f"failed={type(exc).__name__} after "
                    f"{time.monotonic() - started:.3f}s"
                )
                continue

            self._note(
                f"[relay-control] -> {upstream.alias} {method} {path} "
                f"status={response.status_code} bytes={len(content)} "
                f"total_s={time.monotonic() - started:.3f}"
            )
            return RelayedResponse(
                status_code=response.status_code,
                content_type=response.headers.get("content-type", "application/json"),
                body=_buffered_body(content),
                feedback_headers={
                    "x-scitex-admission-mode": "control-plane-bypass",
                    "x-scitex-cache-residency": CacheResidency.UNKNOWN.value,
                    "x-scitex-session-key": "none",
                },
            )

        aliases = ", ".join(upstream.alias for upstream in candidates)
        raise UpstreamUnreachable(
            "No metadata upstream answered within the independent "
            f"{self.metadata_timeout_s:g}s deadline: "
            + "; ".join(failures)
            + f". Eligible upstreams ({len(candidates)}): {aliases}."
        )

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
        cache_priority: bool = False,
        cold_prefill: bool,
        admission_class: str,
        cache_classification: str,
        predicted_uncached_tokens: int,
        selected_alias: str,
        client_disconnected: Callable[[], Awaitable[bool]] | None,
    ) -> InferenceUpstream:
        """Remove a queued admission ticket as soon as its caller disappears."""
        acquire = asyncio.create_task(
            self.pool.acquire(
                session_id,
                exclude=exclude,
                input_tokens=input_tokens,
                priority=priority,
                cache_priority=cache_priority,
                cold_prefill=cold_prefill,
                admission_class=admission_class,
                cache_classification=cache_classification,
                predicted_uncached_tokens=predicted_uncached_tokens,
                selected_alias=selected_alias,
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
        cold_prefill: bool = False,
        request_observation: RequestObservation | None = None,
    ) -> None:
        """Retain admission and retry abort until engine cleanup is confirmed."""
        if not request_id:
            raise ValueError("cleanup reaper requires a non-empty request id")
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
                                cold_prefill=cold_prefill,
                            )
                        )
                        await asyncio.shield(release)
                        if request_observation is not None:
                            self.request_lifecycle.capacity_released(
                                request_observation
                            )
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
        request_started: float | None = None,
        ttft_s: float | None = None,
        input_tokens: int = 0,
        session_id: str = "",
        continuation_handoff: _ContinuationHandoff | None = None,
        successful_session: str = "",
        cold_prefill: bool = False,
        request_id: str = "",
        request_headers: Mapping[str, str] | None = None,
        admission_prediction: AdmissionPrediction | None = None,
        request_observation: RequestObservation | None = None,
    ) -> AsyncIterator[bytes]:
        sent = 0
        outcome = "complete"
        stream = stream or response.aiter_bytes()
        next_chunk: asyncio.Task[bytes] | None = None
        release_capacity = True
        token_report = ResponseTokenReport()

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
                token_report.feed(first_chunk)
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
                token_report.feed(chunk)
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
            token_report.finish()
            self.relay_metrics.observe(token_report, ttft_s=ttft_s)
            if admission_prediction is not None:
                self.admission_predictions.observe(
                    admission_prediction,
                    session_id=session_id,
                    upstream=upstream.alias,
                    reported_input_tokens=token_report.input_tokens,
                    cached_tokens=token_report.observed_cached_tokens(),
                    cache_tier=token_report.observed_cache_tier(),
                )
                self.admission_predictions.save_state(self._engine_generations)
            cache = token_report.cache_observation()
            if tag:
                took = time.monotonic() - started if started is not None else 0.0
                total = (
                    time.monotonic() - request_started
                    if request_started is not None
                    else took
                )
                reported = token_report.fields()
                suffix = f" {reported}" if reported else ""
                self._note(
                    f"[relay] {tag} outcome={outcome} bytes={sent} "
                    f"ttft_s={ttft_s if ttft_s is not None else 0.0:.3f} "
                    f"upstream_s={took:.3f} total_s={total:.3f}{suffix}"
                )
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
                    cold_prefill=cold_prefill,
                    release_capacity=release_capacity,
                    request_id=request_id,
                    request_headers=request_headers or {},
                    request_observation=request_observation,
                )
            )
            if request_observation is not None:
                if outcome == "client_disconnected":
                    self.request_lifecycle.disconnected(
                        request_observation, outcome=outcome, cache=cache
                    )
                else:
                    self.request_lifecycle.completed(
                        request_observation, outcome=outcome, cache=cache
                    )
                self._note_request(request_observation)
            self.continuation_qos.finish_continuation(continuation_handoff)

    async def _finish(
        self,
        client: Any,
        response: Any,
        upstream: InferenceUpstream,
        *,
        input_tokens: int = 0,
        session_id: str = "",
        cold_prefill: bool = False,
        release_capacity: bool = True,
        request_id: str = "",
        request_headers: Mapping[str, str] | None = None,
        request_observation: RequestObservation | None = None,
    ) -> None:
        # Ownership is settled before socket cleanup. A peer stuck in
        # FIN-WAIT/CLOSE-WAIT must never keep a confirmed-aborted or fully
        # consumed request charged against admission indefinitely.
        if release_capacity or not request_id:
            await self.pool.release(
                upstream,
                input_tokens=input_tokens,
                session_id=session_id,
                cold_prefill=cold_prefill,
            )
            if request_observation is not None:
                self.request_lifecycle.capacity_released(request_observation)
            if not release_capacity:
                self.continuation_qos.non_addressable_cleanup_released()
                self._note(
                    "[relay] released admission for non-addressable upstream "
                    "request before transport cleanup; no cleanup reaper created"
                )
        else:
            self._schedule_cleanup_reaper(
                upstream,
                request_id,
                headers=request_headers or {},
                input_tokens=input_tokens,
                session_id=session_id,
                cold_prefill=cold_prefill,
                request_observation=request_observation,
            )
        await response.aclose()
        await client.aclose()

    async def close(self) -> None:
        """Stop admission and wake requests waiting for capacity."""
        tasks = tuple(self._cleanup_reapers)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self.pool.close()
