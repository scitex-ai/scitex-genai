"""Policy-enforcing relay for paid OpenAI-compatible providers.

The boundary is deliberately here, immediately before outbound HTTP.  A
harness model selector is presentation, not enforcement: Hermes can create a
built-in provider from ``DEEPSEEK_API_KEY`` and an operator or model can issue
``/model deepseek-v4-pro``.  Containers therefore receive only the local
gateway credential.  This relay owns the vendor credential, canonicalises an
explicit allowlist, and refuses every other model before opening upstream.

Only payload-free accounting is retained: counts, token usage, estimated cost,
and a digest of the caller's run identity.  Prompts and completions are never
journalled or stored.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import AsyncIterator, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from ._errors import BudgetExceededError, ModelPolicyError
from ._inference import InferenceBackend, RelayedResponse

_RUN_HEADERS = ("x-scitex-run-id", "x-scitex-session-id", "session_id", "x-session-id")
_SAFE_UPSTREAM_HEADERS = frozenset(
    {"accept", "content-type", "user-agent", "anthropic-version", "anthropic-beta"}
)
_RUN_DOMAIN = b"scitex-genai-external-run\0"
_MAX_AUDIT_BUFFER = 8 * 1024 * 1024


def _header(headers: Mapping[str, str], wanted: str) -> str:
    for name, value in headers.items():
        if name.lower() == wanted:
            return value.strip()
    return ""


def _run_key(headers: Mapping[str, str]) -> str:
    raw = next((_header(headers, name) for name in _RUN_HEADERS if _header(headers, name)), "")
    raw = raw or "unattributed"
    return hashlib.sha256(_RUN_DOMAIN + raw.encode()).hexdigest()


def _positive_or_none(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    number = int(value)
    if number < 1:
        raise ValueError(f"{name} must be >= 1 when set")
    return number


@dataclass(frozen=True)
class ExternalProviderPolicy:
    """Outbound model firewall and bounded-spend policy for one provider."""

    provider: str
    upstream_api_key: str = field(repr=False)
    canonical_model: str = "deepseek-flash"
    model_aliases: tuple[str, ...] = ("deepseek-v4-flash",)
    max_tokens_per_request: int = 16_384
    max_requests_per_run: int | None = 100
    max_input_tokens_per_run: int | None = 5_000_000
    max_output_tokens_per_run: int | None = 200_000
    max_total_tokens_per_run: int | None = 5_200_000
    max_estimated_usd_per_run: float | None = None
    input_usd_per_million_tokens: float = 0.0
    output_usd_per_million_tokens: float = 0.0

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider must be non-empty")
        if not self.upstream_api_key.strip():
            raise ValueError("upstream_api_key must be non-empty")
        if not self.canonical_model.strip():
            raise ValueError("canonical_model must be non-empty")
        aliases = tuple(alias.strip() for alias in self.model_aliases)
        if any(not alias for alias in aliases):
            raise ValueError("model_aliases cannot contain empty names")
        if self.canonical_model in aliases:
            raise ValueError("model_aliases must not repeat canonical_model")
        if len(set(aliases)) != len(aliases):
            raise ValueError("model_aliases must be unique")
        object.__setattr__(self, "model_aliases", aliases)
        object.__setattr__(
            self,
            "max_tokens_per_request",
            _positive_or_none("max_tokens_per_request", self.max_tokens_per_request),
        )
        for name in (
            "max_requests_per_run",
            "max_input_tokens_per_run",
            "max_output_tokens_per_run",
            "max_total_tokens_per_run",
        ):
            object.__setattr__(self, name, _positive_or_none(name, getattr(self, name)))
        for name in (
            "input_usd_per_million_tokens",
            "output_usd_per_million_tokens",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
            object.__setattr__(self, name, value)
        if self.max_estimated_usd_per_run is not None:
            value = float(self.max_estimated_usd_per_run)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("max_estimated_usd_per_run must be finite and positive")
            object.__setattr__(self, "max_estimated_usd_per_run", value)

    @property
    def allowed_models(self) -> frozenset[str]:
        return frozenset((self.canonical_model, *self.model_aliases))


@dataclass
class UsageCounters:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reserved_input_tokens: int = 0
    reserved_output_tokens: int = 0
    responses_with_usage: int = 0
    responses_without_usage: int = 0
    reported_model_mismatches: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ExternalUsageLedger:
    """Concurrency-safe, in-memory ceilings for one gateway incarnation."""

    def __init__(self, policy: ExternalProviderPolicy) -> None:
        self.policy = policy
        self.total = UsageCounters()
        self.runs: dict[str, UsageCounters] = {}
        self.last_reported_model = ""
        self._lock = asyncio.Lock()

    async def reserve(self, run_key: str, *, estimated_input: int, output: int) -> None:
        async with self._lock:
            counters = self.runs.setdefault(run_key, UsageCounters())
            projected = UsageCounters(
                requests=counters.requests + 1,
                input_tokens=counters.input_tokens,
                output_tokens=counters.output_tokens,
                reserved_input_tokens=counters.reserved_input_tokens + estimated_input,
                reserved_output_tokens=counters.reserved_output_tokens + output,
            )
            checks = (
                (self.policy.max_requests_per_run, projected.requests, "requests"),
                (
                    self.policy.max_input_tokens_per_run,
                    projected.input_tokens + projected.reserved_input_tokens,
                    "input tokens",
                ),
                (
                    self.policy.max_output_tokens_per_run,
                    projected.output_tokens + projected.reserved_output_tokens,
                    "output tokens",
                ),
                (
                    self.policy.max_total_tokens_per_run,
                    projected.input_tokens
                    + projected.reserved_input_tokens
                    + projected.output_tokens
                    + projected.reserved_output_tokens,
                    "total tokens",
                ),
            )
            for limit, value, label in checks:
                if limit is not None and value > limit:
                    raise BudgetExceededError(
                        f"External-provider run budget would exceed {label}: "
                        f"{value}/{limit}"
                    )
            if self.policy.max_estimated_usd_per_run is not None:
                projected_cost = (
                    (
                        projected.input_tokens + projected.reserved_input_tokens
                    )
                    * self.policy.input_usd_per_million_tokens
                    + (
                        projected.output_tokens + projected.reserved_output_tokens
                    )
                    * self.policy.output_usd_per_million_tokens
                ) / 1_000_000
                if projected_cost > self.policy.max_estimated_usd_per_run:
                    raise BudgetExceededError(
                        "External-provider run budget would exceed estimated cost: "
                        f"${projected_cost:.6f}/${self.policy.max_estimated_usd_per_run:.6f}"
                    )
            counters.requests += 1
            counters.reserved_input_tokens += estimated_input
            counters.reserved_output_tokens += output
            self.total.requests += 1
            self.total.reserved_input_tokens += estimated_input
            self.total.reserved_output_tokens += output

    async def settle(
        self,
        run_key: str,
        *,
        reserved_input: int,
        reserved_output: int,
        input_tokens: int | None,
        output_tokens: int | None,
        reported_model: str,
    ) -> None:
        async with self._lock:
            counters = self.runs[run_key]
            counters.reserved_input_tokens = max(
                0, counters.reserved_input_tokens - reserved_input
            )
            self.total.reserved_input_tokens = max(
                0, self.total.reserved_input_tokens - reserved_input
            )
            counters.reserved_output_tokens = max(
                0, counters.reserved_output_tokens - reserved_output
            )
            self.total.reserved_output_tokens = max(
                0, self.total.reserved_output_tokens - reserved_output
            )
            if input_tokens is None and output_tokens is None:
                # Conservatively keep the reserved output charged when the
                # provider omits usage, so missing telemetry cannot open the
                # budget gate.
                counters.input_tokens += reserved_input
                counters.output_tokens += reserved_output
                self.total.input_tokens += reserved_input
                self.total.output_tokens += reserved_output
                counters.responses_without_usage += 1
                self.total.responses_without_usage += 1
            else:
                input_value = max(0, int(input_tokens or 0))
                output_value = max(0, int(output_tokens or 0))
                counters.input_tokens += input_value
                counters.output_tokens += output_value
                self.total.input_tokens += input_value
                self.total.output_tokens += output_value
                counters.responses_with_usage += 1
                self.total.responses_with_usage += 1
            if reported_model:
                self.last_reported_model = reported_model
                if reported_model != self.policy.canonical_model:
                    counters.reported_model_mismatches += 1
                    self.total.reported_model_mismatches += 1

    def snapshot(self) -> dict[str, Any]:
        usage = asdict(self.total)
        usage["total_tokens"] = self.total.total_tokens
        usage["estimated_cost_usd"] = round(
            (
                self.total.input_tokens * self.policy.input_usd_per_million_tokens
                + self.total.output_tokens * self.policy.output_usd_per_million_tokens
            )
            / 1_000_000,
            8,
        )
        return {
            "provider": self.policy.provider,
            "canonical_model": self.policy.canonical_model,
            "last_reported_model": self.last_reported_model or None,
            "usage": usage,
            "runs_observed": len(self.runs),
        }


def _usage_from_payload(payload: Any) -> tuple[int | None, int | None, str]:
    if not isinstance(payload, dict):
        return None, None, ""
    nested = payload.get("message") or payload.get("response")
    if isinstance(nested, dict):
        nested_input, nested_output, nested_model = _usage_from_payload(nested)
    else:
        nested_input, nested_output, nested_model = None, None, ""
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    input_tokens = usage.get(
        "prompt_tokens", usage.get("input_tokens", nested_input)
    )
    output_tokens = usage.get(
        "completion_tokens", usage.get("output_tokens", nested_output)
    )
    model = payload.get("model")
    if not isinstance(model, str):
        model = nested_model
    return input_tokens, output_tokens, model


class ExternalProviderBackend(InferenceBackend):
    """Inference transport with an outbound credential and model firewall."""

    provider = "external-provider"

    def __init__(self, *args: Any, policy: ExternalProviderPolicy, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if len(self.pool.upstreams) != 1:
            raise ValueError("external-provider relay requires exactly one upstream")
        self.policy = policy
        self.provider = f"external:{policy.provider}"
        self.usage = ExternalUsageLedger(policy)

    def health_status(self) -> dict[str, Any]:
        return self.usage.snapshot()

    def models_payload(self) -> dict[str, Any]:
        """Synthetic discovery: expose the allowlist, never the vendor catalog."""
        return {
            "object": "list",
            "data": [
                {
                    "id": self.policy.canonical_model,
                    "object": "model",
                    "owned_by": self.policy.provider,
                }
            ],
        }

    async def relay(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None,
        headers: Mapping[str, str],
    ) -> RelayedResponse:
        if method.upper() != "POST":
            # Never expose vendor model discovery: it advertises models the
            # policy intentionally makes unreachable.
            raise ModelPolicyError("External-provider gateway allows POST requests only")
        try:
            payload = json.loads(body or b"")
        except (TypeError, ValueError) as exc:
            raise ModelPolicyError("External-provider request must be a JSON object") from exc
        if not isinstance(payload, dict):
            raise ModelPolicyError("External-provider request must be a JSON object")
        requested = payload.get("model")
        if not isinstance(requested, str) or not requested:
            raise ModelPolicyError("External-provider request must name a model")
        if requested not in self.policy.allowed_models:
            raise ModelPolicyError(
                f"Model {requested!r} is blocked by the outbound policy; "
                f"only {self.policy.canonical_model!r} is allowed"
            )
        payload["model"] = self.policy.canonical_model
        max_field = "max_completion_tokens" if "max_completion_tokens" in payload else "max_tokens"
        requested_output = payload.get(max_field, self.policy.max_tokens_per_request)
        if isinstance(requested_output, bool):
            raise ModelPolicyError(f"{max_field} must be a positive integer")
        try:
            requested_output = int(requested_output)
        except (TypeError, ValueError) as exc:
            raise ModelPolicyError(f"{max_field} must be a positive integer") from exc
        if requested_output < 1 or requested_output > self.policy.max_tokens_per_request:
            raise BudgetExceededError(
                f"Request output ceiling exceeded: {requested_output}/"
                f"{self.policy.max_tokens_per_request}"
            )
        payload[max_field] = requested_output
        if payload.get("stream") is True and path.startswith("/v1/chat/completions"):
            stream_options = payload.setdefault("stream_options", {})
            if isinstance(stream_options, dict):
                stream_options["include_usage"] = True
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        # A byte-count plus protocol allowance is deliberately conservative:
        # without DeepSeek's tokenizer locally, chars/4 is not a hard bound.
        # This estimate can reject early but cannot silently multiply spend.
        estimated_input = len(encoded) + 1_024
        run_key = _run_key(headers)
        await self.usage.reserve(
            run_key, estimated_input=estimated_input, output=requested_output
        )
        outbound_headers = {
            name: value
            for name, value in headers.items()
            if name.lower() in _SAFE_UPSTREAM_HEADERS
        }
        outbound_headers["authorization"] = f"Bearer {self.policy.upstream_api_key}"
        try:
            relayed = await super().relay(
                method, path, body=encoded, headers=outbound_headers
            )
        except Exception:
            await self.usage.settle(
                run_key,
                reserved_input=estimated_input,
                reserved_output=requested_output,
                input_tokens=None,
                output_tokens=None,
                reported_model="",
            )
            raise
        relayed.body = self._audit_body(
            relayed.body,
            content_type=relayed.content_type,
            run_key=run_key,
            reserved_output=requested_output,
            reserved_input=estimated_input,
        )
        return relayed

    async def _audit_body(
        self,
        body: AsyncIterator[bytes],
        *,
        content_type: str,
        run_key: str,
        reserved_output: int,
        reserved_input: int,
    ) -> AsyncIterator[bytes]:
        captured = bytearray()
        pending_line = bytearray()
        input_tokens: int | None = None
        output_tokens: int | None = None
        reported_model = ""
        try:
            async for chunk in body:
                if "text/event-stream" in content_type:
                    pending_line.extend(chunk)
                    while b"\n" in pending_line:
                        raw_line, _, remainder = pending_line.partition(b"\n")
                        pending_line = bytearray(remainder)
                        if not raw_line.startswith(b"data:"):
                            continue
                        data = raw_line[5:].strip()
                        if not data or data == b"[DONE]":
                            continue
                        try:
                            candidate = json.loads(data)
                        except ValueError:
                            continue
                        seen_input, seen_output, seen_model = _usage_from_payload(
                            candidate
                        )
                        if seen_input is not None:
                            input_tokens = seen_input
                        if seen_output is not None:
                            output_tokens = seen_output
                        if seen_model:
                            reported_model = seen_model
                    if len(pending_line) > _MAX_AUDIT_BUFFER:
                        pending_line.clear()
                elif len(captured) < _MAX_AUDIT_BUFFER:
                    captured.extend(chunk[: _MAX_AUDIT_BUFFER - len(captured)])
                yield chunk
        finally:
            candidates: list[Any] = []
            raw = bytes(captured)
            if "text/event-stream" not in content_type:
                try:
                    candidates.append(json.loads(raw))
                except ValueError:
                    pass
            for candidate in candidates:
                seen_input, seen_output, seen_model = _usage_from_payload(candidate)
                if seen_input is not None:
                    input_tokens = seen_input
                if seen_output is not None:
                    output_tokens = seen_output
                if seen_model:
                    reported_model = seen_model
            await self.usage.settle(
                run_key,
                reserved_input=reserved_input,
                reserved_output=reserved_output,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reported_model=reported_model,
            )
            self._note(
                f"[external-audit] provider={self.policy.provider} "
                f"requested={self.policy.canonical_model} "
                f"reported={reported_model or 'unreported'} "
                f"input_tokens={input_tokens if input_tokens is not None else 'unknown'} "
                f"output_tokens={output_tokens if output_tokens is not None else 'unknown'}"
            )
