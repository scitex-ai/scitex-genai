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

import hashlib
import json
import math
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from ._errors import BudgetExceededError, ModelPolicyError
from ._external_usage import ExternalUsageLedger
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
    anthropic_path_prefix: str = ""
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
        prefix = self.anthropic_path_prefix.rstrip("/")
        if prefix and not prefix.startswith("/"):
            raise ValueError("anthropic_path_prefix must start with /")
        object.__setattr__(self, "anthropic_path_prefix", prefix)
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
            if not (
                self.input_usd_per_million_tokens
                or self.output_usd_per_million_tokens
            ):
                raise ValueError(
                    "max_estimated_usd_per_run requires a non-zero input or "
                    "output token price"
                )
            object.__setattr__(self, "max_estimated_usd_per_run", value)

    @property
    def allowed_models(self) -> frozenset[str]:
        return frozenset((self.canonical_model, *self.model_aliases))


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
    active_health_probe = False
    health_strategy = "external_provider_status"

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
        if path.startswith("/v1/responses"):
            legacy_fields = {"max_tokens", "max_completion_tokens"}.intersection(
                payload
            )
            if legacy_fields:
                names = ", ".join(sorted(legacy_fields))
                raise ModelPolicyError(
                    f"Responses API does not accept {names}; use max_output_tokens"
                )
            max_field = "max_output_tokens"
        else:
            chat_fields = {"max_tokens", "max_completion_tokens"}.intersection(
                payload
            )
            if "max_output_tokens" in payload or len(chat_fields) > 1:
                raise ModelPolicyError(
                    "Chat request has ambiguous output limits; use exactly one of "
                    "max_tokens or max_completion_tokens"
                )
            max_field = (
                "max_completion_tokens"
                if "max_completion_tokens" in chat_fields
                else "max_tokens"
            )
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
            upstream_path = path
            if path.startswith("/v1/messages") and self.policy.anthropic_path_prefix:
                upstream_path = f"{self.policy.anthropic_path_prefix}{path}"
            relayed = await super().relay(
                method,
                path,
                body=encoded,
                headers=outbound_headers,
                upstream_path=upstream_path,
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
        is_stream = "text/event-stream" in content_type
        model_mismatch = False
        try:
            async for chunk in body:
                if is_stream:
                    pending_line.extend(chunk)
                    emittable = bytearray()
                    while b"\n" in pending_line:
                        raw_line, _, remainder = pending_line.partition(b"\n")
                        pending_line = bytearray(remainder)
                        if len(raw_line) > _MAX_AUDIT_BUFFER:
                            raise ModelPolicyError(
                                "Provider stream event exceeded the effective-model "
                                "audit limit"
                            )
                        emittable.extend(raw_line)
                        emittable.extend(b"\n")
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
                        raise ModelPolicyError(
                            "Provider stream event exceeded the effective-model "
                            "audit limit"
                        )
                    if reported_model and reported_model != self.policy.canonical_model:
                        model_mismatch = True
                        raise ModelPolicyError(
                            "Provider reported a model outside the outbound policy: "
                            f"{reported_model!r}"
                        )
                    if emittable:
                        yield bytes(emittable)
                else:
                    if len(captured) + len(chunk) > _MAX_AUDIT_BUFFER:
                        raise ModelPolicyError(
                            "Provider response exceeded the effective-model audit limit"
                        )
                    captured.extend(chunk)
            if is_stream and pending_line:
                raw_line = bytes(pending_line)
                if raw_line.startswith(b"data:"):
                    data = raw_line[5:].strip()
                    if data and data != b"[DONE]":
                        try:
                            candidate = json.loads(data)
                        except ValueError:
                            candidate = None
                        input_tokens, output_tokens, reported_model = (
                            _usage_from_payload(candidate)
                        )
                if reported_model and reported_model != self.policy.canonical_model:
                    model_mismatch = True
                    raise ModelPolicyError(
                        "Provider reported a model outside the outbound policy: "
                        f"{reported_model!r}"
                    )
                yield raw_line
            if not is_stream:
                try:
                    candidate = json.loads(bytes(captured))
                except ValueError:
                    candidate = None
                input_tokens, output_tokens, reported_model = _usage_from_payload(
                    candidate
                )
                if reported_model and reported_model != self.policy.canonical_model:
                    model_mismatch = True
                    raise ModelPolicyError(
                        "Provider reported a model outside the outbound policy: "
                        f"{reported_model!r}"
                    )
                yield bytes(captured)
        finally:
            await self.usage.settle(
                run_key,
                reserved_input=reserved_input,
                reserved_output=reserved_output,
                input_tokens=None if model_mismatch else input_tokens,
                output_tokens=None if model_mismatch else output_tokens,
                reported_model=reported_model,
            )
            self._note(
                f"[external-audit] provider={self.policy.provider} "
                f"requested={self.policy.canonical_model} "
                f"reported={reported_model or 'unreported'} "
                f"input_tokens={input_tokens if input_tokens is not None else 'unknown'} "
                f"output_tokens={output_tokens if output_tokens is not None else 'unknown'}"
            )
