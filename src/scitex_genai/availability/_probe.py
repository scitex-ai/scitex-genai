"""One bounded key/model check; HTTP codes cross the boundary verbatim."""

from __future__ import annotations

import errno
import json
import socket
import time
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Mapping

import httpx
from scitex_dev.status import Check, StatusCode

from ._routes import provider_route


@dataclass(frozen=True)
class ProviderAvailability:
    provider: str
    model: str
    endpoint_url: str
    check: Check
    reset_at: float | None = None

    @property
    def available(self) -> bool | None:
        """True=verified, False=rejected, None=not established."""
        return self.check.verdict.ok

    @property
    def status(self) -> StatusCode:
        return self.check.cause

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "endpoint_url": self.endpoint_url,
            "check": self.check.to_dict(),
            "reset_at": self.reset_at,
        }


def _reset_at(body: object, headers: Mapping, now: float) -> float | None:
    if isinstance(body, dict):
        for name in ("reset_at", "resets_at", "retry_at", "resetAt"):
            value = body.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value / 1000 if value > 1e12 else float(value)
            if isinstance(value, str):
                try:
                    return datetime.fromisoformat(
                        value.replace("Z", "+00:00")
                    ).timestamp()
                except ValueError:
                    pass
        for value in body.values():
            if (
                isinstance(value, dict)
                and (reset := _reset_at(value, {}, now)) is not None
            ):
                return reset
    retry = headers.get("retry-after") or headers.get("Retry-After")
    if retry:
        try:
            return now + max(0, float(retry))
        except ValueError:
            try:
                return parsedate_to_datetime(retry).timestamp()
            except (TypeError, ValueError):
                pass
    # Some billing providers supply their reset only inside the error message.
    if isinstance(body, dict):
        import re

        for stamp in re.findall(
            r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:\d\d)",
            json.dumps(body),
        ):
            try:
                reset = datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
                if reset > now:
                    return reset
            except ValueError:
                pass
    return None


def _transport_status(error: BaseException, endpoint: str) -> StatusCode:
    if isinstance(error, httpx.TimeoutException):
        return StatusCode(
            kind="errno",
            code="ETIMEDOUT",
            message=f"Probe timed out at {endpoint}; retry the probe",
        )
    current = error
    while current is not None:
        if isinstance(current, socket.gaierror):
            return StatusCode(
                kind="grpc",
                code="UNAVAILABLE",
                message=f"Name resolution failed for {endpoint}; check DNS and retry",
            )
        if isinstance(current, OSError) and current.errno in errno.errorcode:
            return StatusCode(
                kind="errno",
                code=errno.errorcode[current.errno],
                message=f"Probe transport failed at {endpoint}; check connectivity and retry",
            )
        current = current.__cause__ or current.__context__
    return StatusCode(
        kind="grpc",
        code="UNAVAILABLE",
        message=f"Probe received no HTTP response at {endpoint}; check connectivity and retry",
    )


def probe_provider_key(
    provider: str,
    model: str,
    api_key: str,
    *,
    endpoint_url: str | None = None,
    protocol: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    session_id: str | None = None,
    timeout_s: float = 20,
    transport: httpx.BaseTransport | None = None,
) -> ProviderAvailability:
    """Check this key's actual inference access using only ``Reply OK.``.

    A models-list response alone does not establish entitlement or credits.
    The result contains a Check with a native StatusCode cause; transport
    failures have UNKNOWN verdicts and never invent an HTTP status. Secrets
    and full upstream error bodies are excluded from results and logs.
    """
    route = provider_route(
        provider, model, endpoint_url=endpoint_url, protocol=protocol
    )
    if (
        not isinstance(api_key, str)
        or not api_key.strip()
        or any(c in api_key for c in "\r\n")
    ):
        raise ValueError("api_key must be non-empty and contain no newline")
    if not model or timeout_s <= 0:
        raise ValueError("model and a positive timeout_s are required")
    headers = {**(extra_headers or {}), "User-Agent": "scitex-genai/availability-probe"}
    if provider in ("opencode-go", "opencode-zen", "opencode"):
        headers["x-opencode-session"] = session_id or "scitex-genai:availability-probe"
    payload = {"model": model, "stream": False}
    if route.protocol == "openai-responses":
        payload.update(input="Reply OK.", max_output_tokens=16)
        headers["Authorization"] = "Bearer " + api_key
    else:
        payload.update(
            messages=[{"role": "user", "content": "Reply OK."}], max_tokens=16
        )
        if route.protocol == "anthropic-messages":
            headers.update({"x-api-key": api_key, "anthropic-version": "2023-06-01"})
        else:
            headers["Authorization"] = "Bearer " + api_key
    try:
        with httpx.Client(
            timeout=timeout_s, transport=transport, follow_redirects=False
        ) as client:
            with client.stream(
                "POST", route.endpoint_url, json=payload, headers=headers
            ) as response:
                code = response.status_code
                response_headers = response.headers
                data = bytearray()
                for chunk in response.iter_bytes():
                    data.extend(chunk[: max(0, 65536 - len(data))])
                    if len(data) >= 65536:
                        break
        try:
            body = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            body = None
    except httpx.TransportError as error:
        status = _transport_status(error, route.endpoint_url)
        check = Check.unknown(
            "provider_key_available",
            "The probe received no usable HTTP response",
            hint="Retry the availability probe; do not conclude the key is invalid",
            cause=status,
        )
        return ProviderAvailability(provider, model, route.endpoint_url, check)
    status = StatusCode(
        kind="http",
        code=code,
        message=f"Provider probe returned HTTP {code} at {route.endpoint_url}; retry this probe to verify",
    )
    usable = (
        isinstance(body, dict)
        and not body.get("error")
        and (
            bool(body.get("choices"))
            and isinstance(body["choices"], list)
            or isinstance(body.get("output"), list)
            and body.get("status") not in ("failed", "cancelled")
            or body.get("type") == "message"
            and isinstance(body.get("content"), list)
        )
    )
    if 200 <= code < 300 and usable:
        check = Check.ok(
            "provider_key_available",
            "The synthetic inference request completed",
            cause=status,
        )
    elif code in (400, 401, 402, 403, 404, 429):
        # Retain a small set of observed billing phrases for callers handling
        # a provider's HTTP 400. Never return arbitrary body text or echoed keys.
        text = json.dumps(body).lower() if body is not None else ""
        observed = next(
            (
                phrase
                for phrase in (
                    "insufficient credits",
                    "credits exhausted",
                    "out of credits",
                    "usage limit",
                    "active subscription",
                    "trains on request data",
                )
                if phrase in text
            ),
            None,
        )
        detail = f"Provider rejected the synthetic request with HTTP {code}"
        if observed:
            detail += f"; response contains '{observed}'"
        check = Check.not_ok(
            "provider_key_available",
            detail,
            hint="Try another configured key or retry after the provider reset",
            cause=status,
        )
    else:
        check = Check.unknown(
            "provider_key_available",
            f"HTTP {code} did not establish inference availability",
            hint="Retry the probe; check the endpoint and protocol",
            cause=status,
        )
    return ProviderAvailability(
        provider,
        model,
        route.endpoint_url,
        check,
        _reset_at(body, response_headers, time.time()),
    )
