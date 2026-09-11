"""Bounded, content-free reachability probes for inference upstreams."""

from __future__ import annotations

import asyncio
import errno
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit

DEFAULT_HEALTH_PROBE_TIMEOUT_S = 1.0
LOCAL_INFERENCE_HEALTH_PATH = "/v1/models"


@dataclass(frozen=True)
class UpstreamReachability:
    """One observed upstream-health result, safe to expose from ``/health``."""

    reachable: bool
    reason: str
    latency_ms: float
    checked_at: str
    http_status: int | None = None
    ready: bool | None = None
    consecutive_failures: int = 0

    @property
    def readiness(self) -> bool:
        """Effective readiness after the gateway's failure hysteresis."""
        return self.reachable if self.ready is None else self.ready

    def as_dict(self) -> dict[str, Any]:
        """Return the stable JSON shape used by the gateway health endpoint."""
        return asdict(self)


def public_upstream_url(url: str) -> str:
    """Remove credentials, query strings, and fragments from a reported URL."""
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname or ""
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        netloc = hostname
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", ""))
    except ValueError:
        return "<invalid-upstream-url>"


def _checked_at() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _is_connection_refused(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, ConnectionRefusedError):
            return True
        if isinstance(current, OSError) and current.errno == errno.ECONNREFUSED:
            return True
        current = current.__cause__ or current.__context__
    return False


async def probe_upstream(
    base_url: str,
    *,
    path: str = LOCAL_INFERENCE_HEALTH_PATH,
    timeout_s: float = DEFAULT_HEALTH_PROBE_TIMEOUT_S,
    transport: Any = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> UpstreamReachability:
    """Probe the local inference control plane without consuming a response body."""
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError(
            "Gateway health probes require scitex-genai[gateway]"
        ) from exc

    started = monotonic()
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s),
            transport=transport,
            follow_redirects=False,
        ) as client:
            async with client.stream("GET", f"{base_url.rstrip('/')}{path}") as response:
                status_code = response.status_code
        latency_ms = round((monotonic() - started) * 1_000, 1)
        reachable = 200 <= status_code < 300
        return UpstreamReachability(
            reachable=reachable,
            reason="responded" if reachable else f"http_status_{status_code}",
            latency_ms=latency_ms,
            checked_at=_checked_at(),
            http_status=status_code,
        )
    except (asyncio.TimeoutError, httpx.TimeoutException):
        reason = "timeout"
    except httpx.ConnectError as exc:
        reason = (
            "connection_refused" if _is_connection_refused(exc) else "connection_error"
        )
    except httpx.InvalidURL:
        reason = "invalid_url"
    except httpx.TransportError:
        reason = "transport_error"
    return UpstreamReachability(
        reachable=False,
        reason=reason,
        latency_ms=round((monotonic() - started) * 1_000, 1),
        checked_at=_checked_at(),
    )


def timed_out_reachability(started: float, *, monotonic: Callable[[], float]) -> UpstreamReachability:
    """Build the same safe timeout result for an outer end-to-end deadline."""
    return UpstreamReachability(
        reachable=False,
        reason="timeout",
        latency_ms=round((monotonic() - started) * 1_000, 1),
        checked_at=_checked_at(),
    )
