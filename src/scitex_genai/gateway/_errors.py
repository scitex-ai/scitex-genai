"""Errors raised by the structured model gateway."""


class GatewayError(RuntimeError):
    """Base class for gateway failures safe to map to an API response."""


class CredentialError(GatewayError):
    """A provider credential is missing, malformed, or cannot be refreshed."""


class UpstreamError(GatewayError):
    """The upstream model service rejected or failed a request."""

    #: The Anthropic-shaped ``error.type`` a response built from this carries.
    error_type = "api_error"

    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class UpstreamReloading(UpstreamError):
    """503: this conversation's home upstream died moments ago and is reloading.

    Deliberately NOT a failover. Measured 2026-09-05 on the Qwen pair: a
    request that crashes one replica (vLLM EngineCore assert) is retried by
    the harness, the sticky placement has just been forgotten, so the pool
    re-places the conversation on the other replica and the same request
    crashes it too - 8 of 24 double outages that day were within 90 s of each
    other. A recently-died home upstream is far more likely reloading (~90 s)
    than gone, and the request that killed it is the last thing the survivor
    should be handed. Tell the caller to retry instead.
    """

    error_type = "upstream_reloading"

    def __init__(self, message: str, *, retry_after_s: float) -> None:
        super().__init__(message, status_code=503)
        self.retry_after_s = retry_after_s


class UpstreamUnreachable(UpstreamError):
    """No upstream produced an HTTP response at all.

    Distinct from a response with an error status, which is relayed verbatim.
    ``error.type`` is ``upstream_unreachable`` — the wording the fleet's
    agents already know from the hoist proxy this replaces.
    """

    error_type = "upstream_unreachable"

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=502)


class RateLimitError(UpstreamError):
    """An upstream account is temporarily rate limited."""

    def __init__(self, message: str, *, retry_after: float = 60.0) -> None:
        super().__init__(message, status_code=429)
        self.retry_after = max(1.0, retry_after)


class HomeMemberReloading(GatewayError):
    """The session's sticky member is cooling and too RECENTLY to fail over.

    Raised by :class:`~._pool.StickyPool.acquire` instead of re-placing the
    session on another member. Carries ``retry_after_s`` so the relay can
    answer 503 with a Retry-After the harness's own backoff will honour.
    """

    def __init__(self, message: str, *, retry_after_s: float) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class NoAccountAvailable(GatewayError):
    """No configured account can currently accept a request."""
