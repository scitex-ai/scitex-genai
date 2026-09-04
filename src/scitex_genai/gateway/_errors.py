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


class NoAccountAvailable(GatewayError):
    """No configured account can currently accept a request."""
