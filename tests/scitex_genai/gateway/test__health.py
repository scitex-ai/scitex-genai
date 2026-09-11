from __future__ import annotations

import errno

import httpx
import pytest

from scitex_genai.gateway._health import probe_upstream, public_upstream_url


@pytest.mark.asyncio
async def test_probe_reports_reachable_without_reading_response_content() -> None:
    # Arrange
    paths: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, request=request)

    transport = httpx.MockTransport(respond)
    # Act
    result = await probe_upstream(
        "http://engine.internal:18773", transport=transport, monotonic=lambda: 1.0
    )
    # Assert
    assert (
        result.reachable,
        result.reason,
        result.http_status,
        result.latency_ms,
        paths,
    ) == (
        True,
        "responded",
        200,
        0.0,
        ["/v1/models"],
    )


@pytest.mark.asyncio
async def test_probe_reports_connection_refused_without_exception_text() -> None:
    # Arrange
    def refuse(request: httpx.Request) -> httpx.Response:
        error = OSError(errno.ECONNREFUSED, "secret endpoint detail")
        raise httpx.ConnectError("secret transport detail", request=request) from error

    transport = httpx.MockTransport(refuse)
    # Act
    result = await probe_upstream("http://engine.internal", transport=transport)
    # Assert
    assert (result.reachable, result.reason, result.http_status) == (
        False,
        "connection_refused",
        None,
    )


@pytest.mark.asyncio
async def test_probe_reports_timeout_without_exception_text() -> None:
    # Arrange
    def time_out(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("secret timeout detail", request=request)

    transport = httpx.MockTransport(time_out)
    # Act
    result = await probe_upstream("http://engine.internal", transport=transport)
    # Assert
    assert (result.reachable, result.reason, result.http_status) == (
        False,
        "timeout",
        None,
    )


def test_public_upstream_url_removes_every_credential_surface() -> None:
    # Arrange
    url = "https://user:secret@engine.internal:18773/root?token=secret#secret"
    # Act
    public = public_upstream_url(url)
    # Assert
    assert public == "https://engine.internal:18773/root"
