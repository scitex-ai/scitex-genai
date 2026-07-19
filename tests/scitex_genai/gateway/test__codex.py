from __future__ import annotations

from pathlib import Path

import pytest

from scitex_genai.gateway._accounts import CodexAccount, CodexAccountPool
from scitex_genai.gateway._codex import CodexBackend, CodexTransport
from scitex_genai.gateway._credentials import CodexCredential
from scitex_genai.gateway._errors import RateLimitError, UpstreamError


def _account(alias: str) -> CodexAccount:
    return CodexAccount(
        alias,
        CodexCredential(
            Path(f"/{alias}/auth.json"),
            "access",
            "refresh",
            alias,
            4_000_000_000,
        ),
    )


@pytest.mark.asyncio
async def test_backend_rotates_after_rate_limit() -> None:
    # Arrange
    accounts = [_account("alpha"), _account("beta")]
    pool = CodexAccountPool(accounts)

    class Transport:
        calls: list[str] = []

        async def stream(self, payload, account, *, session_id=""):
            self.calls.append(account.alias)
            if account.alias == "alpha":
                raise RateLimitError("limited", retry_after=60)
            yield {"type": "response.completed", "response": {}}

    transport = Transport()
    backend = CodexBackend(pool, transport)
    # Act
    events = [event async for event in backend.stream({}, session_id="session-a")]
    observed = (
        transport.calls,
        events[0]["type"],
        accounts[0].cooldown_until > 0,
        accounts[0].in_flight,
        accounts[1].in_flight,
    )
    # Assert
    assert observed == (["alpha", "beta"], "response.completed", True, 0, 0)


@pytest.mark.asyncio
async def test_backend_refreshes_once_after_unauthorized() -> None:
    # Arrange
    account = _account("alpha")
    pool = CodexAccountPool([account])
    refresh_count = 0

    async def refresh() -> None:
        nonlocal refresh_count
        refresh_count += 1

    account.credential.refresh = refresh  # type: ignore[method-assign]

    class Transport:
        calls = 0

        async def stream(self, payload, selected, *, session_id=""):
            self.calls += 1
            if self.calls == 1:
                raise UpstreamError("expired", status_code=401)
            yield {"type": "response.completed", "response": {}}

    transport = Transport()
    backend = CodexBackend(pool, transport)
    # Act
    events = [event async for event in backend.stream({})]
    # Assert
    assert (transport.calls, refresh_count, events[0]["type"]) == (
        2,
        1,
        "response.completed",
    )


@pytest.mark.asyncio
async def test_backend_rotates_after_transient_upstream_error() -> None:
    # Arrange
    accounts = [_account("alpha"), _account("beta")]
    pool = CodexAccountPool(accounts)

    class Transport:
        calls: list[str] = []

        async def stream(self, payload, account, *, session_id=""):
            self.calls.append(account.alias)
            if account.alias == "alpha":
                raise UpstreamError("unavailable", status_code=503)
            yield {"type": "response.completed", "response": {}}

    transport = Transport()
    backend = CodexBackend(pool, transport)
    # Act
    events = [event async for event in backend.stream({})]
    # Assert
    assert (transport.calls, events[0]["type"], accounts[0].cooldown_until > 0) == (
        ["alpha", "beta"],
        "response.completed",
        True,
    )


@pytest.mark.asyncio
async def test_transport_reads_streamed_error_body_before_decoding() -> None:
    # Arrange
    httpx = pytest.importorskip("httpx")

    async def reject(request):
        return httpx.Response(
            400,
            json={"error": {"message": "unsupported request field"}},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(reject))
    transport = CodexTransport(client=client)
    # Act
    async def consume() -> None:
        async for _ in transport.stream({}, _account("alpha")):
            pass

    # Assert
    with pytest.raises(UpstreamError, match="unsupported request field"):
        await consume()
    await client.aclose()


@pytest.mark.asyncio
async def test_transport_hashes_oversized_session_header() -> None:
    # Arrange
    httpx = pytest.importorskip("httpx")
    observed_session_id = ""

    async def accept(request):
        nonlocal observed_session_id
        observed_session_id = request.headers["session_id"]
        return httpx.Response(
            200,
            text='data: {"type":"response.completed","response":{}}\n\n',
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(accept))
    transport = CodexTransport(client=client)
    # Act
    events = [
        event
        async for event in transport.stream(
            {}, _account("alpha"), session_id="claude-session-" * 10
        )
    ]
    await client.aclose()
    # Assert
    assert (len(observed_session_id), events[0]["type"]) == (
        64,
        "response.completed",
    )
