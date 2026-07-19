"""Streaming client for the ChatGPT Codex Responses transport."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from ._accounts import CodexAccount, CodexAccountPool
from ._errors import CredentialError, NoAccountAvailable, RateLimitError, UpstreamError
from ._usage import CodexUsageClient

DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api"


def codex_url(base_url: str) -> str:
    normalized = (base_url or DEFAULT_CODEX_BASE_URL).rstrip("/")
    if normalized.endswith("/codex/responses"):
        return normalized
    if normalized.endswith("/codex"):
        return f"{normalized}/responses"
    return f"{normalized}/codex/responses"


def _response_error(response: Any) -> str:
    try:
        data = response.json()
    except (ValueError, TypeError):
        return f"Codex upstream returned HTTP {response.status_code}"
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    return f"Codex upstream returned HTTP {response.status_code}"


def _retry_after(response: Any) -> float:
    value = response.headers.get("retry-after", "")
    try:
        return max(1.0, float(value))
    except (TypeError, ValueError):
        return 60.0


async def _parse_sse(response: Any) -> AsyncIterator[dict[str, Any]]:
    buffer = ""
    async for chunk in response.aiter_text():
        buffer += chunk.replace("\r\n", "\n")
        while "\n\n" in buffer:
            frame, buffer = buffer.split("\n\n", 1)
            data = "\n".join(
                line[5:].strip()
                for line in frame.splitlines()
                if line.startswith("data:")
            )
            if not data or data == "[DONE]":
                continue
            try:
                event = json.loads(data)
            except ValueError:
                continue
            if isinstance(event, dict):
                yield event


class CodexTransport:
    """Make raw Codex subscription requests; never run returned tools."""

    def __init__(self, *, base_url: str = DEFAULT_CODEX_BASE_URL, client: Any = None):
        self.base_url = base_url
        self._client = client

    async def stream(
        self,
        payload: dict[str, Any],
        account: CodexAccount,
        *,
        session_id: str = "",
    ) -> AsyncIterator[dict[str, Any]]:
        credential = account.credential
        async with account.refresh_lock:
            if credential.needs_refresh():
                await credential.refresh()

        headers = {
            "Authorization": f"Bearer {credential.access_token}",
            "chatgpt-account-id": credential.account_id,
            "OpenAI-Beta": "responses=experimental",
            "originator": "scitex-genai",
            "User-Agent": "scitex-genai-codex-gateway",
            "accept": "text/event-stream",
            "content-type": "application/json",
        }
        if session_id:
            headers["session_id"] = session_id

        if self._client is None:
            try:
                import httpx
            except ImportError as exc:
                raise CredentialError(
                    "Codex transport requires scitex-genai[gateway]"
                ) from exc
            async with httpx.AsyncClient(timeout=600.0) as client:
                async for event in self._stream_with_client(
                    client, payload, headers
                ):
                    yield event
        else:
            async for event in self._stream_with_client(
                self._client, payload, headers
            ):
                yield event

    async def _stream_with_client(
        self, client: Any, payload: dict[str, Any], headers: dict[str, str]
    ) -> AsyncIterator[dict[str, Any]]:
        async with client.stream(
            "POST", codex_url(self.base_url), headers=headers, json=payload
        ) as response:
            if response.status_code == 429:
                raise RateLimitError(
                    _response_error(response), retry_after=_retry_after(response)
                )
            if response.status_code >= 400:
                raise UpstreamError(
                    _response_error(response), status_code=response.status_code
                )
            async for event in _parse_sse(response):
                yield event


class CodexBackend:
    """Apply account scheduling and failover around :class:`CodexTransport`."""

    def __init__(
        self,
        pool: CodexAccountPool,
        transport: CodexTransport,
        usage_client: CodexUsageClient | None = None,
    ) -> None:
        self.pool = pool
        self.transport = transport
        self.usage_client = usage_client or CodexUsageClient()

    async def refresh_usage(self) -> None:
        await self.usage_client.refresh_pool(self.pool)

    async def stream(
        self, payload: dict[str, Any], *, session_id: str = ""
    ) -> AsyncIterator[dict[str, Any]]:
        attempted: set[str] = set()
        refreshed_after_unauthorized: set[str] = set()
        last_error: Exception | None = None
        while len(attempted) < len(self.pool.accounts):
            try:
                account = await self.pool.acquire(session_id, exclude=attempted)
            except NoAccountAvailable:
                break
            attempted.add(account.alias)
            try:
                async for event in self.transport.stream(
                    payload, account, session_id=session_id
                ):
                    yield event
                return
            except RateLimitError as exc:
                last_error = exc
                await self.pool.cool_down(account, exc.retry_after)
            except CredentialError as exc:
                last_error = exc
                await self.pool.cool_down(account, 60)
            except UpstreamError as exc:
                last_error = exc
                if exc.status_code == 401:
                    if account.alias in refreshed_after_unauthorized:
                        await self.pool.cool_down(account, 60)
                    else:
                        try:
                            async with account.refresh_lock:
                                await account.credential.refresh()
                        except CredentialError as refresh_error:
                            last_error = refresh_error
                            await self.pool.cool_down(account, 60)
                        else:
                            refreshed_after_unauthorized.add(account.alias)
                            attempted.discard(account.alias)
                elif exc.status_code >= 500:
                    await self.pool.cool_down(account, 10)
                else:
                    raise
            finally:
                await self.pool.release(account)
        if last_error is not None:
            raise last_error
        raise NoAccountAvailable("No Codex account could serve the request")
