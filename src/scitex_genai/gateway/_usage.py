"""Codex subscription usage polling for quota-aware account selection."""

from __future__ import annotations

import asyncio
from typing import Any

from ._accounts import CodexAccount, CodexAccountPool
from ._errors import UpstreamError

USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"


def _percent(window: Any) -> float | None:
    if not isinstance(window, dict):
        return None
    value = window.get("used_percent")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return min(100.0, max(0.0, float(value)))


class CodexUsageClient:
    """Read quota windows without exposing tokens or account identifiers."""

    def __init__(self, client: Any = None) -> None:
        self._client = client

    async def fetch(self, account: CodexAccount) -> tuple[float | None, float | None]:
        credential = account.credential
        async with account.refresh_lock:
            if credential.needs_refresh():
                await credential.refresh()
        headers = {
            "Authorization": f"Bearer {credential.access_token}",
            "ChatGPT-Account-Id": credential.account_id,
            "User-Agent": "scitex-genai-codex-gateway",
            "Accept": "application/json",
        }
        if self._client is None:
            try:
                import httpx
            except ImportError as exc:
                raise RuntimeError(
                    "Codex usage polling requires scitex-genai[gateway]"
                ) from exc
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(USAGE_URL, headers=headers)
        else:
            response = await self._client.get(USAGE_URL, headers=headers)
        if response.status_code >= 400:
            raise UpstreamError(
                f"Codex usage endpoint returned HTTP {response.status_code}",
                status_code=response.status_code,
            )
        data = response.json()
        rate_limit = data.get("rate_limit") if isinstance(data, dict) else None
        if not isinstance(rate_limit, dict):
            return None, None
        return _percent(rate_limit.get("primary_window")), _percent(
            rate_limit.get("secondary_window")
        )

    async def refresh_pool(self, pool: CodexAccountPool) -> None:
        async def refresh_one(account: CodexAccount) -> None:
            try:
                primary, secondary = await self.fetch(account)
            except Exception:
                return
            await pool.update_usage(
                account,
                primary_used_percent=primary,
                secondary_used_percent=secondary,
            )

        await asyncio.gather(*(refresh_one(account) for account in pool.accounts))
