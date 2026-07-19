from __future__ import annotations

from pathlib import Path

import pytest

from scitex_genai.gateway._accounts import CodexAccount, CodexAccountPool
from scitex_genai.gateway._credentials import CodexCredential


def _account(alias: str) -> CodexAccount:
    credential = CodexCredential(
        path=Path(f"/{alias}/auth.json"),
        access_token="access",
        refresh_token="refresh",
        account_id=f"id-{alias}",
        expires_at=4_000_000_000,
    )
    return CodexAccount(alias, credential)


@pytest.mark.asyncio
async def test_pool_keeps_sessions_sticky() -> None:
    # Arrange
    pool = CodexAccountPool([_account("alpha"), _account("beta")])
    # Act
    first = await pool.acquire("session-a")
    await pool.release(first)
    second = await pool.acquire("session-a")
    # Assert
    assert second.alias == first.alias


@pytest.mark.asyncio
async def test_pool_spreads_new_sessions_across_in_flight_accounts() -> None:
    # Arrange
    pool = CodexAccountPool([_account("alpha"), _account("beta")])
    # Act
    first = await pool.acquire("session-a")
    second = await pool.acquire("session-b")
    # Assert
    assert second.alias != first.alias


@pytest.mark.asyncio
async def test_cooldown_breaks_sticky_binding() -> None:
    # Arrange
    pool = CodexAccountPool([_account("alpha"), _account("beta")])
    # Act
    first = await pool.acquire("session-a")
    await pool.release(first)
    await pool.cool_down(first, 60)
    second = await pool.acquire("session-a")
    # Assert
    assert second.alias != first.alias


@pytest.mark.asyncio
async def test_pool_prefers_account_with_more_quota_headroom() -> None:
    # Arrange
    alpha = _account("alpha")
    beta = _account("beta")
    pool = CodexAccountPool([alpha, beta])
    await pool.update_usage(
        alpha, primary_used_percent=80, secondary_used_percent=40
    )
    await pool.update_usage(
        beta, primary_used_percent=20, secondary_used_percent=50
    )
    # Act
    selected = await pool.acquire("new-session")
    # Assert
    assert selected.alias == "beta"
