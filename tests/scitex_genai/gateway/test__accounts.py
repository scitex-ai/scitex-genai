from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from scitex_genai.gateway._accounts import CodexAccount, CodexAccountPool
from scitex_genai.gateway._credentials import CodexCredential
from scitex_genai.gateway._errors import NoAccountAvailable


def _account(alias: str) -> CodexAccount:
    credential = CodexCredential(
        path=Path(f"/{alias}/auth.json"),
        access_token="access",
        refresh_token="refresh",
        account_id=f"id-{alias}",
        expires_at=4_000_000_000,
    )
    return CodexAccount(alias, credential)


def _jwt(payload: dict) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    return f"header.{encoded.rstrip('=')}.signature"


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
    await pool.update_usage(alpha, primary_used_percent=80, secondary_used_percent=40)
    await pool.update_usage(beta, primary_used_percent=20, secondary_used_percent=50)
    # Act
    selected = await pool.acquire("new-session")
    # Assert
    assert selected.alias == "beta"


@pytest.mark.asyncio
async def test_single_account_uses_rotation_selector() -> None:
    # Arrange
    candidate_aliases: list[list[str]] = []

    def choose(candidates: list[CodexAccount]) -> CodexAccount:
        candidate_aliases.append([account.alias for account in candidates])
        return candidates[0]

    pool = CodexAccountPool([_account("only")], choose=choose)
    # Act
    selected = await pool.acquire("new-session")
    # Assert
    assert (selected.alias, candidate_aliases) == ("only", [["only"]])


def test_discover_expands_provider_qualified_account_root(tmp_path: Path) -> None:
    # Arrange
    root = tmp_path / "accounts" / "openai"
    account_home = root / "example-account"
    account_home.mkdir(parents=True)
    credential = _account("example-account").credential
    account_home.joinpath("auth.json").write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": _jwt(
                        {
                            "exp": credential.expires_at,
                            "https://api.openai.com/auth": {
                                "chatgpt_account_id": credential.account_id
                            },
                        }
                    ),
                    "refresh_token": credential.refresh_token,
                    "account_id": credential.account_id,
                }
            }
        )
    )
    # Act
    pool = CodexAccountPool.discover([root])
    # Assert
    assert [account.qualified_id for account in pool.accounts] == [
        "openai:example-account"
    ]


def test_discover_fails_loud_when_stored_credential_is_broken(tmp_path: Path) -> None:
    # Arrange
    account_home = tmp_path / "accounts" / "openai" / "broken"
    account_home.mkdir(parents=True)
    account_home.joinpath("auth.json").write_text("{}")

    # Act
    def discover() -> CodexAccountPool:
        return CodexAccountPool.discover([account_home.parent])

    # Assert
    with pytest.raises(NoAccountAvailable, match="Unusable Codex account"):
        discover()
