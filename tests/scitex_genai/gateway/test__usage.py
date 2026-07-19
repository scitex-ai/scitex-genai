from __future__ import annotations

from pathlib import Path

import pytest

from scitex_genai.gateway._accounts import CodexAccount, CodexAccountPool
from scitex_genai.gateway._credentials import CodexCredential
from scitex_genai.gateway._usage import CodexUsageClient


def _account() -> CodexAccount:
    return CodexAccount(
        "alpha",
        CodexCredential(
            Path("/alpha/auth.json"),
            "access-secret",
            "refresh-secret",
            "account-id",
            4_000_000_000,
        ),
    )


@pytest.mark.asyncio
async def test_usage_client_updates_both_windows_without_exposing_tokens() -> None:
    # Arrange
    account = _account()
    seen_headers = {}

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "rate_limit": {
                    "primary_window": {"used_percent": 12.5},
                    "secondary_window": {"used_percent": 34.5},
                }
            }

    class Client:
        async def get(self, url, *, headers):
            seen_headers.update(headers)
            return Response()

    pool = CodexAccountPool([account])
    # Act
    await CodexUsageClient(Client()).refresh_pool(pool)
    # Assert
    assert (
        account.primary_used_percent,
        account.secondary_used_percent,
        seen_headers["Authorization"],
        "access-secret" in repr(account),
    ) == (12.5, 34.5, "Bearer access-secret", False)
