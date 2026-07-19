"""Credential parsing and refresh tests use synthetic JWTs only."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from scitex_genai.gateway._credentials import AUTH_CLAIM, CodexCredential


def _jwt(payload: dict) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{encoded}.signature"


def _write_auth(path: Path, *, access: str, refresh: str = "refresh-secret") -> None:
    path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": access,
                    "refresh_token": refresh,
                    "id_token": "preserve-me",
                },
            }
        ),
        encoding="utf-8",
    )


def test_load_extracts_account_without_exposing_tokens(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "auth.json"
    access = _jwt({"exp": 4_000_000_000, AUTH_CLAIM: {"chatgpt_account_id": "acct-a"}})
    _write_auth(path, access=access)
    # Act
    credential = CodexCredential.load(path)
    # Assert
    assert (
        credential.account_id,
        "refresh-secret" in repr(credential),
        access in repr(credential),
    ) == ("acct-a", False, False)


@pytest.mark.asyncio
async def test_refresh_updates_auth_atomically_and_preserves_other_tokens(
    tmp_path: Path,
) -> None:
    # Arrange
    path = tmp_path / "auth.json"
    old_access = _jwt({"exp": 1, AUTH_CLAIM: {"chatgpt_account_id": "acct-a"}})
    new_access = _jwt({"exp": 4_000_000_000, AUTH_CLAIM: {"chatgpt_account_id": "acct-a"}})
    _write_auth(path, access=old_access)

    class Response:
        status_code = 200

        @staticmethod
        def json() -> dict:
            return {
                "access_token": new_access,
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            }

    async def post(*args, **kwargs):
        return Response()

    credential = CodexCredential.load(path)
    # Act
    await credential.refresh(post=post)
    written = json.loads(path.read_text(encoding="utf-8"))
    # Assert
    assert (
        written["tokens"]["access_token"],
        written["tokens"]["refresh_token"],
        written["tokens"]["id_token"],
        path.stat().st_mode & 0o777,
    ) == (new_access, "new-refresh", "preserve-me", 0o600)
