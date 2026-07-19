"""Codex OAuth credential loading and refresh.

Only the standard Codex ``auth.json`` file is supported. Token values never
leave :class:`CodexCredential` and are never included in its representation.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from ._errors import CredentialError

TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTH_CLAIM = "https://api.openai.com/auth"


def _decode_jwt(token: str) -> dict[str, Any]:
    """Decode public JWT claims without treating them as verified identity."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload.encode("ascii"))
        value = json.loads(decoded)
    except (IndexError, ValueError, TypeError, UnicodeError) as exc:
        raise CredentialError("Codex access token is not a decodable JWT") from exc
    if not isinstance(value, dict):
        raise CredentialError("Codex access token JWT payload is not an object")
    return value


def _claim_string(claims: dict[str, Any], key: str) -> str:
    value = claims.get(key)
    return value if isinstance(value, str) else ""


@dataclass(repr=False)
class CodexCredential:
    """One refreshable Codex subscription credential."""

    path: Path
    access_token: str = field(repr=False)
    refresh_token: str = field(repr=False)
    account_id: str
    expires_at: float
    email: str = ""

    def __repr__(self) -> str:
        return f"CodexCredential(expires_at={self.expires_at!r}, redacted=True)"

    @classmethod
    def load(cls, path: Path | str) -> "CodexCredential":
        auth_path = Path(path).expanduser()
        try:
            raw = json.loads(auth_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise CredentialError(f"Cannot read Codex auth file: {auth_path}") from exc
        except ValueError as exc:
            raise CredentialError(f"Codex auth file is not valid JSON: {auth_path}") from exc

        tokens = raw.get("tokens") if isinstance(raw, dict) else None
        if not isinstance(tokens, dict):
            raise CredentialError(f"Codex auth file has no tokens object: {auth_path}")
        access = tokens.get("access_token")
        refresh = tokens.get("refresh_token")
        if not isinstance(access, str) or not access:
            raise CredentialError(f"Codex auth file has no access token: {auth_path}")
        if not isinstance(refresh, str) or not refresh:
            raise CredentialError(f"Codex auth file has no refresh token: {auth_path}")

        claims = _decode_jwt(access)
        auth_claim = claims.get(AUTH_CLAIM)
        claim_account = (
            auth_claim.get("chatgpt_account_id")
            if isinstance(auth_claim, dict)
            else None
        )
        stored_account = tokens.get("account_id")
        account_id = claim_account or stored_account
        if not isinstance(account_id, str) or not account_id:
            raise CredentialError("Codex token does not identify a ChatGPT account")

        expires_at = claims.get("exp")
        if not isinstance(expires_at, (int, float)):
            raise CredentialError("Codex access token has no expiry claim")
        profile = claims.get("profile")
        email = _claim_string(claims, "email")
        if not email and isinstance(profile, dict):
            email = _claim_string(profile, "email")
        return cls(
            path=auth_path,
            access_token=access,
            refresh_token=refresh,
            account_id=account_id,
            expires_at=float(expires_at),
            email=email,
        )

    def needs_refresh(self, *, now: float | None = None, skew: float = 300.0) -> bool:
        return self.expires_at <= (time.time() if now is None else now) + skew

    async def refresh(
        self,
        *,
        post: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        """Refresh in place and atomically update the source ``auth.json``."""
        if post is None:
            try:
                import httpx
            except ImportError as exc:
                raise CredentialError(
                    "Codex refresh requires scitex-genai[gateway]"
                ) from exc
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    TOKEN_URL,
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": self.refresh_token,
                        "client_id": CODEX_CLIENT_ID,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
        else:
            response = await post(
                TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self.refresh_token,
                    "client_id": CODEX_CLIENT_ID,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        if getattr(response, "status_code", 500) >= 400:
            raise CredentialError(
                f"Codex token refresh failed with HTTP {response.status_code}"
            )
        data = response.json()
        access = data.get("access_token") if isinstance(data, dict) else None
        refresh = data.get("refresh_token") if isinstance(data, dict) else None
        expires_in = data.get("expires_in") if isinstance(data, dict) else None
        if not isinstance(access, str) or not isinstance(refresh, str):
            raise CredentialError("Codex refresh response is missing tokens")
        if not isinstance(expires_in, (int, float)):
            raise CredentialError("Codex refresh response is missing expires_in")

        claims = _decode_jwt(access)
        auth_claim = claims.get(AUTH_CLAIM)
        account_id = (
            auth_claim.get("chatgpt_account_id")
            if isinstance(auth_claim, dict)
            else None
        )
        if not isinstance(account_id, str) or not account_id:
            raise CredentialError("Refreshed Codex token has no account ID")

        self.access_token = access
        self.refresh_token = refresh
        self.account_id = account_id
        self.expires_at = float(claims.get("exp", time.time() + expires_in))
        self._write_tokens()

    def _write_tokens(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CredentialError(f"Cannot update Codex auth file: {self.path}") from exc
        tokens = raw.setdefault("tokens", {})
        tokens.update(
            {
                "access_token": self.access_token,
                "refresh_token": self.refresh_token,
                "account_id": self.account_id,
            }
        )
        raw["last_refresh"] = datetime.now(timezone.utc).isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".auth-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(raw, handle, indent=2)
                handle.write("\n")
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, self.path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
