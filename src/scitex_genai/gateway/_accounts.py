"""Sticky, least-loaded scheduling for Codex subscription accounts."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from ._credentials import CodexCredential
from ._errors import CredentialError, NoAccountAvailable


@dataclass
class CodexAccount:
    """Runtime scheduling state for one provider-qualified account."""

    alias: str
    credential: CodexCredential = field(repr=False)
    in_flight: int = 0
    last_used_at: float = 0.0
    cooldown_until: float = 0.0
    refresh_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    primary_used_percent: float | None = None
    secondary_used_percent: float | None = None
    usage_refreshed_at: float = 0.0

    @property
    def usage_score(self) -> float:
        values = [
            value
            for value in (self.primary_used_percent, self.secondary_used_percent)
            if value is not None
        ]
        return max(values, default=0.0)

    @property
    def qualified_id(self) -> str:
        return f"openai:{self.alias}"


class CodexAccountPool:
    """Select accounts per session and rotate around temporary failures."""

    def __init__(self, accounts: list[CodexAccount]) -> None:
        if not accounts:
            raise NoAccountAvailable("No Codex subscription accounts are configured")
        aliases = [account.alias for account in accounts]
        if len(set(aliases)) != len(aliases):
            raise ValueError("Codex account aliases must be unique")
        self.accounts = accounts
        self._sessions: dict[str, str] = {}
        self._lock = asyncio.Lock()

    @classmethod
    def discover(cls, homes: list[Path | str] | None = None) -> "CodexAccountPool":
        if homes is None:
            configured = os.getenv("SCITEX_GENAI_CODEX_HOMES", "")
            if configured:
                homes = [Path(value) for value in configured.split(os.pathsep) if value]
            else:
                homes = [Path(os.getenv("CODEX_HOME", Path.home() / ".codex"))]

        accounts: list[CodexAccount] = []
        errors: list[str] = []
        for home_value in homes:
            home = Path(home_value).expanduser()
            path = home if home.name == "auth.json" else home / "auth.json"
            try:
                credential = CodexCredential.load(path)
            except CredentialError as exc:
                errors.append(str(exc))
                continue
            alias = home.parent.name if home.name == "auth.json" else home.name
            if alias == ".codex":
                alias = "default"
            accounts.append(CodexAccount(alias=alias, credential=credential))
        if not accounts:
            detail = "; ".join(errors) or "no auth.json files found"
            raise NoAccountAvailable(f"No usable Codex accounts: {detail}")
        return cls(accounts)

    async def acquire(
        self, session_id: str = "", *, exclude: set[str] | None = None
    ) -> CodexAccount:
        excluded = exclude or set()
        async with self._lock:
            now = time.time()
            sticky_alias = self._sessions.get(session_id) if session_id else None
            if sticky_alias and sticky_alias not in excluded:
                sticky = self._by_alias(sticky_alias)
                if sticky is not None and sticky.cooldown_until <= now:
                    sticky.in_flight += 1
                    sticky.last_used_at = now
                    return sticky

            candidates = [
                account
                for account in self.accounts
                if account.alias not in excluded and account.cooldown_until <= now
            ]
            if not candidates:
                raise NoAccountAvailable("All Codex accounts are cooling down")
            selected = min(
                candidates,
                key=lambda account: (
                    account.usage_score,
                    account.in_flight,
                    account.last_used_at,
                    account.alias,
                ),
            )
            selected.in_flight += 1
            selected.last_used_at = now
            if session_id:
                self._sessions[session_id] = selected.alias
            return selected

    async def release(self, account: CodexAccount) -> None:
        async with self._lock:
            account.in_flight = max(0, account.in_flight - 1)

    async def cool_down(self, account: CodexAccount, seconds: float) -> None:
        async with self._lock:
            account.cooldown_until = max(account.cooldown_until, time.time() + seconds)
            stale = [key for key, value in self._sessions.items() if value == account.alias]
            for key in stale:
                self._sessions.pop(key, None)

    async def update_usage(
        self,
        account: CodexAccount,
        *,
        primary_used_percent: float | None,
        secondary_used_percent: float | None,
    ) -> None:
        async with self._lock:
            account.primary_used_percent = primary_used_percent
            account.secondary_used_percent = secondary_used_percent
            account.usage_refreshed_at = time.time()

    def _by_alias(self, alias: str) -> CodexAccount | None:
        return next((account for account in self.accounts if account.alias == alias), None)
