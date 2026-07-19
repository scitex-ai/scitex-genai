"""Sticky, least-loaded scheduling for Codex subscription accounts."""

from __future__ import annotations

import asyncio
import os
import random
import time
from collections.abc import Callable
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
    """Select accounts per session and rotate around temporary failures.

    Every new session uses ``choose`` after quota and concurrent-load
    filtering. This includes a one-account pool: the selector receives a
    one-element list instead of bypassing rotation with a singleton shortcut.
    """

    def __init__(
        self,
        accounts: list[CodexAccount],
        *,
        choose: Callable[[list[CodexAccount]], CodexAccount] | None = None,
    ) -> None:
        if not accounts:
            raise NoAccountAvailable("No Codex subscription accounts are configured")
        aliases = [account.alias for account in accounts]
        if len(set(aliases)) != len(aliases):
            raise ValueError("Codex account aliases must be unique")
        self.accounts = accounts
        self._sessions: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._choose = choose or random.SystemRandom().choice

    @classmethod
    def discover(cls, homes: list[Path | str] | None = None) -> "CodexAccountPool":
        if homes is None:
            configured = os.getenv("SCITEX_GENAI_CODEX_HOMES", "")
            if configured:
                homes = [Path(value) for value in configured.split(os.pathsep) if value]
            else:
                account_root = Path(
                    os.getenv(
                        "SCITEX_GENAI_CODEX_ACCOUNTS_DIR",
                        Path.home()
                        / ".scitex"
                        / "agent-container"
                        / "accounts"
                        / "openai",
                    )
                ).expanduser()
                stored_homes = cls._stored_homes(account_root)
                if account_root.exists() and not stored_homes:
                    raise NoAccountAvailable(
                        f"Codex account store contains no auth files: {account_root}"
                    )
                homes = stored_homes or [
                    Path(os.getenv("CODEX_HOME", Path.home() / ".codex"))
                ]

        accounts: list[CodexAccount] = []
        expanded_homes: list[Path] = []
        for home_value in homes:
            home = Path(home_value).expanduser()
            expanded_homes.extend(cls._stored_homes(home) or [home])
        for home in expanded_homes:
            path = home if home.name == "auth.json" else home / "auth.json"
            try:
                credential = CodexCredential.load(path)
            except CredentialError as exc:
                raise NoAccountAvailable(
                    f"Unusable Codex account credential: {path}"
                ) from exc
            alias = home.parent.name if home.name == "auth.json" else home.name
            if alias == ".codex":
                alias = "default"
            accounts.append(CodexAccount(alias=alias, credential=credential))
        if not accounts:
            raise NoAccountAvailable(
                "No usable Codex accounts: no auth.json files found"
            )
        return cls(accounts)

    @staticmethod
    def _stored_homes(root: Path) -> list[Path]:
        """Return sorted account homes below a provider-qualified store root."""
        if not root.is_dir() or (root / "auth.json").is_file():
            return []
        return sorted(
            child
            for child in root.iterdir()
            if child.is_dir() and (child / "auth.json").is_file()
        )

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
            best_usage = min(account.usage_score for account in candidates)
            quota_candidates = [
                account for account in candidates if account.usage_score == best_usage
            ]
            best_load = min(account.in_flight for account in quota_candidates)
            rotation_candidates = [
                account
                for account in quota_candidates
                if account.in_flight == best_load
            ]
            selected = self._choose(rotation_candidates)
            if all(selected is not candidate for candidate in rotation_candidates):
                raise ValueError(
                    "Codex account selector returned an ineligible account"
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
            stale = [
                key for key, value in self._sessions.items() if value == account.alias
            ]
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
        return next(
            (account for account in self.accounts if account.alias == alias), None
        )
