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
from ._pool import StickyPool


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


class CodexAccountPool(StickyPool[CodexAccount]):
    """Select accounts per session and rotate around temporary failures.

    The scheduling lives in :class:`~scitex_genai.gateway._pool.StickyPool` and
    is shared with any other pool of interchangeable members. What stays here is
    what is genuinely about Codex: discovering accounts on disk, and recording
    the two quota percentages its usage API reports.

    Every new session uses ``choose`` after quota and concurrent-load
    filtering. This includes a one-account pool: the selector receives a
    one-element list instead of bypassing rotation with a singleton shortcut.
    """

    # The wording is unchanged, and it is per-class for a reason: a pool of
    # inference upstreams borrowing this scheduling must not refuse with
    # "All Codex accounts are cooling down".
    empty_message = "No Codex subscription accounts are configured"
    duplicate_message = "Codex account aliases must be unique"
    cooling_message = "All Codex accounts are cooling down"
    ineligible_message = "Codex account selector returned an ineligible account"

    def __init__(
        self,
        accounts: list[CodexAccount],
        *,
        choose: Callable[[list[CodexAccount]], CodexAccount] | None = None,
    ) -> None:
        super().__init__(accounts, choose=choose)

    @property
    def accounts(self) -> list[CodexAccount]:
        """The pool's members, under the name callers already use.

        ``_usage``, ``_codex`` and ``_server`` all read ``pool.accounts``; the
        base class stores ``members``. Kept as a property rather than renaming
        the callers, so this refactor changes no behaviour anywhere else.
        """
        return self.members

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
