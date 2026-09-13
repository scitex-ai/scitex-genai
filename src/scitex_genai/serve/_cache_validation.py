"""Fail-closed validation of SGLang cache tiers.

Configured capacity is not usable capacity.  In particular, hybrid KV+Mamba
restore can fail after SGLang has allocated host/file HiCache.  This observer
therefore reports offload tiers as usable only after an actual successful hit,
and revokes that verdict when the runtime emits a restore failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_DEVICE = re.compile(r"max_total_num_tokens[=: ]+(\d+)", re.I)
_HOST = re.compile(r"HiCache kv host pool \((\d+) tokens\)", re.I)
_MAMBA_FAILURE = re.compile(r"Failed to fetch .*\.mamba from HiCacheFile storage", re.I)
_HYBRID_DISCARD = re.compile(r"HiCache hybrid prefetch discarded", re.I)


@dataclass
class _Tier:
    configured_tokens: int | None = None
    validated: bool = False
    failure: str | None = None

    def report(self, *, device: bool = False) -> dict[str, object]:
        usable = bool(self.configured_tokens) if device else self.validated
        return {
            "status": "failed"
            if self.failure
            else "validated"
            if usable
            else "unverified",
            "configured_tokens": self.configured_tokens,
            "usable_tokens": self.configured_tokens
            if usable and not self.failure
            else 0,
            "failure": self.failure,
        }


class SGLangCacheTierValidator:
    """Consume startup/runtime evidence and expose only validated capacity."""

    def __init__(self) -> None:
        self.device = _Tier()
        self.host = _Tier()
        self.storage = _Tier()

    def observe_log_line(self, line: str) -> None:
        if match := _DEVICE.search(line):
            self.device.configured_tokens = int(match.group(1))
        if match := _HOST.search(line):
            self.host.configured_tokens = int(match.group(1))
        if "Creating storage backend" in line:
            self.storage.configured_tokens = 0
        if _MAMBA_FAILURE.search(line) or _HYBRID_DISCARD.search(line):
            reason = "hybrid_kv_mamba_restore_failed"
            self.host.validated = False
            self.storage.validated = False
            self.host.failure = reason
            self.storage.failure = reason

    def observe_cache_hit(
        self, *, host_tokens: int = 0, storage_tokens: int = 0
    ) -> None:
        if host_tokens > 0 and self.host.failure is None:
            self.host.validated = True
        if storage_tokens > 0 and self.storage.failure is None:
            self.storage.validated = True

    def snapshot(self) -> dict[str, dict[str, object]]:
        return {
            "device": self.device.report(device=True),
            "host": self.host.report(),
            "storage": self.storage.report(),
        }
