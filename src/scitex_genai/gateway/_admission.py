"""Cold-aware admission primitive, inert until residency is authoritative.

Current SAC/Hermes HTTP requests do not expose an exact pre-admission cache
lookup. The controller therefore defaults to observe-only mode. Its enabled
mode is reserved for a future engine-owned residency signal: known-hot work
may pass queued cold work at a slot boundary, while aged cold work receives
the next progressing slot.

This cannot preempt a prefill already admitted by an upstream.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any


class CacheResidency(str, Enum):
    """Residency known before admission, never inferred from request size."""

    HOT = "hot"
    COLD = "cold"
    UNKNOWN = "unknown"


@dataclass
class _Waiter:
    residency: CacheResidency
    queued_at: float
    future: asyncio.Future[None]


class AdmissionPermit:
    """One idempotently releasable admission slot."""

    def __init__(self, controller: "AdmissionController", *, active: bool) -> None:
        self._controller = controller
        self._active = active

    async def release(self) -> None:
        if not self._active:
            return
        self._active = False
        await self._controller._release()

    async def __aenter__(self) -> "AdmissionPermit":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.release()


class AdmissionController:
    """Prefer authoritative hot work without starving older cold work."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        max_concurrent: int = 1,
        max_cold_wait_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        if max_cold_wait_s < 0:
            raise ValueError("max_cold_wait_s must be non-negative")
        self.enabled = enabled
        self.max_concurrent = max_concurrent
        self.max_cold_wait_s = max_cold_wait_s
        self._clock = clock
        self._lock = asyncio.Lock()
        self._running = 0
        self._waiters: list[_Waiter] = []
        self._observed = {residency: 0 for residency in CacheResidency}
        self._admitted = {residency: 0 for residency in CacheResidency}
        self._hot_overtakes = 0

    async def acquire(
        self, residency: CacheResidency = CacheResidency.UNKNOWN
    ) -> AdmissionPermit:
        if not isinstance(residency, CacheResidency):
            residency = CacheResidency(residency)
        async with self._lock:
            self._observed[residency] += 1
            if not self.enabled:
                return AdmissionPermit(self, active=False)
            waiter = _Waiter(
                residency,
                self._clock(),
                asyncio.get_running_loop().create_future(),
            )
            self._waiters.append(waiter)
            self._dispatch()
        try:
            await waiter.future
        except asyncio.CancelledError:
            async with self._lock:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                else:
                    self._running = max(0, self._running - 1)
                self._dispatch()
            raise
        return AdmissionPermit(self, active=True)

    def observe(self, residency: CacheResidency = CacheResidency.UNKNOWN) -> None:
        """Record a request without making an admission decision."""
        if not isinstance(residency, CacheResidency):
            residency = CacheResidency(residency)
        self._observed[residency] += 1

    async def _release(self) -> None:
        async with self._lock:
            self._running = max(0, self._running - 1)
            self._dispatch()

    def _dispatch(self) -> None:
        while self._running < self.max_concurrent and self._waiters:
            now = self._clock()
            aged = next(
                (
                    waiter
                    for waiter in self._waiters
                    if waiter.residency is not CacheResidency.HOT
                    and now - waiter.queued_at >= self.max_cold_wait_s
                ),
                None,
            )
            selected = aged or next(
                (
                    waiter
                    for waiter in self._waiters
                    if waiter.residency is CacheResidency.HOT
                ),
                self._waiters[0],
            )
            selected_index = self._waiters.index(selected)
            if selected.residency is CacheResidency.HOT:
                self._hot_overtakes += sum(
                    waiter.residency is not CacheResidency.HOT
                    for waiter in self._waiters[:selected_index]
                )
            self._waiters.pop(selected_index)
            self._running += 1
            self._admitted[selected.residency] += 1
            if not selected.future.done():
                selected.future.set_result(None)

    def snapshot(self) -> dict[str, Any]:
        """Return content-free counters suitable for health/metrics export."""
        now = self._clock()
        oldest_wait_s = max(
            (max(0.0, now - waiter.queued_at) for waiter in self._waiters),
            default=0.0,
        )
        return {
            "mode": "enabled" if self.enabled else "observe-only",
            "running": self._running,
            "queued": len(self._waiters),
            "oldest_wait_s": oldest_wait_s,
            "hot_overtakes": self._hot_overtakes,
            "observed": {key.value: value for key, value in self._observed.items()},
            "admitted": {key.value: value for key, value in self._admitted.items()},
        }


__all__ = ["AdmissionController", "AdmissionPermit", "CacheResidency"]
