"""Cold-aware admission from generation-bound cache feedback.

The policy defaults to observe-only. Active mode uses the prior response's
engine-owned cache report only for an extension of the same prompt lineage on
the same engine generation. Known-hot work may pass queued cold work at a slot
boundary, while aged cold work receives the next progressing slot.

This cannot preempt a prefill already admitted by an upstream.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CacheResidency(str, Enum):
    """Residency known before admission, never inferred from request size."""

    HOT = "hot"
    COLD = "cold"
    UNKNOWN = "unknown"


class CacheAdmissionSettings(BaseModel):
    """Strict, file-backed policy for cache-aware generation admission."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    mode: Literal["observe_only", "active"] = "observe_only"
    hot_max_uncached_tokens: Annotated[int, Field(ge=0)] = 32_768
    cold_prefill_limit_per_upstream: Annotated[int, Field(ge=1)] = 1
    max_hot_bypasses: Annotated[int, Field(ge=0)] = 4
    starvation_age_s: Annotated[float, Field(ge=0)] = 30.0
    evidence_max_age_s: Annotated[float, Field(gt=0)] = 300.0

    @model_validator(mode="after")
    def active_requires_a_nonzero_hot_boundary(self) -> "CacheAdmissionSettings":
        if self.mode == "active" and self.hot_max_uncached_tokens < 1:
            raise ValueError(
                "active cache admission requires hot_max_uncached_tokens >= 1"
            )
        return self

    @property
    def active(self) -> bool:
        return self.mode == "active"


def classify_cache_prediction(
    *,
    settings: CacheAdmissionSettings,
    engine_generation: str,
    evidence: str,
    prior_cache_tier: str,
    predicted_uncached_tokens: int,
) -> CacheResidency:
    """Classify only generation-bound feedback; reject missing active evidence."""
    if not settings.active:
        return CacheResidency.UNKNOWN
    if not engine_generation or engine_generation == "unavailable":
        raise ValueError("active cache admission lacks an engine generation")
    if evidence == "no-compatible-history":
        # No compatible lineage is a conservative cold classification. It is
        # never promoted from prompt size or an unrelated session.
        return CacheResidency.COLD
    if evidence != "historical-lineage-extension":
        raise ValueError(f"active cache admission lacks cache evidence: {evidence}")
    if prior_cache_tier not in {"device", "host", "storage", "none"}:
        raise ValueError(
            "active cache admission has an invalid prior cache tier: "
            f"{prior_cache_tier}"
        )
    if prior_cache_tier == "none":
        return CacheResidency.COLD
    return (
        CacheResidency.HOT
        if predicted_uncached_tokens <= settings.hot_max_uncached_tokens
        else CacheResidency.COLD
    )


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
            "mode": "active" if self.enabled else "observe-only",
            "admitted": self._running,
            "queued": len(self._waiters),
            "oldest_wait_s": oldest_wait_s,
            "hot_overtakes": self._hot_overtakes,
            "observed": {key.value: value for key, value in self._observed.items()},
            "admitted_total_by_residency": {
                key.value: value for key, value in self._admitted.items()
            },
        }


__all__ = [
    "AdmissionController",
    "AdmissionPermit",
    "CacheAdmissionSettings",
    "CacheResidency",
    "classify_cache_prediction",
]
