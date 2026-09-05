from __future__ import annotations

from dataclasses import dataclass

import pytest

from scitex_genai.gateway._pool import StickyPool


@dataclass
class _Member:
    """The smallest real thing the pool can schedule."""

    alias: str
    in_flight: int = 0
    last_used_at: float = 0.0
    cooldown_until: float = 0.0

    @property
    def usage_score(self) -> float:
        return 0.0


async def _place(pool: StickyPool[_Member], sessions: tuple[str, ...]) -> None:
    for session in sessions:
        member = await pool.acquire(session)
        await pool.release(member)


@pytest.mark.asyncio
async def test_bounded_session_table_forgets_the_oldest_placement() -> None:
    # Arrange
    placements = 0

    def choose(candidates: list[_Member]) -> _Member:
        nonlocal placements
        placements += 1
        return candidates[0]

    pool = StickyPool([_Member("alpha"), _Member("beta")], choose=choose, max_sessions=2)
    # Act
    await _place(pool, ("s1", "s2", "s3", "s1"))
    # Assert
    assert placements == 4


@pytest.mark.asyncio
async def test_unbounded_session_table_keeps_every_placement() -> None:
    # Arrange
    placements = 0

    def choose(candidates: list[_Member]) -> _Member:
        nonlocal placements
        placements += 1
        return candidates[0]

    pool = StickyPool([_Member("alpha"), _Member("beta")], choose=choose)
    # Act
    await _place(pool, ("s1", "s2", "s3", "s1"))
    # Assert
    assert placements == 3
