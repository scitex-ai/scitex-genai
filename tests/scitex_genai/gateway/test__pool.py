from __future__ import annotations

import time
from dataclasses import dataclass

import pytest

from scitex_genai.gateway._errors import HomeMemberReloading
from scitex_genai.gateway._pool import StickyPool


@dataclass
class _Member:
    """The smallest real thing the pool can schedule."""

    alias: str
    in_flight: int = 0
    last_used_at: float = 0.0
    cooldown_until: float = 0.0
    cooling_since: float | None = None

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

    pool = StickyPool(
        [_Member("alpha"), _Member("beta")], choose=choose, max_sessions=2
    )
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


@pytest.mark.asyncio
async def test_a_session_whose_home_just_went_out_is_held_not_moved() -> None:
    # Arrange -- alpha is this session's home; it went out of rotation 5 s ago.
    alpha, beta = _Member("alpha"), _Member("beta")
    pool = StickyPool([alpha, beta], choose=lambda c: c[0], failover_after_s=150.0)
    await pool.release(await pool.acquire("conv-1"))
    await pool.cool_down(alpha, 30.0)
    alpha.cooling_since = time.time() - 5.0
    # Act / Assert -- the session is refused with a retry hint, not handed to beta.
    with pytest.raises(HomeMemberReloading):
        await pool.acquire("conv-1")


@pytest.mark.asyncio
async def test_a_session_whose_home_has_been_out_long_enough_is_re_placed() -> None:
    # Arrange -- alpha has been out for 10 minutes: treated as gone, not reloading.
    alpha, beta = _Member("alpha"), _Member("beta")
    pool = StickyPool([alpha, beta], choose=lambda c: c[0], failover_after_s=150.0)
    await pool.release(await pool.acquire("conv-1"))
    await pool.cool_down(alpha, 30.0)
    alpha.cooling_since = time.time() - 600.0
    # Act
    member = await pool.acquire("conv-1")
    # Assert
    assert member is beta


@pytest.mark.asyncio
async def test_a_new_session_is_never_held_by_another_sessions_home() -> None:
    # Arrange -- alpha is out; a brand-new session has no home to wait for.
    alpha, beta = _Member("alpha"), _Member("beta")
    pool = StickyPool([alpha, beta], choose=lambda c: c[0], failover_after_s=150.0)
    await pool.cool_down(alpha, 30.0)
    # Act
    member = await pool.acquire("conv-new")
    # Assert
    assert member is beta


@pytest.mark.asyncio
async def test_without_a_hold_window_the_old_immediate_failover_stands() -> None:
    # Arrange -- failover_after_s left at 0: cool_down forgets the pin as before.
    alpha, beta = _Member("alpha"), _Member("beta")
    pool = StickyPool([alpha, beta], choose=lambda c: c[0])
    await pool.release(await pool.acquire("conv-1"))
    await pool.cool_down(alpha, 30.0)
    # Act
    member = await pool.acquire("conv-1")
    # Assert
    assert member is beta


@pytest.mark.asyncio
async def test_a_member_handed_out_healthy_again_clears_its_cooling_clock() -> None:
    # Arrange -- alpha cooled, the cooldown expired, and it is chosen again.
    alpha = _Member("alpha")
    pool = StickyPool([alpha], choose=lambda c: c[0], failover_after_s=150.0)
    await pool.cool_down(alpha, 0.0)
    alpha.cooldown_until = 0.0
    # Act
    await pool.acquire("conv-1")
    # Assert
    assert alpha.cooling_since is None
