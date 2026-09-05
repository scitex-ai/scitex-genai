"""Sticky, least-loaded scheduling over a set of interchangeable members.

Extracted from ``_accounts.py`` unchanged in behaviour. The scheduling never
knew anything about Codex: it needs a member with an ``alias``, an ``in_flight``
count, a ``cooldown_until`` stamp, a ``last_used_at`` stamp and a
``usage_score``. Accounts satisfy that; so does an inference upstream.

WHY IT IS BEING SHARED. The hoist proxy on compute-04 grew its own copy of this
— a sticky per-conversation router with in-flight accounting — in 369 lines of
unpackaged, untested code outside any package. One of the two is going to be
deleted, and the one with tests should be the survivor.

THE THREE RULES THE SCHEDULING ENCODES, none of them obvious:

* STICKINESS IS FOR THE CACHE, not for fairness. A session returns to the member
  it used last so the prefix cache is warm; scattering a conversation across
  members re-pays the prefill every turn. Measured on this fleet: 60.60 s cold
  against 1.54 s warm.
* MEMBERS ARE ASSUMED INTERCHANGEABLE. Selection is least-usage, then
  least-in-flight, then a caller-supplied tiebreak. That is correct for a
  homogeneous set and wrong for a mixed one, where a slow member drains its
  queue faster than work arrives and therefore keeps *looking* least-loaded.
  Exclude the odd member; do not try to fix it here.
* A COOLDOWN DROPS STICKINESS. Cooling a member also forgets every session
  pinned to it, so those sessions re-select rather than queueing behind a member
  that is deliberately out of rotation.

MESSAGES BELONG TO THE SUBCLASS. Every refusal here is worded by a class
attribute, so a pool of inference upstreams cannot report "All Codex accounts
are cooling down". An error that names the wrong subject is how an operator
spends an afternoon on the wrong component; there is no reason to build one in.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable
from typing import Generic, Protocol, TypeVar

from ._errors import NoAccountAvailable


class PoolMember(Protocol):
    """What the scheduling needs from anything it schedules."""

    alias: str
    in_flight: int
    last_used_at: float
    cooldown_until: float

    @property
    def usage_score(self) -> float:
        """Lower is preferred. Return 0.0 when the notion does not apply."""


M = TypeVar("M", bound=PoolMember)


class StickyPool(Generic[M]):
    """Pick a member per session, and rotate around temporary failures.

    Every new session goes through ``choose`` after usage and load filtering,
    INCLUDING a one-member pool: the selector receives a one-element list rather
    than a singleton shortcut. That matters more than it looks — an early
    version of the hoist proxy returned the sole member without incrementing its
    in-flight count while a matching release still decremented, which is
    symmetric only while the pool cannot change size.
    """

    # Overridden per domain so refusals name the right subject.
    empty_message = "No pool members are configured"
    duplicate_message = "Pool member aliases must be unique"
    cooling_message = "All pool members are cooling down"
    ineligible_message = "Pool selector returned an ineligible member"

    def __init__(
        self,
        members: list[M],
        *,
        choose: Callable[[list[M]], M] | None = None,
        max_sessions: int | None = None,
    ) -> None:
        if not members:
            raise NoAccountAvailable(self.empty_message)
        aliases = [member.alias for member in members]
        if len(set(aliases)) != len(aliases):
            raise ValueError(self.duplicate_message)
        self.members = members
        self._sessions: dict[str, str] = {}
        # ``None`` keeps the table unbounded, which is what every existing
        # caller had. A bound evicts the OLDEST placement first: the hoist
        # proxy capped its route table at 512 because a long-lived process
        # must not grow without limit, and losing a placement only costs a
        # prefix-cache miss on that session's next turn, never correctness.
        self._max_sessions = max_sessions
        self._lock = asyncio.Lock()
        self._choose = choose or random.SystemRandom().choice

    async def acquire(self, session_id: str = "", *, exclude: set[str] | None = None) -> M:
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
                member
                for member in self.members
                if member.alias not in excluded and member.cooldown_until <= now
            ]
            if not candidates:
                raise NoAccountAvailable(self.cooling_message)
            best_usage = min(member.usage_score for member in candidates)
            usage_candidates = [
                member for member in candidates if member.usage_score == best_usage
            ]
            best_load = min(member.in_flight for member in usage_candidates)
            rotation_candidates = [
                member for member in usage_candidates if member.in_flight == best_load
            ]
            selected = self._choose(rotation_candidates)
            if all(selected is not candidate for candidate in rotation_candidates):
                raise ValueError(self.ineligible_message)
            selected.in_flight += 1
            selected.last_used_at = now
            if session_id:
                if (
                    self._max_sessions is not None
                    and session_id not in self._sessions
                    and len(self._sessions) >= self._max_sessions
                ):
                    self._sessions.pop(next(iter(self._sessions)))
                self._sessions[session_id] = selected.alias
            return selected

    async def release(self, member: M) -> None:
        async with self._lock:
            member.in_flight = max(0, member.in_flight - 1)

    async def cool_down(self, member: M, seconds: float) -> None:
        async with self._lock:
            member.cooldown_until = max(member.cooldown_until, time.time() + seconds)
            # Forget the sessions pinned here, or they queue behind a member
            # that was deliberately taken out of rotation.
            stale = [
                key for key, value in self._sessions.items() if value == member.alias
            ]
            for key in stale:
                self._sessions.pop(key, None)

    def _by_alias(self, alias: str) -> M | None:
        return next(
            (member for member in self.members if member.alias == alias), None
        )
