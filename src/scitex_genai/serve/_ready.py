"""Wait for an engine to answer ``/health`` -- bounded by PROGRESS, not by a clock.

THE DEFECT THIS REPLACES. The captured script waited ``240 x 15 s = 60 min``
for ``/health`` and then gave up: it skipped its READY line, its tunnel and
its discovery record, and left the engine running unsupervised. A cold
FlashInfer JIT on the same engine took 2 h 27 m (measured), so the wrapper
abandoned a healthy engine exactly when the machine was doing the most work.
The fleet had two working engines that nothing could reach.

A fixed bound answers the wrong question. The question is "is the engine
still getting somewhere?", and the JIT answers it on disk: while it compiles,
files keep appearing under the engine's cache directory. So this waits as
long as EITHER the process is alive and the cache is still changing, and
declares failure only after ``idle_limit_s`` of no ``/health`` AND no cache
progress. The answer is a fixed dataclass with a three-valued ``state``, so a
caller never has to guess which of "up", "dead" and "could not tell" it got.

Every side effect is a parameter -- the HTTP probe, the liveness check, the
clock, the sleep -- so the loop is asserted against in tests with real
directories and hand-written doubles, never with mocks.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

READY = "ready"
FAILED = "failed"
UNKNOWN = "unknown"
STATES = (READY, FAILED, UNKNOWN)


@dataclass(frozen=True)
class Readiness:
    """The declared answer: ``state`` is one of ``ready`` / ``failed`` / ``unknown``."""

    state: str
    reason: str
    waited_s: float
    checks: int

    def __post_init__(self) -> None:
        if self.state not in STATES:
            raise ValueError(f"state must be one of {STATES}, got {self.state!r}")
        if self.waited_s < 0 or self.checks < 0:
            raise ValueError("waited_s and checks must be non-negative")


def probe_health(url: str, timeout_s: float = 5.0) -> bool:
    """One real GET; True only on a 2xx answer."""
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:  # noqa: S310 -- loopback URL from the conf
            return 200 <= response.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def newest_mtime(directory: Path) -> float:
    """The most recent modification time under ``directory``; 0.0 when empty or absent."""
    root = Path(directory)
    if not root.is_dir():
        return 0.0
    newest = 0.0
    for path in root.rglob("*"):
        if not path.is_file():
            continue  # a directory's mtime moves when we create it; only files mean work
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:
            continue
    return newest


def wait_ready(
    health_url: str,
    *,
    process_alive: Callable[[], bool],
    cache_dir: Path,
    idle_limit_s: float = 1800.0,
    poll_s: float = 15.0,
    http_get: Callable[[str], bool] = probe_health,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Readiness:
    """Block until ``/health`` answers, the process dies, or progress stops.

    ``idle_limit_s`` is measured from the LAST sign of progress (the newest
    file under ``cache_dir``, or the start), never from the start alone.
    """
    started = clock()
    last_progress = started
    seen_mtime = newest_mtime(cache_dir)
    checks = 0
    while True:
        checks += 1
        now = clock()
        if not process_alive():
            return Readiness(
                FAILED,
                "engine process exited before /health answered",
                now - started,
                checks,
            )
        if http_get(health_url):
            return Readiness(READY, f"{health_url} answered", now - started, checks)
        mtime = newest_mtime(cache_dir)
        if mtime > seen_mtime:
            seen_mtime = mtime
            last_progress = now
        idle = now - last_progress
        if idle >= idle_limit_s:
            return Readiness(
                FAILED,
                f"no /health and no new file under {cache_dir} for {int(idle)} s",
                now - started,
                checks,
            )
        sleep(poll_s)
