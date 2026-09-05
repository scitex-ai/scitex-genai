"""Readiness is bounded by progress on disk, not by a clock -- asserted with doubles.

No mocks (PA-306): the probe, the liveness check, the clock and the sleep are
the function's own parameters, and the cache directory is a real ``tmp_path``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from scitex_genai.serve._ready import (
    FAILED,
    READY,
    Readiness,
    newest_mtime,
    wait_ready,
)

URL = "http://127.0.0.1:8768/health"


class _Clock:
    """A clock that advances only when the loop sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _answers_after(count: int):
    """A probe that fails ``count`` times, then answers."""
    calls = {"n": 0}

    def probe(url: str) -> bool:
        calls["n"] += 1
        return calls["n"] > count

    return probe


def _touch(directory: Path, name: str, mtime: float) -> None:
    path = directory / name
    path.write_text("x")
    os.utime(path, (mtime, mtime))


def _raised(call) -> BaseException | None:
    try:
        call()
    except Exception as exc:  # noqa: BLE001 -- the test names the type it expects
        return exc
    return None


def test_ready_when_health_answers(tmp_path: Path):
    # Arrange
    clock = _Clock()

    # Act
    result = wait_ready(
        URL,
        process_alive=lambda: True,
        cache_dir=tmp_path,
        http_get=_answers_after(2),
        clock=clock,
        sleep=clock.sleep,
        poll_s=15,
    )

    # Assert
    assert (result.state, result.checks, result.waited_s) == (READY, 3, 30.0)


def test_failed_as_soon_as_the_process_dies(tmp_path: Path):
    # Arrange
    clock = _Clock()

    # Act
    result = wait_ready(
        URL,
        process_alive=lambda: False,
        cache_dir=tmp_path,
        http_get=lambda url: False,
        clock=clock,
        sleep=clock.sleep,
    )

    # Assert
    assert (result.state, result.checks) == (FAILED, 1)


def test_failed_after_idle_limit_with_no_progress(tmp_path: Path):
    # Arrange
    clock = _Clock()

    # Act
    result = wait_ready(
        URL,
        process_alive=lambda: True,
        cache_dir=tmp_path,
        idle_limit_s=60,
        poll_s=15,
        http_get=lambda url: False,
        clock=clock,
        sleep=clock.sleep,
    )

    # Assert
    assert (result.state, result.waited_s) == (FAILED, 60.0)


def test_a_growing_cache_keeps_the_wait_alive_past_the_idle_limit(tmp_path: Path):
    # Arrange
    clock = _Clock()
    ticks = {"n": 0}

    def probe(url: str) -> bool:
        ticks["n"] += 1
        _touch(tmp_path, f"kernel-{ticks['n']}.o", 1_000_000 + ticks["n"])
        return ticks["n"] >= 10

    # Act
    result = wait_ready(
        URL,
        process_alive=lambda: True,
        cache_dir=tmp_path,
        idle_limit_s=60,
        poll_s=15,
        http_get=probe,
        clock=clock,
        sleep=clock.sleep,
    )

    # Assert
    assert (result.state, result.waited_s) == (READY, 135.0)


def test_the_failure_reason_names_the_cache_dir_and_the_idle_seconds(tmp_path: Path):
    # Arrange
    clock = _Clock()

    # Act
    result = wait_ready(
        URL,
        process_alive=lambda: True,
        cache_dir=tmp_path,
        idle_limit_s=30,
        poll_s=15,
        http_get=lambda url: False,
        clock=clock,
        sleep=clock.sleep,
    )

    # Assert
    assert result.reason == f"no /health and no new file under {tmp_path} for 30 s"


def test_newest_mtime_of_an_absent_dir_is_zero(tmp_path: Path):
    # Arrange
    absent = tmp_path / "none"

    # Act
    value = newest_mtime(absent)

    # Assert
    assert value == 0.0


def test_newest_mtime_sees_nested_files(tmp_path: Path):
    # Arrange
    (tmp_path / "a" / "b").mkdir(parents=True)
    _touch(tmp_path / "a" / "b", "deep", 2_000_000)
    _touch(tmp_path, "shallow", 1_000_000)

    # Act
    value = newest_mtime(tmp_path)

    # Assert
    assert value == 2_000_000


@pytest.mark.parametrize("state", ["up", "", "READY"])
def test_an_undeclared_state_is_refused(state):
    # Arrange
    given = dict(state=state, reason="r", waited_s=0.0, checks=0)

    # Act
    raised = _raised(lambda: Readiness(**given))

    # Assert
    assert isinstance(raised, ValueError)
