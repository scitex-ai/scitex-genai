from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from scitex_genai.gateway._drain import DrainError, DrainState, restart_when_drained
from scitex_genai.gateway._unit import UNIT_NAME


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_restart_closes_admission_then_waits_for_zero_counts() -> None:
    # Arrange
    calls: list[tuple[str, str]] = []
    restarts: list[list[str]] = []
    states = iter(
        [
            {"draining": True, "in_flight": 1, "queued": 0},
            {"draining": True, "in_flight": 0, "queued": 0},
        ]
    )

    def http(method: str, url: str, key: str) -> Mapping[str, object]:
        calls.append((method, url))
        if url.endswith("/admin/drain"):
            return {"draining": True, "in_flight": 1, "queued": 0}
        return next(states)

    # Act
    restart_when_drained(
        health_url="http://gateway.test/health",
        api_key="secret",
        http_call=http,
        systemctl_call=lambda argv: restarts.append(list(argv)),
        sleep=lambda _: None,
    )

    # Assert
    assert (calls, restarts) == (
        [
            ("POST", "http://gateway.test/admin/drain"),
            ("GET", "http://gateway.test/health"),
            ("GET", "http://gateway.test/health"),
        ],
        [["systemctl", "--user", "restart", UNIT_NAME]],
    )


def test_timeout_refuses_restart_and_reopens_admission() -> None:
    # Arrange
    clock = Clock()
    calls: list[tuple[str, str]] = []
    restarts: list[Sequence[str]] = []
    error = ""

    def http(method: str, url: str, key: str) -> Mapping[str, object]:
        calls.append((method, url))
        return {"draining": method == "GET", "in_flight": 1, "queued": 0}

    # Act
    try:
        restart_when_drained(
            health_url="http://gateway.test/health",
            api_key="secret",
            timeout_s=2,
            poll_interval_s=1,
            http_call=http,
            systemctl_call=restarts.append,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
    except DrainError as exc:
        error = str(exc)
    # Assert
    assert ("deadline expired" in error, calls[-1], restarts) == (
        True,
        ("POST", "http://gateway.test/admin/resume"),
        [],
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"draining": True, "in_flight": "1", "queued": 0},
        {"draining": True, "in_flight": -1, "queued": 0},
    ],
)
def test_unverifiable_health_is_never_treated_as_empty(payload) -> None:
    # Arrange
    # Act
    # Assert
    with pytest.raises(DrainError):
        DrainState.from_payload(payload)


def test_restart_is_refused_if_gateway_did_not_close_admission() -> None:
    # Arrange
    calls: list[tuple[str, str]] = []
    error = ""

    def http(method: str, url: str, key: str) -> Mapping[str, object]:
        calls.append((method, url))
        return {"draining": False, "in_flight": 0, "queued": 0}

    def unexpected_restart(_: Sequence[str]) -> None:
        raise RuntimeError("systemctl must not run")

    # Act
    try:
        restart_when_drained(
            health_url="http://gateway.test/health",
            api_key="secret",
            http_call=http,
            systemctl_call=unexpected_restart,
        )
    except DrainError as exc:
        error = str(exc)
    # Assert
    assert ("did not enter draining" in error, calls[-1]) == (
        True,
        ("POST", "http://gateway.test/admin/resume"),
    )
