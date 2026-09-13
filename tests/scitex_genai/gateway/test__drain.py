from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from scitex_genai.gateway._drain import (
    DrainError,
    DrainState,
    restart_when_drained,
)
from scitex_genai.gateway._unit import UNIT_NAME


def test_restart_crosses_atomic_server_barrier_before_systemctl() -> None:
    # Arrange
    calls: list[tuple[str, str]] = []
    restarts: list[list[str]] = []

    def http(method: str, url: str, key: str, timeout_s: float) -> Mapping[str, object]:
        calls.append((method, url))
        return {"draining": True, "ready": False, "in_flight": 0, "queued": 0}

    # Act
    restart_when_drained(
        health_url="http://gateway.test/health",
        api_key="secret",
        http_call=http,
        systemctl_call=lambda argv: restarts.append(list(argv)),
    )

    # Assert
    assert (calls, restarts) == (
        [
            ("POST", "http://gateway.test/admin/drain?timeout_s=1800"),
        ],
        [["systemctl", "--user", "restart", UNIT_NAME]],
    )


def test_server_refusal_never_restarts_or_attempts_resume() -> None:
    # Arrange
    calls: list[tuple[str, str]] = []
    restarts: list[Sequence[str]] = []
    error = ""

    def http(method: str, url: str, key: str, timeout_s: float) -> Mapping[str, object]:
        calls.append((method, url))
        raise DrainError(
            "gateway refused drain: HTTP 409: drain deadline expired; "
            "admission remains closed"
        )

    # Act
    try:
        restart_when_drained(
            health_url="http://gateway.test/health",
            api_key="secret",
            timeout_s=2,
            poll_interval_s=1,
            http_call=http,
            systemctl_call=restarts.append,
        )
    except DrainError as exc:
        error = str(exc)
    # Assert
    assert ("deadline expired" in error, calls, restarts) == (
        True,
        [("POST", "http://gateway.test/admin/drain?timeout_s=2")],
        [],
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"draining": True, "ready": False, "in_flight": "1", "queued": 0},
        {"draining": True, "ready": False, "in_flight": -1, "queued": 0},
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

    def http(method: str, url: str, key: str, timeout_s: float) -> Mapping[str, object]:
        calls.append((method, url))
        return {"draining": False, "ready": True, "in_flight": 0, "queued": 0}

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
    assert ("did not confirm" in error, calls[-1]) == (
        True,
        ("POST", "http://gateway.test/admin/drain?timeout_s=1800"),
    )


def test_systemctl_failure_leaves_barrier_closed() -> None:
    # Arrange
    error = ""

    def http(method: str, url: str, key: str, timeout_s: float) -> Mapping[str, object]:
        return {"draining": True, "ready": False, "in_flight": 0, "queued": 0}

    # Act
    try:
        restart_when_drained(
            health_url="http://gateway.test/health",
            api_key="secret",
            http_call=http,
            systemctl_call=lambda _: (_ for _ in ()).throw(OSError("denied")),
        )
    except DrainError as exc:
        error = str(exc)

    # Assert
    assert "admission remains closed" in error
