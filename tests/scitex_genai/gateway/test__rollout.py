from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

import pytest

from scitex_genai.gateway._rollout import RolloutError, rollback, rollout, wait_healthy
from scitex_genai.gateway._unit import (
    FRONTEND_SOCKET_UNIT,
    UNIT_NAME,
    backend_unit_name,
)


def _candidate(
    generation: str, build: str, incarnation: str, fingerprint: str | None = None
) -> dict:
    return {
        "status": "ok",
        "ready": True,
        "members": [{"ready": True}],
        "gateway": {
            "frontend_generation": generation,
            "build": build,
            "incarnation": incarnation,
            "code_fingerprint": fingerprint or _fingerprint(),
        },
    }


def _fingerprint() -> str:
    from scitex_genai.gateway._identity import gateway_code_fingerprint

    return gateway_code_fingerprint()


def test_wait_healthy_rejects_the_wrong_build_even_when_status_is_ok():
    # Arrange
    def health(*_):
        return _candidate("new", "wrong", "process")

    # Act
    # Assert
    with pytest.raises(RolloutError, match="identity mismatch"):
        wait_healthy(
            Path("/tmp/candidate.sock"),
            generation="new",
            build="expected",
            incarnation="process",
            timeout_s=0.01,
            health_call=health,
            sleep=lambda _: None,
        )


def test_wait_healthy_rejects_explicit_not_ready_even_when_status_is_ok():
    # Arrange
    health = _candidate("new", "expected", "process")
    health["ready"] = False

    # Act
    # Assert
    with pytest.raises(RolloutError, match="not ready"):
        wait_healthy(
            Path("/tmp/candidate.sock"),
            generation="new",
            build="expected",
            incarnation="process",
            timeout_s=0.01,
            health_call=lambda *_: health,
            sleep=lambda _: None,
        )


def test_wait_healthy_rejects_a_different_installed_code_fingerprint():
    # Arrange
    def health(*_):
        return _candidate("new", "expected", "process", "different-code")

    # Act
    # Assert
    with pytest.raises(RolloutError, match="code fingerprint mismatch"):
        wait_healthy(
            Path("/tmp/candidate.sock"),
            generation="new",
            build="expected",
            incarnation="process",
            code_fingerprint="controller-code",
            timeout_s=0.01,
            health_call=health,
            sleep=lambda _: None,
        )


def test_concurrent_rollout_is_refused_before_starting_a_candidate(tmp_path: Path):
    # Arrange
    state = tmp_path / "state.json"
    current = tmp_path / "current.sock"
    lock_path = current.with_suffix(".sock.lock")
    lock_path.touch()

    # Act
    # Assert
    with (
        lock_path.open("r+") as lock,
        pytest.raises(RolloutError, match="already running"),
    ):
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        rollout(
            generation="new",
            build="build",
            config=tmp_path / "config.yaml",
            public_health_url="http://127.0.0.1:18772/health",
            runtime_dir=tmp_path / "run",
            unit_dir=tmp_path / "units",
            state_path=state,
            current_socket_path=current,
        )


def test_wait_healthy_requires_exact_member_set_and_capacities():
    # Arrange
    health = _candidate("new", "expected", "process")
    health["members"] = [
        {"label": "qwen-a", "ready": True, "capacity": 8, "token_capacity": 500000}
    ]

    # Act
    # Assert
    with pytest.raises(RolloutError, match="token_capacity mismatch"):
        wait_healthy(
            Path("/tmp/candidate.sock"),
            generation="new",
            build="expected",
            incarnation="process",
            expected_members={"qwen-a": {"capacity": 8, "token_capacity": 250000}},
            timeout_s=0.01,
            health_call=lambda *_: health,
            sleep=lambda _: None,
        )


def test_rollout_verifies_private_candidate_switches_then_stops_old(tmp_path: Path):
    # Arrange
    runtime = tmp_path / "run"
    units = tmp_path / "units"
    state = tmp_path / "state.json"
    runtime.mkdir()
    old_socket = runtime / "old.sock"
    os.symlink(old_socket, runtime / "current.sock")
    state.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "active": "old",
                "previous": None,
                "generations": {
                    "old": {
                        "build": "old-build",
                        "incarnation": "old-process",
                        "code_fingerprint": "old-code",
                        "socket": str(old_socket),
                        "unit": backend_unit_name("old"),
                    }
                },
            }
        )
    )
    calls: list[list[str]] = []

    def health(target, _timeout):
        if isinstance(target, Path):
            generation = target.stem
        else:
            generation = Path(os.readlink(runtime / "current.sock")).stem
        if generation == "new":
            return _candidate("new", "new-build", "new-process")
        return _candidate("old", "old-build", "old-process")

    # Act
    rollout(
        generation="new",
        build="new-build",
        config=tmp_path / "config.yaml",
        public_health_url="http://127.0.0.1:18772/health",
        runtime_dir=runtime,
        unit_dir=units,
        state_path=state,
        current_socket_path=runtime / "current.sock",
        systemctl=lambda argv: calls.append(list(argv)),
        health_call=health,
    )

    # Assert
    saved = json.loads(state.read_text())
    unit_text = (units / backend_unit_name("new")).read_text()
    assert (
        Path(os.readlink(runtime / "current.sock")).stem,
        saved["active"],
        saved["previous"],
        set(saved["generations"]),
        "--graceful-rollout-shutdown" in unit_text,
        calls[-1],
    ) == (
        "new",
        "new",
        "old",
        {"old", "new"},
        True,
        [
            "systemctl",
            "--user",
            "disable",
            "--now",
            backend_unit_name("old"),
        ],
    )


def test_failed_public_verification_atomically_restores_old_target(tmp_path: Path):
    # Arrange
    runtime = tmp_path / "run"
    runtime.mkdir()
    old_socket = runtime / "old.sock"
    os.symlink(old_socket, runtime / "current.sock")
    calls: list[list[str]] = []

    def health(target, _timeout):
        if isinstance(target, Path):
            return _candidate("new", "build", "new-process")
        raise RolloutError("proxy smoke failed")

    # Act
    error = None
    try:
        rollout(
            generation="new",
            build="build",
            config=tmp_path / "config.yaml",
            public_health_url="http://127.0.0.1:18772/health",
            runtime_dir=runtime,
            unit_dir=tmp_path / "units",
            state_path=tmp_path / "state.json",
            current_socket_path=runtime / "current.sock",
            health_timeout_s=0.01,
            systemctl=lambda argv: calls.append(list(argv)),
            health_call=health,
        )
    except RolloutError as exc:
        error = exc

    # Assert
    assert (
        "deadline" in str(error),
        Path(os.readlink(runtime / "current.sock")).stem,
        calls[-1],
    ) == (
        True,
        "old",
        ["systemctl", "--user", "disable", "--now", backend_unit_name("new")],
    )


def test_bootstrap_requires_empty_legacy_gateway_before_stopping_it(tmp_path: Path):
    # Arrange
    calls: list[list[str]] = []

    def health(target, _timeout):
        if isinstance(target, Path):
            return _candidate("new", "build", "new-process")
        return {"status": "ok", "in_flight": 1, "queued": 0, "held": 0}

    # Act
    error = None
    try:
        rollout(
            generation="new",
            build="build",
            config=tmp_path / "config.yaml",
            public_health_url="http://127.0.0.1:18772/health",
            runtime_dir=tmp_path / "run",
            unit_dir=tmp_path / "units",
            state_path=tmp_path / "state.json",
            current_socket_path=tmp_path / "current.sock",
            bootstrap_coordinated=True,
            systemctl=lambda argv: calls.append(list(argv)),
            health_call=health,
        )
    except RolloutError as exc:
        error = exc

    # Assert
    assert (
        "not empty" in str(error),
        ["systemctl", "--user", "disable", "--now", UNIT_NAME] not in calls,
        ["systemctl", "--user", "enable", "--now", FRONTEND_SOCKET_UNIT] not in calls,
    ) == (True, True, True)


def test_bootstrap_rejects_nonzero_legacy_held_before_stopping_it(tmp_path: Path):
    # Arrange
    calls: list[list[str]] = []

    def health(target, _timeout):
        if isinstance(target, Path):
            return _candidate("new", "build", "new-process")
        return {"status": "ok", "in_flight": 0, "queued": 0, "held": 1}

    # Act
    error = None
    try:
        rollout(
            generation="new",
            build="build",
            config=tmp_path / "config.yaml",
            public_health_url="http://127.0.0.1:18772/health",
            runtime_dir=tmp_path / "run",
            unit_dir=tmp_path / "units",
            state_path=tmp_path / "state.json",
            current_socket_path=tmp_path / "current.sock",
            bootstrap_coordinated=True,
            systemctl=lambda argv: calls.append(list(argv)),
            health_call=health,
        )
    except RolloutError as exc:
        error = exc

    # Assert
    assert (
        "not empty" in str(error),
        ["systemctl", "--user", "disable", "--now", UNIT_NAME] not in calls,
    ) == (True, True)


def test_empty_legacy_schema_without_held_bootstraps_durable_frontend(
    tmp_path: Path,
):
    # Arrange
    runtime = tmp_path / "run"
    current = tmp_path / "current.sock"
    calls: list[list[str]] = []

    def health(target, _timeout):
        if isinstance(target, Path) or current.is_symlink():
            return _candidate("new", "build", "new-process")
        return {"status": "ok", "in_flight": 0, "queued": 0}

    # Act
    rollout(
        generation="new",
        build="build",
        config=tmp_path / "config.yaml",
        public_health_url="http://127.0.0.1:18772/health",
        runtime_dir=runtime,
        unit_dir=tmp_path / "units",
        state_path=tmp_path / "state.json",
        current_socket_path=current,
        bootstrap_coordinated=True,
        systemctl=lambda argv: calls.append(list(argv)),
        health_call=health,
    )

    # Assert
    assert (
        Path(os.readlink(current)).stem,
        calls[-3:],
    ) == (
        "new",
        [
            ["systemctl", "--user", "enable", backend_unit_name("new")],
            ["systemctl", "--user", "disable", "--now", UNIT_NAME],
            ["systemctl", "--user", "enable", "--now", FRONTEND_SOCKET_UNIT],
        ],
    )


def test_rollback_starts_previous_verifies_switches_and_drains_current(tmp_path: Path):
    # Arrange
    runtime = tmp_path / "run"
    runtime.mkdir()
    new_socket = runtime / "new.sock"
    old_socket = runtime / "old.sock"
    os.symlink(new_socket, runtime / "current.sock")
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "active": "new",
                "previous": "old",
                "generations": {
                    "new": {"build": "new", "incarnation": "n"},
                    "old": {
                        "build": "old",
                        "incarnation": "o",
                        "code_fingerprint": _fingerprint(),
                        "socket": str(old_socket),
                        "unit": backend_unit_name("old"),
                    },
                },
            }
        )
    )
    calls: list[list[str]] = []

    def health(target, _timeout):
        generation = (
            target.stem
            if isinstance(target, Path)
            else Path(os.readlink(runtime / "current.sock")).stem
        )
        return _candidate(generation, generation, generation[0])

    # Act
    rollback(
        public_health_url="http://127.0.0.1:18772/health",
        runtime_dir=runtime,
        state_path=state,
        current_socket_path=runtime / "current.sock",
        systemctl=lambda argv: calls.append(list(argv)),
        health_call=health,
    )

    # Assert
    assert (
        Path(os.readlink(runtime / "current.sock")).stem,
        json.loads(state.read_text())["active"],
        calls[-1],
    ) == (
        "old",
        "old",
        ["systemctl", "--user", "disable", "--now", backend_unit_name("new")],
    )
