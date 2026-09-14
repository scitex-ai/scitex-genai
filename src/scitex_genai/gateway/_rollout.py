"""Blue/green gateway rollout behind systemd's socket proxy primitive."""

from __future__ import annotations

import fcntl
import json
import os
import re
import secrets
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any

from ._identity import gateway_code_fingerprint
from ._settings import default_gateway_session_state_path
from ._unit import (
    DEFAULT_UNIT_DIR,
    FRONTEND_SERVICE_UNIT,
    FRONTEND_SOCKET_UNIT,
    UNIT_NAME,
    backend_unit_name,
    gateway_command,
    render_backend_unit,
)

SCHEMA_VERSION = 1
GENERATION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
DEFAULT_HEALTH_TIMEOUT_S = 60.0


class RolloutError(RuntimeError):
    """A candidate could not be safely promoted."""


Systemctl = Callable[[Sequence[str]], None]
HealthCall = Callable[[str | Path, float], Mapping[str, Any]]


def default_runtime_dir() -> Path:
    root = os.getenv("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    return Path(root) / "scitex-genai-gateway"


def default_rollout_state_path() -> Path:
    return default_gateway_session_state_path().with_name("gateway-rollout.json")


def default_current_socket_path() -> Path:
    """Durable selector whose target socket is recreated after reboot."""
    return default_rollout_state_path().with_name("gateway-current.sock")


@contextmanager
def _rollout_lock(state_path: Path):
    lock_path = state_path.with_suffix(f"{state_path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with lock_path.open("a+", encoding="utf-8") as lock:
        os.chmod(lock_path, 0o600)
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RolloutError("another gateway rollout is already running") from exc
        yield


def _serialized(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        selector = kwargs.get("current_socket_path") or default_current_socket_path()
        with _rollout_lock(Path(selector)):
            return function(*args, **kwargs)

    return wrapped


def validate_generation(value: str) -> str:
    if not GENERATION_RE.fullmatch(value):
        raise ValueError(
            "generation must be 1..64 letters, digits, dots, underscores or hyphens"
        )
    return value


def _systemctl(argv: Sequence[str]) -> None:
    subprocess.run(list(argv), check=True)


def _health(target: str | Path, timeout_s: float) -> Mapping[str, Any]:
    try:
        import httpx
    except ImportError as exc:
        raise RolloutError("rollout requires scitex-genai[gateway]") from exc
    if isinstance(target, Path):
        transport = httpx.HTTPTransport(uds=str(target))
        with httpx.Client(transport=transport, base_url="http://gateway") as client:
            response = client.get("/health", timeout=timeout_s)
    else:
        with httpx.Client() as client:
            response = client.get(target, timeout=timeout_s)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RolloutError("health response is not an object")
    return payload


def _verify_health(
    payload: Mapping[str, Any],
    *,
    generation: str,
    build: str,
    incarnation: str | None,
    code_fingerprint: str | None,
    expected_members: Mapping[str, Mapping[str, int]] | None,
) -> None:
    identity = payload.get("gateway")
    if not isinstance(identity, Mapping):
        raise RolloutError("candidate health lacks gateway identity")
    observed = (
        identity.get("frontend_generation"),
        identity.get("build"),
        identity.get("incarnation"),
    )
    expected = (generation, build)
    if observed[:2] != expected or not isinstance(observed[2], str) or not observed[2]:
        raise RolloutError(
            f"candidate identity mismatch: expected generation/build {expected!r}, "
            f"observed {observed!r}"
        )
    if incarnation is not None and observed[2] != incarnation:
        raise RolloutError(
            f"candidate incarnation mismatch: expected {incarnation!r}, "
            f"observed {observed[2]!r}"
        )
    observed_fingerprint = identity.get("code_fingerprint")
    if code_fingerprint is not None and observed_fingerprint != code_fingerprint:
        raise RolloutError(
            "candidate code fingerprint mismatch: "
            f"expected {code_fingerprint!r}, observed {observed_fingerprint!r}"
        )
    if "ready" in payload:
        if payload.get("ready") is not True:
            raise RolloutError("candidate is not ready")
    elif payload.get("status") != "ok":
        raise RolloutError("candidate is not ready")
    members = payload.get("members")
    if isinstance(members, list) and (
        not members
        or any(
            not isinstance(item, Mapping) or item.get("ready") is not True
            for item in members
        )
    ):
        raise RolloutError("not every configured inference member is ready")
    if expected_members is not None:
        observed_members = {
            str(item.get("label")): item
            for item in members or []
            if isinstance(item, Mapping) and item.get("label") is not None
        }
        if set(observed_members) != set(expected_members):
            raise RolloutError(
                "candidate member set mismatch: "
                f"expected {sorted(expected_members)!r}, "
                f"observed {sorted(observed_members)!r}"
            )
        for label, expected_member in expected_members.items():
            observed_member = observed_members[label]
            for field in ("capacity", "token_capacity"):
                if observed_member.get(field) != expected_member.get(field):
                    raise RolloutError(
                        f"candidate member {label!r} {field} mismatch: "
                        f"expected {expected_member.get(field)!r}, "
                        f"observed {observed_member.get(field)!r}"
                    )


def wait_healthy(
    target: str | Path,
    *,
    generation: str,
    build: str,
    incarnation: str | None = None,
    code_fingerprint: str | None = None,
    expected_members: Mapping[str, Mapping[str, int]] | None = None,
    timeout_s: float = DEFAULT_HEALTH_TIMEOUT_S,
    health_call: HealthCall = _health,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Mapping[str, Any]:
    """Wait for exact candidate identity and readiness, never another process."""
    deadline = monotonic() + timeout_s
    last: Exception | None = None
    while monotonic() < deadline:
        try:
            payload = health_call(target, min(5.0, max(0.1, deadline - monotonic())))
            _verify_health(
                payload,
                generation=generation,
                build=build,
                incarnation=incarnation,
                code_fingerprint=code_fingerprint,
                expected_members=expected_members,
            )
            return payload
        except Exception as exc:  # candidate may still be starting
            last = exc
            sleep(min(0.25, max(0.0, deadline - monotonic())))
    raise RolloutError(f"candidate did not become ready before deadline: {last}")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_state(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise RolloutError(f"cannot read rollout state {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise RolloutError("rollout state has an unsupported schema")
    return payload


def _switch(current: Path, target: Path) -> Path | None:
    """Atomically change which Unix socket new proxy connections resolve."""
    previous: Path | None = None
    if current.is_symlink():
        previous = Path(os.readlink(current))
        if not previous.is_absolute():
            previous = current.parent / previous
    temporary = current.with_name(f".{current.name}.{secrets.token_hex(8)}")
    os.symlink(target, temporary)
    os.replace(temporary, current)
    return previous


def _generation_from_socket(path: Path | None) -> str | None:
    if path is None or path.suffix != ".sock":
        return None
    try:
        return validate_generation(path.stem)
    except ValueError:
        return None


@_serialized
def rollout(
    *,
    generation: str,
    build: str,
    config: Path | str,
    public_health_url: str,
    runtime_dir: Path | None = None,
    unit_dir: Path | None = None,
    state_path: Path | None = None,
    current_socket_path: Path | None = None,
    expected_members: Mapping[str, Mapping[str, int]] | None = None,
    interpreter: str | None = None,
    health_timeout_s: float = DEFAULT_HEALTH_TIMEOUT_S,
    bootstrap_coordinated: bool = False,
    systemctl: Systemctl = _systemctl,
    health_call: HealthCall = _health,
    report: Callable[[str], None] = print,
) -> None:
    """Verify a private candidate, atomically promote it, then drain the old one."""
    generation = validate_generation(generation)
    if not build.strip():
        raise ValueError("build must be non-empty")
    runtime = runtime_dir or default_runtime_dir()
    units = unit_dir or DEFAULT_UNIT_DIR
    state_file = state_path or default_rollout_state_path()
    current_socket = current_socket_path or default_current_socket_path()
    runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
    units.mkdir(parents=True, exist_ok=True)
    current_socket.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    candidate_socket = runtime / f"{generation}.sock"
    if (
        _generation_from_socket(
            Path(os.readlink(current_socket)) if current_socket.is_symlink() else None
        )
        == generation
    ):
        raise RolloutError(f"generation {generation} is already active")
    command = gateway_command(
        config=config,
        interpreter=interpreter or sys.executable,
        uds=candidate_socket,
        gateway_build=build,
        frontend_generation=generation,
        graceful_rollout_shutdown=True,
    )
    unit = backend_unit_name(generation)
    (units / unit).write_text(
        render_backend_unit(
            command=command, socket_path=candidate_socket, generation=generation
        )
    )
    systemctl(["systemctl", "--user", "daemon-reload"])
    systemctl(["systemctl", "--user", "start", unit])
    report(f"candidate {generation}: started {unit}")
    expected_fingerprint = gateway_code_fingerprint()
    try:
        candidate_health = wait_healthy(
            candidate_socket,
            generation=generation,
            build=build,
            incarnation=None,
            code_fingerprint=expected_fingerprint,
            expected_members=expected_members,
            timeout_s=health_timeout_s,
            health_call=health_call,
        )
    except Exception:
        systemctl(["systemctl", "--user", "stop", unit])
        raise
    incarnation = str(candidate_health["gateway"]["incarnation"])
    report(f"candidate {generation}: identity and upstream readiness verified")
    try:
        systemctl(["systemctl", "--user", "enable", unit])
    except Exception:
        systemctl(["systemctl", "--user", "stop", unit])
        raise

    previous_socket = None
    if current_socket.is_symlink():
        previous_socket = Path(os.readlink(current_socket))
        if not previous_socket.is_absolute():
            previous_socket = current_socket.parent / previous_socket
    elif not bootstrap_coordinated:
        systemctl(["systemctl", "--user", "disable", "--now", unit])
        raise RolloutError(
            "no active rollout generation; use the coordinated bootstrap"
        )

    if bootstrap_coordinated:
        legacy = health_call(public_health_url, min(5.0, health_timeout_s))
        if any(legacy.get(name) != 0 for name in ("in_flight", "queued", "held")):
            systemctl(["systemctl", "--user", "disable", "--now", unit])
            raise RolloutError(
                "legacy gateway is not empty; client pause is not complete"
            )
    legacy_stopped = False
    try:
        if bootstrap_coordinated:
            systemctl(["systemctl", "--user", "disable", "--now", UNIT_NAME])
            legacy_stopped = True
        _switch(current_socket, candidate_socket)
        if bootstrap_coordinated:
            systemctl(["systemctl", "--user", "enable", "--now", FRONTEND_SOCKET_UNIT])
        wait_healthy(
            public_health_url,
            generation=generation,
            build=build,
            incarnation=incarnation,
            code_fingerprint=expected_fingerprint,
            expected_members=expected_members,
            timeout_s=health_timeout_s,
            health_call=health_call,
        )
    except Exception:
        try:
            if previous_socket is not None:
                _switch(current_socket, previous_socket)
            else:
                current_socket.unlink(missing_ok=True)
        except OSError:
            pass
        systemctl(["systemctl", "--user", "disable", "--now", unit])
        if legacy_stopped:
            systemctl(["systemctl", "--user", "disable", "--now", FRONTEND_SOCKET_UNIT])
            systemctl(["systemctl", "--user", "stop", FRONTEND_SERVICE_UNIT])
            systemctl(["systemctl", "--user", "enable", "--now", UNIT_NAME])
        raise

    previous_generation = _generation_from_socket(previous_socket)
    try:
        prior_state = _read_state(state_file)
    except RolloutError:
        prior_state = {"generations": {}}
    generations = prior_state.get("generations", {})
    if not isinstance(generations, dict):
        generations = {}
    generations[generation] = {
        "build": build,
        "incarnation": incarnation,
        "code_fingerprint": expected_fingerprint,
        "members": expected_members,
        "socket": str(candidate_socket),
        "unit": unit,
    }
    _write_json_atomic(
        state_file,
        {
            "schema_version": SCHEMA_VERSION,
            "active": generation,
            "previous": previous_generation,
            "generations": generations,
        },
    )
    report(f"candidate {generation}: frontend cut over atomically")
    if previous_generation is not None and previous_generation != generation:
        systemctl(
            [
                "systemctl",
                "--user",
                "disable",
                "--now",
                backend_unit_name(previous_generation),
            ]
        )
        report(f"generation {previous_generation}: drained and stopped")


@_serialized
def rollback(
    *,
    public_health_url: str,
    runtime_dir: Path | None = None,
    state_path: Path | None = None,
    current_socket_path: Path | None = None,
    health_timeout_s: float = DEFAULT_HEALTH_TIMEOUT_S,
    systemctl: Systemctl = _systemctl,
    health_call: HealthCall = _health,
    report: Callable[[str], None] = print,
) -> None:
    """Re-promote the recorded previous generation, then drain the current one."""
    runtime = runtime_dir or default_runtime_dir()
    state_file = state_path or default_rollout_state_path()
    state = _read_state(state_file)
    active = validate_generation(str(state.get("active", "")))
    previous = validate_generation(str(state.get("previous", "")))
    generations = state.get("generations")
    record = generations.get(previous) if isinstance(generations, dict) else None
    if not isinstance(record, Mapping):
        raise RolloutError(f"no retained metadata for previous generation {previous}")
    build = str(record.get("build", ""))
    fingerprint = str(record.get("code_fingerprint", ""))
    if not fingerprint:
        raise RolloutError(f"no retained code fingerprint for generation {previous}")
    recorded_members = record.get("members")
    expected_members = (
        recorded_members if isinstance(recorded_members, Mapping) else None
    )
    target = Path(str(record.get("socket", runtime / f"{previous}.sock")))
    unit = str(record.get("unit", backend_unit_name(previous)))
    systemctl(["systemctl", "--user", "enable", "--now", unit])
    try:
        restored_health = wait_healthy(
            target,
            generation=previous,
            build=build,
            incarnation=None,
            code_fingerprint=fingerprint,
            expected_members=expected_members,
            timeout_s=health_timeout_s,
            health_call=health_call,
        )
    except Exception:
        systemctl(["systemctl", "--user", "disable", "--now", unit])
        raise
    current = current_socket_path or default_current_socket_path()
    current.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    active_socket = _switch(current, target)
    try:
        wait_healthy(
            public_health_url,
            generation=previous,
            build=build,
            incarnation=str(restored_health["gateway"]["incarnation"]),
            code_fingerprint=fingerprint,
            expected_members=expected_members,
            timeout_s=health_timeout_s,
            health_call=health_call,
        )
    except Exception:
        if active_socket is not None:
            _switch(current, active_socket)
        systemctl(["systemctl", "--user", "disable", "--now", unit])
        raise
    state["active"] = previous
    state["previous"] = active
    record["incarnation"] = str(restored_health["gateway"]["incarnation"])
    _write_json_atomic(state_file, state)
    systemctl(["systemctl", "--user", "disable", "--now", backend_unit_name(active)])
    report(f"rollback: {previous} is active; {active} drained and stopped")
