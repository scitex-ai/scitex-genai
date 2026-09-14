"""Small payload-free session metadata shared by gateway generations."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
MAX_STATE_BYTES = 256 * 1024


def _label(session_id: str) -> str:
    return hashlib.sha256(
        b"scitex-genai-gateway-session-v1\0" + session_id.encode("utf-8")
    ).hexdigest()[:24]


class GatewaySessionState:
    """Locked atomic storage for hashed route and continuation metadata."""

    def __init__(self, path: Path | str, *, max_sessions: int = 512) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(f"{self.path.suffix}.lock")
        self.max_sessions = max_sessions

    def route(self, session_id: str, allowed: set[str]) -> str | None:
        if not session_id:
            return None
        value = self._read().get("routes", {}).get(_label(session_id))
        return value if isinstance(value, str) and value in allowed else None

    def remember_route(self, session_id: str, alias: str) -> None:
        if session_id:
            self._update("routes", _label(session_id), alias)

    def successful(self, session_id: str) -> bool:
        return bool(
            session_id and _label(session_id) in self._read().get("successful", {})
        )

    def remember_success(self, session_id: str) -> None:
        if session_id:
            self._update("successful", _label(session_id), True)

    def _read(self) -> dict[str, Any]:
        try:
            if self.path.stat().st_size > MAX_STATE_BYTES:
                return {}
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            return {}
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != SCHEMA_VERSION
        ):
            return {}
        return payload

    def _update(self, namespace: str, key: str, value: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            os.chmod(self.lock_path, 0o600)
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            payload = self._read()
            payload["schema_version"] = SCHEMA_VERSION
            table = payload.setdefault(namespace, {})
            if not isinstance(table, dict):
                table = payload[namespace] = {}
            if table.get(key) == value:
                return
            table.pop(key, None)
            table[key] = value
            while len(table) > self.max_sessions:
                table.pop(next(iter(table)))
            self._atomic_write(payload)

    def _atomic_write(self, payload: dict[str, Any]) -> None:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=self.path.parent
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
