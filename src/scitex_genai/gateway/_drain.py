"""Deterministic gateway drain before a systemd restart."""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ._unit import UNIT_NAME

DEFAULT_DRAIN_TIMEOUT_S = 1800.0
DEFAULT_POLL_INTERVAL_S = 2.0


class DrainError(RuntimeError):
    """The gateway could not be proven empty, so restart was refused."""


@dataclass(frozen=True)
class DrainState:
    in_flight: int
    queued: int
    draining: bool

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DrainState":
        try:
            in_flight = payload["in_flight"]
            queued = payload["queued"]
            draining = payload["draining"]
        except KeyError as exc:
            raise DrainError(
                "gateway health lacks drain counters; deploy matching scitex-genai "
                "source before requesting a drained restart"
            ) from exc
        if (
            not isinstance(in_flight, int)
            or isinstance(in_flight, bool)
            or in_flight < 0
            or not isinstance(queued, int)
            or isinstance(queued, bool)
            or queued < 0
            or not isinstance(draining, bool)
        ):
            raise DrainError("gateway health returned invalid drain counters")
        return cls(in_flight=in_flight, queued=queued, draining=draining)

    @property
    def empty(self) -> bool:
        return self.in_flight == 0 and self.queued == 0


HttpCall = Callable[[str, str, str], Mapping[str, Any]]
SystemctlCall = Callable[[Sequence[str]], None]


def _http_json(method: str, url: str, api_key: str) -> Mapping[str, Any]:
    request = urllib.request.Request(
        url,
        method=method,
        headers={"authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise DrainError(
            f"gateway control request failed: {method} {url}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise DrainError(f"gateway returned a non-object response: {method} {url}")
    return payload


def _systemctl(argv: Sequence[str]) -> None:
    subprocess.run(list(argv), check=True)


def restart_when_drained(
    *,
    health_url: str,
    api_key: str,
    timeout_s: float = DEFAULT_DRAIN_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    http_call: HttpCall = _http_json,
    systemctl_call: SystemctlCall = _systemctl,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    report: Callable[[str], None] = print,
) -> None:
    """Close admission, prove zero work, then restart the user unit."""
    if timeout_s <= 0:
        raise ValueError("timeout_s must be > 0")
    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be > 0")
    base_url = health_url.removesuffix("/health").rstrip("/")
    drain_url = f"{base_url}/admin/drain"
    resume_url = f"{base_url}/admin/resume"
    http_call("POST", drain_url, api_key)
    deadline = monotonic() + timeout_s
    last: DrainState | None = None
    try:
        while True:
            last = DrainState.from_payload(http_call("GET", health_url, api_key))
            if not last.draining:
                raise DrainError(
                    "gateway did not enter draining state; restart refused"
                )
            report(
                "scitex-genai-gateway: draining "
                f"in_flight={last.in_flight} queued={last.queued}"
            )
            if last.empty:
                systemctl_call(["systemctl", "--user", "restart", UNIT_NAME])
                report(f"scitex-genai-gateway: drained; restarted {UNIT_NAME}")
                return
            if monotonic() >= deadline:
                raise DrainError(
                    "drain deadline expired with "
                    f"in_flight={last.in_flight} queued={last.queued}; restart refused"
                )
            sleep(poll_interval_s)
    except BaseException:
        try:
            http_call("POST", resume_url, api_key)
        except DrainError as resume_error:
            raise DrainError(
                "drained restart failed and admission could not be reopened; "
                f"check {health_url} immediately: {resume_error}"
            ) from None
        raise
