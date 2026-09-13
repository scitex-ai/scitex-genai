"""Deterministic gateway drain before a systemd restart."""

from __future__ import annotations

import json
import math
import subprocess
import urllib.error
import urllib.parse
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
    ready: bool

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DrainState":
        try:
            in_flight = payload["in_flight"]
            queued = payload["queued"]
            draining = payload["draining"]
            ready = payload["ready"]
        except KeyError as exc:
            raise DrainError(
                "gateway response lacks drain-barrier fields; deploy matching scitex-genai "
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
            or not isinstance(ready, bool)
        ):
            raise DrainError("gateway returned invalid drain-barrier fields")
        return cls(
            in_flight=in_flight,
            queued=queued,
            draining=draining,
            ready=ready,
        )

    @property
    def empty(self) -> bool:
        return self.in_flight == 0 and self.queued == 0


HttpCall = Callable[[str, str, str, float], Mapping[str, Any]]
SystemctlCall = Callable[[Sequence[str]], None]


def _http_json(
    method: str, url: str, api_key: str, timeout_s: float
) -> Mapping[str, Any]:
    request = urllib.request.Request(
        url,
        method=method,
        headers={"authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            failure = json.loads(exc.read())
            message = failure["error"]["message"]
        except (json.JSONDecodeError, KeyError, TypeError):
            message = str(exc.reason)
        raise DrainError(f"gateway refused drain: HTTP {exc.code}: {message}") from exc
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
    report: Callable[[str], None] = print,
) -> None:
    """Cross the server's atomic empty barrier, then restart the user unit."""
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be finite and > 0")
    if not math.isfinite(poll_interval_s) or poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be finite and > 0")
    base_url = health_url.removesuffix("/health").rstrip("/")
    query = urllib.parse.urlencode({"timeout_s": f"{timeout_s:g}"})
    drain_url = f"{base_url}/admin/drain?{query}"
    state = DrainState.from_payload(
        http_call("POST", drain_url, api_key, timeout_s + 5.0)
    )
    if not state.draining or state.ready or not state.empty:
        raise DrainError(
            "gateway did not confirm a closed, empty admission barrier; "
            "restart refused and admission was not reopened"
        )
    report("scitex-genai-gateway: admission closed; in_flight=0 queued=0")
    try:
        systemctl_call(["systemctl", "--user", "restart", UNIT_NAME])
    except Exception as exc:
        raise DrainError(
            f"restart failed while admission remains closed: {exc}"
        ) from exc
    report(f"scitex-genai-gateway: drained; restarted {UNIT_NAME}")
