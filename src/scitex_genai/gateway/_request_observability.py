"""Bounded, payload-free lifecycle observations for inference requests."""

from __future__ import annotations

import hashlib
import secrets
import time
from collections import Counter, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

MAX_RECENT_REQUESTS = 256
AGENT_ID_HEADER = "x-scitex-agent-id"
_AGENT_LABEL_DOMAIN = b"scitex-genai-agent-label-v1\0"
_SESSION_LABEL_DOMAIN = b"scitex-genai-status-session-v1\0"


def _stable_label(domain: bytes, value: str, *, missing: str) -> str:
    normalized = value.strip()
    if not normalized:
        return missing
    return hashlib.sha256(domain + normalized.encode("utf-8")).hexdigest()[:12]


def request_agent_label(headers: Mapping[str, str]) -> str:
    """Hash SAC's stable agent identity without retaining its raw value."""
    for name, value in headers.items():
        if name.lower() == AGENT_ID_HEADER and isinstance(value, str):
            return _stable_label(_AGENT_LABEL_DOMAIN, value, missing="unknown")
    return "unknown"


def request_session_label(session_id: str) -> str:
    """Return one stable, non-secret label for an opaque routing session."""
    return _stable_label(_SESSION_LABEL_DOMAIN, session_id, missing="anonymous")


@dataclass
class RequestObservation:
    """One request's current phase and payload-free resource accounting."""

    request_label: str
    agent_label: str
    session_label: str
    estimated_input_tokens: int
    started_at: float
    phase: str = "admission_queued"
    phase_started_at: float = 0.0
    queue_elapsed_s: float = 0.0
    admitted_input_tokens: int | None = None
    gateway_input_tokens_admitted: int | None = None
    predicted_uncached_tokens: int | None = None
    cache_classification: str = "unknown"
    upstream: str | None = None
    outcome: str | None = None
    cache: dict[str, Any] = field(default_factory=dict)
    finished_at: float | None = None
    gateway_capacity_owned: bool = False

    def status(self, now: float) -> dict[str, Any]:
        observed_at = self.finished_at if self.finished_at is not None else now
        queue_elapsed_s = (
            max(0.0, now - self.started_at)
            if self.phase == "admission_queued"
            else self.queue_elapsed_s
        )
        row: dict[str, Any] = {
            "request_label": self.request_label,
            "agent_label": self.agent_label,
            "session_label": self.session_label,
            "phase": self.phase,
            "elapsed_s": max(0.0, observed_at - self.started_at),
            "phase_elapsed_s": (
                0.0
                if self.finished_at is not None
                else max(0.0, now - self.phase_started_at)
            ),
            "queue_elapsed_s": queue_elapsed_s,
            "estimated_input_tokens": self.estimated_input_tokens,
            "admitted_input_tokens": self.admitted_input_tokens,
            "gateway_input_tokens_admitted": self.gateway_input_tokens_admitted,
            "gateway_capacity_owned": self.gateway_capacity_owned,
            "predicted_uncached_tokens": self.predicted_uncached_tokens,
            "cache_classification": self.cache_classification,
            "upstream": self.upstream,
            "outcome": self.outcome,
        }
        if self.cache:
            row["cache"] = dict(self.cache)
        return row


class RequestLifecycleRegistry:
    """Track every active request and a bounded tail of terminal requests.

    Mutations and snapshots are synchronous because they run on the gateway's
    single asyncio event-loop thread. Active records are never evicted; only
    completed/disconnected history is bounded.
    """

    def __init__(
        self,
        *,
        max_recent: int = MAX_RECENT_REQUESTS,
        clock: Callable[[], float] = time.monotonic,
        request_label_factory: Callable[[], str] | None = None,
    ) -> None:
        if max_recent < 0:
            raise ValueError("max_recent must be >= 0")
        self.max_recent = max_recent
        self._clock = clock
        self._request_label_factory = request_label_factory or (
            lambda: secrets.token_hex(8)
        )
        self._active: dict[str, RequestObservation] = {}
        self._recent: deque[RequestObservation] = deque(maxlen=max_recent or None)
        self._terminal_total: Counter[str] = Counter()

    def start(
        self, *, agent_label: str, session_label: str, estimated_input_tokens: int
    ) -> RequestObservation:
        now = self._clock()
        record = RequestObservation(
            request_label=self._request_label_factory(),
            agent_label=agent_label,
            session_label=session_label,
            estimated_input_tokens=max(0, estimated_input_tokens),
            started_at=now,
            phase_started_at=now,
        )
        self._active[record.request_label] = record
        return record

    def admission_queued(self, record: RequestObservation) -> None:
        now = self._clock()
        record.phase = "admission_queued"
        record.phase_started_at = now
        record.upstream = None
        record.admitted_input_tokens = None
        record.gateway_input_tokens_admitted = None
        record.predicted_uncached_tokens = None
        record.cache_classification = "unknown"
        record.gateway_capacity_owned = False

    def upstream_inflight(
        self,
        record: RequestObservation,
        *,
        upstream: str,
        admitted_input_tokens: int,
        gateway_input_tokens_admitted: int,
        predicted_uncached_tokens: int,
        cache_classification: str,
    ) -> None:
        now = self._clock()
        record.queue_elapsed_s = max(0.0, now - record.started_at)
        record.phase = "upstream_inflight"
        record.phase_started_at = now
        record.upstream = upstream
        record.admitted_input_tokens = max(0, admitted_input_tokens)
        record.gateway_input_tokens_admitted = max(0, gateway_input_tokens_admitted)
        record.predicted_uncached_tokens = max(0, predicted_uncached_tokens)
        record.cache_classification = cache_classification
        record.gateway_capacity_owned = True

    def capacity_released(self, record: RequestObservation) -> None:
        """Settle gateway admission ownership, including delayed reapers."""
        record.gateway_capacity_owned = False
        if record.finished_at is not None:
            self._active.pop(record.request_label, None)
            if self.max_recent and record not in self._recent:
                self._recent.append(record)

    def completed(
        self,
        record: RequestObservation,
        *,
        outcome: str = "complete",
        cache: Mapping[str, Any] | None = None,
    ) -> None:
        self._terminal(record, phase="completed", outcome=outcome, cache=cache)

    def disconnected(
        self,
        record: RequestObservation,
        *,
        outcome: str = "client_disconnected",
        cache: Mapping[str, Any] | None = None,
    ) -> None:
        self._terminal(record, phase="disconnected", outcome=outcome, cache=cache)

    def _terminal(
        self,
        record: RequestObservation,
        *,
        phase: str,
        outcome: str,
        cache: Mapping[str, Any] | None,
    ) -> None:
        if record.request_label not in self._active or record.finished_at is not None:
            return
        now = self._clock()
        if record.phase == "admission_queued":
            record.queue_elapsed_s = max(0.0, now - record.started_at)
        record.phase = phase
        record.phase_started_at = now
        record.finished_at = now
        record.outcome = outcome
        record.cache = {
            key: value for key, value in (cache or {}).items() if value is not None
        }
        self._terminal_total[phase] += 1
        if not record.gateway_capacity_owned:
            self._active.pop(record.request_label, None)
            if self.max_recent:
                self._recent.append(record)

    def snapshot(self) -> dict[str, Any]:
        now = self._clock()
        active = list(self._active.values())
        recent = list(reversed(self._recent)) if self.max_recent else []
        phases = Counter(record.phase for record in active)
        return {
            "active": len(active),
            "active_by_phase": dict(sorted(phases.items())),
            "terminal_total": dict(sorted(self._terminal_total.items())),
            "requests": [record.status(now) for record in active + recent],
            "recent_limit": self.max_recent,
        }

    def health_snapshot(self) -> dict[str, Any]:
        """Aggregate-only request state for the unauthenticated health view."""
        active = list(self._active.values())
        phases = Counter(record.phase for record in active)
        return {
            "active": len(active),
            "active_by_phase": dict(sorted(phases.items())),
            "terminal_total": dict(sorted(self._terminal_total.items())),
        }


__all__ = [
    "AGENT_ID_HEADER",
    "MAX_RECENT_REQUESTS",
    "RequestLifecycleRegistry",
    "RequestObservation",
    "request_agent_label",
    "request_session_label",
]
