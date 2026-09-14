"""Payload-free admission prediction and actual-report feedback.

The predictor records what an upstream *reported* after a request.  Historical
reports are useful evidence, but never prove that pages are still resident.
Callers may conservatively budget predicted *uncached prefill work*; the
prediction is not an authoritative cache-residency lookup.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DOMAIN = b"scitex-genai-lineage-v1\0"
_SESSION_DOMAIN = b"scitex-genai-prediction-session-v1\0"
_UPSTREAM_DOMAIN = b"scitex-genai-upstream-v1\0"
_STATE_SCHEMA_VERSION = 1
_MAX_STATE_BYTES = 512 * 1024
_UNAVAILABLE_GENERATION = "unavailable"


def _digest(parts: list[bytes]) -> str:
    value = hashlib.sha256(_DOMAIN)
    for part in parts:
        value.update(len(part).to_bytes(8, "big"))
        value.update(part)
    return value.hexdigest()[:16]


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


@dataclass(frozen=True)
class RequestLineage:
    """Digests for a full request and every message-boundary prefix."""

    full_digest: str
    prefix_digests: frozenset[str]


def request_lineage(body: bytes | None) -> RequestLineage:
    """Hash the entire prompt structure without retaining prompt content.

    A digest of only the first bytes cannot detect compaction or a changed tool
    schema later in a large request.  The non-message request structure is
    included in every digest; message prefixes are hashed incrementally, so
    this remains linear in payload size.
    """
    try:
        payload = json.loads(body or b"")
    except (TypeError, ValueError):
        raw = body or b""
        digest = _digest([raw])
        return RequestLineage(digest, frozenset({digest}))
    if not isinstance(payload, dict):
        digest = _digest([_canonical(payload)])
        return RequestLineage(digest, frozenset({digest}))

    sequence_name = next(
        (name for name in ("messages", "input") if isinstance(payload.get(name), list)),
        None,
    )
    if sequence_name is None:
        digest = _digest([_canonical(payload)])
        return RequestLineage(digest, frozenset({digest}))

    sequence = payload[sequence_name]
    fixed = dict(payload)
    fixed.pop(sequence_name, None)
    # Transport/reporting controls do not change the model-visible prefix.
    fixed.pop("stream", None)
    fixed.pop("stream_options", None)
    fixed.pop("return_cached_tokens_details", None)
    parts = [_canonical({"sequence": sequence_name, "fixed": fixed})]
    prefixes = {_digest(parts)}
    for item in sequence:
        parts.append(_canonical(item))
        prefixes.add(_digest(parts))
    return RequestLineage(_digest(parts), frozenset(prefixes))


@dataclass(frozen=True)
class AdmissionPrediction:
    session_label: str
    upstream_label: str
    engine_generation: str
    lineage_digest: str
    predecessor_digest: str | None
    estimated_input_tokens: int
    prior_reported_input_tokens: int | None
    prior_cached_tokens: int | None
    prior_cache_tier: str
    predicted_uncached_tokens: int
    evidence: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_label": self.session_label,
            "upstream_label": self.upstream_label,
            "engine_generation": self.engine_generation,
            "lineage_digest": self.lineage_digest,
            "predecessor_digest": self.predecessor_digest,
            "estimated_input_tokens": self.estimated_input_tokens,
            "prior_reported_input_tokens": self.prior_reported_input_tokens,
            "prior_cached_tokens": self.prior_cached_tokens,
            "prior_cache_tier": self.prior_cache_tier,
            "predicted_uncached_tokens": self.predicted_uncached_tokens,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class _Observation:
    lineage_digest: str
    estimated_input_tokens: int
    reported_input_tokens: int
    cached_tokens: int
    cache_tier: str
    observed_at: float
    observed_wall_at: float


class AdmissionPredictionTelemetry:
    """Bounded historical predictor with an explicit non-authoritative mode."""

    def __init__(
        self,
        *,
        max_sessions: int = 512,
        max_recent: int = 32,
        max_observation_age_s: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        state_path: Path | str | None = None,
    ) -> None:
        if max_sessions < 1 or max_recent < 1:
            raise ValueError("prediction bounds must be positive")
        if max_observation_age_s < 0:
            raise ValueError("max_observation_age_s must be non-negative")
        self.max_sessions = max_sessions
        self.max_observation_age_s = max_observation_age_s
        self._clock = clock
        self._wall_clock = wall_clock
        self._state_path = Path(state_path) if state_path is not None else None
        self._observations: OrderedDict[tuple[str, str, str], _Observation] = (
            OrderedDict()
        )
        self._recent: deque[dict[str, Any]] = deque(maxlen=max_recent)
        self._counters = {
            "predictions_total": 0,
            "lineage_extensions_total": 0,
            "lineage_breaks_total": 0,
            "cache_reports_total": 0,
            "missing_cache_reports_total": 0,
            "prediction_error_tokens_abs_sum": 0,
            "prediction_error_samples": 0,
        }
        self._state_restored = 0
        self._state_write_failures = 0

    @classmethod
    def _session_label(cls, session_id: str) -> str:
        return cls._label(_SESSION_DOMAIN, session_id or "anonymous")

    @classmethod
    def _upstream_label(cls, upstream: str) -> str:
        return cls._label(_UPSTREAM_DOMAIN, upstream)

    @classmethod
    def _key(
        cls, session_id: str, upstream: str, generation: str
    ) -> tuple[str, str, str]:
        return (
            cls._session_label(session_id),
            cls._upstream_label(upstream),
            generation,
        )

    @staticmethod
    def _label(domain: bytes, value: str) -> str:
        return hashlib.sha256(domain + value.encode()).hexdigest()[:12]

    def predict(
        self,
        *,
        session_id: str,
        upstream: str,
        engine_generation: str | None,
        body: bytes | None,
        estimated_input_tokens: int,
    ) -> AdmissionPrediction:
        lineage = request_lineage(body)
        if engine_generation == "unavailable":
            engine_generation = None
        key = (
            self._key(session_id, upstream, engine_generation)
            if engine_generation is not None
            else None
        )
        prior = self._observations.get(key) if key is not None else None
        if (
            prior is not None
            and self._clock() - prior.observed_at > self.max_observation_age_s
        ):
            assert key is not None
            self._observations.pop(key, None)
            prior = None
        predecessor = (
            prior.lineage_digest
            if prior is not None and prior.lineage_digest in lineage.prefix_digests
            else None
        )
        if predecessor is not None:
            self._counters["lineage_extensions_total"] += 1
            # Translate byte-estimator growth onto the last tokenizer report.
            growth = max(0, estimated_input_tokens - prior.estimated_input_tokens)
            predicted_input = prior.reported_input_tokens + growth
            # The last actual cache report, rather than the full prompt, is the
            # evidence for reusable work.  A 640k/640k report predicts only the
            # appended suffix; a 243k/103k report keeps the unreported 140k in
            # the next admission budget instead of incorrectly calling it hot.
            predicted_uncached = max(0, predicted_input - prior.cached_tokens)
            evidence = "historical-lineage-extension"
        else:
            predicted_uncached = estimated_input_tokens
            evidence = "no-compatible-history"
            if prior is not None:
                self._counters["lineage_breaks_total"] += 1
        self._counters["predictions_total"] += 1
        return AdmissionPrediction(
            session_label=self._session_label(session_id),
            upstream_label=self._upstream_label(upstream),
            engine_generation=engine_generation or "unavailable",
            lineage_digest=lineage.full_digest,
            predecessor_digest=predecessor,
            estimated_input_tokens=estimated_input_tokens,
            prior_reported_input_tokens=(
                prior.reported_input_tokens if prior else None
            ),
            prior_cached_tokens=(prior.cached_tokens if prior else None),
            prior_cache_tier=(prior.cache_tier if prior else "unknown"),
            predicted_uncached_tokens=predicted_uncached,
            evidence=evidence,
        )

    def observe(
        self,
        prediction: AdmissionPrediction,
        *,
        session_id: str,
        upstream: str,
        reported_input_tokens: int | None,
        cached_tokens: int | None,
        cache_tier: str,
    ) -> None:
        actual_uncached = (
            max(0, reported_input_tokens - cached_tokens)
            if reported_input_tokens is not None and cached_tokens is not None
            else None
        )
        missing = reported_input_tokens is None or cached_tokens is None
        generation_available = prediction.engine_generation != "unavailable"
        if missing:
            self._counters["missing_cache_reports_total"] += 1
            # Absence is evidence of nothing.  Do not let an older successful
            # report make the next turn look hot after the feedback chain broke.
            if generation_available:
                key = self._key(session_id, upstream, prediction.engine_generation)
                self._observations.pop(key, None)
        else:
            self._counters["cache_reports_total"] += 1
            error = abs(prediction.predicted_uncached_tokens - actual_uncached)
            self._counters["prediction_error_tokens_abs_sum"] += error
            self._counters["prediction_error_samples"] += 1
            if generation_available:
                key = self._key(session_id, upstream, prediction.engine_generation)
                self._observations[key] = _Observation(
                    lineage_digest=prediction.lineage_digest,
                    estimated_input_tokens=prediction.estimated_input_tokens,
                    reported_input_tokens=reported_input_tokens,
                    cached_tokens=cached_tokens,
                    cache_tier=cache_tier,
                    observed_at=self._clock(),
                    observed_wall_at=self._wall_clock(),
                )
                self._observations.move_to_end(key)
                while len(self._observations) > self.max_sessions:
                    self._observations.popitem(last=False)
        row = prediction.as_dict()
        row.update(
            reported_input_tokens=reported_input_tokens,
            actual_cached_tokens=cached_tokens,
            actual_uncached_tokens=actual_uncached,
            actual_cache_tier=cache_tier,
            cache_report_missing=missing,
        )
        self._recent.append(row)

    @staticmethod
    def _authoritative_labels(
        engine_generations: Mapping[str, str | None],
    ) -> dict[str, str]:
        """Hash upstream names and reject the placeholder as cache identity."""
        return {
            AdmissionPredictionTelemetry._upstream_label(upstream): generation
            for upstream, generation in engine_generations.items()
            if generation and generation != _UNAVAILABLE_GENERATION
        }

    def restore_state(self, engine_generations: Mapping[str, str | None]) -> int:
        """Restore fresh observations for exactly the engines now running.

        A gateway restart is not evidence that an engine survived.  Callers must
        first supply an authoritative generation for each upstream they want to
        restore; placeholder or mismatched generations are ignored.
        """
        if self._state_path is None:
            return 0
        allowed = self._authoritative_labels(engine_generations)
        if not allowed:
            return 0
        try:
            if self._state_path.stat().st_size > _MAX_STATE_BYTES:
                return 0
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            return 0
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != _STATE_SCHEMA_VERSION
        ):
            return 0
        rows = payload.get("observations")
        if not isinstance(rows, list):
            return 0
        now = self._wall_clock()
        restored = 0
        for row in rows[: self.max_sessions]:
            observation = self._validated_state_row(row, allowed=allowed, now=now)
            if observation is None:
                continue
            key, value = observation
            current = self._observations.get(key)
            if current is None or current.observed_at < value.observed_at:
                self._observations[key] = value
                self._observations.move_to_end(key)
                restored += 1
        while len(self._observations) > self.max_sessions:
            self._observations.popitem(last=False)
        self._state_restored += restored
        return restored

    def _validated_state_row(
        self,
        row: Any,
        *,
        allowed: Mapping[str, str],
        now: float,
    ) -> tuple[tuple[str, str, str], _Observation] | None:
        if not isinstance(row, dict):
            return None
        session_label = row.get("session_label")
        upstream_label = row.get("upstream_label")
        generation = row.get("engine_generation")
        lineage_digest = row.get("lineage_digest")
        cache_tier = row.get("cache_tier")
        observed_at = row.get("observed_at")
        integers = (
            row.get("estimated_input_tokens"),
            row.get("reported_input_tokens"),
            row.get("cached_tokens"),
        )

        def is_hex(value: Any, length: int) -> bool:
            return (
                isinstance(value, str)
                and len(value) == length
                and all(character in "0123456789abcdef" for character in value)
            )

        if not (
            is_hex(session_label, 12)
            and is_hex(upstream_label, 12)
            and isinstance(generation, str)
            and allowed.get(upstream_label) == generation
            and is_hex(lineage_digest, 16)
            and isinstance(cache_tier, str)
            and isinstance(observed_at, (int, float))
            and not isinstance(observed_at, bool)
            and math.isfinite(observed_at)
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in integers
            )
        ):
            return None
        age = now - float(observed_at)
        if age < 0 or age > self.max_observation_age_s:
            return None
        key = (session_label, upstream_label, generation)
        return key, _Observation(
            lineage_digest=lineage_digest,
            estimated_input_tokens=integers[0],
            reported_input_tokens=integers[1],
            cached_tokens=integers[2],
            cache_tier=cache_tier,
            observed_at=self._clock() - age,
            observed_wall_at=float(observed_at),
        )

    def save_state(self, engine_generations: Mapping[str, str | None]) -> bool:
        """Atomically hand off bounded, payload-free history for this engine set."""
        if self._state_path is None:
            return False
        allowed = self._authoritative_labels(engine_generations)
        # A partial identity set must not overwrite a complete prior handoff.
        if len(allowed) != len(engine_generations):
            return False
        now = self._clock()
        wall_now = self._wall_clock()
        anonymous = self._session_label("")
        rows = []
        for (
            session_label,
            upstream_label,
            generation,
        ), observation in self._observations.items():
            if (
                session_label == anonymous
                or allowed.get(upstream_label) != generation
                or now - observation.observed_at > self.max_observation_age_s
                or now < observation.observed_at
                or wall_now < observation.observed_wall_at
            ):
                continue
            rows.append(
                {
                    "session_label": session_label,
                    "upstream_label": upstream_label,
                    "engine_generation": generation,
                    "lineage_digest": observation.lineage_digest,
                    "estimated_input_tokens": observation.estimated_input_tokens,
                    "reported_input_tokens": observation.reported_input_tokens,
                    "cached_tokens": observation.cached_tokens,
                    "cache_tier": observation.cache_tier,
                    "observed_at": observation.observed_wall_at,
                }
            )
        payload = {
            "schema_version": _STATE_SCHEMA_VERSION,
            "written_at": wall_now,
            "observations": rows[-self.max_sessions :],
        }
        try:
            self._atomic_write(payload)
        except OSError:
            self._state_write_failures += 1
            return False
        return True

    def _atomic_write(self, payload: dict[str, Any]) -> None:
        assert self._state_path is not None
        self._state_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self._state_path.name}.", dir=self._state_path.parent
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._state_path)
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

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": "admission-feedback",
            "authoritative_for_admission": False,
            "observations": len(self._observations),
            "cumulative": dict(self._counters),
            "recent": list(self._recent),
            "state_handoff": {
                "enabled": self._state_path is not None,
                "restored": self._state_restored,
                "write_failures": self._state_write_failures,
            },
        }
