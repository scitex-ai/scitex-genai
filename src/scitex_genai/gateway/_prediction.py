"""Payload-free admission prediction and actual-report feedback.

The predictor records what an upstream *reported* after a request.  Historical
reports are useful evidence, but never prove that pages are still resident.
Callers may conservatively budget predicted *uncached prefill work*; the
prediction is not an authoritative cache-residency lookup.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_DOMAIN = b"scitex-genai-lineage-v1\0"
_SESSION_DOMAIN = b"scitex-genai-prediction-session-v1\0"


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


class AdmissionPredictionTelemetry:
    """Bounded historical predictor with an explicit non-authoritative mode."""

    def __init__(
        self,
        *,
        max_sessions: int = 512,
        max_recent: int = 32,
        max_observation_age_s: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_sessions < 1 or max_recent < 1:
            raise ValueError("prediction bounds must be positive")
        if max_observation_age_s < 0:
            raise ValueError("max_observation_age_s must be non-negative")
        self.max_sessions = max_sessions
        self.max_observation_age_s = max_observation_age_s
        self._clock = clock
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
            (session_id, upstream, engine_generation)
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
            session_label=self._label(_SESSION_DOMAIN, session_id or "anonymous"),
            upstream_label=self._label(b"scitex-genai-upstream-v1\0", upstream),
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
                key = (session_id, upstream, prediction.engine_generation)
                self._observations.pop(key, None)
        else:
            self._counters["cache_reports_total"] += 1
            error = abs(prediction.predicted_uncached_tokens - actual_uncached)
            self._counters["prediction_error_tokens_abs_sum"] += error
            self._counters["prediction_error_samples"] += 1
            if generation_available:
                key = (session_id, upstream, prediction.engine_generation)
                self._observations[key] = _Observation(
                    lineage_digest=prediction.lineage_digest,
                    estimated_input_tokens=prediction.estimated_input_tokens,
                    reported_input_tokens=reported_input_tokens,
                    cached_tokens=cached_tokens,
                    cache_tier=cache_tier,
                    observed_at=self._clock(),
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

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": "admission-feedback",
            "authoritative_for_admission": False,
            "observations": len(self._observations),
            "cumulative": dict(self._counters),
            "recent": list(self._recent),
        }
