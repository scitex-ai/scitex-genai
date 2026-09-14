import json

import pytest

from scitex_genai.gateway._inference import InferenceBackend, InferenceUpstreamPool
from scitex_genai.gateway._prediction import AdmissionPredictionTelemetry


def _body(messages: list[dict[str, str]], *, tools: str = "stable") -> bytes:
    return json.dumps(
        {"model": "qwen", "tools": [{"name": tools}], "messages": messages}
    ).encode()


def _observe(
    telemetry: AdmissionPredictionTelemetry,
    *,
    messages: list[dict[str, str]],
    estimated: int,
    reported: int,
    cached: int,
    generation: str = "engine-a",
):
    body = _body(messages)
    prediction = telemetry.predict(
        session_id="session",
        upstream="upstream",
        engine_generation=generation,
        body=body,
        estimated_input_tokens=estimated,
    )
    telemetry.observe(
        prediction,
        session_id="session",
        upstream="upstream",
        reported_input_tokens=reported,
        cached_tokens=cached,
        cache_tier="device" if cached else "none",
    )
    return prediction


def test_observed_app_compaction_is_a_lineage_break_not_a_hot_prediction() -> None:
    # Arrange
    telemetry = AdmissionPredictionTelemetry()
    common = {"role": "system", "content": "x" * 20_000}
    old = [common, {"role": "user", "content": "old history"}]
    _observe(
        telemetry, messages=old, estimated=671_627, reported=687_453, cached=687_040
    )

    # Act
    compacted = [common, {"role": "user", "content": "compacted summary"}]
    prediction = telemetry.predict(
        session_id="session",
        upstream="upstream",
        engine_generation="engine-a",
        body=_body(compacted),
        estimated_input_tokens=665_653,
    )

    # Assert
    assert (
        prediction.evidence,
        prediction.predecessor_digest,
        prediction.predicted_uncached_tokens,
    ) == ("no-compatible-history", None, 665_653)


def test_observed_hub_append_only_turn_predicts_only_growth() -> None:
    # Arrange
    telemetry = AdmissionPredictionTelemetry()
    prior = [{"role": "system", "content": "rules"}]
    _observe(
        telemetry, messages=prior, estimated=404_000, reported=450_000, cached=449_000
    )
    current = prior + [{"role": "user", "content": "continue"}]

    # Act
    prediction = telemetry.predict(
        session_id="session",
        upstream="upstream",
        engine_generation="engine-a",
        body=_body(current),
        estimated_input_tokens=405_000,
    )

    # Assert
    assert (
        prediction.evidence,
        prediction.predecessor_digest is not None,
        prediction.predicted_uncached_tokens,
    ) == ("historical-lineage-extension", True, 2_000)


def test_partial_actual_cache_report_remains_in_the_next_uncached_budget() -> None:
    # Arrange
    telemetry = AdmissionPredictionTelemetry()
    messages = [{"role": "user", "content": "large turn"}]
    _observe(
        telemetry,
        messages=messages,
        estimated=243_434,
        reported=243_434,
        cached=103_296,
    )

    # Act
    prediction = telemetry.predict(
        session_id="session",
        upstream="upstream",
        engine_generation="engine-a",
        body=_body(messages),
        estimated_input_tokens=243_434,
    )

    # Assert
    assert prediction.predicted_uncached_tokens == 140_138


def test_hot_history_expires_before_the_observed_residency_loss_window() -> None:
    # Arrange
    now = [0.0]
    telemetry = AdmissionPredictionTelemetry(
        max_observation_age_s=300.0, clock=lambda: now[0]
    )
    messages = [{"role": "user", "content": "large turn"}]
    _observe(
        telemetry,
        messages=messages,
        estimated=642_616,
        reported=642_616,
        cached=642_616,
    )
    now[0] = 362.803

    # Act
    prediction = telemetry.predict(
        session_id="session",
        upstream="upstream",
        engine_generation="engine-a",
        body=_body(messages),
        estimated_input_tokens=642_616,
    )

    # Assert
    assert (prediction.evidence, prediction.predicted_uncached_tokens) == (
        "no-compatible-history",
        642_616,
    )


def test_observed_ui_historical_hot_can_still_miss_and_feedback_records_error() -> None:
    # Arrange
    telemetry = AdmissionPredictionTelemetry()
    prior = [{"role": "system", "content": "rules"}]
    _observe(
        telemetry, messages=prior, estimated=500_735, reported=546_784, cached=545_856
    )
    current = prior + [{"role": "user", "content": "next"}]
    # Act
    prediction = telemetry.predict(
        session_id="session",
        upstream="upstream",
        engine_generation="engine-a",
        body=_body(current),
        estimated_input_tokens=501_171,
    )
    telemetry.observe(
        prediction,
        session_id="session",
        upstream="upstream",
        reported_input_tokens=547_199,
        cached_tokens=13_184,
        cache_tier="device",
    )

    snapshot = telemetry.snapshot()

    # Assert
    assert (
        snapshot["authoritative_for_admission"],
        snapshot["recent"][-1]["actual_uncached_tokens"],
        snapshot["cumulative"]["prediction_error_tokens_abs_sum"] >= 533_000,
    ) == (False, 534_015, True)


def test_generation_change_and_missing_report_never_reuse_history() -> None:
    # Arrange
    telemetry = AdmissionPredictionTelemetry()
    prior = [{"role": "system", "content": "rules"}]
    _observe(telemetry, messages=prior, estimated=100, reported=110, cached=100)
    current = prior + [{"role": "user", "content": "next"}]
    # Act
    prediction = telemetry.predict(
        session_id="session",
        upstream="upstream",
        engine_generation="engine-b",
        body=_body(current),
        estimated_input_tokens=120,
    )
    telemetry.observe(
        prediction,
        session_id="session",
        upstream="upstream",
        reported_input_tokens=130,
        cached_tokens=None,
        cache_tier="unknown",
    )

    snapshot = telemetry.snapshot()

    # Assert
    assert (
        prediction.evidence,
        prediction.predicted_uncached_tokens,
        snapshot["cumulative"]["missing_cache_reports_total"],
        snapshot["observations"],
    ) == ("no-compatible-history", 120, 1, 1)


def test_missing_cache_report_invalidates_same_generation_history() -> None:
    # Arrange
    telemetry = AdmissionPredictionTelemetry()
    messages = [{"role": "user", "content": "one"}]
    _observe(telemetry, messages=messages, estimated=100, reported=100, cached=100)
    # Act
    missing = telemetry.predict(
        session_id="session",
        upstream="upstream",
        engine_generation="engine-a",
        body=_body(messages + [{"role": "user", "content": "two"}]),
        estimated_input_tokens=110,
    )
    telemetry.observe(
        missing,
        session_id="session",
        upstream="upstream",
        reported_input_tokens=110,
        cached_tokens=None,
        cache_tier="unknown",
    )

    next_prediction = telemetry.predict(
        session_id="session",
        upstream="upstream",
        engine_generation="engine-a",
        body=_body(messages + [{"role": "user", "content": "three"}]),
        estimated_input_tokens=120,
    )

    # Assert
    assert (next_prediction.evidence, next_prediction.predicted_uncached_tokens) == (
        "no-compatible-history",
        120,
    )


def test_observations_and_recent_rows_are_bounded() -> None:
    # Arrange
    telemetry = AdmissionPredictionTelemetry(max_sessions=2, max_recent=2)

    # Act
    for index in range(3):
        body = _body([{"role": "user", "content": str(index)}])
        prediction = telemetry.predict(
            session_id=f"session-{index}",
            upstream="upstream",
            engine_generation="engine",
            body=body,
            estimated_input_tokens=10,
        )
        telemetry.observe(
            prediction,
            session_id=f"session-{index}",
            upstream="upstream",
            reported_input_tokens=10,
            cached_tokens=0,
            cache_tier="none",
        )

    snapshot = telemetry.snapshot()

    # Assert
    assert (
        snapshot["observations"],
        len(snapshot["recent"]),
        "session-" not in json.dumps(snapshot),
    ) == (2, 2, True)


@pytest.mark.asyncio
async def test_operator_snapshot_distinguishes_gateway_admitted_from_backend_queue() -> (
    None
):
    # Arrange
    pool = InferenceUpstreamPool.from_urls("http://engine:1")
    backend = InferenceBackend(pool)
    member = await pool.acquire("session", input_tokens=500_000)
    backend.observe_backend_scheduler(
        upstream=member.alias,
        engine_generation="process-123",
        running=0,
        queued=2,
        token_usage=0.22,
    )

    # Act
    snapshot = await backend.observability_snapshot()
    comparison = snapshot["admission_prediction"]["gateway_backend_comparison"]
    await pool.release(member, input_tokens=500_000, session_id="session")

    # Assert
    assert (
        comparison["gateway_admitted"],
        comparison["gateway_queued"],
        comparison["backend"]["running"],
        comparison["backend"]["queued"],
        comparison["backend"]["token_usage"],
        comparison["block_reason"],
        comparison["backend"]["engine_generation"] != "process-123",
    ) == (1, 0, 0, 2, 0.22, "backend-scheduler-queue", True)
