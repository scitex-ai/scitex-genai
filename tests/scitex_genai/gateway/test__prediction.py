import json
import stat
from pathlib import Path

import pytest

from scitex_genai.gateway._inference import InferenceBackend, InferenceUpstreamPool
from scitex_genai.gateway._prediction import AdmissionPredictionTelemetry
from scitex_genai.gateway._sglang_metrics import SGLangSchedulerObservation


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


def test_unavailable_generation_never_records_reusable_history() -> None:
    # Arrange
    telemetry = AdmissionPredictionTelemetry()
    first = telemetry.predict(
        session_id="session",
        upstream="engine",
        engine_generation=None,
        body=b'{"messages":[{"role":"user","content":"first"}]}',
        estimated_input_tokens=100,
    )
    telemetry.observe(
        first,
        session_id="session",
        upstream="engine",
        reported_input_tokens=100,
        cached_tokens=90,
        cache_tier="device",
    )

    # Act
    second = telemetry.predict(
        session_id="session",
        upstream="engine",
        engine_generation=None,
        body=(
            b'{"messages":[{"role":"user","content":"first"},'
            b'{"role":"user","content":"second"}]}'
        ),
        estimated_input_tokens=120,
    )

    # Assert
    assert (
        first.engine_generation,
        second.evidence,
        second.predicted_uncached_tokens,
        telemetry.snapshot()["observations"],
    ) == ("unavailable", "no-compatible-history", 120, 0)


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


def test_fresh_matching_generation_history_survives_gateway_restart(
    tmp_path: Path,
) -> None:
    # Arrange
    now = [1_000.0]
    path = tmp_path / "runtime" / "admission-history.json"
    first = AdmissionPredictionTelemetry(
        clock=lambda: now[0], wall_clock=lambda: now[0], state_path=path
    )
    prior = [{"role": "user", "content": "private prompt"}]
    _observe(
        first,
        messages=prior,
        estimated=640_000,
        reported=640_000,
        cached=639_000,
        generation="engine-a",
    )
    saved = first.save_state({"upstream": "engine-a"})
    now[0] += 1
    restarted = AdmissionPredictionTelemetry(
        clock=lambda: now[0], wall_clock=lambda: now[0], state_path=path
    )

    # Act
    restored = restarted.restore_state({"upstream": "engine-a"})
    prediction = restarted.predict(
        session_id="session",
        upstream="upstream",
        engine_generation="engine-a",
        body=_body(prior + [{"role": "user", "content": "next"}]),
        estimated_input_tokens=641_000,
    )

    # Assert
    serialized = path.read_text(encoding="utf-8")
    assert (
        saved,
        restored,
        prediction.evidence,
        prediction.predicted_uncached_tokens,
        "private prompt" in serialized,
        '"session"' in serialized,
        "http://" in serialized,
        stat.S_IMODE(path.stat().st_mode),
    ) == (True, 1, "historical-lineage-extension", 2_000, False, False, False, 0o600)


@pytest.mark.parametrize(
    ("generation", "elapsed"),
    (("engine-b", 1.0), ("engine-a", 300.001), ("unavailable", 1.0)),
)
def test_history_is_not_restored_for_wrong_unknown_or_expired_engine(
    tmp_path: Path, generation: str, elapsed: float
) -> None:
    # Arrange
    now = [1_000.0]
    path = tmp_path / "admission-history.json"
    first = AdmissionPredictionTelemetry(
        clock=lambda: now[0], wall_clock=lambda: now[0], state_path=path
    )
    messages = [{"role": "user", "content": "one"}]
    _observe(
        first,
        messages=messages,
        estimated=100,
        reported=100,
        cached=100,
        generation="engine-a",
    )
    first.save_state({"upstream": "engine-a"})
    now[0] += elapsed
    restarted = AdmissionPredictionTelemetry(
        clock=lambda: now[0], wall_clock=lambda: now[0], state_path=path
    )

    # Act
    restored = restarted.restore_state({"upstream": generation})
    prediction = restarted.predict(
        session_id="session",
        upstream="upstream",
        engine_generation=generation,
        body=_body(messages),
        estimated_input_tokens=100,
    )

    # Assert
    assert (restored, prediction.evidence, prediction.predicted_uncached_tokens) == (
        0,
        "no-compatible-history",
        100,
    )


def test_partial_engine_identity_never_overwrites_prior_handoff(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "admission-history.json"
    telemetry = AdmissionPredictionTelemetry(state_path=path)
    path.write_text("keep until every engine is identified", encoding="utf-8")

    # Act
    saved = telemetry.save_state(
        {"http://one:1": "generation-one", "http://two:2": "unavailable"}
    )

    # Assert
    assert (saved, path.read_text(encoding="utf-8")) == (
        False,
        "keep until every engine is identified",
    )


def test_backend_generation_hook_restores_only_after_authoritative_identity(
    tmp_path: Path,
) -> None:
    # Arrange
    path = tmp_path / "admission-history.json"
    upstream = "http://engine:1"
    first = InferenceBackend(
        InferenceUpstreamPool.from_urls(upstream), admission_history_path=path
    )
    generation = first.observe_engine_generation(
        upstream=upstream, engine_generation="pid-123-start-456"
    )
    messages = [{"role": "user", "content": "one"}]
    prediction = first.admission_predictions.predict(
        session_id="session",
        upstream=upstream,
        engine_generation=generation,
        body=_body(messages),
        estimated_input_tokens=100,
    )
    first.admission_predictions.observe(
        prediction,
        session_id="session",
        upstream=upstream,
        reported_input_tokens=100,
        cached_tokens=100,
        cache_tier="device",
    )
    first.admission_predictions.save_state(first._engine_generations)
    restarted = InferenceBackend(
        InferenceUpstreamPool.from_urls(upstream), admission_history_path=path
    )

    # Act
    before = restarted.admission_predictions.snapshot()["observations"]
    restarted.observe_engine_generation(
        upstream=upstream, engine_generation="pid-123-start-456"
    )
    after = restarted.admission_predictions.snapshot()

    # Assert
    assert (before, after["observations"], after["state_handoff"]["restored"]) == (
        0,
        1,
        1,
    )


@pytest.mark.asyncio
async def test_operator_snapshot_distinguishes_gateway_admitted_from_backend_queue() -> (
    None
):
    # Arrange
    pool = InferenceUpstreamPool.from_urls("http://engine:1")

    async def scheduler_probe(
        upstream: str, timeout_s: float
    ) -> SGLangSchedulerObservation:
        return SGLangSchedulerObservation("1" * 32, 0, 2, 0.22)

    backend = InferenceBackend(pool, scheduler_probe=scheduler_probe)
    member = await pool.acquire("session", input_tokens=500_000)

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
        comparison["backend"]["engine_generation"] != "1" * 32,
    ) == (1, 0, 0, 2, 0.22, "backend-scheduler-queue", True)


@pytest.mark.asyncio
async def test_scheduler_generation_change_invalidates_prediction_history() -> None:
    # Arrange
    generations = iter(("1" * 32, "2" * 32))

    async def scheduler_probe(
        upstream: str, timeout_s: float
    ) -> SGLangSchedulerObservation:
        return SGLangSchedulerObservation(next(generations), 0, 0, 0.0)

    backend = InferenceBackend(
        InferenceUpstreamPool.from_urls("http://engine:1"),
        scheduler_probe=scheduler_probe,
    )
    # Act
    await backend.refresh_backend_scheduler("http://engine:1")
    first = backend.admission_predictions.predict(
        session_id="session",
        upstream="http://engine:1",
        engine_generation=backend._engine_generations["http://engine:1"],
        body=b'{"messages":[{"role":"user","content":"one"}]}',
        estimated_input_tokens=100,
    )
    backend.admission_predictions.observe(
        first,
        session_id="session",
        upstream="http://engine:1",
        reported_input_tokens=100,
        cached_tokens=90,
        cache_tier="device",
    )

    await backend.refresh_backend_scheduler("http://engine:1")
    second = backend.admission_predictions.predict(
        session_id="session",
        upstream="http://engine:1",
        engine_generation=backend._engine_generations["http://engine:1"],
        body=(
            b'{"messages":[{"role":"user","content":"one"},'
            b'{"role":"user","content":"two"}]}'
        ),
        estimated_input_tokens=120,
    )

    # Assert
    assert (first.engine_generation != second.engine_generation, second.evidence) == (
        True,
        "no-compatible-history",
    )


@pytest.mark.asyncio
async def test_failed_generation_probe_makes_prior_history_non_reusable() -> None:
    # Arrange
    calls = 0

    async def scheduler_probe(
        upstream: str, timeout_s: float
    ) -> SGLangSchedulerObservation:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TimeoutError
        return SGLangSchedulerObservation("1" * 32, 0, 0, 0.0)

    backend = InferenceBackend(
        InferenceUpstreamPool.from_urls("http://engine:1"),
        scheduler_probe=scheduler_probe,
    )
    # Act
    await backend.refresh_backend_scheduler("http://engine:1")
    first = backend.admission_predictions.predict(
        session_id="session",
        upstream="http://engine:1",
        engine_generation=backend._engine_generations["http://engine:1"],
        body=b'{"messages":[{"role":"user","content":"one"}]}',
        estimated_input_tokens=100,
    )
    backend.admission_predictions.observe(
        first,
        session_id="session",
        upstream="http://engine:1",
        reported_input_tokens=100,
        cached_tokens=90,
        cache_tier="device",
    )

    await backend.refresh_backend_scheduler("http://engine:1")
    second = backend.admission_predictions.predict(
        session_id="session",
        upstream="http://engine:1",
        engine_generation=backend._engine_generations["http://engine:1"],
        body=(
            b'{"messages":[{"role":"user","content":"one"},'
            b'{"role":"user","content":"two"}]}'
        ),
        estimated_input_tokens=120,
    )

    # Assert
    assert (
        second.engine_generation,
        second.evidence,
        second.predicted_uncached_tokens,
    ) == ("unavailable", "no-compatible-history", 120)
