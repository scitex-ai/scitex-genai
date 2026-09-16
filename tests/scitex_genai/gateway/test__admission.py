from __future__ import annotations

import asyncio
import json
import threading
import time

import httpx
import pytest

from scitex_genai.gateway._admission import (
    AdmissionController,
    CacheAdmissionSettings,
    CacheResidency,
    classify_cache_prediction,
)
from scitex_genai.gateway._errors import InferenceAdmissionError, UpstreamUnreachable
from scitex_genai.gateway._inference import (
    InferenceBackend,
    InferenceUpstreamPool,
    is_read_only_control_route,
    request_session_key,
)
from scitex_genai.gateway._server import create_app
from scitex_genai.gateway._sglang_metrics import SGLangSchedulerObservation


@pytest.mark.asyncio
async def test_admission_is_observe_only_by_default() -> None:
    # Arrange
    controller = AdmissionController(max_concurrent=1)

    # Act
    first = await controller.acquire(CacheResidency.COLD)
    second = await asyncio.wait_for(controller.acquire(CacheResidency.HOT), timeout=0.1)
    snapshot = controller.snapshot()
    await second.release()
    await first.release()

    # Assert
    assert (
        snapshot["mode"],
        snapshot["admitted"],
        snapshot["admitted_total_by_residency"],
        snapshot["queued"],
        "running" in snapshot,
    ) == (
        "observe-only",
        0,
        {"hot": 0, "cold": 0, "unknown": 0},
        0,
        False,
    )


@pytest.mark.asyncio
async def test_known_hot_work_overtakes_queued_cold_work() -> None:
    # Arrange
    controller = AdmissionController(enabled=True, max_concurrent=1)
    blocker = await controller.acquire(CacheResidency.HOT)
    cold_task = asyncio.create_task(controller.acquire(CacheResidency.COLD))
    await asyncio.sleep(0)
    hot_task = asyncio.create_task(controller.acquire(CacheResidency.HOT))
    await asyncio.sleep(0)

    # Act
    await blocker.release()
    hot = await asyncio.wait_for(hot_task, timeout=0.1)
    result = (cold_task.done(), controller.snapshot()["hot_overtakes"])
    await hot.release()
    cold = await asyncio.wait_for(cold_task, timeout=0.1)
    await cold.release()

    # Assert
    assert result == (False, 1)


@pytest.mark.asyncio
async def test_aged_cold_work_receives_the_next_progressing_slot() -> None:
    # Arrange
    now = [10.0]
    controller = AdmissionController(
        enabled=True,
        max_concurrent=1,
        max_cold_wait_s=5.0,
        clock=lambda: now[0],
    )
    blocker = await controller.acquire(CacheResidency.HOT)
    cold_task = asyncio.create_task(controller.acquire(CacheResidency.COLD))
    await asyncio.sleep(0)
    now[0] = 16.0
    hot_task = asyncio.create_task(controller.acquire(CacheResidency.HOT))
    await asyncio.sleep(0)

    # Act
    await blocker.release()
    cold = await asyncio.wait_for(cold_task, timeout=0.1)
    hot_was_waiting = hot_task.done() is False
    await cold.release()
    hot = await asyncio.wait_for(hot_task, timeout=0.1)
    await hot.release()

    # Assert
    assert hot_was_waiting is True


@pytest.mark.parametrize(
    ("evidence", "tier", "uncached", "expected"),
    [
        ("no-compatible-history", "unknown", 400_000, CacheResidency.COLD),
        ("historical-lineage-extension", "device", 2_000, CacheResidency.HOT),
        ("historical-lineage-extension", "host", 40_000, CacheResidency.COLD),
        ("historical-lineage-extension", "storage", 32_768, CacheResidency.HOT),
        ("historical-lineage-extension", "none", 1, CacheResidency.COLD),
    ],
)
def test_active_classification_uses_generation_bound_cache_feedback(
    evidence: str, tier: str, uncached: int, expected: CacheResidency
) -> None:
    # Arrange
    settings = CacheAdmissionSettings(mode="active")

    # Act
    actual = classify_cache_prediction(
        settings=settings,
        engine_generation="engine-1",
        evidence=evidence,
        prior_cache_tier=tier,
        predicted_uncached_tokens=uncached,
    )

    # Assert
    assert actual is expected


@pytest.mark.parametrize(
    ("generation", "evidence", "tier"),
    [
        ("unavailable", "historical-lineage-extension", "device"),
        ("engine-1", "missing-prior-cache-report", "device"),
        ("engine-1", "historical-lineage-extension", "unknown"),
    ],
)
def test_active_classification_fails_fast_without_authoritative_cache_evidence(
    generation: str, evidence: str, tier: str
) -> None:
    # Arrange
    settings = CacheAdmissionSettings(mode="active")

    # Act
    with pytest.raises(ValueError, match="active cache admission"):
        classify_cache_prediction(
            settings=settings,
            engine_generation=generation,
            evidence=evidence,
            prior_cache_tier=tier,
            predicted_uncached_tokens=1,
        )

    # Assert


@pytest.mark.asyncio
async def test_active_relay_fails_before_dispatch_when_engine_evidence_is_missing(
    upstream_factory,
) -> None:
    # Arrange
    upstream = upstream_factory()

    async def unavailable(_url: str, _timeout_s: float):
        raise RuntimeError("metrics unavailable")

    backend = InferenceBackend(
        InferenceUpstreamPool.from_urls(
            upstream.url,
            cold_prefill_limit_per_upstream=1,
            cold_prefill_min_tokens=32_768,
        ),
        scheduler_probe=unavailable,
        cache_report_enabled=True,
        cache_admission_settings=CacheAdmissionSettings(mode="active"),
    )

    # Act
    raised = None
    try:
        await backend.relay(
            "POST",
            "/v1/chat/completions",
            body=b'{"model":"m","messages":[{"role":"user","content":"hello"}]}',
            headers={"X-SciTeX-Session-ID": "session"},
        )
    except InferenceAdmissionError as exc:
        raised = exc

    # Assert
    assert (
        "lacks an engine generation" in str(raised),
        len(upstream.requests),
        backend.pool.status()[0]["in_flight"],
    ) == (True, 0, 0)


def test_active_backend_rejects_missing_cache_reports() -> None:
    # Arrange
    settings = CacheAdmissionSettings(mode="active")
    matching = InferenceUpstreamPool.from_urls(
        "http://only:1",
        max_admission_bypasses=settings.max_hot_bypasses,
        priority_aging_s=settings.starvation_age_s,
        cache_prediction_max_age_s=settings.evidence_max_age_s,
        cold_prefill_limit_per_upstream=settings.cold_prefill_limit_per_upstream,
        cold_prefill_min_tokens=settings.hot_max_uncached_tokens,
    )
    # Act
    with pytest.raises(ValueError, match="requires cache_report_enabled"):
        InferenceBackend(matching, cache_admission_settings=settings)

    # Assert


def test_active_backend_rejects_unvalidated_pool_policy() -> None:
    # Arrange
    settings = CacheAdmissionSettings(mode="active")
    mismatched = InferenceUpstreamPool.from_urls(
        "http://only:1",
        cold_prefill_limit_per_upstream=1,
        cold_prefill_min_tokens=64_000,
    )

    # Act
    with pytest.raises(ValueError, match="pool policy does not match"):
        InferenceBackend(
            mismatched,
            cache_report_enabled=True,
            cache_admission_settings=settings,
        )

    # Assert


@pytest.mark.asyncio
async def test_active_relay_promotes_only_feedback_backed_lineage_to_hot(
    upstream_factory,
) -> None:
    # Arrange
    report = {
        "usage": {
            "prompt_tokens": 1_000,
            "prompt_tokens_details": {"cached_tokens": 990},
        },
        "sglext": {"cached_tokens_details": {"device": 990, "host": 0}},
    }
    upstream = upstream_factory(chunks=(json.dumps(report).encode(),))

    async def scheduler_probe(
        _url: str, _timeout_s: float
    ) -> SGLangSchedulerObservation:
        return SGLangSchedulerObservation("1" * 32, 0, 0, 0.0)

    settings = CacheAdmissionSettings(mode="active")
    backend = InferenceBackend(
        InferenceUpstreamPool.from_urls(
            upstream.url,
            cold_prefill_limit_per_upstream=1,
            cold_prefill_min_tokens=settings.hot_max_uncached_tokens,
        ),
        scheduler_probe=scheduler_probe,
        cache_report_enabled=True,
        cache_admission_settings=settings,
    )
    body = b'{"model":"m","messages":[{"role":"user","content":"same"}]}'
    headers = {"X-SciTeX-Session-ID": "session"}
    first = await backend.relay(
        "POST", "/v1/chat/completions", body=body, headers=headers
    )
    _ = b"".join([chunk async for chunk in first.body])

    # Act
    second = await backend.relay(
        "POST", "/v1/chat/completions", body=body, headers=headers
    )
    _ = b"".join([chunk async for chunk in second.body])
    status = await backend.observability_snapshot()

    # Assert
    assert (
        first.feedback_headers["x-scitex-admission-mode"],
        first.feedback_headers["x-scitex-cache-residency"],
        second.feedback_headers["x-scitex-cache-residency"],
        status["admission_prediction"]["authoritative_for_admission"],
        backend.cache_admission.snapshot()["mode"],
    ) == ("active", "cold", "hot", True, "active")


@pytest.mark.asyncio
async def test_keyless_relay_reports_unknown_without_failing(upstream_factory) -> None:
    # Arrange
    upstream = upstream_factory(chunks=(b'{"data":[]}',))
    backend = InferenceBackend(InferenceUpstreamPool.from_urls(upstream.url))

    # Act
    reply = await backend.relay("GET", "/v1/models/props", body=None, headers={})
    content = b"".join([chunk async for chunk in reply.body])

    # Assert
    assert (
        reply.status_code,
        content,
        reply.feedback_headers,
    ) == (
        200,
        b'{"data":[]}',
        {
            "x-scitex-admission-mode": "control-plane-bypass",
            "x-scitex-cache-residency": "unknown",
            "x-scitex-session-key": "none",
        },
    )


@pytest.mark.asyncio
async def test_explicit_session_has_hashed_feedback(upstream_factory) -> None:
    # Arrange
    upstream = upstream_factory()
    backend = InferenceBackend(InferenceUpstreamPool.from_urls(upstream.url))
    raw = "sac:scholar:durable-session"
    headers = {"X-SciTeX-Session-ID": raw}

    # Act
    reply = await backend.relay(
        "POST",
        "/v1/responses",
        body=b'{"model":"m","input":"hello"}',
        headers=headers,
    )
    await reply.body.aclose()
    feedback = reply.feedback_headers

    # Assert
    assert (feedback["x-scitex-session-key"], raw in str(feedback)) == (
        request_session_key(headers)[:12],
        False,
    )


@pytest.mark.asyncio
async def test_streaming_response_propagates_admission_feedback(
    upstream_factory,
) -> None:
    # Arrange
    upstream = upstream_factory(chunks=(b'{"id":"response-1"}',))
    backend = InferenceBackend(InferenceUpstreamPool.from_urls(upstream.url))
    app = create_app(backend, api_key="relay-secret")
    transport = httpx.ASGITransport(app=app)

    # Act
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://gateway.test"
        ) as client:
            response = await client.post(
                "/v1/responses",
                json={"model": "m", "input": "hello"},
                headers={"Authorization": "Bearer relay-secret"},
            )

    # Assert
    assert (
        response.headers["x-scitex-admission-mode"],
        response.headers["x-scitex-cache-residency"],
        len(response.headers["x-scitex-session-key"]),
    ) == ("observe-only", "unknown", 12)


@pytest.mark.asyncio
async def test_metadata_does_not_enter_cache_or_generation_admission_snapshot(
    upstream_factory,
) -> None:
    # Arrange
    upstream = upstream_factory()
    backend = InferenceBackend(InferenceUpstreamPool.from_urls(upstream.url))
    app = create_app(backend, api_key="relay-secret")
    transport = httpx.ASGITransport(app=app)

    # Act
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://gateway.test"
        ) as client:
            await client.get(
                "/v1/models/props",
                headers={"Authorization": "Bearer relay-secret"},
            )
            health = await client.get("/health")
    snapshot = health.json()["cache_admission"]

    # Assert
    assert (
        snapshot["mode"],
        snapshot["observed"],
        snapshot["admitted"],
        snapshot["admitted_total_by_residency"],
        snapshot["queued"],
    ) == (
        "observe-only",
        {"hot": 0, "cold": 0, "unknown": 0},
        0,
        {"hot": 0, "cold": 0, "unknown": 0},
        0,
    )


@pytest.mark.asyncio
async def test_non_model_get_remains_in_generation_admission(upstream_factory) -> None:
    # Arrange: /v1/{path} is a GET catch-all. Keep the exemption explicit so a
    # future compute-bearing GET cannot silently bypass generation admission.
    upstream = upstream_factory()
    backend = InferenceBackend(InferenceUpstreamPool.from_urls(upstream.url))

    # Act
    reply = await backend.relay("GET", "/v1/batches/work", body=None, headers={})
    _ = b"".join([chunk async for chunk in reply.body])

    # Assert
    assert (
        reply.feedback_headers["x-scitex-admission-mode"],
        "x-scitex-request-label" in reply.feedback_headers,
        backend.cache_admission.snapshot()["observed"],
    ) == (
        "observe-only",
        True,
        {"hot": 0, "cold": 0, "unknown": 1},
    )


@pytest.mark.parametrize(
    ("method", "path", "expected"),
    [
        ("GET", "/v1/models", True),
        ("HEAD", "/v1/models/", True),
        ("GET", "/v1/models/props?refresh=false", True),
        ("POST", "/v1/models", False),
        ("GET", "/v1/batches/work", False),
        ("HEAD", "/v1/responses", False),
    ],
)
def test_only_model_discovery_is_read_only_control_plane(
    method: str, path: str, expected: bool
) -> None:
    # Arrange
    route = (method, path)

    # Act
    actual = is_read_only_control_route(*route)

    # Assert
    assert actual is expected


def _body_with_estimated_tokens(tokens: int) -> bytes:
    prefix = b'{"model":"m","messages":[{"role":"user","content":"'
    suffix = b'"}]}'
    target_bytes = tokens * 4
    return prefix + (b"x" * (target_bytes - len(prefix) - len(suffix))) + suffix


@pytest.mark.asyncio
async def test_reproduces_stale_zero_token_ticket_blocking_sticky_224793_turn() -> None:
    # Arrange: reproduce the operator snapshot independently of HTTP cleanup.
    pool = InferenceUpstreamPool.from_urls(
        ["http://qwen-tp1-256k:1", "http://qwen-tp2:2"],
        capacity_per_upstream=8,
        cold_prefill_limit_per_upstream=1,
        cold_prefill_min_tokens=128_000,
    )
    pool.upstreams[0].token_capacity = 250_000
    pool.upstreams[1].token_capacity = 1_600_000
    pinned = await pool.acquire("cards", input_tokens=1, predicted_uncached_tokens=1)
    await pool.release(pinned, input_tokens=1, session_id="cards")
    stale_get = await pool.acquire(
        "",
        input_tokens=0,
        cold_prefill=False,
        selected_alias=pool.upstreams[0].alias,
    )
    continuation = asyncio.create_task(
        pool.acquire(
            "cards",
            input_tokens=224_793,
            predicted_uncached_tokens=224_793,
        )
    )
    await _wait_for_pool_queue(pool, 1)

    # Act
    snapshot = await pool.observability_snapshot()
    queued = next(
        ticket for ticket in snapshot["tickets"] if ticket["state"] == "queued"
    )
    admitted = next(
        ticket for ticket in snapshot["tickets"] if ticket["state"] == "admitted"
    )

    # Assert: stickiness, not member capacity, keeps the turn behind the stale
    # zero-token hot ticket while the 1.6m-token second member is completely free.
    assert (
        admitted["input_tokens"],
        queued["input_tokens"],
        queued["cold_prefill"],
        queued["block_reason"],
        pool.upstreams[1].in_flight,
        continuation.done(),
    ) == (0, 224_793, True, "hot-work-in-flight", 0, False)

    await pool.release(stale_get, input_tokens=0, session_id="")
    recovered = await asyncio.wait_for(continuation, 0.5)
    await pool.release(
        recovered, input_tokens=224_793, session_id="cards", cold_prefill=True
    )


@pytest.mark.asyncio
async def test_hung_metadata_cannot_block_sticky_224793_token_continuation(
    upstream_factory,
) -> None:
    # Arrange: this is the production incident shape. A discovery GET hangs on
    # the 256k member while a session already pinned there sends a cold turn.
    metadata_release = threading.Event()
    tp1 = upstream_factory(
        block_until=metadata_release,
        block_path="/v1/models",
    )
    tp2 = upstream_factory()
    pool = InferenceUpstreamPool.from_urls(
        [tp1.url, tp2.url],
        capacity_per_upstream=8,
        cold_prefill_limit_per_upstream=1,
        cold_prefill_min_tokens=128_000,
    )
    pool.upstreams[0].token_capacity = 250_000
    pool.upstreams[1].token_capacity = 1_600_000
    backend = InferenceBackend(pool, metadata_timeout_s=1.0)
    headers = {"X-SciTeX-Session-ID": "cards-session"}
    first = await backend.relay(
        "POST",
        "/v1/chat/completions",
        body=b'{"model":"m","messages":[{"role":"user","content":"pin"}]}',
        headers=headers,
    )
    _ = b"".join([chunk async for chunk in first.body])
    tp1.request_started.clear()
    metadata = asyncio.create_task(
        backend.relay("GET", "/v1/models", body=None, headers={})
    )
    metadata_started = await asyncio.to_thread(tp1.request_started.wait, 1.0)
    body = _body_with_estimated_tokens(224_793)

    # Act
    continuation = await asyncio.wait_for(
        backend.relay("POST", "/v1/chat/completions", body=body, headers=headers),
        timeout=1.0,
    )
    _ = b"".join([chunk async for chunk in continuation.body])
    snapshot = await pool.observability_snapshot()

    # Assert: metadata owns no ticket, the sticky request used tp1, and tp2
    # remained free instead of being needed to hide a leaked admission slot.
    assert (
        len(body),
        snapshot["admitted"],
        snapshot["queued"],
        pool.upstreams[1].in_flight,
        metadata_started,
        metadata.done(),
        tp1.requests[-1]["path"],
    ) == (224_793 * 4, 0, 0, 0, True, False, "/v1/chat/completions")

    metadata_release.set()
    metadata_reply = await asyncio.wait_for(metadata, timeout=1.0)
    _ = b"".join([chunk async for chunk in metadata_reply.body])


@pytest.mark.asyncio
async def test_metadata_absolute_deadline_never_consumes_generation_slot(
    upstream_factory,
) -> None:
    # Arrange
    metadata_release = threading.Event()
    upstream = upstream_factory(
        block_until=metadata_release,
        block_path="/v1/models",
    )
    pool = InferenceUpstreamPool.from_urls(upstream.url)
    backend = InferenceBackend(pool, timeout_s=60.0, metadata_timeout_s=0.02)
    started = time.monotonic()

    # Act
    error = None
    try:
        await backend.relay("GET", "/v1/models", body=None, headers={})
    except UpstreamUnreachable as exc:
        error = exc
    elapsed = time.monotonic() - started
    snapshot = await pool.observability_snapshot()

    # Assert: the independent wall-clock deadline wins over the generation
    # timeout and cannot leave an admitted or queued ticket behind.
    assert (
        isinstance(error, UpstreamUnreachable),
        elapsed < 1.0,
        snapshot["admitted"],
        snapshot["queued"],
        pool.status()[0]["in_flight"],
        backend.request_health_snapshot()["active"],
    ) == (True, True, 0, 0, 0, 0)
    metadata_release.set()


@pytest.mark.asyncio
async def test_legacy_non_addressable_ticket_is_reaped_on_caller_deadline(
    upstream_factory,
) -> None:
    # Arrange: before read-only admission was bypassed, the incident's GET
    # followed this non-addressable cleanup path. Model it with a POST route
    # that likewise has no injectable SGLang request id.
    release = threading.Event()
    upstream = upstream_factory(block_until=release)
    pool = InferenceUpstreamPool.from_urls(upstream.url)
    backend = InferenceBackend(pool, continuation_qos_enabled=True)
    request = asyncio.create_task(
        backend.relay(
            "POST",
            "/v1/messages",
            body=b'{"model":"m","messages":[]}',
            headers={},
        )
    )
    request_started = await asyncio.to_thread(upstream.request_started.wait, 1.0)

    # Act: this is the client's 900-second deadline, compressed for the test.
    request.cancel()
    cancelled = None
    try:
        await asyncio.wait_for(request, timeout=0.5)
    except asyncio.CancelledError as exc:
        cancelled = exc
    snapshot = await pool.observability_snapshot()

    # Assert: cleanup is bounded and no stale zero-token admission survives.
    assert (
        request_started,
        isinstance(cancelled, asyncio.CancelledError),
        snapshot["admitted"],
        snapshot["queued"],
        pool.status()[0]["in_flight"],
        backend.request_health_snapshot()["active"],
    ) == (True, True, 0, 0, 0, 0)
    release.set()


@pytest.mark.asyncio
async def test_actual_partial_cache_report_drives_next_relay_admission(
    upstream_factory,
) -> None:
    # Arrange
    report = {
        "usage": {
            "prompt_tokens": 243_434,
            "prompt_tokens_details": {"cached_tokens": 103_296},
        }
    }
    upstream = upstream_factory(chunks=(json.dumps(report).encode(),))
    pool = InferenceUpstreamPool.from_urls(
        upstream.url,
        capacity_per_upstream=2,
        cold_prefill_limit_per_upstream=1,
        cold_prefill_min_tokens=128_000,
    )
    backend = InferenceBackend(pool)
    body = b'{"model":"m","messages":[{"role":"user","content":"same"}]}'
    headers = {"X-SciTeX-Session-ID": "session"}
    first = await backend.relay(
        "POST", "/v1/chat/completions", body=body, headers=headers
    )
    _ = b"".join([chunk async for chunk in first.body])

    # Act
    second = await backend.relay(
        "POST", "/v1/chat/completions", body=body, headers=headers
    )
    snapshot = await pool.observability_snapshot()
    ticket = snapshot["tickets"][0]

    # Assert
    assert (
        second.feedback_headers["x-scitex-cache-residency"],
        ticket["predicted_uncached_tokens"],
        ticket["cold_prefill"],
    ) == ("cold", 140_138, True)
    _ = b"".join([chunk async for chunk in second.body])


async def _wait_for_pool_queue(pool: InferenceUpstreamPool, size: int) -> None:
    for _ in range(100):
        if pool.status()[0]["queued"] == size:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"pool queue did not reach {size}")


@pytest.mark.asyncio
async def test_predicted_hot_and_hot_prefills_coexist() -> None:
    # Arrange
    pool = InferenceUpstreamPool.from_urls(
        "http://only:1",
        capacity_per_upstream=2,
        cold_prefill_limit_per_upstream=1,
        cold_prefill_min_tokens=128_000,
    )

    # Act
    first = await pool.acquire(
        "hot-1",
        input_tokens=640_000,
        predicted_uncached_tokens=0,
        cache_classification="hot",
    )
    second = await pool.acquire(
        "hot-2",
        input_tokens=640_000,
        predicted_uncached_tokens=4_000,
        cache_classification="hot",
    )

    # Assert
    assert (first.in_flight, first.cold_prefills_in_flight) == (2, 0)
    await pool.release(first, input_tokens=640_000, session_id="hot-1")
    await pool.release(second, input_tokens=640_000, session_id="hot-2")


@pytest.mark.asyncio
async def test_predicted_hot_and_large_uncached_prefill_coexist() -> None:
    # Arrange
    pool = InferenceUpstreamPool.from_urls(
        "http://only:1",
        capacity_per_upstream=2,
        cold_prefill_limit_per_upstream=1,
        cold_prefill_min_tokens=128_000,
    )

    # Act
    cold = await pool.acquire(
        "cold",
        input_tokens=243_434,
        predicted_uncached_tokens=140_138,
        cache_classification="cold",
    )
    hot = await pool.acquire(
        "hot",
        input_tokens=640_000,
        predicted_uncached_tokens=0,
        cache_classification="hot",
    )

    # Assert
    assert (cold.in_flight, cold.cold_prefills_in_flight) == (2, 1)
    await pool.release(cold, input_tokens=243_434, session_id="cold")
    await pool.release(hot, input_tokens=640_000, session_id="hot")


@pytest.mark.asyncio
async def test_large_uncached_prefills_serialize_and_unknown_is_conservative() -> None:
    # Arrange
    pool = InferenceUpstreamPool.from_urls(
        "http://only:1",
        capacity_per_upstream=3,
        cold_prefill_limit_per_upstream=1,
        cold_prefill_min_tokens=128_000,
    )
    first = await pool.acquire(
        "cold",
        input_tokens=677_000,
        predicted_uncached_tokens=677_000,
        cache_classification="cold",
    )
    unknown = asyncio.create_task(
        pool.acquire(
            "unknown",
            input_tokens=243_434,
            predicted_uncached_tokens=243_434,
            cache_classification="unknown",
        )
    )
    await _wait_for_pool_queue(pool, 1)

    # Act
    hot = await pool.acquire(
        "hot",
        input_tokens=640_000,
        predicted_uncached_tokens=0,
        cache_classification="hot",
    )
    blocked = unknown.done() is False
    await pool.release(hot, input_tokens=640_000, session_id="hot")
    await pool.release(first, input_tokens=677_000, session_id="cold")
    admitted = await asyncio.wait_for(unknown, 1)
    await pool.release(admitted, input_tokens=243_434, session_id="unknown")

    # Assert
    assert blocked is True


@pytest.mark.asyncio
async def test_cancelled_uncached_waiter_releases_its_queue_ownership() -> None:
    # Arrange
    pool = InferenceUpstreamPool.from_urls(
        "http://only:1",
        capacity_per_upstream=2,
        cold_prefill_limit_per_upstream=1,
        cold_prefill_min_tokens=100,
    )
    first = await pool.acquire("first", input_tokens=200, predicted_uncached_tokens=200)
    waiting = asyncio.create_task(
        pool.acquire("cancelled", input_tokens=200, predicted_uncached_tokens=200)
    )
    await _wait_for_pool_queue(pool, 1)

    # Act
    waiting.cancel()
    await asyncio.gather(waiting, return_exceptions=True)

    queued = pool.status()[0]["queued"]
    await pool.release(first, input_tokens=200, session_id="first")

    # Assert
    assert queued == 0


@pytest.mark.asyncio
async def test_finished_upstream_releases_when_slow_client_closes_stream() -> None:
    # Arrange
    pool = InferenceUpstreamPool.from_urls(
        "http://only:1",
        cold_prefill_limit_per_upstream=1,
        cold_prefill_min_tokens=100,
    )
    member = await pool.acquire(
        "session", input_tokens=200, predicted_uncached_tokens=200
    )
    backend = InferenceBackend(pool)

    async def finished_stream():
        if False:
            yield b""

    class FinishedResponse:
        async def aclose(self) -> None:
            return None

    class ClosingClient:
        async def aclose(self) -> None:
            return None

    relayed = backend._drain(
        ClosingClient(),
        FinishedResponse(),
        member,
        stream=finished_stream(),
        first_chunk=b"already-buffered",
        input_tokens=200,
        session_id="session",
        cold_prefill=True,
    )
    # Act
    first_chunk = await anext(relayed)
    held_while_client_is_slow = member.in_flight
    await relayed.aclose()

    # Assert
    assert (
        first_chunk,
        held_while_client_is_slow,
        member.in_flight,
        member.cold_prefills_in_flight,
    ) == (
        b"already-buffered",
        1,
        0,
        0,
    )


@pytest.mark.asyncio
async def test_aged_uncached_work_waits_for_all_non_cold_work_to_finish() -> None:
    # Arrange
    pool = InferenceUpstreamPool.from_urls(
        "http://only:1",
        capacity_per_upstream=2,
        priority_aging_s=0,
        cold_prefill_limit_per_upstream=1,
        cold_prefill_min_tokens=100,
    )
    cold = await pool.acquire("cold", input_tokens=200, predicted_uncached_tokens=200)
    blocker = await pool.acquire("blocker", input_tokens=1, predicted_uncached_tokens=1)
    aged = asyncio.create_task(
        pool.acquire("aged", input_tokens=200, predicted_uncached_tokens=200)
    )
    await _wait_for_pool_queue(pool, 1)
    priority = asyncio.create_task(
        pool.acquire(
            "priority", input_tokens=1, predicted_uncached_tokens=1, priority=True
        )
    )
    await _wait_for_pool_queue(pool, 2)

    # Act
    await pool.release(cold, input_tokens=200, session_id="cold")
    next_priority = await asyncio.wait_for(priority, 1)
    cold_waited = aged.done() is False
    await pool.release(blocker, input_tokens=1, session_id="blocker")
    await pool.release(next_priority, input_tokens=1, session_id="priority")
    next_member = await asyncio.wait_for(aged, 1)
    await pool.release(next_member, input_tokens=200, session_id="aged")

    # Assert
    assert cold_waited is True


@pytest.mark.asyncio
async def test_queued_hot_prediction_expires_to_unknown_full_prefill() -> None:
    # Arrange
    pool = InferenceUpstreamPool.from_urls(
        "http://only:1",
        capacity_per_upstream=2,
        cache_prediction_max_age_s=0,
        cold_prefill_limit_per_upstream=1,
        cold_prefill_min_tokens=128_000,
    )
    cold = await pool.acquire(
        "cold", input_tokens=500_000, predicted_uncached_tokens=500_000
    )
    blocker = await pool.acquire("blocker", input_tokens=1, predicted_uncached_tokens=1)
    stale_hot = asyncio.create_task(
        pool.acquire(
            "stale-hot",
            input_tokens=642_616,
            predicted_uncached_tokens=0,
            cache_classification="hot",
            cache_priority=True,
        )
    )
    await _wait_for_pool_queue(pool, 1)

    # Act
    await pool.release(blocker, input_tokens=1, session_id="blocker")
    await asyncio.sleep(0)
    snapshot = await pool.observability_snapshot()

    queued = next(
        ticket for ticket in snapshot["tickets"] if ticket["state"] == "queued"
    )
    observed = (
        stale_hot.done(),
        queued["cache_classification"],
        queued["priority"],
        queued["cache_priority"],
        queued["predicted_uncached_tokens"],
        queued["cold_prefill"],
    )
    stale_hot.cancel()
    await asyncio.gather(stale_hot, return_exceptions=True)
    await pool.release(cold, input_tokens=500_000, session_id="cold")

    # Assert
    assert observed == (False, "unknown", False, False, 642_616, True)


@pytest.mark.asyncio
async def test_single_large_cold_prefill_drains_then_makes_progress() -> None:
    # Arrange
    pool = InferenceUpstreamPool.from_urls(
        "http://only:1",
        capacity_per_upstream=3,
        token_capacity_per_upstream=1_600_000,
        cold_prefill_limit_per_upstream=1,
        cold_prefill_min_tokens=128_000,
    )
    hot = await pool.acquire(
        "hot", input_tokens=144_000, predicted_uncached_tokens=1_000
    )
    large = asyncio.create_task(
        pool.acquire(
            "large-cold",
            input_tokens=1_590_000,
            predicted_uncached_tokens=1_590_000,
        )
    )
    await _wait_for_pool_queue(pool, 1)

    # Act
    backfill = await pool.acquire(
        "backfill", input_tokens=144_000, predicted_uncached_tokens=1_000
    )
    await pool.release(backfill, input_tokens=144_000, session_id="backfill")
    remained_queued = large.done() is False
    await pool.release(hot, input_tokens=144_000, session_id="hot")
    admitted = await asyncio.wait_for(large, 1)
    await pool.release(admitted, input_tokens=1_590_000, session_id="large-cold")

    # Assert
    assert (remained_queued, pool.status()[0]["in_flight"]) == (True, 0)
