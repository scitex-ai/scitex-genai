from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from scitex_genai.gateway._admission import AdmissionController, CacheResidency
from scitex_genai.gateway._inference import (
    InferenceBackend,
    InferenceUpstreamPool,
    request_session_key,
)
from scitex_genai.gateway._server import create_app


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
        len(reply.feedback_headers["x-scitex-request-label"]),
    ) == (
        200,
        b'{"data":[]}',
        {
            "x-scitex-admission-mode": "observe-only",
            "x-scitex-cache-residency": "unknown",
            "x-scitex-session-key": "none",
            "x-scitex-request-label": reply.feedback_headers["x-scitex-request-label"],
            "x-scitex-agent-label": "unknown",
            "x-scitex-session-label": "anonymous",
        },
        16,
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
async def test_health_exposes_incremented_observe_only_snapshot(
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
        {"hot": 0, "cold": 0, "unknown": 1},
        0,
        {"hot": 0, "cold": 0, "unknown": 0},
        0,
    )


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
        queued["predicted_uncached_tokens"],
        queued["cold_prefill"],
    )
    stale_hot.cancel()
    await asyncio.gather(stale_hot, return_exceptions=True)
    await pool.release(cold, input_tokens=500_000, session_id="cold")

    # Assert
    assert observed == (False, "unknown", 642_616, True)


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
