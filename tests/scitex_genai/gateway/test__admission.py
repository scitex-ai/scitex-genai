from __future__ import annotations

import asyncio

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
    assert (snapshot["mode"], snapshot["running"], snapshot["queued"]) == (
        "observe-only",
        0,
        0,
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
    assert (reply.status_code, content, reply.feedback_headers) == (
        200,
        b'{"data":[]}',
        {
            "x-scitex-admission-mode": "observe-only",
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
async def test_streaming_response_propagates_admission_feedback(upstream_factory) -> None:
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
async def test_health_exposes_incremented_observe_only_snapshot(upstream_factory) -> None:
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
        snapshot["running"],
        snapshot["queued"],
    ) == ("observe-only", {"hot": 0, "cold": 0, "unknown": 1}, 0, 0)
