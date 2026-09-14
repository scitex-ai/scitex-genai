from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("fastapi")

from scitex_genai.gateway._errors import InferenceAdmissionError
from scitex_genai.gateway._health import UpstreamReachability
from scitex_genai.gateway._inference import (
    InferenceBackend,
    InferenceMemberResumeError,
    InferenceUpstreamPool,
    _ClientDisconnected,
)
from scitex_genai.gateway._server import create_app
from scitex_genai.gateway._sglang_metrics import SGLangSchedulerObservation


async def _wait_until(predicate) -> None:
    for _ in range(1_000):
        if predicate():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("condition did not become true")


async def _raised_async(awaitable) -> BaseException | None:
    try:
        await awaitable
    except BaseException as exc:
        return exc
    return None


def _heterogeneous_pool(*, capacity: int = 1) -> InferenceUpstreamPool:
    return InferenceUpstreamPool.from_specs(
        (
            SimpleNamespace(label="qwen-tp1", url="http://tp1:1", token_capacity=100),
            SimpleNamespace(label="qwen-tp2", url="http://tp2:2", token_capacity=1_000),
        ),
        capacity_per_upstream=capacity,
        max_queue_size=8,
    )


@pytest.mark.asyncio
async def test_pre_cutoff_waiter_finishes_before_member_quiesce_barrier() -> None:
    # Arrange
    pool = InferenceUpstreamPool.from_urls(
        "http://only:1", capacity_per_upstream=1, max_queue_size=4
    )
    first = await pool.acquire("first")
    waiter = asyncio.create_task(pool.acquire("waiter"))
    await _wait_until(lambda: pool.upstreams[0].queued == 1)

    # Act
    barrier = asyncio.create_task(pool.begin_member_quiesce("http://only:1", 1))
    await _wait_until(lambda: pool.upstreams[0].quiesced)
    await pool.release(first, session_id="first")
    admitted = await asyncio.wait_for(waiter, 0.2)
    barrier_waiting = not barrier.done()
    await pool.release(admitted, session_id="waiter")
    state = await asyncio.wait_for(barrier, 0.2)

    # Assert
    assert (barrier_waiting, state.empty, state.held, state.quiesced) == (
        True,
        True,
        0,
        True,
    )


@pytest.mark.asyncio
async def test_post_cutoff_sticky_work_is_held_without_repin_then_resumed() -> None:
    # Arrange
    pool = _heterogeneous_pool()
    home_alias = await pool.route_alias("sticky", input_tokens=500)
    home = await pool.acquire("sticky", input_tokens=500, selected_alias="qwen-tp2")
    await pool.release(home, input_tokens=500, session_id="sticky")
    await pool.begin_member_quiesce("qwen-tp2", 0.2)

    # Act
    held = asyncio.create_task(pool.acquire("sticky", input_tokens=50))
    await _wait_until(lambda: pool.upstreams[1].held == 1)
    state = await pool.member_state("qwen-tp2")
    held_before_resume = held.done()
    pinned_alias = pool._sessions["sticky"]
    await pool._resume_member_after_validation("qwen-tp2")
    admitted = await asyncio.wait_for(held, 0.2)
    await pool.release(admitted, input_tokens=50, session_id="sticky")

    # Assert
    assert (
        home_alias,
        held_before_resume,
        state.empty,
        state.held,
        pinned_alias,
        admitted.alias,
    ) == (
        "qwen-tp2",
        False,
        True,
        1,
        "qwen-tp2",
        "qwen-tp2",
    )


@pytest.mark.asyncio
async def test_only_capable_request_is_held_for_quiesced_tp2() -> None:
    # Arrange
    pool = _heterogeneous_pool()
    await pool.begin_member_quiesce("qwen-tp2", 0.2)

    # Act
    large = asyncio.create_task(pool.acquire("large", input_tokens=500))
    await _wait_until(lambda: pool.upstreams[1].held == 1)
    was_held = not large.done()
    pinned_alias = pool._sessions["large"]
    large.cancel()
    cancelled = await _raised_async(large)

    # Assert
    assert (was_held, pinned_alias, type(cancelled)) == (
        True,
        "qwen-tp2",
        asyncio.CancelledError,
    )


@pytest.mark.asyncio
async def test_unpinned_capable_request_routes_around_quiesced_member() -> None:
    # Arrange
    pool = _heterogeneous_pool()
    await pool.begin_member_quiesce("qwen-tp2", 0.2)

    # Act
    selected = await asyncio.wait_for(pool.acquire("small", input_tokens=50), 0.2)
    await pool.release(selected, input_tokens=50, session_id="small")

    # Assert
    assert selected.alias == "qwen-tp1"


@pytest.mark.asyncio
async def test_tp1_queue_progress_is_unaffected_by_tp2_quiesce() -> None:
    # Arrange
    pool = _heterogeneous_pool()
    first = await pool.acquire("tp1-first", input_tokens=50, selected_alias="qwen-tp1")
    waiter = asyncio.create_task(
        pool.acquire("tp1-waiter", input_tokens=50, selected_alias="qwen-tp1")
    )
    await _wait_until(lambda: pool.upstreams[0].queued == 1)

    # Act
    await pool.begin_member_quiesce("qwen-tp2", 0.2)
    await pool.release(first, input_tokens=50, session_id="tp1-first")
    admitted = await asyncio.wait_for(waiter, 0.2)
    await pool.release(admitted, input_tokens=50, session_id="tp1-waiter")

    # Assert
    assert admitted.alias == "qwen-tp1"


@pytest.mark.asyncio
async def test_cutoff_is_atomic_and_barrier_ignores_later_held_work() -> None:
    # Arrange
    pool = InferenceUpstreamPool.from_urls("http://only:1")
    owned = await pool.acquire("before")

    # Act
    barrier = asyncio.create_task(pool.begin_member_quiesce("http://only:1", 1))
    await _wait_until(lambda: pool.upstreams[0].quiesced)
    later = asyncio.create_task(pool.acquire("after"))
    await _wait_until(lambda: pool.upstreams[0].held == 1)
    await pool.release(owned, session_id="before")

    state = await asyncio.wait_for(barrier, 0.2)
    later_was_held = not later.done()
    later.cancel()
    cancelled = await _raised_async(later)

    # Assert
    assert (
        state.in_flight,
        state.queued,
        state.held,
        later_was_held,
        type(cancelled),
    ) == (0, 0, 1, True, asyncio.CancelledError)


@pytest.mark.asyncio
async def test_quiesce_timeout_preserves_fence_and_owned_work() -> None:
    # Arrange
    async def scheduler_probe(url: str, timeout_s: float):
        return SGLangSchedulerObservation("1" * 32, 0, 0, 0.0)

    pool = InferenceUpstreamPool.from_specs(
        (SimpleNamespace(label="only", url="http://only:1", token_capacity=1_000),)
    )
    owned = await pool.acquire("owned")
    backend = InferenceBackend(pool, scheduler_probe=scheduler_probe)
    app = create_app(backend, api_key="secret")
    transport = httpx.ASGITransport(app=app)

    # Act
    async with httpx.AsyncClient(
        transport=transport, base_url="http://gateway"
    ) as client:
        response = await client.post(
            "/admin/members/only/quiesce?timeout_s=0.001",
            headers={"x-api-key": "secret"},
        )
    state = await pool.member_state("only")
    await pool.release(owned, session_id="owned")

    # Assert
    assert (response.status_code, response.json()["in_flight"], state.quiesced) == (
        409,
        1,
        True,
    )


@pytest.mark.asyncio
async def test_disconnected_held_ticket_cleans_all_counters() -> None:
    # Arrange
    pool = InferenceUpstreamPool.from_urls("http://only:1")
    await pool.begin_member_quiesce("http://only:1", 0.2)
    backend = InferenceBackend(pool)
    disconnected = asyncio.Event()

    async def client_disconnected() -> bool:
        return disconnected.is_set()

    # Act
    held = asyncio.create_task(
        backend._acquire_while_connected(
            "gone",
            exclude=set(),
            input_tokens=7,
            priority=False,
            cold_prefill=False,
            admission_class="unclassified",
            cache_classification="unknown",
            predicted_uncached_tokens=7,
            selected_alias="http://only:1",
            client_disconnected=client_disconnected,
        )
    )
    await _wait_until(lambda: pool.upstreams[0].held == 1)
    disconnected.set()
    error = await _raised_async(held)
    snapshot = await pool.observability_snapshot()

    # Assert
    assert (
        type(error),
        pool.upstreams[0].held,
        pool.upstreams[0].input_tokens_held,
        snapshot["held"],
        snapshot["input_tokens_held"],
    ) == (_ClientDisconnected, 0, 0, 0, 0)


@pytest.mark.asyncio
async def test_global_drain_still_wakes_held_work_and_reaches_zero() -> None:
    # Arrange
    pool = InferenceUpstreamPool.from_urls("http://only:1")
    await pool.begin_member_quiesce("http://only:1", 0.2)
    held = asyncio.create_task(pool.acquire("held"))
    await _wait_until(lambda: pool.upstreams[0].held == 1)

    # Act
    drained = await asyncio.wait_for(pool.begin_drain(0.2), 0.2)
    error = await _raised_async(held)

    # Assert
    assert (drained.empty, type(error), pool.upstreams[0].held) == (
        True,
        InferenceAdmissionError,
        0,
    )


@pytest.mark.asyncio
async def test_admin_member_quiesce_requires_auth_and_valid_alias() -> None:
    # Arrange
    pool = InferenceUpstreamPool.from_specs(
        (SimpleNamespace(label="only", url="http://only:1", token_capacity=1_000),)
    )
    backend = InferenceBackend(pool)
    app = create_app(backend, api_key="secret")
    transport = httpx.ASGITransport(app=app)

    # Act
    async with httpx.AsyncClient(
        transport=transport, base_url="http://gateway"
    ) as client:
        denied = await client.post("/admin/members/only/quiesce")
        missing = await client.post(
            "/admin/members/missing/quiesce", headers={"x-api-key": "secret"}
        )

    # Assert
    assert (
        denied.status_code,
        missing.status_code,
        missing.json()["error"]["type"],
    ) == (
        401,
        404,
        "member_not_found",
    )


@pytest.mark.asyncio
async def test_resume_requires_health_and_new_authoritative_generation() -> None:
    # Arrange
    generations = iter(("1" * 32, "1" * 32, "2" * 32))
    health = iter((False, True, True))

    async def scheduler_probe(url: str, timeout_s: float):
        return SGLangSchedulerObservation(next(generations), 0, 0, 0.0)

    async def health_probe(url: str, timeout_s: float):
        reachable = next(health)
        return UpstreamReachability(
            reachable,
            "responded" if reachable else "connection_refused",
            0.1,
            "now",
            http_status=200 if reachable else None,
        )

    pool = InferenceUpstreamPool.from_specs(
        (SimpleNamespace(label="qwen-tp2", url="http://tp2:2", token_capacity=1_000),)
    )
    backend = InferenceBackend(
        pool,
        scheduler_probe=scheduler_probe,
        health_probe=health_probe,
        health_cache_ttl_s=0,
    )
    old_generation = backend.observe_engine_generation(
        upstream="qwen-tp2", engine_generation="1" * 32
    )
    body = b'{"messages":[{"role":"user","content":"hello"}]}'
    prior = backend.admission_predictions.predict(
        session_id="sticky",
        upstream="qwen-tp2",
        engine_generation=old_generation,
        body=body,
        estimated_input_tokens=10,
    )
    backend.admission_predictions.observe(
        prior,
        session_id="sticky",
        upstream="qwen-tp2",
        reported_input_tokens=10,
        cached_tokens=10,
        cache_tier="device",
    )

    # Act
    await backend.quiesce_member("qwen-tp2", 0.2)
    during_cutover = backend.admission_predictions.predict(
        session_id="sticky",
        upstream="qwen-tp2",
        engine_generation=backend._engine_generations["qwen-tp2"],
        body=body,
        estimated_input_tokens=10,
    )
    held = asyncio.create_task(pool.acquire("sticky", input_tokens=10))
    await _wait_until(lambda: pool.upstreams[0].held == 1)

    health_error = await _raised_async(backend.resume_member("qwen-tp2"))
    generation_error = await _raised_async(backend.resume_member("qwen-tp2"))
    held_before_valid_resume = not held.done()
    remained_quiesced = pool.upstreams[0].quiesced
    await backend.resume_member("qwen-tp2")
    admitted = await asyncio.wait_for(held, 0.2)
    after_cutover = backend.admission_predictions.predict(
        session_id="sticky",
        upstream="qwen-tp2",
        engine_generation=backend._engine_generations["qwen-tp2"],
        body=body,
        estimated_input_tokens=10,
    )
    await pool.release(admitted, input_tokens=10, session_id="sticky")

    # Assert
    assert (
        type(health_error),
        "fresh health validation" in str(health_error),
        type(generation_error),
        "pre-quiesce engine generation" in str(generation_error),
        held_before_valid_resume,
        remained_quiesced,
        during_cutover.evidence,
        after_cutover.evidence,
        after_cutover.engine_generation != old_generation,
    ) == (
        InferenceMemberResumeError,
        True,
        InferenceMemberResumeError,
        True,
        True,
        True,
        "no-compatible-history",
        "no-compatible-history",
        True,
    )
