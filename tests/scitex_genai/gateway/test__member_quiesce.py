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
    pool = InferenceUpstreamPool.from_urls(
        "http://only:1", capacity_per_upstream=1, max_queue_size=4
    )
    first = await pool.acquire("first")
    waiter = asyncio.create_task(pool.acquire("waiter"))
    await _wait_until(lambda: pool.upstreams[0].queued == 1)

    barrier = asyncio.create_task(pool.begin_member_quiesce("http://only:1", 1))
    await _wait_until(lambda: pool.upstreams[0].quiesced)
    await pool.release(first, session_id="first")
    admitted = await asyncio.wait_for(waiter, 0.2)
    assert barrier.done() is False
    await pool.release(admitted, session_id="waiter")

    state = await asyncio.wait_for(barrier, 0.2)
    assert (state.empty, state.held, state.quiesced) == (True, 0, True)


@pytest.mark.asyncio
async def test_post_cutoff_sticky_work_is_held_without_repin_then_resumed() -> None:
    pool = _heterogeneous_pool()
    assert await pool.route_alias("sticky", input_tokens=500) == "qwen-tp2"
    home = await pool.acquire("sticky", input_tokens=500, selected_alias="qwen-tp2")
    await pool.release(home, input_tokens=500, session_id="sticky")
    await pool.begin_member_quiesce("qwen-tp2", 0.2)

    held = asyncio.create_task(pool.acquire("sticky", input_tokens=50))
    await _wait_until(lambda: pool.upstreams[1].held == 1)
    state = await pool.member_state("qwen-tp2")
    assert (held.done(), state.empty, state.held, pool._sessions["sticky"]) == (
        False,
        True,
        1,
        "qwen-tp2",
    )

    await pool._resume_member_after_validation("qwen-tp2")
    admitted = await asyncio.wait_for(held, 0.2)
    assert admitted.alias == "qwen-tp2"
    await pool.release(admitted, input_tokens=50, session_id="sticky")


@pytest.mark.asyncio
async def test_only_capable_request_is_held_for_quiesced_tp2() -> None:
    pool = _heterogeneous_pool()
    await pool.begin_member_quiesce("qwen-tp2", 0.2)

    large = asyncio.create_task(pool.acquire("large", input_tokens=500))
    await _wait_until(lambda: pool.upstreams[1].held == 1)
    assert (large.done(), pool._sessions["large"]) == (False, "qwen-tp2")
    large.cancel()
    with pytest.raises(asyncio.CancelledError):
        await large


@pytest.mark.asyncio
async def test_unpinned_capable_request_routes_around_quiesced_member() -> None:
    pool = _heterogeneous_pool()
    await pool.begin_member_quiesce("qwen-tp2", 0.2)
    selected = await asyncio.wait_for(pool.acquire("small", input_tokens=50), 0.2)
    assert selected.alias == "qwen-tp1"
    await pool.release(selected, input_tokens=50, session_id="small")


@pytest.mark.asyncio
async def test_tp1_queue_progress_is_unaffected_by_tp2_quiesce() -> None:
    pool = _heterogeneous_pool()
    first = await pool.acquire("tp1-first", input_tokens=50, selected_alias="qwen-tp1")
    waiter = asyncio.create_task(
        pool.acquire("tp1-waiter", input_tokens=50, selected_alias="qwen-tp1")
    )
    await _wait_until(lambda: pool.upstreams[0].queued == 1)
    await pool.begin_member_quiesce("qwen-tp2", 0.2)
    await pool.release(first, input_tokens=50, session_id="tp1-first")
    admitted = await asyncio.wait_for(waiter, 0.2)
    assert admitted.alias == "qwen-tp1"
    await pool.release(admitted, input_tokens=50, session_id="tp1-waiter")


@pytest.mark.asyncio
async def test_cutoff_is_atomic_and_barrier_ignores_later_held_work() -> None:
    pool = InferenceUpstreamPool.from_urls("http://only:1")
    owned = await pool.acquire("before")
    barrier = asyncio.create_task(pool.begin_member_quiesce("http://only:1", 1))
    await _wait_until(lambda: pool.upstreams[0].quiesced)
    later = asyncio.create_task(pool.acquire("after"))
    await _wait_until(lambda: pool.upstreams[0].held == 1)
    await pool.release(owned, session_id="before")

    state = await asyncio.wait_for(barrier, 0.2)
    assert (state.in_flight, state.queued, state.held, later.done()) == (0, 0, 1, False)
    later.cancel()
    with pytest.raises(asyncio.CancelledError):
        await later


@pytest.mark.asyncio
async def test_quiesce_timeout_preserves_fence_and_owned_work() -> None:
    async def scheduler_probe(url: str, timeout_s: float):
        return SGLangSchedulerObservation("1" * 32, 0, 0, 0.0)

    pool = InferenceUpstreamPool.from_specs(
        (SimpleNamespace(label="only", url="http://only:1", token_capacity=1_000),)
    )
    owned = await pool.acquire("owned")
    backend = InferenceBackend(pool, scheduler_probe=scheduler_probe)
    app = create_app(backend, api_key="secret")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://gateway"
    ) as client:
        response = await client.post(
            "/admin/members/only/quiesce?timeout_s=0.001",
            headers={"x-api-key": "secret"},
        )
    state = await pool.member_state("only")
    assert (response.status_code, response.json()["in_flight"], state.quiesced) == (
        409,
        1,
        True,
    )
    await pool.release(owned, session_id="owned")


@pytest.mark.asyncio
async def test_disconnected_held_ticket_cleans_all_counters() -> None:
    pool = InferenceUpstreamPool.from_urls("http://only:1")
    await pool.begin_member_quiesce("http://only:1", 0.2)
    backend = InferenceBackend(pool)
    disconnected = asyncio.Event()

    async def client_disconnected() -> bool:
        return disconnected.is_set()

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
    with pytest.raises(_ClientDisconnected):
        await held
    snapshot = await pool.observability_snapshot()
    assert (
        pool.upstreams[0].held,
        pool.upstreams[0].input_tokens_held,
        snapshot["held"],
        snapshot["input_tokens_held"],
    ) == (0, 0, 0, 0)


@pytest.mark.asyncio
async def test_global_drain_still_wakes_held_work_and_reaches_zero() -> None:
    pool = InferenceUpstreamPool.from_urls("http://only:1")
    await pool.begin_member_quiesce("http://only:1", 0.2)
    held = asyncio.create_task(pool.acquire("held"))
    await _wait_until(lambda: pool.upstreams[0].held == 1)

    drained = await asyncio.wait_for(pool.begin_drain(0.2), 0.2)
    error = None
    try:
        await held
    except Exception as exc:  # expected gateway shutdown refusal
        error = exc
    assert (drained.empty, type(error), pool.upstreams[0].held) == (
        True,
        InferenceAdmissionError,
        0,
    )


@pytest.mark.asyncio
async def test_admin_member_quiesce_requires_auth_and_valid_alias() -> None:
    pool = InferenceUpstreamPool.from_specs(
        (SimpleNamespace(label="only", url="http://only:1", token_capacity=1_000),)
    )
    backend = InferenceBackend(pool)
    app = create_app(backend, api_key="secret")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://gateway"
    ) as client:
        denied = await client.post("/admin/members/only/quiesce")
        missing = await client.post(
            "/admin/members/missing/quiesce", headers={"x-api-key": "secret"}
        )
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

    with pytest.raises(InferenceMemberResumeError, match="fresh health validation"):
        await backend.resume_member("qwen-tp2")
    with pytest.raises(
        InferenceMemberResumeError, match="pre-quiesce engine generation"
    ):
        await backend.resume_member("qwen-tp2")
    assert (held.done(), pool.upstreams[0].quiesced) == (False, True)
    await backend.resume_member("qwen-tp2")
    admitted = await asyncio.wait_for(held, 0.2)
    after_cutover = backend.admission_predictions.predict(
        session_id="sticky",
        upstream="qwen-tp2",
        engine_generation=backend._engine_generations["qwen-tp2"],
        body=body,
        estimated_input_tokens=10,
    )
    assert (
        during_cutover.evidence,
        after_cutover.evidence,
        after_cutover.engine_generation != old_generation,
    ) == ("no-compatible-history", "no-compatible-history", True)
    await pool.release(admitted, input_tokens=10, session_id="sticky")
