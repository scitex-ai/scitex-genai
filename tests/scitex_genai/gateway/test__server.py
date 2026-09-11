from __future__ import annotations

import asyncio
import json
import signal
import threading
from contextlib import asynccontextmanager, suppress

import httpx
import pytest
import pytest_asyncio

# The gateway app is FastAPI-based (`create_app` resolves fastapi lazily);
# skip cleanly on installs without the [gateway] extra.
pytest.importorskip("fastapi")

from scitex_genai.gateway._errors import InferenceAdmissionError
from scitex_genai.gateway._health import UpstreamReachability
from scitex_genai.gateway._inference import InferenceBackend, InferenceUpstreamPool
from scitex_genai.gateway._server import _build_uvicorn_server, create_app


class _Pool:
    accounts = [object()]


class _Backend:
    pool = _Pool()
    refreshed = 0

    async def refresh_usage(self) -> None:
        self.refreshed += 1

    async def stream(self, payload, *, session_id=""):
        yield {"type": "response.created", "response": {"id": "resp-1"}}
        yield {
            "type": "response.output_item.added",
            "item": {"type": "message", "id": "item-1"},
        }
        yield {"type": "response.output_text.delta", "delta": "Hello"}
        yield {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "id": "item-1",
                "content": [{"type": "output_text", "text": "Hello"}],
            },
        }
        yield {
            "type": "response.completed",
            "response": {"usage": {"input_tokens": 3, "output_tokens": 1}},
        }


def _body(*, stream: bool) -> dict:
    return {
        "model": "gpt-5.4",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": stream,
    }


@pytest.fixture
def app():
    return create_app(_Backend(), api_key="relay-secret")


@pytest_asyncio.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://gateway.test"
        ) as test_client:
            yield test_client


@pytest.mark.asyncio
async def test_messages_rejects_missing_api_key(client) -> None:
    # Arrange
    # Act
    response = await client.post("/v1/messages", json=_body(stream=False))
    # Assert
    assert (response.status_code, response.json()["error"]["type"]) == (
        401,
        "authentication_error",
    )


@pytest.mark.asyncio
async def test_nonstream_messages_returns_anthropic_shape(client) -> None:
    # Arrange
    # Act
    response = await client.post(
        "/v1/messages",
        json=_body(stream=False),
        headers={"x-api-key": "relay-secret", "session_id": "session-a"},
    )
    # Assert
    assert (
        response.status_code,
        response.json()["content"],
        response.json()["usage"],
    ) == (
        200,
        [{"type": "text", "text": "Hello"}],
        {"input_tokens": 3, "output_tokens": 1},
    )


@pytest.mark.asyncio
async def test_stream_messages_returns_anthropic_sse(client) -> None:
    # Arrange
    # Act
    response = await client.post(
        "/v1/messages",
        json=_body(stream=True),
        headers={"Authorization": "Bearer relay-secret"},
    )
    # Assert
    assert (
        response.status_code,
        response.headers["content-type"].startswith("text/event-stream"),
        "event: message_start" in response.text,
        "event: content_block_delta" in response.text,
        "event: message_stop" in response.text,
    ) == (200, True, True, True, True)


@pytest.mark.asyncio
async def test_count_tokens_is_authenticated_and_positive(client) -> None:
    # Arrange
    # Act
    response = await client.post(
        "/v1/messages/count_tokens",
        content=json.dumps(_body(stream=False)),
        headers={"x-api-key": "relay-secret", "content-type": "application/json"},
    )
    # Assert
    assert (response.status_code, response.json()["input_tokens"] > 0) == (200, True)


def test_created_app_is_accepted_by_uvicorn_config(app) -> None:
    """The gateway CLI hands ``create_app``'s result straight to uvicorn."""
    # Arrange
    uvicorn = pytest.importorskip("uvicorn")
    # Act
    config = uvicorn.Config(app)
    # Assert
    assert config.app is app


# ---------------------------------------------------------------- inference
# The same app, an InferenceBackend behind it, and a REAL upstream listener.


def _relay_body() -> dict:
    return {
        "model": "local-model",
        "max_tokens": 8,
        "system": "Top",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Available agent types for the Agent tool"},
        ],
    }


@asynccontextmanager
async def _serving(backend: InferenceBackend):
    app = create_app(backend, api_key="relay-secret")
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://gateway.test"
        ) as test_client:
            yield test_client


@pytest.mark.asyncio
async def test_relay_app_serves_messages_from_the_upstream_pool(
    upstream_factory,
) -> None:
    # Arrange
    reply = b'{"id":"msg_1","type":"message","role":"assistant","content":[]}'
    upstream = upstream_factory(chunks=(reply,))
    backend = InferenceBackend(InferenceUpstreamPool.from_urls(upstream.url))
    # Act
    async with _serving(backend) as test_client:
        response = await test_client.post(
            "/v1/messages?beta=true",
            json=_relay_body(),
            headers={"x-api-key": "relay-secret"},
        )
    forwarded = json.loads(upstream.requests[0]["body"])
    # Assert
    assert (
        response.status_code,
        response.content,
        upstream.requests[0]["path"],
        [message["role"] for message in forwarded["messages"]],
    ) == (200, reply, "/v1/messages?beta=true", ["user"])


@pytest.mark.asyncio
async def test_relay_app_streams_sse_from_the_upstream(upstream_factory) -> None:
    # Arrange
    frames = (
        b'event: message_start\ndata: {"type":"message_start"}\n\n',
        b'event: content_block_delta\ndata: {"delta":{"text":"Hi"}}\n\n',
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    )
    upstream = upstream_factory(content_type="text/event-stream", chunks=frames)
    backend = InferenceBackend(InferenceUpstreamPool.from_urls(upstream.url))
    # Act
    async with _serving(backend) as test_client:
        response = await test_client.post(
            "/v1/messages",
            json={**_relay_body(), "stream": True},
            headers={"x-api-key": "relay-secret"},
        )
    # Assert
    assert (
        response.status_code,
        response.headers["content-type"].startswith("text/event-stream"),
        response.content,
    ) == (200, True, b"".join(frames))


@pytest.mark.asyncio
async def test_relay_app_refuses_with_inference_wording_when_no_upstream_answers(
    dead_url_factory,
) -> None:
    # Arrange
    backend = InferenceBackend(InferenceUpstreamPool.from_urls(dead_url_factory()))
    # Act
    async with _serving(backend) as test_client:
        response = await test_client.post(
            "/v1/messages", json=_relay_body(), headers={"x-api-key": "relay-secret"}
        )
    error = response.json()["error"]
    # Assert
    assert (
        response.status_code,
        error["type"],
        error["message"].startswith("No inference upstream answered"),
        "account" in error["message"].lower(),
    ) == (502, "upstream_unreachable", True, False)


@pytest.mark.asyncio
async def test_relay_app_still_requires_the_api_key(upstream_factory) -> None:
    # Arrange
    upstream = upstream_factory()
    backend = InferenceBackend(InferenceUpstreamPool.from_urls(upstream.url))
    # Act
    async with _serving(backend) as test_client:
        response = await test_client.post("/v1/messages", json=_relay_body())
    # Assert
    assert (response.status_code, len(upstream.requests)) == (401, 0)


@pytest.mark.asyncio
async def test_relay_app_health_names_the_upstreams(upstream_factory) -> None:
    # Arrange
    upstream = upstream_factory()
    backend = InferenceBackend(InferenceUpstreamPool.from_urls(upstream.url))
    # Act
    async with _serving(backend) as test_client:
        response = await test_client.get("/health")
    payload = response.json()
    member = payload["members"][0]
    # Assert
    assert (
        payload["status"],
        payload["health_strategy"],
        payload["upstreams"],
        payload["configured_members"],
        payload["admission_eligible_members"],
        payload["reachable_members"],
        payload["active_members"],
        member["url"],
        member["configured"],
        member["admission_eligible"],
        member["reachable"],
        member["active"],
        member["reachability"]["reason"],
    ) == (
        "ok",
        "local_control_plane",
        [upstream.url],
        1,
        1,
        1,
        1,
        upstream.url,
        True,
        True,
        True,
        True,
        "responded",
    )


@pytest.mark.asyncio
async def test_relay_health_degrades_after_consecutive_unreachable_observations() -> (
    None
):
    # Arrange
    async def unreachable(_url: str, _timeout_s: float) -> UpstreamReachability:
        return UpstreamReachability(
            False, "connection_refused", 1.2, "2026-09-12T00:00:00Z"
        )

    backend = InferenceBackend(
        InferenceUpstreamPool.from_urls("http://127.0.0.1:18773"),
        health_probe=unreachable,
        health_cache_ttl_s=0,
    )
    # Act
    async with _serving(backend) as test_client:
        first = await test_client.get("/health")
        response = await test_client.get("/health")
    first_payload = first.json()
    payload = response.json()
    # Assert
    assert (
        response.status_code,
        first.status_code,
        first_payload["members"][0]["reachable"],
        first_payload["members"][0]["ready"],
        first_payload["members"][0]["reachability"]["consecutive_failures"],
        payload["status"],
        payload["reason"],
        payload["configured_members"],
        payload["admission_eligible_members"],
        payload["reachable_members"],
        payload["active_members"],
        payload["members"][0]["reachability"]["reason"],
    ) == (
        503,
        200,
        False,
        True,
        1,
        "degraded",
        "no_inference_upstream_reachable",
        1,
        1,
        0,
        0,
        "connection_refused",
    )


@pytest.mark.asyncio
async def test_relay_health_clears_recovered_cooldown_and_wakes_waiter() -> None:
    # Arrange
    async def reachable(_url: str, _timeout_s: float) -> UpstreamReachability:
        return UpstreamReachability(True, "responded", 2.4, "2026-09-12T00:00:00Z", 200)

    pool = InferenceUpstreamPool.from_urls(
        "http://127.0.0.1:18773", capacity_per_upstream=1
    )
    admitted = await pool.acquire("first")
    waiting = asyncio.create_task(pool.acquire("second"))
    for _ in range(100):
        if pool.upstreams[0].queued == 1:
            break
        await asyncio.sleep(0)
    await pool.cool_down(pool.upstreams[0], 30)
    await pool.release(admitted)
    backend = InferenceBackend(pool, health_probe=reachable)
    # Act
    async with _serving(backend) as test_client:
        response = await test_client.get("/health")
        recovered = await asyncio.wait_for(waiting, timeout=0.1)
    payload = response.json()
    member = payload["members"][0]
    await pool.release(recovered)
    # Assert
    assert (
        response.status_code,
        payload["status"],
        payload["reachable_members"],
        payload["admission_eligible_members"],
        payload["active_members"],
        member["reachable"],
        member["admission_eligible"],
        pool.upstreams[0].cooldown_until,
    ) == (
        200,
        "ok",
        1,
        1,
        1,
        True,
        True,
        0.0,
    )


@pytest.mark.asyncio
async def test_relay_health_failed_probe_preserves_cooldown() -> None:
    # Arrange
    async def failed(_url: str, _timeout_s: float) -> UpstreamReachability:
        return UpstreamReachability(
            False, "connection_refused", 1.1, "2026-09-12T00:00:00Z"
        )

    pool = InferenceUpstreamPool.from_urls("http://127.0.0.1:18773")
    await pool.cool_down(pool.upstreams[0], 30)
    cooldown_until = pool.upstreams[0].cooldown_until
    backend = InferenceBackend(
        pool, health_probe=failed, health_failure_threshold=1
    )
    # Act
    async with _serving(backend) as test_client:
        response = await test_client.get("/health")
    # Assert
    assert (response.status_code, pool.upstreams[0].cooldown_until) == (
        503,
        cooldown_until,
    )


@pytest.mark.asyncio
async def test_relay_health_timeout_probe_preserves_cooldown() -> None:
    # Arrange
    async def timed_out(_url: str, _timeout_s: float) -> UpstreamReachability:
        return UpstreamReachability(False, "timeout", 1_000.0, "2026-09-12T00:00:00Z")

    pool = InferenceUpstreamPool.from_urls("http://127.0.0.1:18773")
    await pool.cool_down(pool.upstreams[0], 30)
    cooldown_until = pool.upstreams[0].cooldown_until
    backend = InferenceBackend(
        pool, health_probe=timed_out, health_failure_threshold=1
    )
    # Act
    async with _serving(backend) as test_client:
        response = await test_client.get("/health")
    # Assert
    assert (response.status_code, pool.upstreams[0].cooldown_until) == (
        503,
        cooldown_until,
    )


@pytest.mark.asyncio
async def test_recovery_probe_cannot_clear_a_newer_concurrent_failure() -> None:
    # Arrange
    started = asyncio.Event()
    finish = asyncio.Event()

    async def delayed_recovery(_url: str, _timeout_s: float) -> UpstreamReachability:
        started.set()
        await finish.wait()
        return UpstreamReachability(True, "responded", 2.4, "2026-09-12T00:00:00Z", 200)

    pool = InferenceUpstreamPool.from_urls("http://127.0.0.1:18773")
    upstream = pool.upstreams[0]
    await pool.cool_down(upstream, 30)
    backend = InferenceBackend(pool, health_probe=delayed_recovery)
    probe = asyncio.create_task(backend.probe_upstreams())
    await started.wait()
    await pool.cool_down(upstream, 60)
    newer_cooldown = upstream.cooldown_until
    finish.set()
    # Act
    await probe
    # Assert
    assert (upstream.cooldown_generation, upstream.cooldown_until) == (
        2,
        newer_cooldown,
    )


@pytest.mark.asyncio
async def test_concurrent_health_callers_share_one_probe_generation() -> None:
    # Arrange
    called = 0
    started = asyncio.Event()
    finish = asyncio.Event()

    async def slow_probe(_url: str, _timeout_s: float) -> UpstreamReachability:
        nonlocal called
        called += 1
        started.set()
        await finish.wait()
        return UpstreamReachability(True, "responded", 1.0, "2026-09-12T00:00:00Z", 200)

    backend = InferenceBackend(
        InferenceUpstreamPool.from_urls("http://127.0.0.1:18773"),
        health_probe=slow_probe,
    )
    tasks = [asyncio.create_task(backend.probe_upstreams()) for _ in range(25)]
    await started.wait()
    finish.set()
    # Act
    results = await asyncio.gather(*tasks)
    cached = await backend.probe_upstreams()
    # Assert
    assert (
        called,
        len(results),
        all(result[0].reachable for result in results),
        cached[0].reachable,
    ) == (
        1,
        25,
        True,
        True,
    )


@pytest.mark.asyncio
async def test_injected_health_probe_obeys_end_to_end_timeout() -> None:
    # Arrange
    cancelled = asyncio.Event()

    async def stalled(_url: str, _timeout_s: float) -> UpstreamReachability:
        try:
            await asyncio.sleep(60)
        finally:
            cancelled.set()
        raise AssertionError("unreachable")

    backend = InferenceBackend(
        InferenceUpstreamPool.from_urls("http://127.0.0.1:18773"),
        health_probe=stalled,
        health_probe_timeout_s=0.01,
        health_failure_threshold=1,
    )
    # Act
    result = (await backend.probe_upstreams())[0]
    # Assert
    assert (result.reason, result.readiness, cancelled.is_set()) == (
        "timeout",
        False,
        True,
    )


@pytest.mark.asyncio
async def test_relay_health_exposes_live_admission_counts(upstream_factory) -> None:
    # Arrange
    upstream = upstream_factory()
    pool = InferenceUpstreamPool.from_urls(
        upstream.url, capacity_per_upstream=1, max_queue_size=1
    )
    backend = InferenceBackend(pool)
    admitted = await pool.acquire("first")
    waiting = asyncio.create_task(pool.acquire("second"))
    for _ in range(100):
        if pool.upstreams[0].queued == 1:
            break
        await asyncio.sleep(0)

    # Act
    async with _serving(backend) as test_client:
        response = await test_client.get("/health")
        waiting.cancel()
        with suppress(asyncio.CancelledError):
            await waiting
        await pool.release(admitted)

    # Assert
    member = response.json()["members"][0]
    assert (
        member["url"],
        member["active"],
        member["in_flight"],
        member["queued"],
        member["capacity"],
    ) == (upstream.url, True, 1, 1, 1)


@pytest.mark.asyncio
async def test_relay_health_exposes_live_token_admission_counts(
    upstream_factory,
) -> None:
    # Arrange
    upstream = upstream_factory()
    pool = InferenceUpstreamPool.from_urls(
        upstream.url, token_capacity_per_upstream=1_000
    )
    backend = InferenceBackend(pool)
    admitted = await pool.acquire("first", input_tokens=700)
    waiting = asyncio.create_task(pool.acquire("second", input_tokens=400))
    for _ in range(100):
        if pool.upstreams[0].queued == 1:
            break
        await asyncio.sleep(0)

    # Act
    async with _serving(backend) as test_client:
        response = await test_client.get("/health")
        waiting.cancel()
        with suppress(asyncio.CancelledError):
            await waiting
        await pool.release(admitted, input_tokens=700)

    # Assert
    assert (
        response.json()["input_tokens_in_flight"],
        response.json()["input_tokens_queued"],
        response.json()["members"][0]["token_capacity"],
    ) == (700, 400, 1_000)


@pytest.mark.asyncio
async def test_relay_lifespan_closes_admission_on_shutdown(upstream_factory) -> None:
    # Arrange
    pool = InferenceUpstreamPool.from_urls(upstream_factory().url)
    backend = InferenceBackend(pool)

    # Act
    async with _serving(backend):
        pass

    # Assert
    with pytest.raises(InferenceAdmissionError, match="shutting down"):
        await pool.acquire("after-shutdown")


@pytest.mark.asyncio
async def test_relay_returns_503_when_the_bounded_queue_is_full(
    upstream_factory,
) -> None:
    # Arrange
    upstream = upstream_factory()
    pool = InferenceUpstreamPool.from_urls(
        upstream.url, capacity_per_upstream=1, max_queue_size=0
    )
    admitted = await pool.acquire("occupying")
    backend = InferenceBackend(pool)

    # Act
    async with _serving(backend) as test_client:
        response = await test_client.post(
            "/v1/messages",
            json=_relay_body(),
            headers={"x-api-key": "relay-secret", "x-session-id": "overload"},
        )
        await pool.release(admitted)

    # Assert
    assert (
        response.status_code,
        response.json()["error"]["type"],
        len(upstream.requests),
    ) == (503, "inference_admission", 0)


@pytest.mark.asyncio
async def test_sigterm_closes_queue_before_uvicorn_drains_admitted_request(
    upstream_factory,
) -> None:
    # Arrange
    release_upstream = threading.Event()
    upstream = upstream_factory(block_until=release_upstream)
    pool = InferenceUpstreamPool.from_urls(
        upstream.url, capacity_per_upstream=1, max_queue_size=1
    )
    app = create_app(InferenceBackend(pool), api_key="relay-secret")
    server = _build_uvicorn_server(
        app,
        host="127.0.0.1",
        port=0,
        log_level="warning",
        timeout_graceful_shutdown=5,
    )
    serve_task = asyncio.create_task(server.serve())
    for _ in range(1000):
        if server.started:
            break
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]

    # Act
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
        admitted_task = asyncio.create_task(
            client.post(
                "/v1/messages",
                json=_relay_body(),
                headers={"x-api-key": "relay-secret", "x-session-id": "admitted"},
            )
        )
        started = await asyncio.to_thread(upstream.request_started.wait, 2)
        queued_task = asyncio.create_task(
            client.post(
                "/v1/messages",
                json=_relay_body(),
                headers={"x-api-key": "relay-secret", "x-session-id": "queued"},
            )
        )
        for _ in range(1000):
            if pool.upstreams[0].queued == 1:
                break
            await asyncio.sleep(0.001)
        server.handle_exit(signal.SIGTERM, None)
        # Uvicorn re-raises captured OS signals after ``serve`` returns. This
        # test invokes the real handler directly, so consume that bookkeeping
        # entry rather than terminating the pytest process afterward.
        server._captured_signals.clear()
        queued_response = await asyncio.wait_for(queued_task, timeout=2)
        draining = not serve_task.done()
        release_upstream.set()
        admitted_response = await asyncio.wait_for(admitted_task, timeout=2)
        await asyncio.wait_for(serve_task, timeout=2)

    # Assert
    assert (
        started,
        queued_response.status_code,
        queued_response.json()["error"]["type"],
        draining,
        admitted_response.status_code,
        serve_task.done(),
    ) == (True, 503, "inference_admission", True, 200, True)


@pytest.mark.asyncio
async def test_relay_app_relays_get_paths_to_the_upstream(upstream_factory) -> None:
    # Arrange
    upstream = upstream_factory(chunks=(b'{"data":[]}',))
    backend = InferenceBackend(InferenceUpstreamPool.from_urls(upstream.url))
    # Act
    async with _serving(backend) as test_client:
        response = await test_client.get(
            "/v1/models", headers={"x-api-key": "relay-secret"}
        )
    # Assert
    assert (response.status_code, upstream.requests[0]["path"]) == (200, "/v1/models")


# ---------------------------------------------------------------------------
# The OpenAI protocol routes (2026-09-05, Codex).
# ---------------------------------------------------------------------------


def _chat_body() -> dict:
    return {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": "Keep me"},
            {"role": "user", "content": "Hello"},
        ],
    }


@pytest.mark.asyncio
async def test_relay_app_serves_chat_completions_untouched(upstream_factory) -> None:
    # Arrange
    reply = b'{"id":"chatcmpl-1","object":"chat.completion","choices":[]}'
    upstream = upstream_factory(chunks=(reply,))
    backend = InferenceBackend(InferenceUpstreamPool.from_urls(upstream.url))
    # Act
    async with _serving(backend) as test_client:
        response = await test_client.post(
            "/v1/chat/completions",
            json=_chat_body(),
            headers={"Authorization": "Bearer relay-secret"},
        )
    forwarded = json.loads(upstream.requests[0]["body"])
    # Assert
    assert (
        response.status_code,
        response.content,
        upstream.requests[0]["path"],
        [message["role"] for message in forwarded["messages"]],
    ) == (200, reply, "/v1/chat/completions", ["system", "user"])


@pytest.mark.asyncio
async def test_relay_app_serves_responses(upstream_factory) -> None:
    # Arrange -- Codex's only wire_api in 0.153.4 is "responses".
    reply = b'{"id":"resp_1","object":"response","output":[]}'
    upstream = upstream_factory(chunks=(reply,))
    backend = InferenceBackend(InferenceUpstreamPool.from_urls(upstream.url))
    # Act
    async with _serving(backend) as test_client:
        response = await test_client.post(
            "/v1/responses",
            json={"model": "local-model", "instructions": "Keep me", "input": "Hello"},
            headers={"Authorization": "Bearer relay-secret"},
        )
    # Assert
    assert (response.status_code, response.content, upstream.requests[0]["path"]) == (
        200,
        reply,
        "/v1/responses",
    )


@pytest.mark.asyncio
async def test_relay_app_refuses_an_openai_route_in_the_openai_envelope(
    upstream_factory,
) -> None:
    # Arrange -- a wrong key must reach Codex as "Invalid API key", not as an
    # Anthropic envelope it cannot parse.
    upstream = upstream_factory()
    backend = InferenceBackend(InferenceUpstreamPool.from_urls(upstream.url))
    # Act
    async with _serving(backend) as test_client:
        response = await test_client.post(
            "/v1/responses",
            json={"model": "local-model", "input": "Hello"},
            headers={"Authorization": "Bearer wrong"},
        )
    # Assert
    assert (response.status_code, response.json()) == (
        401,
        {
            "error": {
                "message": "Invalid API key",
                "type": "authentication_error",
                "code": 401,
            }
        },
    )
