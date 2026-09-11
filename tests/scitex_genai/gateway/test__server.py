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
from scitex_genai.gateway._inference import InferenceBackend, InferenceUpstreamPool
from scitex_genai.gateway._server import (
    _build_uvicorn_server,
    _until_response_or_disconnect,
    create_app,
)


class _Pool:
    accounts = [object()]


async def _capture_base_exception(awaitable) -> BaseException | None:
    try:
        await awaitable
    except BaseException as exc:
        return exc
    return None


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
    # Assert
    assert response.json() == {
        "status": "ok",
        "provider": "inference-upstream",
        "upstreams": [upstream.url],
        "members": [
            {
                "url": upstream.url,
                "active": True,
                "in_flight": 0,
                "queued": 0,
                "capacity": 8,
                "admitted_total": 0,
                "cancelled_total": 0,
            }
        ],
        "active_members": 1,
        "in_flight": 0,
        "queued": 0,
        "admitted_total": 0,
        "cancelled_total": 0,
    }


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
    assert response.json()["members"][0] == {
        "url": upstream.url,
        "active": True,
        "in_flight": 1,
        "queued": 1,
        "capacity": 1,
        "admitted_total": 1,
        "cancelled_total": 0,
    }


@pytest.mark.asyncio
async def test_asgi_disconnect_cancels_a_queued_relay_and_removes_its_ticket(
    upstream_factory,
) -> None:
    # Arrange -- call the ASGI surface directly so ``http.disconnect`` is
    # real protocol input rather than client-task cancellation by a test SDK.
    upstream = upstream_factory()
    pool = InferenceUpstreamPool.from_urls(
        upstream.url, capacity_per_upstream=1, max_queue_size=1
    )
    occupying = await pool.acquire("occupying")
    journal: list[str] = []
    app = create_app(InferenceBackend(pool, journal=journal.append), api_key="secret")
    incoming: asyncio.Queue[dict] = asyncio.Queue()
    await incoming.put(
        {
            "type": "http.request",
            "body": json.dumps(_relay_body()).encode(),
            "more_body": False,
        }
    )
    sent: list[dict] = []
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/messages",
        "raw_path": b"/v1/messages",
        "query_string": b"",
        "headers": [
            (b"x-api-key", b"secret"),
            (b"x-session-id", b"private-session-name"),
            (b"content-type", b"application/json"),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("gateway.test", 80),
    }

    async def receive() -> dict:
        return await incoming.get()

    async def send(message: dict) -> None:
        sent.append(message)

    request_task = asyncio.create_task(app(scope, receive, send))
    for _ in range(100):
        if pool.upstreams[0].queued == 1:
            break
        await asyncio.sleep(0)
    else:
        raise AssertionError(f"relay did not queue: {pool.status()}")

    # Act
    await incoming.put({"type": "http.disconnect"})
    await asyncio.wait_for(request_task, timeout=1)
    state = pool.status()[0]
    await pool.release(occupying)

    # Assert -- only counts and route metadata are observable; neither body
    # text nor caller-declared session identity appears in the journal.
    journal_text = "\n".join(journal)
    assert (
        state["in_flight"],
        state["queued"],
        state["cancelled_total"],
        sent[0]["status"],
        len(upstream.requests),
        "private-session-name" in journal_text,
        "Hello" in journal_text,
    ) == (1, 0, 1, 499, 0, False, False)


@pytest.mark.asyncio
async def test_outer_cancellation_during_response_handoff_releases_capacity(
    upstream_factory,
) -> None:
    # Arrange -- make observer cancellation pause after the response task has
    # won, reproducing the ownership-transfer window from the reviewer probe.
    upstream = upstream_factory(chunks=(b"reply",))
    pool = InferenceUpstreamPool.from_urls(upstream.url, capacity_per_upstream=1)
    backend = InferenceBackend(pool)
    relayed = await backend.relay(
        "POST", "/v1/messages", body=b"{}", headers={"x-session-id": "handoff"}
    )
    receive_started = asyncio.Event()
    observer_cleanup_started = asyncio.Event()

    class BlockingCancellationRequest:
        async def receive(self) -> dict:
            receive_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                observer_cleanup_started.set()
                await asyncio.Future()
                raise

    async def completed_response():
        await receive_started.wait()
        return relayed

    handoff = asyncio.create_task(
        _until_response_or_disconnect(
            BlockingCancellationRequest(), completed_response()
        )
    )
    await observer_cleanup_started.wait()

    # Act
    handoff.cancel()
    handoff_result = await _capture_base_exception(handoff)
    state_after_cancel = pool.status()[0]
    await relayed.aclose(cancelled=True)  # idempotency / no negative counter
    state_after_second_close = pool.status()[0]

    # Assert
    assert (
        isinstance(handoff_result, asyncio.CancelledError),
        state_after_cancel["in_flight"],
        state_after_cancel["cancelled_total"],
        state_after_second_close["in_flight"],
        state_after_second_close["cancelled_total"],
    ) == (True, 0, 1, 0, 1)


@pytest.mark.asyncio
async def test_response_exception_always_settles_disconnect_observer() -> None:
    # Arrange
    receive_started = asyncio.Event()
    receive_finished = asyncio.Event()

    class Request:
        async def receive(self) -> dict:
            receive_started.set()
            try:
                await asyncio.Future()
            finally:
                receive_finished.set()

    async def failing_response():
        await receive_started.wait()
        raise RuntimeError("upstream failure")

    # Act
    result = await _capture_base_exception(
        _until_response_or_disconnect(Request(), failing_response())
    )
    await asyncio.sleep(0)

    # Assert
    assert (type(result), str(result), receive_finished.is_set()) == (
        RuntimeError,
        "upstream failure",
        True,
    )


@pytest.mark.asyncio
async def test_repeated_outer_cancel_cannot_interrupt_pre_response_release(
) -> None:
    # Arrange
    send_started = asyncio.Event()
    close_started = asyncio.Event()
    allow_close = asyncio.Event()

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def build_request(self, *args, **kwargs):
            return object()

        async def send(self, request, *, stream):
            send_started.set()
            await asyncio.Future()

        async def aclose(self) -> None:
            close_started.set()
            await allow_close.wait()

    pool = InferenceUpstreamPool.from_urls(
        "http://only:1", capacity_per_upstream=1
    )
    backend = InferenceBackend(pool, client_factory=Client)

    class Request:
        async def receive(self) -> dict:
            await send_started.wait()
            return {"type": "http.disconnect"}

    handoff = asyncio.create_task(
        _until_response_or_disconnect(
            Request(),
            backend.relay("POST", "/v1/messages", body=b"{}", headers={}),
        )
    )
    await asyncio.wait_for(close_started.wait(), 1)
    state_during_close = pool.status()[0]

    # Act
    handoff.cancel()
    result = await _capture_base_exception(handoff)
    allow_close.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    state_after_close = pool.status()[0]

    # Assert
    assert (
        state_during_close["in_flight"],
        isinstance(result, asyncio.CancelledError),
        state_after_close["in_flight"],
        state_after_close["cancelled_total"],
    ) == (1, True, 0, 1)


@pytest.mark.asyncio
async def test_repeated_cancel_during_close_does_not_poison_finalizer(
) -> None:
    # Arrange
    response_close_started = asyncio.Event()

    class Response:
        status_code = 200
        headers = {"content-type": "application/json"}

        async def aclose(self) -> None:
            response_close_started.set()
            await asyncio.Future()

        async def aiter_bytes(self):
            yield b"ok"

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def build_request(self, *args, **kwargs):
            return object()

        async def send(self, request, *, stream):
            return Response()

        async def aclose(self) -> None:
            pass

    pool = InferenceUpstreamPool.from_urls(
        "http://only:1", capacity_per_upstream=1
    )
    backend = InferenceBackend(pool, client_factory=Client)
    relayed = await backend.relay("POST", "/v1/messages", body=b"{}", headers={})

    # Act
    async with pool._admission:
        close_task = asyncio.create_task(relayed.aclose(cancelled=True))
        await response_close_started.wait()
        close_task.cancel()
        await asyncio.sleep(0)
        close_task.cancel()
    first_result = await _capture_base_exception(close_task)
    await relayed.aclose(cancelled=True)
    state = pool.status()[0]

    # Assert
    assert (
        isinstance(first_result, asyncio.CancelledError),
        state["in_flight"],
        state["cancelled_total"],
    ) == (True, 0, 1)


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
