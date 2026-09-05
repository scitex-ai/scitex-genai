from __future__ import annotations

import json
from contextlib import asynccontextmanager

import httpx
import pytest
import pytest_asyncio

# The gateway app is FastAPI-based (`create_app` resolves fastapi lazily);
# skip cleanly on installs without the [gateway] extra.
pytest.importorskip("fastapi")

from scitex_genai.gateway._inference import InferenceBackend, InferenceUpstreamPool
from scitex_genai.gateway._server import create_app


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
    assert (response.status_code, response.json()["content"], response.json()["usage"]) == (
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
async def test_relay_app_serves_messages_from_the_upstream_pool(upstream_factory) -> None:
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
    }


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

