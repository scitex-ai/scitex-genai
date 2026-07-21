from __future__ import annotations

import json

import httpx
import pytest
import pytest_asyncio

# The gateway app is FastAPI-based (`create_app` resolves fastapi lazily);
# skip cleanly on installs without the [gateway] extra.
pytest.importorskip("fastapi")

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
