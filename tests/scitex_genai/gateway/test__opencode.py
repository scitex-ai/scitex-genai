"""Tests for the OpenCode serve-harness backend (no network: stub client)."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("fastapi")

from scitex_genai.gateway._opencode import (
    OpenCodeBackend,
    _text_of,
    openai_messages_to_text,
)
from scitex_genai.gateway._server import create_app


def test_openai_messages_to_text_flattens_string_content() -> None:
    # Arrange
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Reply OK"},
    ]
    # Act
    text = openai_messages_to_text(messages)
    # Assert
    assert text == "Be brief.\n\nReply OK"


def test_openai_messages_to_text_flattens_block_content() -> None:
    # Arrange
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
    # Act
    text = openai_messages_to_text(messages)
    # Assert
    assert text == "Hi"


def test_text_of_picks_first_text_part() -> None:
    # Arrange
    message = {"parts": [{"type": "step-start"}, {"type": "text", "text": "DONE"}]}
    # Act
    text = _text_of(message)
    # Assert
    assert text == "DONE"


def test_text_of_returns_empty_without_text_part() -> None:
    # Arrange
    message = {"parts": [{"type": "step-start"}]}
    # Act
    text = _text_of(message)
    # Assert
    assert text == ""


class _Response:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status
        self.text = str(payload)

    def json(self) -> dict:
        return self._payload


class _Client:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def post(self, url: str, json: dict, timeout: float) -> _Response:
        self.calls.append(url)
        if url.endswith("/session"):
            return _Response({"id": "ses_test123"})
        return _Response(
            {
                "info": {"modelID": "m1"},
                "parts": [{"type": "text", "text": "HELLO_BACK"}],
            }
        )

    def close(self) -> None:
        pass


def test_complete_returns_reply_text() -> None:
    # Arrange
    import asyncio

    backend = OpenCodeBackend()
    client = _Client()
    body = {"model": "m1", "messages": [{"role": "user", "content": "Hi"}]}
    # Act
    result = asyncio.run(backend.complete(body, client=client))
    # Assert
    assert result["text"] == "HELLO_BACK"


def test_complete_hits_session_endpoint_first() -> None:
    # Arrange
    import asyncio

    backend = OpenCodeBackend()
    client = _Client()
    body = {"model": "m1", "messages": [{"role": "user", "content": "Hi"}]}
    # Act
    asyncio.run(backend.complete(body, client=client))
    # Assert
    assert client.calls[0].endswith("/session")


def test_chat_completions_route_answers_reply() -> None:
    # Arrange
    from fastapi.testclient import TestClient

    backend = OpenCodeBackend()

    async def fake_complete(body: dict) -> dict:
        return {"text": "GW_OK", "model": "muse-spark-1.3-contributor-free"}

    backend.complete = fake_complete  # type: ignore[method-assign]
    try:
        app = create_app(backend, api_key="k")
        client = TestClient(app)
        # Act
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer k"},
            json={"model": "muse-spark-1.3-contributor-free", "messages": []},
        )
        # Assert
        assert response.json()["choices"][0]["message"]["content"] == "GW_OK"
    finally:
        pass


def test_chat_completions_route_status_ok() -> None:
    # Arrange
    from fastapi.testclient import TestClient

    backend = OpenCodeBackend()

    async def fake_complete(body: dict) -> dict:
        return {"text": "GW_OK", "model": "muse-spark-1.3-contributor-free"}

    backend.complete = fake_complete  # type: ignore[method-assign]
    app = create_app(backend, api_key="k")
    client = TestClient(app)
    # Act
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer k"},
        json={"model": "muse-spark-1.3-contributor-free", "messages": []},
    )
    # Assert
    assert response.status_code == 200


def test_chat_completions_route_rejects_bad_key() -> None:
    # Arrange
    from fastapi.testclient import TestClient

    app = create_app(OpenCodeBackend(), api_key="k")
    client = TestClient(app)
    # Act
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer wrong"},
        json={"model": "m", "messages": []},
    )
    # Assert
    assert response.status_code == 401


def _streaming_app() -> Any:
    from fastapi.testclient import TestClient  # noqa: F401

    backend = OpenCodeBackend()

    async def fake_complete(body: dict) -> dict:
        return {"text": "GW_OK", "model": "muse-spark-1.3-contributor-free"}

    backend.complete = fake_complete  # type: ignore[method-assign]
    return create_app(backend, api_key="k")


def _stream_response_parts() -> tuple[int, str, list[str], list[dict], list[dict]]:
    import json as _json

    from fastapi.testclient import TestClient

    client = TestClient(_streaming_app())
    # Act
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer k"},
        json={
            "model": "muse-spark-1.3-contributor-free",
            "messages": [],
            "stream": True,
        },
    )
    frames = [frame for frame in response.text.split("\n\n") if frame.strip()]
    payloads = [
        _json.loads(frame.split("data:", 1)[1])
        for frame in frames[:-1]
        if frame.strip().startswith("data:")
    ]
    deltas = [p["choices"][0]["delta"] for p in payloads]
    return (
        response.status_code,
        response.headers["content-type"],
        frames,
        payloads,
        deltas,
    )


def test_stream_completions_report_success_status() -> None:
    # Arrange
    # Act
    status, _content_type, _frames, _payloads, _deltas = _stream_response_parts()
    # Assert
    assert status == 200


def test_stream_completions_use_event_stream_content_type() -> None:
    # Arrange
    # Act
    _status, content_type, _frames, _payloads, _deltas = _stream_response_parts()
    # Assert
    assert "text/event-stream" in content_type


def test_stream_completions_end_with_done_frame() -> None:
    # Arrange
    # Act
    _status, _content_type, frames, _payloads, _deltas = _stream_response_parts()
    # Assert
    assert frames[-1].strip() == "data: [DONE]"


def test_stream_completions_emit_sse_data_frames() -> None:
    # Arrange
    # Act
    _status, _content_type, _frames, payloads, _deltas = _stream_response_parts()
    # Assert
    assert payloads, "expected at least one SSE data frame"


def test_stream_completions_send_completion_chunk_objects() -> None:
    # Arrange
    # Act
    _status, _content_type, _frames, payloads, _deltas = _stream_response_parts()
    # Assert
    assert all(p["object"] == "chat.completion.chunk" for p in payloads)


def test_stream_completions_open_with_assistant_role() -> None:
    # Arrange
    # Act
    _status, _content_type, _frames, _payloads, deltas = _stream_response_parts()
    # Assert
    assert deltas[0] == {"role": "assistant"}


def test_stream_completions_deliver_reply_text_delta() -> None:
    # Arrange
    # Act
    _status, _content_type, _frames, _payloads, deltas = _stream_response_parts()
    # Assert
    assert {"content": "GW_OK"} in deltas


def test_stream_completions_mark_final_chunk_stop() -> None:
    # Arrange
    # Act
    _status, _content_type, _frames, payloads, _deltas = _stream_response_parts()
    # Assert
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"


def _nonstream_response_parts() -> tuple[int, str, dict]:
    from fastapi.testclient import TestClient

    client = TestClient(_streaming_app())
    # Act
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer k"},
        json={"model": "muse-spark-1.3-contributor-free", "messages": []},
    )
    return (
        response.status_code,
        response.headers.get("content-type", ""),
        response.json(),
    )


def test_nonstream_completions_report_success_status() -> None:
    # Arrange
    # Act
    status, _content_type, _payload = _nonstream_response_parts()
    # Assert
    assert status == 200


def test_nonstream_completions_avoid_event_stream_content() -> None:
    # Arrange
    # Act
    _status, content_type, _payload = _nonstream_response_parts()
    # Assert
    assert "text/event-stream" not in content_type


def test_nonstream_completions_return_completion_object() -> None:
    # Arrange
    # Act
    _status, _content_type, payload = _nonstream_response_parts()
    # Assert
    assert payload["object"] == "chat.completion"


def test_nonstream_completions_return_reply_text() -> None:
    # Arrange
    # Act
    _status, _content_type, payload = _nonstream_response_parts()
    # Assert
    assert payload["choices"][0]["message"]["content"] == "GW_OK"
