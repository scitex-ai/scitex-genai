"""Tests for the OpenCode serve-harness backend (no network: stub client)."""

from __future__ import annotations

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
    # Act / Assert
    assert _text_of(message) == "DONE"


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
        assert "/session/ses_test123/message" in url
        assert json["model"] == {"providerID": "opencode", "modelID": "m1"}
        assert json["parts"] == [{"type": "text", "text": "Hi"}]
        return _Response(
            {
                "info": {"modelID": "m1"},
                "parts": [{"type": "text", "text": "HELLO_BACK"}],
            }
        )

    def close(self) -> None:
        pass


def test_complete_drives_session_api() -> None:
    # Arrange
    import asyncio

    backend = OpenCodeBackend()
    client = _Client()
    body = {"model": "m1", "messages": [{"role": "user", "content": "Hi"}]}
    # Act
    result = asyncio.run(backend.complete(body, client=client))
    # Assert
    assert result == {"text": "HELLO_BACK", "model": "m1"}
    assert client.calls[0].endswith("/session")


def test_chat_completions_route_answers_openai_envelope() -> None:
    # Arrange
    from fastapi.testclient import TestClient

    backend = OpenCodeBackend()
    async_orig = backend.complete

    async def fake_complete(body: dict) -> dict:
        assert body["model"] == "muse-spark-1.3-contributor-free"
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
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "GW_OK"
    finally:
        backend.complete = async_orig  # type: ignore[method-assign]


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
