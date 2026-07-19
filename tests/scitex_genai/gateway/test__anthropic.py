from __future__ import annotations

import json
from hashlib import sha256

from scitex_genai.gateway._anthropic import (
    AnthropicStreamTranslator,
    anthropic_to_codex,
    codex_events_to_anthropic,
)


def _event_data(chunk: str) -> dict:
    data_line = next(line for line in chunk.splitlines() if line.startswith("data: "))
    return json.loads(data_line[6:])


def test_anthropic_to_codex_preserves_tool_cycle_and_images() -> None:
    # Arrange
    body = {
        "model": "gpt-5.4",
        "system": [{"type": "text", "text": "Use tools carefully."}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect this"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "aW1hZ2U=",
                        },
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call-1|fc-1",
                        "name": "read_file",
                        "input": {"path": "README.md"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call-1|fc-1",
                        "content": "contents",
                    }
                ],
            },
        ],
        "tools": [
            {
                "name": "read_file",
                "description": "Read a file",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            }
        ],
        "stream": True,
    }
    # Act
    payload = anthropic_to_codex(body, session_id="session-a")
    observed = (
        payload["instructions"],
        payload["input"][0]["content"][1]["type"],
        payload["input"][1]["call_id"],
        payload["input"][1]["id"],
        payload["input"][2],
        payload["tools"][0]["parameters"],
        payload["prompt_cache_key"],
    )
    expected = (
        "Use tools carefully.",
        "input_image",
        "call-1",
        "fc-1",
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "contents",
        },
        body["tools"][0]["input_schema"],
        "session-a",
    )
    # Assert
    assert observed == expected


def test_anthropic_to_codex_hashes_oversized_prompt_cache_key() -> None:
    # Arrange
    session_id = "claude-session-" * 10
    body = {"model": "gpt-5.6-sol", "messages": []}
    # Act
    payload = anthropic_to_codex(body, session_id=session_id)
    # Assert
    assert payload["prompt_cache_key"] == sha256(session_id.encode()).hexdigest()


def test_stream_translator_emits_anthropic_tool_events_and_usage() -> None:
    # Arrange
    translator = AnthropicStreamTranslator("gpt-5.4")
    events = [
        {"type": "response.created", "response": {"id": "resp-1"}},
        {
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "id": "fc-1",
                "call_id": "call-1",
                "name": "read_file",
            },
        },
        {"type": "response.function_call_arguments.delta", "delta": '{"path":'},
        {"type": "response.function_call_arguments.delta", "delta": '"README.md"}'},
        {
            "type": "response.output_item.done",
            "item": {"type": "function_call", "id": "fc-1", "call_id": "call-1"},
        },
        {
            "type": "response.completed",
            "response": {"usage": {"input_tokens": 10, "output_tokens": 5}},
        },
    ]
    # Act
    chunks = [chunk for event in events for chunk in translator.translate(event)]
    payloads = [_event_data(chunk) for chunk in chunks]
    observed = (
        payloads[1]["content_block"]["id"],
        payloads[2]["delta"]["partial_json"],
        payloads[-2]["delta"]["stop_reason"],
        payloads[-2]["usage"]["output_tokens"],
        payloads[-1],
    )
    # Assert
    assert observed == (
        "call-1|fc-1",
        '{"path":',
        "tool_use",
        5,
        {"type": "message_stop"},
    )


def test_nonstream_collector_preserves_text_and_tool_calls() -> None:
    # Arrange
    events = [
        {"type": "response.created", "response": {"id": "resp-1"}},
        {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "content": [{"type": "output_text", "text": "Checking."}],
            },
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "id": "fc-1",
                "call_id": "call-1",
                "name": "read_file",
                "arguments": '{"path":"README.md"}',
            },
        },
        {
            "type": "response.completed",
            "response": {"usage": {"input_tokens": 10, "output_tokens": 5}},
        },
    ]
    # Act
    response = codex_events_to_anthropic(events, model="gpt-5.4")
    observed = (
        response["id"],
        response["content"][0],
        response["content"][1]["id"],
        response["stop_reason"],
        response["usage"],
    )
    # Assert
    assert observed == (
        "resp-1",
        {"type": "text", "text": "Checking."},
        "call-1|fc-1",
        "tool_use",
        {"input_tokens": 10, "output_tokens": 5},
    )
