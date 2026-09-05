"""One bad tool call must not poison a Codex conversation.

Measured 2026-09-05 17:58Z: vLLM 0.28.0 (qwen3_xml parser) emitted a
function_call whose ``arguments`` was cut off at 97 characters with a leaked
``</parameter`` tag; every later request carrying that item answered
``400 Expecting value: line 1 column 98 (char 97)``. The relay now repairs such
strings before forwarding. Real bytes through ``prepare``; nothing mocked.
"""

from __future__ import annotations

import json

from scitex_genai.gateway._inference import (
    InferenceBackend,
    InferenceUpstreamPool,
    repair_tool_call_arguments,
)

# The exact string the model produced (97 characters, then nothing).
BROKEN = (
    '{"cmd": "timeout 7\\n7", "echo SAC_NAME=$SAC_NAME; '
    'echo CCA=$SCITEX_CARDS_AGENT_ID\\n</parameter": '
)


def _responses_body(arguments: str) -> bytes:
    return json.dumps(
        {
            "model": "qwen38-27b",
            "input": [
                {"role": "user", "content": "hi"},
                {
                    "type": "function_call",
                    "call_id": "call_b064aed114762414",
                    "name": "exec_command",
                    "arguments": arguments,
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_b064aed114762414",
                    "output": "…",
                },
            ],
        }
    ).encode()


def test_broken_arguments_are_repaired_into_json_through_prepare() -> None:
    # Arrange
    backend = InferenceBackend(InferenceUpstreamPool.from_urls("http://127.0.0.1:9"))
    # Act
    forwarded, _ = backend.prepare(_responses_body(BROKEN), hoist=False)
    # Assert
    item = json.loads(forwarded)["input"][1]
    assert json.loads(item["arguments"]) == {"_invalid_arguments": BROKEN}


def test_valid_arguments_are_forwarded_byte_for_byte() -> None:
    # Arrange
    body = _responses_body('{"cmd": "ls -la"}')
    backend = InferenceBackend(InferenceUpstreamPool.from_urls("http://127.0.0.1:9"))
    # Act
    forwarded, _ = backend.prepare(body, hoist=False)
    # Assert
    assert forwarded == body


def test_repair_counts_only_the_strings_it_changed() -> None:
    # Arrange
    payload = json.loads(_responses_body(BROKEN))
    # Act
    _, repaired = repair_tool_call_arguments(payload)
    # Assert
    assert repaired == 1


def test_chat_shape_tool_calls_are_repaired_too() -> None:
    # Arrange
    payload = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": BROKEN},
                    }
                ],
            }
        ]
    }
    # Act
    repaired_payload, _ = repair_tool_call_arguments(payload)
    # Assert
    arguments = repaired_payload["messages"][0]["tool_calls"][0]["function"][
        "arguments"
    ]
    assert json.loads(arguments) == {"_invalid_arguments": BROKEN}


def test_relay_journal_line_names_the_repair() -> None:
    # Arrange
    lines: list[str] = []
    backend = InferenceBackend(
        InferenceUpstreamPool.from_urls("http://127.0.0.1:9"), journal=lines.append
    )
    # Act
    backend.prepare(_responses_body(BROKEN), hoist=False)
    # Assert
    assert lines == [
        "[relay] repaired 1 tool-call argument string(s) that were not JSON (kept under _invalid_arguments)"
    ]
