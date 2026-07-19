"""Anthropic Messages to Codex Responses protocol translation."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from ._errors import GatewayError


def _system_text(system: Any) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "\n\n".join(
            block.get("text", "")
            for block in system
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _user_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    parts: list[dict[str, Any]] = []
    if not isinstance(content, list):
        return parts
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            parts.append({"type": "input_text", "text": block.get("text", "")})
        elif block_type == "image":
            source = block.get("source")
            if not isinstance(source, dict):
                continue
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/png")
                parts.append(
                    {
                        "type": "input_image",
                        "detail": "auto",
                        "image_url": f"data:{media_type};base64,{source.get('data', '')}",
                    }
                )
            elif source.get("type") == "url":
                parts.append(
                    {
                        "type": "input_image",
                        "detail": "auto",
                        "image_url": source.get("url", ""),
                    }
                )
    return parts


def _assistant_items(content: Any, message_index: int) -> list[dict[str, Any]]:
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
    if not isinstance(blocks, list):
        return []
    items: list[dict[str, Any]] = []
    for block_index, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text" and block.get("text"):
            items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": block.get("text", ""),
                            "annotations": [],
                        }
                    ],
                    "status": "completed",
                    "id": f"msg_{message_index}_{block_index}",
                }
            )
        elif block_type == "tool_use":
            tool_id = str(block.get("id", ""))
            call_id, separator, item_id = tool_id.partition("|")
            item: dict[str, Any] = {
                "type": "function_call",
                "call_id": call_id,
                "name": block.get("name", ""),
                "arguments": json.dumps(block.get("input", {}), separators=(",", ":")),
            }
            if separator and item_id:
                item["id"] = item_id
            items.append(item)
    return items


def _tool_results(content: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return Responses function outputs and remaining ordinary user content."""
    if not isinstance(content, list):
        return [], _user_content(content)
    outputs: list[dict[str, Any]] = []
    ordinary: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "tool_result":
            ordinary.extend(_user_content([block]))
            continue
        call_id = str(block.get("tool_use_id", "")).split("|", 1)[0]
        result_content = block.get("content", "")
        text = _text_from_content(result_content)
        outputs.append(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": text or "(no output)",
            }
        )
    return outputs, ordinary


def anthropic_to_codex(body: dict[str, Any], *, session_id: str = "") -> dict[str, Any]:
    """Build a Codex Responses request without executing any tools."""
    model = body.get("model")
    messages = body.get("messages")
    if not isinstance(model, str) or not model:
        raise GatewayError("Anthropic request requires a model")
    if not isinstance(messages, list):
        raise GatewayError("Anthropic request requires a messages array")

    inputs: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content", "")
        if role == "assistant":
            inputs.extend(_assistant_items(content, message_index))
        elif role == "user":
            outputs, ordinary = _tool_results(content)
            inputs.extend(outputs)
            if ordinary:
                inputs.append({"role": "user", "content": ordinary})

    tools = []
    for tool in body.get("tools", []) or []:
        if not isinstance(tool, dict):
            continue
        tools.append(
            {
                "type": "function",
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object"}),
                "strict": None,
            }
        )

    tool_choice: Any = "auto"
    anthropic_choice = body.get("tool_choice")
    if isinstance(anthropic_choice, dict):
        choice_type = anthropic_choice.get("type")
        if choice_type == "none":
            tool_choice = "none"
        elif choice_type == "any":
            tool_choice = "required"
        elif choice_type == "tool":
            tool_choice = {
                "type": "function",
                "name": anthropic_choice.get("name", ""),
            }

    request: dict[str, Any] = {
        "model": model,
        "store": False,
        "stream": True,
        "instructions": _system_text(body.get("system")),
        "input": inputs,
        "text": {"verbosity": "medium"},
        "include": ["reasoning.encrypted_content"],
        "tool_choice": tool_choice,
        "parallel_tool_calls": not (
            isinstance(anthropic_choice, dict)
            and anthropic_choice.get("disable_parallel_tool_use") is True
        ),
    }
    if session_id:
        request["prompt_cache_key"] = session_id
    if tools:
        request["tools"] = tools
    if "temperature" in body:
        request["temperature"] = body["temperature"]
    thinking = body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "enabled":
        request["reasoning"] = {"summary": "auto"}
    return request


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


@dataclass
class AnthropicStreamTranslator:
    """Statefully translate Codex Responses SSE objects to Anthropic SSE."""

    requested_model: str
    message_id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex}")
    started: bool = False
    stopped: bool = False
    next_index: int = 0
    open_blocks: dict[str, int] = field(default_factory=dict)
    block_types: dict[int, str] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0

    def translate(self, event: dict[str, Any]) -> list[str]:
        event_type = event.get("type")
        chunks: list[str] = []
        if not self.started:
            chunks.append(self._message_start(event))
            self.started = True

        if event_type == "response.output_item.added":
            item = event.get("item", {})
            item_type = item.get("type") if isinstance(item, dict) else None
            if item_type == "message":
                self._open_block(str(item.get("id", "text")), "text")
            elif item_type == "function_call":
                key = str(item.get("id") or item.get("call_id") or self.next_index)
                index = self._open_block(key, "tool_use")
                tool_id = f"{item.get('call_id', '')}|{item.get('id', '')}".rstrip("|")
                chunks.append(
                    _sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": index,
                            "content_block": {
                                "type": "tool_use",
                                "id": tool_id,
                                "name": item.get("name", ""),
                                "input": {},
                            },
                        },
                    )
                )
        elif event_type == "response.output_text.delta":
            index = self._latest_block("text")
            if index is None:
                index = self._open_block("text", "text")
            if not self._block_announced(index):
                chunks.append(
                    _sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": index,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                )
                self.block_types[index] = "text:announced"
            chunks.append(
                _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "text_delta", "text": event.get("delta", "")},
                    },
                )
            )
        elif event_type == "response.function_call_arguments.delta":
            index = self._latest_block("tool_use")
            if index is not None:
                chunks.append(
                    _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": event.get("delta", ""),
                            },
                        },
                    )
                )
        elif event_type == "response.output_item.done":
            item = event.get("item", {})
            if isinstance(item, dict) and item.get("type") in {"message", "function_call"}:
                key = str(item.get("id") or item.get("call_id") or "text")
                index = self.open_blocks.pop(key, None)
                if index is None:
                    index = self._latest_block(
                        "tool_use" if item.get("type") == "function_call" else "text"
                    )
                if index is not None:
                    chunks.append(
                        _sse(
                            "content_block_stop",
                            {"type": "content_block_stop", "index": index},
                        )
                    )
        elif event_type in {"response.completed", "response.done"}:
            chunks.extend(self._finish(event.get("response", {})))
        elif event_type in {"response.failed", "error"}:
            message = event.get("message") or (
                event.get("response", {}).get("error", {}).get("message")
                if isinstance(event.get("response"), dict)
                else None
            )
            raise GatewayError(str(message or "Codex response failed"))
        return chunks

    def _message_start(self, event: dict[str, Any]) -> str:
        response = event.get("response", {})
        if isinstance(response, dict):
            self.message_id = str(response.get("id") or self.message_id)
            usage = response.get("usage", {})
            if isinstance(usage, dict):
                self.input_tokens = int(usage.get("input_tokens", 0) or 0)
        return _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.requested_model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": self.input_tokens, "output_tokens": 0},
                },
            },
        )

    def _finish(self, response: Any) -> list[str]:
        if self.stopped:
            return []
        self.stopped = True
        chunks: list[str] = []
        for index in sorted(set(self.open_blocks.values())):
            chunks.append(
                _sse(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": index},
                )
            )
        self.open_blocks.clear()
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        if isinstance(usage, dict):
            self.input_tokens = int(usage.get("input_tokens", self.input_tokens) or 0)
            self.output_tokens = int(usage.get("output_tokens", 0) or 0)
        stop_reason = (
            "tool_use"
            if any(value.startswith("tool_use") for value in self.block_types.values())
            else "end_turn"
        )
        chunks.append(
            _sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": self.output_tokens},
                },
            )
        )
        chunks.append(_sse("message_stop", {"type": "message_stop"}))
        return chunks

    def _open_block(self, key: str, block_type: str) -> int:
        existing = self.open_blocks.get(key)
        if existing is not None:
            return existing
        index = self.next_index
        self.next_index += 1
        self.open_blocks[key] = index
        self.block_types[index] = block_type
        return index

    def _latest_block(self, block_type: str) -> int | None:
        candidates = [
            index
            for index, value in self.block_types.items()
            if value.startswith(block_type)
        ]
        return max(candidates) if candidates else None

    def _block_announced(self, index: int) -> bool:
        return self.block_types.get(index, "").endswith(":announced")


def codex_events_to_anthropic(
    events: list[dict[str, Any]], *, model: str
) -> dict[str, Any]:
    """Collect Codex events into one non-streaming Anthropic message."""
    message_id = f"msg_{uuid.uuid4().hex}"
    content: list[dict[str, Any]] = []
    input_tokens = 0
    output_tokens = 0
    for event in events:
        event_type = event.get("type")
        response = event.get("response")
        if event_type == "response.created" and isinstance(response, dict):
            message_id = str(response.get("id") or message_id)
        elif event_type == "response.output_item.done":
            item = event.get("item")
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                text = "".join(
                    str(part.get("text", ""))
                    for part in item.get("content", [])
                    if isinstance(part, dict) and part.get("type") == "output_text"
                )
                if text:
                    content.append({"type": "text", "text": text})
            elif item.get("type") == "function_call":
                try:
                    arguments = json.loads(item.get("arguments", "{}"))
                except (TypeError, ValueError):
                    arguments = {}
                tool_id = f"{item.get('call_id', '')}|{item.get('id', '')}".rstrip("|")
                content.append(
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": item.get("name", ""),
                        "input": arguments,
                    }
                )
        elif event_type in {"response.completed", "response.done"}:
            usage = response.get("usage", {}) if isinstance(response, dict) else {}
            if isinstance(usage, dict):
                input_tokens = int(usage.get("input_tokens", 0) or 0)
                output_tokens = int(usage.get("output_tokens", 0) or 0)
        elif event_type in {"response.failed", "error"}:
            raise GatewayError("Codex response failed")
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": (
            "tool_use" if any(block.get("type") == "tool_use" for block in content) else "end_turn"
        ),
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }
