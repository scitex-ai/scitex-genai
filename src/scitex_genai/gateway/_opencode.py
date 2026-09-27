"""Client for a local ``opencode serve`` harness, exposing the OpenAI shape.

The Zen free SKUs (e.g. ``muse-spark-1.3-contributor-free``) are app-locked:
raw API calls 403 with FreeTierError, but the OpenCode app carries the
client identity the SKU demands. ``opencode serve`` fronts the same engine
over a control-plane session API (``POST /session`` + ``POST
/session/:id/message``); this module drives that API and yields OpenAI
``chat.completions``-shaped events so :func:`gateway._server.create_app`
can serve Hermes' ``custom`` provider with config only.

Verified 2026-09-21 on scitex-compute-04: ``SERVE_API_OK`` round-trip
~3 s, cost 0, through ``opencode serve :4096`` (systemd opencode-serve).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from ._errors import UpstreamError

DEFAULT_OPENCODE_SERVE_URL = "http://127.0.0.1:4096"
DEFAULT_PROVIDER_ID = "opencode"
DEFAULT_AGENT = "build"


def _text_of(message: dict[str, Any]) -> str:
    for part in message.get("parts", []):
        if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
            return str(part["text"])
    return ""


def openai_messages_to_text(messages: Any) -> str:
    """Flatten OpenAI chat messages to the single text part serve expects."""
    chunks: list[str] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            chunks.append(content.strip())
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    chunks.append(block["text"].strip())
    return "\n\n".join(chunk for chunk in chunks if chunk)


class OpenCodeBackend:
    """Serve OpenAI chat completions through a local ``opencode serve``."""

    def __init__(
        self,
        *,
        serve_url: str = DEFAULT_OPENCODE_SERVE_URL,
        provider_id: str = DEFAULT_PROVIDER_ID,
        agent: str = DEFAULT_AGENT,
        client: Any = None,
    ) -> None:
        self.serve_url = (serve_url or DEFAULT_OPENCODE_SERVE_URL).rstrip("/")
        self.provider_id = provider_id
        self.agent = agent
        self._client = client

    async def refresh_usage(self) -> None:
        """No quota to poll on a local harness; satisfies the server lifespan."""

    def _post(self, client: Any, path: str, payload: dict[str, Any]) -> Any:
        # Status-code cascade: every upstream HTTP status is preserved, never
        # collapsed. A response with an error status raises UpstreamError
        # carrying the UPSTREAM code (401/403/429/...) so the server relays it
        # verbatim; only the no-response case (timeout/transport) maps to
        # 504/502 in complete(). Reliable systems need the real codes.
        response = client.post(f"{self.serve_url}{path}", json=payload, timeout=600.0)
        if response.status_code >= 400:
            raise UpstreamError(
                f"opencode serve returned HTTP {response.status_code}: "
                f"{response.text[:300]}",
                status_code=response.status_code,
            )
        return response

    async def complete(
        self, body: dict[str, Any], *, client: Any = None
    ) -> dict[str, Any]:
        """One OpenAI chat-completions body -> (reply_text, serve_model_id)."""
        model = str(body.get("model") or "")
        prompt = openai_messages_to_text(body.get("messages"))
        if not prompt:
            raise UpstreamError("chat completions request has no text content", status_code=400)
        own_client = False
        if client is None:
            try:
                import httpx
            except ImportError as exc:
                raise UpstreamError(
                    "OpenCode backend requires scitex-genai[gateway]"
                ) from exc
            client = httpx.Client(timeout=600.0)
            own_client = True
        try:
            # Sync httpx inside async handlers blocks the event loop for up to
            # the 600s timeout (c04 2026-09-27: wedged gateway, loud 504s, no
            # hangs). Run upstream I/O in a thread; map transport failures to
            # UpstreamError (504 timeout / 502 other) — fail loud, never hang.
            session = await asyncio.to_thread(
                self._post, client, "/session", {"title": "genai-gateway"}
            )
            sid = session.json().get("id", "")
            if not sid:
                raise UpstreamError("opencode serve returned no session id", status_code=502)
            message = await asyncio.to_thread(
                self._post,
                client,
                f"/session/{sid}/message",
                {
                    "parts": [{"type": "text", "text": prompt}],
                    "model": {"providerID": self.provider_id, "modelID": model},
                    "agent": self.agent,
                },
            )
            data = message.json()
            text = _text_of(data)
            if not text:
                raise UpstreamError("opencode serve returned no text reply", status_code=502)
            info = data.get("info", {}) if isinstance(data, dict) else {}
            return {"text": text, "model": str(info.get("modelID") or model)}
        except UpstreamError:
            raise
        except Exception as exc:
            try:
                import httpx as _httpx

                is_timeout = isinstance(exc, _httpx.TimeoutException)
            except ImportError:
                is_timeout = "timeout" in type(exc).__name__.lower()
            raise UpstreamError(
                f"opencode serve call failed "
                f"({type(exc).__name__}): {str(exc)[:200]}",
                status_code=504 if is_timeout else 502,
            ) from exc
        finally:
            if own_client:
                client.close()

    async def stream(
        self, payload: dict[str, Any], *, session_id: str = ""
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield one OpenAI chat-completion-chunk-shaped event, then done."""
        result = await self.complete(payload)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": result["model"],
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": result["text"]},
                    "finish_reason": "stop",
                }
            ],
        }
