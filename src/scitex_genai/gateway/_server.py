"""Authenticated Anthropic-compatible HTTP surface for model backends."""

import asyncio
import hmac
import json
import math
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from ._anthropic import (
    AnthropicStreamTranslator,
    anthropic_to_codex,
    codex_events_to_anthropic,
)
from ._codex import CodexBackend
from ._errors import GatewayError, UpstreamError


def _request_token(request: Any) -> str:
    api_key = request.headers.get("x-api-key", "")
    if api_key:
        return api_key
    authorization = request.headers.get("authorization", "")
    return authorization[7:] if authorization.lower().startswith("bearer ") else ""


def _session_id(request: Any, body: dict[str, Any]) -> str:
    for name in ("session_id", "x-session-id"):
        value = request.headers.get(name, "")
        if value:
            return value
    metadata = body.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("user_id"), str):
        return metadata["user_id"]
    return ""


def _anthropic_error(message: str, error_type: str = "api_error") -> dict[str, Any]:
    return {"type": "error", "error": {"type": error_type, "message": message}}


def _estimate_tokens(body: dict[str, Any]) -> int:
    """Conservative fallback until a Codex tokenizer is exposed."""
    serialized = json.dumps(
        {"system": body.get("system"), "messages": body.get("messages")},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return max(1, math.ceil(len(serialized.encode("utf-8")) / 4))


def create_app(backend: CodexBackend, *, api_key: str | None = None) -> Any:
    """Create the FastAPI app without importing server dependencies at import time."""
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:
        raise RuntimeError("Gateway server requires scitex-genai[gateway]") from exc

    expected_key = api_key or os.getenv("SCITEX_GENAI_GATEWAY_API_KEY", "")
    if not expected_key:
        raise RuntimeError("SCITEX_GENAI_GATEWAY_API_KEY must be set")

    @asynccontextmanager
    async def lifespan(app: Any) -> AsyncIterator[None]:
        async def poll_usage() -> None:
            while True:
                await backend.refresh_usage()
                await asyncio.sleep(60)

        task = asyncio.create_task(poll_usage())
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app = FastAPI(
        title="SciTeX GenAI Gateway",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    def authorized(request: Request) -> bool:
        return hmac.compare_digest(_request_token(request), expected_key)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "provider": "openai-codex",
            "accounts": len(backend.pool.accounts),
        }

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request) -> Any:
        if not authorized(request):
            return JSONResponse(_anthropic_error("Invalid API key", "authentication_error"), 401)
        body = await request.json()
        return {"input_tokens": _estimate_tokens(body)}

    @app.post("/v1/messages")
    async def messages(request: Request) -> Any:
        if not authorized(request):
            return JSONResponse(_anthropic_error("Invalid API key", "authentication_error"), 401)
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise GatewayError("Request body must be a JSON object")
            session_id = _session_id(request, body)
            payload = anthropic_to_codex(body, session_id=session_id)
        except (ValueError, GatewayError) as exc:
            return JSONResponse(_anthropic_error(str(exc), "invalid_request_error"), 400)

        if body.get("stream") is True:

            async def stream_response() -> AsyncIterator[str]:
                translator = AnthropicStreamTranslator(str(body["model"]))
                try:
                    async for event in backend.stream(payload, session_id=session_id):
                        for chunk in translator.translate(event):
                            yield chunk
                except GatewayError as exc:
                    yield f"event: error\ndata: {json.dumps(_anthropic_error(str(exc)))}\n\n"

            return StreamingResponse(stream_response(), media_type="text/event-stream")

        try:
            events = [
                event
                async for event in backend.stream(payload, session_id=session_id)
            ]
            return codex_events_to_anthropic(events, model=str(body["model"]))
        except UpstreamError as exc:
            return JSONResponse(_anthropic_error(str(exc)), exc.status_code)
        except GatewayError as exc:
            return JSONResponse(_anthropic_error(str(exc)), 503)

    return app
