"""Authenticated Anthropic-compatible HTTP surface for model backends."""

import asyncio
import hmac
import json
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
from ._health import public_upstream_url
from ._inference import InferenceBackend, estimate_input_tokens
from ._secrets import resolve_gateway_key


def _build_uvicorn_server(app: Any, **kwargs: Any) -> Any:
    """Build uvicorn with inference admission closed before request draining.

    Uvicorn normally waits for active request tasks before running the app's
    lifespan shutdown. Capacity waiters are active tasks, so lifespan alone
    cannot wake them. Closing admission at the start of ``shutdown`` makes
    those waiters return 503 before uvicorn waits, while already admitted
    streams remain request tasks and retain uvicorn's normal graceful drain.
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Gateway server requires scitex-genai[gateway]") from exc

    backend = app.state.scitex_backend

    class AdmissionAwareServer(uvicorn.Server):
        async def shutdown(self, sockets=None) -> None:
            if isinstance(backend, InferenceBackend):
                await backend.close()
            await super().shutdown(sockets)

    return AdmissionAwareServer(uvicorn.Config(app, **kwargs))


def run_uvicorn(app: Any, **kwargs: Any) -> None:
    """Run the admission-aware uvicorn server used by the console command."""
    _build_uvicorn_server(app, **kwargs).run()


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


def _openai_error(message: str, error_type: str, status: int) -> dict[str, Any]:
    """The OpenAI error envelope, for the routes an OpenAI-protocol client calls.

    A gateway-generated refusal on ``/v1/chat/completions`` or
    ``/v1/responses`` wrapped in the Anthropic envelope reaches Codex as an
    opaque deserialisation failure instead of "Invalid API key" or the
    worded upstream refusal. Upstream errors are relayed verbatim either way.
    """
    return {"error": {"message": message, "type": error_type, "code": status}}


def _error_for(path: str, message: str, error_type: str, status: int) -> dict[str, Any]:
    if path.startswith("/v1/messages"):
        return _anthropic_error(message, error_type)
    return _openai_error(message, error_type, status)


def _estimate_tokens(body: dict[str, Any]) -> int:
    """Conservative fallback until a Codex tokenizer is exposed."""
    serialized = json.dumps(
        {"system": body.get("system"), "messages": body.get("messages")},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return estimate_input_tokens(serialized.encode("utf-8"))


def create_app(
    backend: CodexBackend | InferenceBackend, *, api_key: str | None = None
) -> Any:
    """Create the FastAPI app without importing server dependencies at import time.

    Two kinds of backend, one surface. A :class:`CodexBackend` has
    ``/v1/messages`` translated to the Codex Responses protocol; an
    :class:`InferenceBackend` has it relayed verbatim (after the system hoist)
    to a pool of inference upstreams that speak BOTH protocols, and
    additionally relays ``POST /v1/chat/completions`` and ``POST
    /v1/responses`` untouched (the OpenAI protocol, for Codex — no hoist, no
    translation) plus ``GET /v1/*`` so ``/v1/models`` and the like reach the
    upstream as they did through the hoist proxy. Authentication, ``/health``
    and ``/v1/messages/count_tokens`` are the same for both.
    """
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:
        raise RuntimeError("Gateway server requires scitex-genai[gateway]") from exc

    expected_key = api_key or resolve_gateway_key().value

    relaying = isinstance(backend, InferenceBackend)

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

    @asynccontextmanager
    async def inference_lifespan(app: Any) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await backend.close()

    app = FastAPI(
        title="SciTeX GenAI Gateway",
        docs_url=None,
        redoc_url=None,
        # An inference pool has no quota to poll; only Codex accounts do.
        lifespan=inference_lifespan if relaying else lifespan,
    )
    app.state.scitex_backend = backend

    def authorized(request: Request) -> bool:
        return hmac.compare_digest(_request_token(request), expected_key)

    @app.get("/health")
    async def health() -> Any:
        if relaying:
            members = backend.pool.status()
            if not backend.active_health_probe:
                for member in members:
                    member["url"] = public_upstream_url(member["url"])
                status = {
                    "status": "ok",
                    "provider": backend.provider,
                    "health_strategy": backend.health_strategy,
                    "upstreams": [
                        public_upstream_url(upstream.alias)
                        for upstream in backend.pool.upstreams
                    ],
                    "members": members,
                    "active_members": sum(member["active"] for member in members),
                    "in_flight": sum(member["in_flight"] for member in members),
                    "queued": sum(member["queued"] for member in members),
                    "draining": backend.pool.draining,
                    "cache_admission": backend.cache_admission.snapshot(),
                    "continuation_qos": backend.continuation_qos.snapshot(),
                    "cache_report": {
                        "mode": (
                            "sglang-openai-enabled"
                            if backend.cache_report_enabled
                            else "disabled"
                        )
                    },
                    "external": backend.health_status(),
                }
                if any("token_capacity" in member for member in members):
                    status["input_tokens_in_flight"] = sum(
                        member["input_tokens_in_flight"] for member in members
                    )
                    status["input_tokens_queued"] = sum(
                        member["input_tokens_queued"] for member in members
                    )
                return status
            reachability = await backend.probe_upstreams()
            members = backend.pool.status()
            for member, observed in zip(members, reachability, strict=True):
                admission_eligible = member["active"]
                member["url"] = public_upstream_url(member["url"])
                member["configured"] = True
                member["admission_eligible"] = admission_eligible
                member["reachable"] = observed.reachable
                member["ready"] = observed.readiness
                member["active"] = admission_eligible and observed.readiness
                member["reachability"] = observed.as_dict()
            active_members = sum(member["active"] for member in members)
            reachable_members = sum(member["reachable"] for member in members)
            ready_members = sum(member["ready"] for member in members)
            admission_eligible_members = sum(
                member["admission_eligible"] for member in members
            )
            draining = backend.pool.draining
            status = {
                "status": (
                    "draining" if draining else ("ok" if active_members else "degraded")
                ),
                "provider": backend.provider,
                "health_strategy": backend.health_strategy,
                "upstreams": [
                    public_upstream_url(upstream.alias)
                    for upstream in backend.pool.upstreams
                ],
                "members": members,
                "configured_members": len(members),
                "admission_eligible_members": admission_eligible_members,
                "reachable_members": reachable_members,
                "ready_members": ready_members,
                "active_members": active_members,
                "in_flight": sum(member["in_flight"] for member in members),
                "queued": sum(member["queued"] for member in members),
                "draining": draining,
                "cache_admission": backend.cache_admission.snapshot(),
                "continuation_qos": backend.continuation_qos.snapshot(),
                "cache_report": {
                    "mode": (
                        "sglang-openai-enabled"
                        if backend.cache_report_enabled
                        else "disabled"
                    )
                },
            }
            if not active_members and not draining:
                status["reason"] = (
                    "no_inference_upstream_reachable"
                    if not reachable_members
                    else "reachable_upstreams_not_admission_eligible"
                )
            if any("token_capacity" in member for member in members):
                status["input_tokens_in_flight"] = sum(
                    member["input_tokens_in_flight"] for member in members
                )
                status["input_tokens_queued"] = sum(
                    member["input_tokens_queued"] for member in members
                )
            return (
                status
                if active_members or draining
                else JSONResponse(status, status_code=503)
            )
        return {
            "status": "ok",
            "provider": "openai-codex",
            "accounts": len(backend.pool.accounts),
        }

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request) -> Any:
        if not authorized(request):
            return JSONResponse(
                _anthropic_error("Invalid API key", "authentication_error"), 401
            )
        body = await request.json()
        return {"input_tokens": _estimate_tokens(body)}

    if relaying:

        @app.post("/admin/drain")
        async def begin_drain(request: Request) -> Any:
            if not authorized(request):
                return JSONResponse(
                    _openai_error("Invalid API key", "authentication_error", 401),
                    401,
                )
            await backend.pool.close()
            members = backend.pool.status()
            return {
                "draining": True,
                "in_flight": sum(member["in_flight"] for member in members),
                "queued": sum(member["queued"] for member in members),
            }

        @app.post("/admin/resume")
        async def cancel_drain(request: Request) -> Any:
            if not authorized(request):
                return JSONResponse(
                    _openai_error("Invalid API key", "authentication_error", 401),
                    401,
                )
            await backend.pool.resume()
            return {"draining": False}

        async def relay(request: Request) -> Any:
            path = request.url.path
            if not authorized(request):
                return JSONResponse(
                    _error_for(path, "Invalid API key", "authentication_error", 401),
                    401,
                )
            # The query string travels with the path: Claude Code posts to
            # ``/v1/messages?beta=true`` and the upstream sees exactly that.
            target = path
            if request.url.query:
                target = f"{target}?{request.url.query}"
            body = await request.body()
            try:
                relayed = await backend.relay(
                    request.method,
                    target,
                    body=body or None,
                    headers=request.headers,
                    client_disconnected=request.is_disconnected,
                )
            except UpstreamError as exc:
                return JSONResponse(
                    _error_for(path, str(exc), exc.error_type, exc.status_code),
                    exc.status_code,
                )
            return StreamingResponse(
                relayed.body,
                status_code=relayed.status_code,
                media_type=relayed.content_type,
                headers=relayed.feedback_headers,
            )

        @app.post("/v1/messages")
        async def relay_messages(request: Request) -> Any:
            return await relay(request)

        # The OpenAI protocol, for Codex (2026-09-05). vLLM serves both
        # routes natively, so the body passes through untouched — no
        # translation, and no hoist (see ``_inference.hoists_on``). Explicit
        # routes rather than a POST catch-all: every other POST path stays a
        # 405 on purpose.
        @app.post("/v1/chat/completions")
        async def relay_chat_completions(request: Request) -> Any:
            return await relay(request)

        @app.post("/v1/responses")
        async def relay_responses(request: Request) -> Any:
            return await relay(request)

        @app.get("/v1/{path:path}")
        async def relay_get(request: Request, path: str) -> Any:
            models_payload = getattr(backend, "models_payload", None)
            if path == "models" and models_payload is not None:
                if not authorized(request):
                    return JSONResponse(
                        _openai_error("Invalid API key", "authentication_error", 401),
                        401,
                    )
                return JSONResponse(models_payload())
            return await relay(request)

        return app

    @app.post("/v1/messages")
    async def messages(request: Request) -> Any:
        if not authorized(request):
            return JSONResponse(
                _anthropic_error("Invalid API key", "authentication_error"), 401
            )
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise GatewayError("Request body must be a JSON object")
            session_id = _session_id(request, body)
            payload = anthropic_to_codex(body, session_id=session_id)
        except (ValueError, GatewayError) as exc:
            return JSONResponse(
                _anthropic_error(str(exc), "invalid_request_error"), 400
            )

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
                event async for event in backend.stream(payload, session_id=session_id)
            ]
            return codex_events_to_anthropic(events, model=str(body["model"]))
        except UpstreamError as exc:
            return JSONResponse(_anthropic_error(str(exc)), exc.status_code)
        except GatewayError as exc:
            return JSONResponse(_anthropic_error(str(exc)), 503)

    return app
