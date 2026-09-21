"""Authenticated Anthropic-compatible HTTP surface for model backends."""

import asyncio
import hmac
import json
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from ._anthropic import (
    AnthropicStreamTranslator,
    anthropic_to_codex,
    codex_events_to_anthropic,
)
from ._codex import CodexBackend
from ._drain import DEFAULT_DRAIN_TIMEOUT_S
from ._errors import GatewayError, UpstreamError
from ._health import public_upstream_url
from ._identity import GatewayIdentity, gateway_identity
from ._opencode import OpenCodeBackend
from ._inference import (
    InferenceBackend,
    InferenceDrainTimeout,
    InferenceMemberQuiesceTimeout,
    InferenceMemberResumeError,
    estimate_input_tokens,
)
from ._secrets import resolve_gateway_key


def _build_uvicorn_server(
    app: Any, *, close_admission_on_shutdown: bool = True, **kwargs: Any
) -> Any:
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
            if close_admission_on_shutdown and isinstance(backend, InferenceBackend):
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


def _codex_responses_payload(body: dict[str, Any]) -> dict[str, Any]:
    """Normalize public Responses shorthand for the Codex transport."""
    value = body.get("input")
    if isinstance(value, str):
        value = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": value}],
            }
        ]
    elif isinstance(value, list):
        normalized = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("content"), str):
                content_type = (
                    "output_text"
                    if item.get("role") == "assistant"
                    else "input_text"
                )
                item = {
                    **item,
                    "content": [{"type": content_type, "text": item["content"]}],
                }
            normalized.append(item)
        value = normalized
    return {**body, "input": value, "stream": True, "store": False}


def _codex_failure_message(event: dict[str, Any]) -> str:
    """The upstream wording carried by a Codex failure event.

    ``error`` events put the message at the top level, ``response.failed``
    events nest it under ``response.error`` — the same two shapes
    :meth:`~._anthropic.AnthropicStreamTranslator.translate` reads before it
    raises, so a non-streaming Responses client hears the upstream's refusal
    instead of a generic gateway failure.
    """
    message = event.get("message")
    if not message:
        response = event.get("response")
        if isinstance(response, dict):
            error = response.get("error")
            if isinstance(error, dict):
                message = error.get("message")
    return str(message or "Codex response failed")


def _openai_sse_error(exc: GatewayError) -> str:
    """The ``error`` frame an OpenAI-protocol stream carries for a failure.

    An :class:`UpstreamError` knows its own status and error type, and the
    non-streaming branch of the same route relays both, so a stream that
    fails on 401 or 429 must not reach the client as a generic 503.
    """
    if isinstance(exc, UpstreamError):
        error = _openai_error(str(exc), exc.error_type, exc.status_code)
    else:
        error = _openai_error(str(exc), "api_error", 503)
    return f"event: error\ndata: {json.dumps(error, separators=(',', ':'))}\n\n"


def create_app(
    backend: CodexBackend | InferenceBackend | OpenCodeBackend,
    *,
    api_key: str | None = None,
    identity: GatewayIdentity | None = None,
) -> Any:
    """Create the FastAPI app without importing server dependencies at import time.

    Three kinds of backend, one surface. A :class:`CodexBackend` has
    ``/v1/messages`` translated to the Codex Responses protocol; an
    :class:`InferenceBackend` has it relayed verbatim (after the system hoist)
    to a pool of inference upstreams that speak BOTH protocols, and
    additionally relays ``POST /v1/chat/completions`` and ``POST
    /v1/responses`` untouched (the OpenAI protocol, for Codex — no hoist, no
    translation) plus ``GET /v1/*`` so ``/v1/models`` and the like reach the
    upstream as they did through the hoist proxy. An :class:`OpenCodeBackend`
    serves ``POST /v1/chat/completions`` by driving a local ``opencode
    serve`` harness (the app identity Zen free SKUs demand) and answering in
    the OpenAI envelope. Authentication, ``/health`` and
    ``/v1/messages/count_tokens`` are the same for all three.
    """
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:
        raise RuntimeError("Gateway server requires scitex-genai[gateway]") from exc

    expected_key = api_key or resolve_gateway_key().value
    process_identity = identity or gateway_identity()

    relaying = isinstance(backend, InferenceBackend)
    opencode = isinstance(backend, OpenCodeBackend)

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
        if opencode:
            assert isinstance(backend, OpenCodeBackend)
            return {
                "status": "ok",
                "provider": "opencode-serve",
                "serve_url": backend.serve_url,
            }
        if relaying:
            members = backend.pool.status()
            if not backend.active_health_probe:
                drain = await backend.pool.drain_state()
                for member in members:
                    member["url"] = public_upstream_url(member["url"])
                status = {
                    "status": "draining" if drain.draining else "ok",
                    "ready": drain.ready,
                    "provider": backend.provider,
                    "health_strategy": backend.health_strategy,
                    "upstreams": [
                        public_upstream_url(upstream.base_url)
                        for upstream in backend.pool.upstreams
                    ],
                    "members": members,
                    "active_members": sum(member["active"] for member in members),
                    "in_flight": drain.in_flight,
                    "queued": drain.queued,
                    "held": sum(member.get("held", 0) for member in members),
                    "draining": drain.draining,
                    "cache_admission": backend.cache_admission.snapshot(),
                    "continuation_qos": backend.continuation_qos.snapshot(),
                    "request_lifecycle": backend.request_health_snapshot(),
                    "cache_report": {
                        "mode": (
                            "sglang-openai-enabled"
                            if backend.cache_report_enabled
                            else "disabled"
                        )
                    },
                    "external": backend.health_status(),
                    "gateway": process_identity.as_dict(),
                }
                if any("token_capacity" in member for member in members):
                    status["input_tokens_in_flight"] = sum(
                        member["input_tokens_in_flight"] for member in members
                    )
                    status["input_tokens_queued"] = sum(
                        member["input_tokens_queued"] for member in members
                    )
                return status if drain.ready else JSONResponse(status, status_code=503)
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
            drain = await backend.pool.drain_state()
            status = {
                "status": (
                    "draining"
                    if drain.draining
                    else ("ok" if active_members else "degraded")
                ),
                "ready": drain.ready and bool(active_members),
                "provider": backend.provider,
                "health_strategy": backend.health_strategy,
                "upstreams": [
                    public_upstream_url(upstream.base_url)
                    for upstream in backend.pool.upstreams
                ],
                "members": members,
                "configured_members": len(members),
                "admission_eligible_members": admission_eligible_members,
                "reachable_members": reachable_members,
                "ready_members": ready_members,
                "active_members": active_members,
                "in_flight": drain.in_flight,
                "queued": drain.queued,
                "held": sum(member.get("held", 0) for member in members),
                "draining": drain.draining,
                "cache_admission": backend.cache_admission.snapshot(),
                "continuation_qos": backend.continuation_qos.snapshot(),
                "request_lifecycle": backend.request_health_snapshot(),
                "cache_report": {
                    "mode": (
                        "sglang-openai-enabled"
                        if backend.cache_report_enabled
                        else "disabled"
                    )
                },
                "gateway": process_identity.as_dict(),
            }
            if not active_members and not drain.draining:
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
            return status if status["ready"] else JSONResponse(status, status_code=503)
        if opencode:
            assert isinstance(backend, OpenCodeBackend)
            return {
                "status": "ok",
                "provider": "opencode-serve",
                "serve_url": backend.serve_url,
                "model": backend.model,
                "gateway": process_identity.as_dict(),
            }
        return {
            "status": "ok",
            "provider": "openai-codex",
            "accounts": len(backend.pool.accounts),
            "gateway": process_identity.as_dict(),
        }

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request) -> Any:
        if not authorized(request):
            return JSONResponse(
                _anthropic_error("Invalid API key", "authentication_error"), 401
            )
        body = await request.json()
        return {"input_tokens": _estimate_tokens(body)}

    if opencode:
        assert isinstance(backend, OpenCodeBackend)

        @app.post("/v1/chat/completions")
        async def opencode_chat_completions(request: Request) -> Any:
            """OpenAI chat completions through the local opencode harness."""
            if not authorized(request):
                return JSONResponse(
                    _openai_error("Invalid API key", "authentication_error", 401),
                    401,
                )
            try:
                body = await request.json()
                if not isinstance(body, dict):
                    raise GatewayError("Request body must be a JSON object")
                if not isinstance(body.get("model"), str) or not body["model"]:
                    raise GatewayError("Chat completions request requires a model")
            except (ValueError, GatewayError) as exc:
                return JSONResponse(
                    _openai_error(str(exc), "invalid_request_error", 400), 400
                )
            try:
                result = await backend.complete(body)
            except UpstreamError as exc:
                return JSONResponse(
                    _openai_error(str(exc), exc.error_type, exc.status_code),
                    exc.status_code,
                )
            except GatewayError as exc:
                return JSONResponse(_openai_error(str(exc), "api_error", 503), 503)
            import time as _time
            import uuid as _uuid

            completion_id = _uuid.uuid4().hex[:12]
            return {
                "id": f"chatcmpl-{completion_id}",
                "object": "chat.completion",
                "created": int(_time.time()),
                "model": result["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result["text"]},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

        @app.get("/v1/models")
        async def opencode_models(request: Request) -> Any:
            if not authorized(request):
                return JSONResponse(
                    _openai_error("Invalid API key", "authentication_error", 401),
                    401,
                )
            return {
                "object": "list",
                "data": [
                    {
                        "id": "muse-spark-1.3-contributor-free",
                        "object": "model",
                        "owned_by": "opencode",
                    }
                ],
            }

        return app

    if relaying:

        @app.get("/admin/status")
        async def operator_status(request: Request) -> Any:
            """Authenticated, payload-free gateway admission/relay metrics."""
            if not authorized(request):
                return JSONResponse(
                    _openai_error("Invalid API key", "authentication_error", 401),
                    401,
                )
            status = await backend.observability_snapshot()
            status["gateway"] = process_identity.as_dict()
            return status

        @app.post("/admin/drain")
        async def begin_drain(
            request: Request, timeout_s: float = DEFAULT_DRAIN_TIMEOUT_S
        ) -> Any:
            if not authorized(request):
                return JSONResponse(
                    _openai_error("Invalid API key", "authentication_error", 401),
                    401,
                )
            if not math.isfinite(timeout_s) or timeout_s <= 0:
                return JSONResponse(
                    _openai_error(
                        "drain timeout_s must be finite and > 0",
                        "invalid_request_error",
                        400,
                    ),
                    400,
                )
            try:
                state = await backend.pool.begin_drain(timeout_s)
            except InferenceDrainTimeout as exc:
                return JSONResponse(
                    {
                        **_openai_error(str(exc), "drain_timeout", 409),
                        **exc.state.as_dict(),
                    },
                    409,
                )
            return state.as_dict()

        @app.post("/admin/resume")
        async def cancel_drain(request: Request) -> Any:
            if not authorized(request):
                return JSONResponse(
                    _openai_error("Invalid API key", "authentication_error", 401),
                    401,
                )
            await backend.pool.resume()
            return (await backend.pool.drain_state()).as_dict()

        @app.post("/admin/members/{alias}/quiesce")
        async def quiesce_member(
            alias: str, request: Request, timeout_s: float = DEFAULT_DRAIN_TIMEOUT_S
        ) -> Any:
            if not authorized(request):
                return JSONResponse(
                    _openai_error("Invalid API key", "authentication_error", 401),
                    401,
                )
            if not math.isfinite(timeout_s) or timeout_s <= 0:
                return JSONResponse(
                    _openai_error(
                        "quiesce timeout_s must be finite and > 0",
                        "invalid_request_error",
                        400,
                    ),
                    400,
                )
            try:
                state = await backend.quiesce_member(alias, timeout_s)
            except KeyError:
                return JSONResponse(
                    _openai_error(
                        f"Unknown inference member: {alias}",
                        "member_not_found",
                        404,
                    ),
                    404,
                )
            except InferenceMemberQuiesceTimeout as exc:
                return JSONResponse(
                    {
                        **_openai_error(str(exc), "member_quiesce_timeout", 409),
                        **exc.state.as_dict(),
                    },
                    409,
                )
            return state.as_dict()

        @app.post("/admin/members/{alias}/resume")
        async def resume_member(alias: str, request: Request) -> Any:
            if not authorized(request):
                return JSONResponse(
                    _openai_error("Invalid API key", "authentication_error", 401),
                    401,
                )
            try:
                state = await backend.resume_member(alias)
            except KeyError:
                return JSONResponse(
                    _openai_error(
                        f"Unknown inference member: {alias}",
                        "member_not_found",
                        404,
                    ),
                    404,
                )
            except InferenceMemberResumeError as exc:
                return JSONResponse(
                    _openai_error(str(exc), "member_resume_validation_failed", 409),
                    409,
                )
            return state.as_dict()

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

    @app.post("/v1/responses")
    async def responses(request: Request) -> Any:
        if not authorized(request):
            return JSONResponse(
                _openai_error("Invalid API key", "authentication_error", 401),
                401,
            )
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise GatewayError("Request body must be a JSON object")
            if not isinstance(body.get("model"), str) or not body["model"]:
                raise GatewayError("Responses request requires a model")
            if "input" not in body:
                raise GatewayError("Responses request requires input")
            requested_stream = body.get("stream") is True
            payload = _codex_responses_payload(body)
            session_id = _session_id(request, body)
        except (ValueError, GatewayError) as exc:
            return JSONResponse(
                _openai_error(str(exc), "invalid_request_error", 400), 400
            )

        if requested_stream:

            async def stream_response() -> AsyncIterator[str]:
                try:
                    async for event in backend.stream(
                        payload, session_id=session_id
                    ):
                        event_type = str(event.get("type", "message"))
                        data = json.dumps(event, separators=(",", ":"))
                        yield f"event: {event_type}\ndata: {data}\n\n"
                except GatewayError as exc:
                    yield _openai_sse_error(exc)

            return StreamingResponse(stream_response(), media_type="text/event-stream")

        try:
            completed = None
            async for event in backend.stream(payload, session_id=session_id):
                event_type = event.get("type")
                if event_type == "response.completed":
                    candidate = event.get("response")
                    if isinstance(candidate, dict):
                        completed = candidate
                elif event_type in {"response.failed", "error"}:
                    raise GatewayError(_codex_failure_message(event))
            if completed is None:
                raise GatewayError("Codex response ended without response.completed")
            return completed
        except UpstreamError as exc:
            return JSONResponse(
                _openai_error(str(exc), exc.error_type, exc.status_code),
                exc.status_code,
            )
        except GatewayError as exc:
            return JSONResponse(_openai_error(str(exc), "api_error", 503), 503)

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
