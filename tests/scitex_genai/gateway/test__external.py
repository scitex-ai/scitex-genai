"""Adversarial tests for the paid-provider outbound boundary."""

from __future__ import annotations

import json

import httpx
import pytest

pytest.importorskip("fastapi")

from scitex_genai.gateway._external import (
    ExternalProviderBackend,
    ExternalProviderPolicy,
)
from scitex_genai.gateway._inference import InferenceUpstreamPool
from scitex_genai.gateway._server import create_app


def _policy(**overrides):
    values = {
        "provider": "deepseek",
        "upstream_api_key": "vendor-secret",
        "canonical_model": "deepseek-flash",
        "model_aliases": ("deepseek-v4-flash",),
        "max_tokens_per_request": 100,
        "max_requests_per_run": 2,
        "max_input_tokens_per_run": 10_000,
        "max_output_tokens_per_run": 150,
        "max_total_tokens_per_run": 10_000,
        "input_usd_per_million_tokens": 0.1,
        "output_usd_per_million_tokens": 0.2,
    }
    values.update(overrides)
    return ExternalProviderPolicy(**values)


def _body(model="deepseek-flash", *, stream=False, max_tokens=20):
    return {
        "model": model,
        "messages": [{"role": "user", "content": "secret prompt"}],
        "max_tokens": max_tokens,
        "stream": stream,
    }


def _backend(upstream, **policy):
    return ExternalProviderBackend(
        InferenceUpstreamPool.from_urls([upstream.url]),
        policy=_policy(**policy),
    )


@pytest.mark.asyncio
async def test_alias_is_normalized_and_credentials_are_separated(upstream_factory):
    # Arrange
    upstream = upstream_factory(
        chunks=(
            json.dumps(
                {
                    "model": "deepseek-flash",
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                    "choices": [],
                }
            ).encode(),
        )
    )
    app = create_app(_backend(upstream), api_key="local-gateway-secret")
    # Act
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_body("deepseek-v4-flash"),
            headers={
                "authorization": "Bearer local-gateway-secret",
                "x-scitex-run-id": "run-containing-private-identity",
            },
        )
    sent = upstream.requests[0]
    # Assert
    assert (
        response.status_code,
        json.loads(sent["body"])["model"],
        sent["headers"]["authorization"],
        "local-gateway-secret" in str(sent),
        "run-containing-private-identity" in str(sent),
    ) == (200, "deepseek-flash", "Bearer vendor-secret", False, False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "forbidden", ["deepseek-v4-pro", "deepseek-pro", "deepseek-reasoner", "DeepSeek-Flash"]
)
async def test_explicit_model_switch_cannot_bypass_firewall(
    upstream_factory, forbidden
):
    # Arrange
    upstream = upstream_factory()
    app = create_app(_backend(upstream), api_key="local")
    # Act
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_body(forbidden),
            headers={"authorization": "Bearer local"},
        )
    # Assert
    assert (
        response.status_code,
        response.json()["error"]["type"],
        upstream.requests,
    ) == (400, "model_policy", [])


@pytest.mark.asyncio
async def test_model_discovery_is_synthetic_and_does_not_expose_vendor(upstream_factory):
    # Arrange
    upstream = upstream_factory()
    app = create_app(_backend(upstream), api_key="local")
    # Act
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
    ) as client:
        response = await client.get(
            "/v1/models", headers={"authorization": "Bearer local"}
        )
    # Assert
    assert (
        response.status_code,
        [row["id"] for row in response.json()["data"]],
        upstream.requests,
    ) == (200, ["deepseek-flash"], [])


@pytest.mark.asyncio
async def test_usage_and_response_reported_model_are_audited_without_payload(
    upstream_factory,
):
    # Arrange
    upstream = upstream_factory(
        chunks=(
            b'{"model":"unexpected-provider-label","usage":'
            b'{"prompt_tokens":7,"completion_tokens":3},"choices":[]}',
        )
    )
    backend = _backend(upstream)
    app = create_app(backend, api_key="local")
    # Act
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_body(),
            headers={"authorization": "Bearer local", "x-scitex-run-id": "private"},
        )
        health = (await client.get("/health")).json()["external"]
    # Assert
    assert (
        response.status_code,
        health["last_reported_model"],
        health["usage"]["input_tokens"],
        health["usage"]["output_tokens"],
        health["usage"]["reported_model_mismatches"],
        health["usage"]["estimated_cost_usd"],
        "secret prompt" in json.dumps(health),
        "private" in json.dumps(health),
    ) == (200, "unexpected-provider-label", 7, 3, 1, 0.0000013, False, False)


@pytest.mark.asyncio
async def test_request_and_output_budgets_reject_before_upstream(upstream_factory):
    # Arrange
    response_body = b'{"model":"deepseek-flash","usage":{"prompt_tokens":1,"completion_tokens":1}}'
    upstream = upstream_factory(chunks=(response_body,))
    app = create_app(
        _backend(
            upstream,
            max_requests_per_run=1,
            max_output_tokens_per_run=30,
            max_total_tokens_per_run=10_000,
        ),
        api_key="local",
    )
    headers = {"authorization": "Bearer local", "x-scitex-run-id": "same-run"}
    # Act
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
    ) as client:
        first = await client.post("/v1/chat/completions", json=_body(), headers=headers)
        second = await client.post("/v1/chat/completions", json=_body(), headers=headers)
        oversized = await client.post(
            "/v1/chat/completions", json=_body(max_tokens=101), headers=headers
        )
    # Assert
    assert (
        first.status_code,
        second.status_code,
        oversized.status_code,
        len(upstream.requests),
    ) == (200, 429, 429, 1)


@pytest.mark.asyncio
async def test_cost_budget_rejects_before_upstream(upstream_factory):
    # Arrange
    upstream = upstream_factory()
    app = create_app(
        _backend(
            upstream,
            input_usd_per_million_tokens=1.0,
            output_usd_per_million_tokens=1.0,
            max_estimated_usd_per_run=0.000001,
        ),
        api_key="local",
    )
    # Act
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_body(),
            headers={"authorization": "Bearer local"},
        )
    # Assert
    assert (
        response.status_code,
        response.json()["error"]["type"],
        upstream.requests,
    ) == (429, "budget_exceeded", [])


def test_cost_ceiling_without_prices_is_refused():
    # Arrange
    values = {
        "provider": "deepseek",
        "upstream_api_key": "vendor-secret",
        "max_estimated_usd_per_run": 1.0,
    }
    # Act
    # Assert
    with pytest.raises(ValueError, match="requires a non-zero"):
        ExternalProviderPolicy(**values)


@pytest.mark.asyncio
async def test_stream_usage_is_collected_and_include_usage_is_forced(upstream_factory):
    # Arrange
    upstream = upstream_factory(
        content_type="text/event-stream",
        chunks=(
            b'data: {"model":"deepseek-flash","choices":[]}\n\n',
            b'data: {"model":"deepseek-flash","usage":{"prompt_tokens":9,"completion_tokens":4}}\n\n',
            b"data: [DONE]\n\n",
        ),
    )
    backend = _backend(upstream)
    app = create_app(backend, api_key="local")
    # Act
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_body(stream=True),
            headers={"authorization": "Bearer local"},
        )
    sent = json.loads(upstream.requests[0]["body"])
    # Assert
    assert (
        response.status_code,
        sent["stream_options"],
        backend.usage.total.input_tokens,
        backend.usage.total.output_tokens,
    ) == (200, {"include_usage": True}, 9, 4)


@pytest.mark.asyncio
async def test_partial_stream_usage_keeps_missing_output_reservation(upstream_factory):
    # Arrange
    upstream = upstream_factory(
        content_type="text/event-stream",
        chunks=(
            b'data: {"model":"deepseek-flash","usage":{"prompt_tokens":9}}\n\n',
            b"data: [DONE]\n\n",
        ),
    )
    backend = _backend(upstream, max_output_tokens_per_run=30)
    app = create_app(backend, api_key="local")
    headers = {"authorization": "Bearer local", "x-scitex-run-id": "same-run"}
    # Act
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
    ) as client:
        first = await client.post(
            "/v1/chat/completions", json=_body(stream=True), headers=headers
        )
        second = await client.post(
            "/v1/chat/completions", json=_body(stream=True), headers=headers
        )
    # Assert
    assert (
        first.status_code,
        second.status_code,
        backend.usage.total.input_tokens,
        backend.usage.total.output_tokens,
        backend.usage.total.responses_without_usage,
        len(upstream.requests),
    ) == (200, 429, 9, 20, 1, 1)


@pytest.mark.asyncio
async def test_responses_api_uses_protocol_specific_output_field(upstream_factory):
    # Arrange
    upstream = upstream_factory(
        chunks=(b'{"model":"deepseek-flash","usage":{"input_tokens":2,"output_tokens":1}}',)
    )
    app = create_app(_backend(upstream), api_key="local")
    body = {"model": "deepseek-flash", "input": "hello", "max_output_tokens": 12}
    # Act
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
    ) as client:
        response = await client.post(
            "/v1/responses",
            json=body,
            headers={"authorization": "Bearer local"},
        )
    sent = json.loads(upstream.requests[0]["body"])
    # Assert
    assert (
        response.status_code,
        sent.get("max_output_tokens"),
        "max_tokens" in sent,
        "max_completion_tokens" in sent,
    ) == (200, 12, False, False)


@pytest.mark.asyncio
async def test_responses_api_refuses_legacy_output_field_before_upstream(
    upstream_factory,
):
    # Arrange
    upstream = upstream_factory()
    app = create_app(_backend(upstream), api_key="local")
    body = {"model": "deepseek-flash", "input": "hello", "max_tokens": 99}
    # Act
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
    ) as client:
        response = await client.post(
            "/v1/responses",
            json=body,
            headers={"authorization": "Bearer local"},
        )
    # Assert
    assert (
        response.status_code,
        response.json()["error"]["type"],
        upstream.requests,
    ) == (400, "model_policy", [])


@pytest.mark.asyncio
async def test_anthropic_stream_audits_nested_reported_model(upstream_factory):
    # Arrange
    upstream = upstream_factory(
        content_type="text/event-stream",
        chunks=(
            b'data: {"type":"message_start","message":{"model":"deepseek-v4.1-flash","usage":{"input_tokens":8,"output_tokens":0}}}\n\n',
            b'data: {"type":"message_delta","usage":{"output_tokens":5}}\n\n',
        ),
    )
    backend = _backend(upstream, anthropic_path_prefix="/anthropic")
    app = create_app(backend, api_key="local")
    # Act
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
    ) as client:
        response = await client.post(
            "/v1/messages",
            json=_body(stream=True),
            headers={"authorization": "Bearer local"},
        )
    # Assert
    assert (
        response.status_code,
        upstream.requests[0]["path"],
        backend.usage.last_reported_model,
        backend.usage.total.input_tokens,
        backend.usage.total.output_tokens,
        backend.usage.total.reported_model_mismatches,
    ) == (200, "/anthropic/v1/messages", "deepseek-v4.1-flash", 8, 5, 1)
