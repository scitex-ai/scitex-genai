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
    assert response.status_code == 200
    sent = upstream.requests[0]
    assert json.loads(sent["body"])["model"] == "deepseek-flash"
    assert sent["headers"]["authorization"] == "Bearer vendor-secret"
    assert "local-gateway-secret" not in str(sent)
    assert "run-containing-private-identity" not in str(sent)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "forbidden", ["deepseek-v4-pro", "deepseek-pro", "deepseek-reasoner", "DeepSeek-Flash"]
)
async def test_explicit_model_switch_cannot_bypass_firewall(
    upstream_factory, forbidden
):
    upstream = upstream_factory()
    app = create_app(_backend(upstream), api_key="local")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_body(forbidden),
            headers={"authorization": "Bearer local"},
        )
    assert (response.status_code, response.json()["error"]["type"]) == (
        400,
        "model_policy",
    )
    assert upstream.requests == []


@pytest.mark.asyncio
async def test_model_discovery_is_synthetic_and_does_not_expose_vendor(upstream_factory):
    upstream = upstream_factory()
    app = create_app(_backend(upstream), api_key="local")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
    ) as client:
        response = await client.get(
            "/v1/models", headers={"authorization": "Bearer local"}
        )
    assert response.status_code == 200
    assert [row["id"] for row in response.json()["data"]] == ["deepseek-flash"]
    assert upstream.requests == []


@pytest.mark.asyncio
async def test_usage_and_response_reported_model_are_audited_without_payload(
    upstream_factory,
):
    upstream = upstream_factory(
        chunks=(
            b'{"model":"unexpected-provider-label","usage":'
            b'{"prompt_tokens":7,"completion_tokens":3},"choices":[]}',
        )
    )
    backend = _backend(upstream)
    app = create_app(backend, api_key="local")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_body(),
            headers={"authorization": "Bearer local", "x-scitex-run-id": "private"},
        )
        health = (await client.get("/health")).json()["external"]
    assert response.status_code == 200
    assert health["last_reported_model"] == "unexpected-provider-label"
    assert health["usage"]["input_tokens"] == 7
    assert health["usage"]["output_tokens"] == 3
    assert health["usage"]["reported_model_mismatches"] == 1
    assert health["usage"]["estimated_cost_usd"] == 0.0000013
    assert "secret prompt" not in json.dumps(health)
    assert "private" not in json.dumps(health)


@pytest.mark.asyncio
async def test_request_and_output_budgets_reject_before_upstream(upstream_factory):
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
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
    ) as client:
        first = await client.post("/v1/chat/completions", json=_body(), headers=headers)
        second = await client.post("/v1/chat/completions", json=_body(), headers=headers)
        oversized = await client.post(
            "/v1/chat/completions", json=_body(max_tokens=101), headers=headers
        )
    assert first.status_code == 200
    assert second.status_code == 429
    assert oversized.status_code == 429
    assert len(upstream.requests) == 1


@pytest.mark.asyncio
async def test_cost_budget_rejects_before_upstream(upstream_factory):
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
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_body(),
            headers={"authorization": "Bearer local"},
        )
    assert (response.status_code, response.json()["error"]["type"]) == (
        429,
        "budget_exceeded",
    )
    assert upstream.requests == []


@pytest.mark.asyncio
async def test_stream_usage_is_collected_and_include_usage_is_forced(upstream_factory):
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
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_body(stream=True),
            headers={"authorization": "Bearer local"},
        )
    sent = json.loads(upstream.requests[0]["body"])
    assert response.status_code == 200
    assert sent["stream_options"] == {"include_usage": True}
    assert backend.usage.total.input_tokens == 9
    assert backend.usage.total.output_tokens == 4
