import errno
import json

import httpx
import pytest

from scitex_genai.availability import probe_provider_key, provider_route


@pytest.mark.parametrize("code", [400, 401, 402, 403, 404, 429, 500, 503])
def test_native_provider_codes_survive_without_secret_or_body(code):
    token = "fake-sensitive-key"
    body = {
        "error": {
            "message": f"insufficient credits; bearer={token}",
            "code": "upstream-original",
        }
    }
    result = probe_provider_key(
        "command-code",
        "meta/muse",
        token,
        transport=httpx.MockTransport(lambda request: httpx.Response(code, json=body)),
    )
    wire = result.to_dict()
    assert wire["check"]["cause"]["kind"] == "http"
    assert wire["check"]["cause"]["code"] == code
    assert set(wire["check"]["cause"]) == {"kind", "code", "message"}
    assert token not in json.dumps(wire) + repr(result)
    assert "upstream-original" not in json.dumps(wire)
    assert result.available is (False if code < 500 else None)


@pytest.mark.parametrize(
    "model,suffix",
    [
        ("kimi-k3", "/chat/completions"),
        ("muse-spark-1.3-contributor", "/responses"),
        ("qwen3.8-max", "/messages"),
    ],
)
def test_official_endpoint_protocol_and_minimal_request(model, suffix):
    def handle(request):
        assert str(request.url) == "https://opencode.ai/zen/go/v1" + suffix
        assert request.headers["x-opencode-session"] == "stable-diagnostic-session"
        assert request.headers["user-agent"] == "scitex-genai/availability-probe"
        data = json.loads(request.content)
        assert data["model"] == model
        assert "Reply OK." in request.content.decode()
        assert "private conversation" not in request.content.decode()
        if suffix == "/responses":
            assert data["max_output_tokens"] == 16
            assert request.headers["authorization"] == "Bearer fake-key"
            return httpx.Response(200, json={"status": "completed", "output": []})
        assert data["max_tokens"] == 16
        if suffix == "/messages":
            assert request.headers["x-api-key"] == "fake-key"
            return httpx.Response(200, json={"type": "message", "content": []})
        return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})

    result = probe_provider_key(
        "opencode-go",
        model,
        "fake-key",
        session_id="stable-diagnostic-session",
        transport=httpx.MockTransport(handle),
    )
    assert result.available is True
    assert result.status.code == 200


def test_timeout_reports_native_errno_and_unknown_verdict():
    def handle(request):
        raise httpx.ReadTimeout(
            "untrusted exception includes fake-key", request=request
        )

    result = probe_provider_key(
        "opencode-go", "kimi-k3", "fake-key", transport=httpx.MockTransport(handle)
    )
    assert result.available is None
    assert result.status.kind == "errno" and result.status.code == "ETIMEDOUT"
    assert "fake-key" not in json.dumps(result.to_dict())


def test_connection_refused_is_not_an_invented_http_response():
    def handle(request):
        try:
            raise ConnectionRefusedError(errno.ECONNREFUSED, "refused")
        except OSError as cause:
            raise httpx.ConnectError("connect failed", request=request) from cause

    result = probe_provider_key(
        "opencode-go", "kimi-k3", "fake-key", transport=httpx.MockTransport(handle)
    )
    assert result.status.kind == "errno" and result.status.code == "ECONNREFUSED"
    assert result.available is None


@pytest.mark.parametrize(
    "body,headers",
    [
        ({"error": {"resetAt": "2030-01-01T00:00:00Z"}}, {}),
        ({"error": {"reset_at": 1893456000000}}, {}),
        ({"error": {"message": "weekly cap resets at 2030-01-01T00:00:00Z"}}, {}),
        ({}, {"Retry-After": "Tue, 01 Jan 2030 00:00:00 GMT"}),
    ],
)
def test_reset_timestamp_is_retained(body, headers):
    result = probe_provider_key(
        "opencode-go",
        "kimi-k3",
        "fake-key",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(429, json=body, headers=headers)
        ),
    )
    assert result.status.code == 429
    assert result.reset_at == 1893456000


def test_redirect_does_not_forward_credentials():
    calls = []

    def handle(request):
        calls.append(request.url)
        return httpx.Response(
            307, headers={"Location": "https://another-provider.invalid/"}
        )

    result = probe_provider_key(
        "opencode-go", "kimi-k3", "fake-key", transport=httpx.MockTransport(handle)
    )
    assert len(calls) == 1
    assert result.status.code == 307
    assert result.available is None


@pytest.mark.parametrize(
    "body",
    [
        {"choices": []},
        {"error": "rejected"},
        {"status": "failed", "output": []},
        "not-json",
    ],
)
def test_http_success_without_valid_inference_is_unknown(body):
    result = probe_provider_key(
        "opencode-go",
        "kimi-k3",
        "fake-key",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)),
    )
    assert result.status.code == 200
    assert result.available is None


@pytest.mark.parametrize(
    "url",
    [
        "https://user:secret@provider.invalid/v1",
        "https://provider.invalid/v1?token=secret",
        "file:///tmp/provider",
    ],
)
def test_endpoint_rejects_embedded_credentials(url):
    with pytest.raises(ValueError, match="without embedded credentials"):
        provider_route(
            "custom", "model", endpoint_url=url, protocol="openai-chat-completions"
        )
