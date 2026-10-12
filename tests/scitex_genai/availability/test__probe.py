import errno
import json

import httpx
import pytest

from scitex_genai.availability import probe_provider_key


@pytest.fixture
def rejected_request():
    def probe(code):
        return probe_provider_key(
            "command-code",
            "meta/muse",
            "fake-sensitive-key",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    code,
                    json={
                        "error": {
                            "message": "insufficient credits; bearer=fake-sensitive-key",
                            "code": "upstream-original",
                        }
                    },
                )
            ),
        )

    return probe


@pytest.mark.parametrize("code", [400, 401, 402, 403, 404, 429, 500, 503])
def test_native_http_code_survives_boundary(code, rejected_request):
    # Arrange
    expected = code
    # Act
    result = rejected_request(code)
    # Assert
    assert result.status.code == expected


def test_http_cause_uses_canonical_fields(rejected_request):
    # Arrange
    expected = {"kind", "code", "message"}
    # Act
    cause = rejected_request(401).to_dict()["check"]["cause"]
    # Assert
    assert set(cause) == expected


@pytest.mark.parametrize("secret", ["fake-sensitive-key", "upstream-original"])
def test_untrusted_body_never_reaches_result(secret, rejected_request):
    # Arrange
    code = 401
    # Act
    result = rejected_request(code)
    # Assert
    assert secret not in json.dumps(result.to_dict()) + repr(result)


@pytest.mark.parametrize(
    "code,expected",
    [(400, False), (401, False), (429, False), (500, None), (503, None)],
)
def test_provider_rejection_preserves_availability_verdict(
    code, expected, rejected_request
):
    # Arrange
    probe = rejected_request
    # Act
    result = probe(code)
    # Assert
    assert result.available is expected


@pytest.fixture(
    params=[
        (
            "kimi-k3",
            "/chat/completions",
            {"choices": [{"message": {"content": "OK"}}]},
            "max_tokens",
        ),
        (
            "muse-spark-1.3-contributor",
            "/responses",
            {"status": "completed", "output": []},
            "max_output_tokens",
        ),
        ("qwen3.8-max", "/messages", {"type": "message", "content": []}, "max_tokens"),
    ]
)
def official_request(request):
    model, suffix, body, limit = request.param
    calls = []

    def handle(req):
        calls.append(req)
        return httpx.Response(200, json=body)

    return model, suffix, limit, calls, httpx.MockTransport(handle)


def _official_probe(case):
    return probe_provider_key(
        "opencode-go",
        case[0],
        "fake-key",
        session_id="stable-diagnostic-session",
        transport=case[4],
    )


def test_probe_uses_official_model_endpoint(official_request):
    # Arrange
    case = official_request
    # Act
    _official_probe(case)
    # Assert
    assert str(case[3][0].url) == "https://opencode.ai/zen/go/v1" + case[1]


def test_probe_uses_only_synthetic_input(official_request):
    # Arrange
    case = official_request
    # Act
    _official_probe(case)
    data = json.loads(case[3][0].content)
    # Assert
    assert data.get("input", data.get("messages")) in (
        "Reply OK.",
        [{"role": "user", "content": "Reply OK."}],
    )


def test_probe_bounds_requested_output_tokens(official_request):
    # Arrange
    case = official_request
    # Act
    _official_probe(case)
    # Assert
    assert json.loads(case[3][0].content)[case[2]] == 16


def test_official_request_completes_availability_check(official_request):
    # Arrange
    case = official_request
    # Act
    result = _official_probe(case)
    # Assert
    assert result.available is True


def test_probe_sends_stable_diagnostic_session(official_request):
    # Arrange
    case = official_request
    # Act
    _official_probe(case)
    # Assert
    assert case[3][0].headers["x-opencode-session"] == "stable-diagnostic-session"


@pytest.fixture(params=["timeout", "refused"])
def transport_failure(request):
    kind = request.param

    def handle(req):
        if kind == "timeout":
            raise httpx.ReadTimeout("untrusted fake-key", request=req)
        try:
            raise ConnectionRefusedError(errno.ECONNREFUSED, "refused")
        except OSError as cause:
            raise httpx.ConnectError("untrusted fake-key", request=req) from cause

    return kind, httpx.MockTransport(handle)


def test_transport_failure_retains_native_errno(transport_failure):
    # Arrange
    kind, transport = transport_failure
    expected = {"timeout": "ETIMEDOUT", "refused": "ECONNREFUSED"}[kind]
    # Act
    result = probe_provider_key(
        "opencode-go", "kimi-k3", "fake-key", transport=transport
    )
    # Assert
    assert (result.status.kind, result.status.code) == ("errno", expected)


def test_transport_failure_keeps_unknown_verdict(transport_failure):
    # Arrange
    transport = transport_failure[1]
    # Act
    result = probe_provider_key(
        "opencode-go", "kimi-k3", "fake-key", transport=transport
    )
    # Assert
    assert result.available is None


def test_transport_exception_never_exposes_secret(transport_failure):
    # Arrange
    transport = transport_failure[1]
    # Act
    result = probe_provider_key(
        "opencode-go", "kimi-k3", "fake-key", transport=transport
    )
    # Assert
    assert "fake-key" not in json.dumps(result.to_dict())


@pytest.mark.parametrize(
    "body,headers",
    [
        ({"error": {"resetAt": "2030-01-01T00:00:00Z"}}, {}),
        ({"error": {"reset_at": 1893456000000}}, {}),
        ({"error": {"message": "weekly cap resets at 2030-01-01T00:00:00Z"}}, {}),
        ({}, {"Retry-After": "Tue, 01 Jan 2030 00:00:00 GMT"}),
    ],
)
def test_reset_timestamp_survives_provider_boundary(body, headers):
    # Arrange
    transport = httpx.MockTransport(
        lambda request: httpx.Response(429, json=body, headers=headers)
    )
    # Act
    result = probe_provider_key(
        "opencode-go", "kimi-k3", "fake-key", transport=transport
    )
    # Assert
    assert result.reset_at == 1893456000


def test_redirect_never_forwards_credentials():
    # Arrange
    calls = []

    def handle(request):
        calls.append(request.url)
        return httpx.Response(
            307, headers={"Location": "https://another-provider.invalid/"}
        )

    # Act
    probe_provider_key(
        "opencode-go", "kimi-k3", "fake-key", transport=httpx.MockTransport(handle)
    )
    # Assert
    assert len(calls) == 1


@pytest.mark.parametrize(
    "body",
    [
        {"choices": []},
        {"error": "rejected"},
        {"status": "failed", "output": []},
        "not-json",
    ],
)
def test_http_success_without_inference_stays_unknown(body):
    # Arrange
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    # Act
    result = probe_provider_key(
        "opencode-go", "kimi-k3", "fake-key", transport=transport
    )
    # Assert
    assert result.available is None
