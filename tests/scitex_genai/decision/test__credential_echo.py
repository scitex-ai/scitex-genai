"""Known credential echoes must not become retained decision evidence."""

import json

import pytest

from scitex_genai.decision import (
    ChoiceDecision,
    DecisionBudget,
    DecisionRefusal,
    RefusalReason,
    SystemOneResponse,
    SystemOneTarget,
    decide,
)

_KEY = "test-only-credential-not-a-real-provider-key"
_TARGET = SystemOneTarget("offline", "model", "account", "https://provider.invalid/")


class _ReplyTransport:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def post(self, **kwargs):
        self.calls += 1
        return self.response


def _call(transport):
    return decide(
        "original request",
        target=_TARGET,
        api_key=_KEY,
        choices={"figure": "Figure", "hold": "Hold"},
        instructions="Unedited",
        hold_choice="hold",
        budget=DecisionBudget(2, 4096),
        transport=transport,
    )


@pytest.mark.parametrize(
    "response",
    [
        SystemOneResponse(401, b"Unauthorized: " + _KEY.encode()),
        SystemOneResponse(200, json.dumps({"model": _KEY}).encode()),
        SystemOneResponse(
            200,
            (
                '{"error":"' + "".join("\\u%04x" % ord(char) for char in _KEY) + '"}'
            ).encode(),
        ),
        SystemOneResponse(200, b"{}", headers=(("X-Key", _KEY),)),
        SystemOneResponse(200, b"{}", headers=((_KEY, "header-name-echo"),)),
        SystemOneResponse(200, _KEY.encode() * 200, body_complete=False),
        SystemOneResponse(
            200,
            (
                '{"duplicate":"'
                + "".join("\\u%04x" % ord(char) for char in _KEY)
                + '","duplicate":"later-value"}'
            ).encode(),
        ),
    ],
)
def test_known_credential_echo_withholds_original_response(response):
    # Arrange
    transport = _ReplyTransport(response)
    # Act
    result = _call(transport)
    # Assert
    assert (
        type(result),
        result.reason,
        result.response,
        transport.calls,
        _KEY in repr(result),
        hasattr(result, "probabilities"),
    ) == (
        DecisionRefusal,
        RefusalReason.CREDENTIAL_ECHO,
        None,
        1,
        False,
        False,
    )


def test_untainted_raw_response_is_unchanged_after_echo_admission():
    # Arrange
    raw = (
        b' {"model":"model","answers":{"decision":{"type":"choice",'
        b'"choice":"hold","confidence":0.1,"probabilities":{"figure":0.3,'
        b'"hold":0.7}}},"usage":{"input_tokens":5,"output_tokens":2}}\n'
    )
    response = SystemOneResponse(200, raw, headers=(("X-Request-ID", "unchanged"),))
    transport = _ReplyTransport(response)
    # Act
    result = _call(transport)
    # Assert
    assert (
        type(result),
        result.response is response,
        result.response.body,
        result.reported_model,
        result.is_hold,
        result.confidence,
        result.usage.input_tokens,
        result.usage.output_tokens,
        transport.calls,
    ) == (
        ChoiceDecision,
        True,
        raw,
        "model",
        True,
        0.1,
        5,
        2,
        1,
    )
