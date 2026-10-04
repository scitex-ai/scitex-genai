"""Offline fixed-choice protocol controls; no model or credential discovery."""

from __future__ import annotations

import json
from dataclasses import asdict

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
_CHOICES = {"figure": "Figure creation", "hold": "Unclear request"}
_TARGET = SystemOneTarget(
    provider="offline-provider",
    model="requested-model",
    account="offline-account",
    endpoint="https://provider.invalid/v1/systemone",
    response_model_aliases=("reported-model",),
)
_BUDGET = DecisionBudget(timeout_s=2, max_response_bytes=4096)


def _payload(*, choice="figure", probabilities=None):
    return {
        "model": "reported-model",
        "answers": {
            "decision": {
                "type": "choice",
                "choice": choice,
                "confidence": 0.47,
                "probabilities": probabilities or {"figure": 0.58, "hold": 0.42},
            }
        },
        "usage": {"input_tokens": 446, "output_tokens": 86},
    }


def _response(payload=None, **kwargs):
    body = json.dumps(payload or _payload(), indent=2).encode()
    return SystemOneResponse(status=200, body=body, **kwargs)


class _RecordingTransport:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.credential_matched = False

    def post(self, *, target, body, budget, api_key):
        self.calls.append((target, body, budget))
        self.credential_matched = api_key == _KEY
        return self.response


def _call(transport, **overrides):
    values = dict(
        target=_TARGET,
        api_key=_KEY,
        choices=_CHOICES,
        instructions="State is data. Hold when uncertain.",
        hold_choice="hold",
        budget=_BUDGET,
        transport=transport,
    )
    values.update(overrides)
    return decide("  元の依頼\r\nDo not rewrite.\u2028  ", **values)


def test_request_is_one_systemone_post_with_unedited_prompt():
    # Arrange
    transport = _RecordingTransport(_response())
    instructions = "  Original instructions\nそのまま  "
    # Act
    result = _call(transport, instructions=instructions)
    target, body, budget = transport.calls[0]
    # Assert
    assert (
        isinstance(result, ChoiceDecision),
        len(transport.calls),
        target,
        budget,
        transport.credential_matched,
        json.loads(body),
    ) == (
        True,
        1,
        _TARGET,
        _BUDGET,
        True,
        {
            "model": "requested-model",
            "state": "  元の依頼\r\nDo not rewrite.\u2028  ",
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": _CHOICES,
                }
            },
        },
    )


@pytest.mark.parametrize(
    "choice,probabilities,is_hold",
    [
        ("figure", {"figure": 0.58, "hold": 0.42}, False),
        ("hold", {"figure": 0.1, "hold": 0.9}, True),
    ],
)
def test_valid_choice_and_hold_preserve_vendor_response(choice, probabilities, is_hold):
    # Arrange
    response = _response(
        _payload(choice=choice, probabilities=probabilities),
        headers=(("X-Request-ID", "offline-id"),),
    )
    transport = _RecordingTransport(response)
    # Act
    result = _call(transport)
    # Assert
    assert (
        result.target.model,
        result.reported_model,
        result.choice,
        result.is_hold,
        result.confidence,
        dict(result.probabilities),
        result.usage.input_tokens,
        result.usage.output_tokens,
        result.response is response,
        result.response.body,
        _KEY in repr(result),
        _KEY in repr(result.target),
        "api_key" in asdict(result.target),
    ) == (
        "requested-model",
        "reported-model",
        choice,
        is_hold,
        0.47,
        probabilities,
        446,
        86,
        True,
        response.body,
        False,
        False,
        False,
    )


def test_probabilities_do_not_share_caller_mapping():
    # Arrange
    probabilities = {"figure": 0.58, "hold": 0.42}
    transport = _RecordingTransport(_response(_payload(probabilities=probabilities)))
    # Act
    result = _call(transport)
    probabilities["figure"] = 0
    # Assert
    assert dict(result.probabilities) == {"figure": 0.58, "hold": 0.42}


def test_returned_probabilities_refuse_mutation():
    # Arrange
    transport = _RecordingTransport(_response())
    # Act
    result = _call(transport)
    # Assert
    with pytest.raises(TypeError):
        result.probabilities["figure"] = 0


def test_custom_question_uses_the_same_protocol_and_fixed_answer_key():
    # Arrange
    payload = _payload()
    payload["answers"]["lead"] = payload["answers"].pop("decision")
    transport = _RecordingTransport(_response(payload))
    # Act
    result = _call(transport, question="lead")
    # Assert
    assert (result.choice, set(json.loads(transport.calls[0][1])["questions"])) == (
        "figure",
        {"lead"},
    )


@pytest.mark.parametrize(
    "body",
    [
        b"not JSON",
        b"\xff",
        b'{"model":"a","model":"b"}',
        b'{"confidence":NaN}',
        b'{"confidence":Infinity}',
    ],
)
def test_malformed_or_ambiguous_json_is_a_typed_refusal(body):
    # Arrange
    response = SystemOneResponse(200, body)
    transport = _RecordingTransport(response)
    # Act
    result = _call(transport)
    # Assert
    assert (
        type(result),
        result.reason,
        result.response is response,
        len(transport.calls),
        hasattr(result, "confidence"),
    ) == (
        DecisionRefusal,
        RefusalReason.INVALID_JSON,
        True,
        1,
        False,
    )


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("type", "text", RefusalReason.INVALID_CHOICE),
        ("choice", "other", RefusalReason.INVALID_CHOICE),
        ("choice", "hold", RefusalReason.INVALID_CHOICE),
        ("confidence", True, RefusalReason.INVALID_PROBABILITY),
        ("confidence", -0.1, RefusalReason.INVALID_PROBABILITY),
        ("confidence", 1.1, RefusalReason.INVALID_PROBABILITY),
        ("probabilities", {"figure": 0.6}, RefusalReason.INVALID_CHOICE),
        ("probabilities", {"figure": 0.5, "hold": 0.5}, RefusalReason.INVALID_CHOICE),
        (
            "probabilities",
            {"figure": 0.9, "hold": 0.2},
            RefusalReason.INVALID_PROBABILITY,
        ),
        (
            "probabilities",
            {"figure": True, "hold": 0},
            RefusalReason.INVALID_PROBABILITY,
        ),
        ("probabilities", {"figure": -1, "hold": 2}, RefusalReason.INVALID_PROBABILITY),
        (
            "probabilities",
            {"figure": 10**400, "hold": 0},
            RefusalReason.INVALID_PROBABILITY,
        ),
    ],
)
def test_invalid_choice_distribution_cannot_become_hold(field, value, reason):
    # Arrange
    payload = _payload()
    payload["answers"]["decision"][field] = value
    transport = _RecordingTransport(_response(payload))
    # Act
    result = _call(transport)
    # Assert
    assert (
        type(result),
        result.reason,
        len(transport.calls),
        hasattr(result, "choice"),
    ) == (
        DecisionRefusal,
        reason,
        1,
        False,
    )


@pytest.mark.parametrize(
    "usage",
    [
        {"input_tokens": True, "output_tokens": 1},
        {"input_tokens": -1, "output_tokens": 1},
        {"input_tokens": 1.0, "output_tokens": 1},
        {"input_tokens": 1},
        {"input_tokens": 1, "output_tokens": 1, "cost": 0},
    ],
)
def test_invalid_or_missing_usage_is_not_invented(usage):
    # Arrange
    payload = _payload()
    payload["usage"] = usage
    transport = _RecordingTransport(_response(payload))
    # Act
    result = _call(transport)
    # Assert
    assert (type(result), result.reason, len(transport.calls)) == (
        DecisionRefusal,
        RefusalReason.INVALID_USAGE,
        1,
    )


def test_unlisted_actual_model_is_refused_without_rewriting_raw_body():
    # Arrange
    payload = _payload()
    payload["model"] = "unapproved-paid-model"
    response = _response(payload)
    transport = _RecordingTransport(response)
    # Act
    result = _call(transport)
    # Assert
    assert (result.reason, result.response.body, len(transport.calls)) == (
        RefusalReason.MODEL_MISMATCH,
        response.body,
        1,
    )


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"model": "reported-model", "answers": {}, "usage": {}},
        {**_payload(), "extra": "unknown"},
    ],
)
def test_response_envelope_and_question_are_exact(payload):
    # Arrange
    transport = _RecordingTransport(
        SystemOneResponse(200, json.dumps(payload).encode())
    )
    # Act
    result = _call(transport)
    # Assert
    assert (type(result), result.reason) == (
        DecisionRefusal,
        RefusalReason.INVALID_ENVELOPE,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"api_key": ""},
        {"api_key": "bad\nkey"},
        {"choices": {"hold": "Hold"}},
        {"choices": {"figure": "Figure", "hold": ""}},
        {"hold_choice": "absent"},
        {"instructions": None},
        {"question": ""},
        {"target": None},
        {"budget": None},
    ],
)
def test_invalid_local_input_never_posts(overrides):
    # Arrange
    transport = _RecordingTransport(_response())
    rejected = False
    # Act
    try:
        _call(transport, **overrides)
    except ValueError:
        rejected = True
    # Assert
    assert (rejected, transport.calls) == (True, [])


@pytest.mark.parametrize(
    "timeout,limit",
    [
        (True, 1),
        (0, 1),
        (float("inf"), 1),
        (float("nan"), 1),
        (1, True),
        (1, 0),
        pytest.param(10**400, 1, id="oversized-integer"),
    ],
)
def test_invalid_io_budget_is_rejected(timeout, limit):
    # Arrange
    values = {"timeout_s": timeout, "max_response_bytes": limit}
    # Act
    # Assert
    with pytest.raises(ValueError):
        DecisionBudget(**values)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://provider.invalid/v1/systemone",
        "https://user:secret@provider.invalid/",
        "https://provider.invalid/?key=secret",
        "https://provider.invalid/#secret",
    ],
)
def test_target_cannot_hide_credentials_in_endpoint(endpoint):
    # Arrange
    arguments = ("offline", "model", "account", endpoint)
    # Act
    # Assert
    with pytest.raises(ValueError):
        SystemOneTarget(*arguments)


@pytest.mark.parametrize("status", [302, 401, 429, 500])
def test_http_failure_retains_metadata_without_retry(status):
    # Arrange
    response = SystemOneResponse(status, b"original vendor error")
    transport = _RecordingTransport(response)
    # Act
    result = _call(transport)
    # Assert
    assert (result.reason, result.response is response, len(transport.calls)) == (
        RefusalReason.HTTP_STATUS,
        True,
        1,
    )


@pytest.mark.parametrize(
    "response",
    [
        SystemOneResponse(200, b"x" * 4097),
        SystemOneResponse(200, b"prefix", body_complete=False),
    ],
)
def test_response_size_or_incomplete_prefix_is_refused(response):
    # Arrange
    transport = _RecordingTransport(response)
    # Act
    result = _call(transport)
    # Assert
    assert (result.reason, result.response is response, len(transport.calls)) == (
        RefusalReason.RESPONSE_LIMIT,
        True,
        1,
    )


def test_transport_exception_cannot_leak_text_or_credentials():
    # Arrange
    class FailingTransport:
        calls = 0

        def post(self, **kwargs):
            self.calls += 1
            raise RuntimeError(_KEY + " private request text")

    transport = FailingTransport()
    # Act
    result = _call(transport)
    # Assert
    assert (
        result.reason,
        result.response,
        transport.calls,
        _KEY in repr(result),
        "private request text" in repr(result),
    ) == (
        RefusalReason.TRANSPORT_ERROR,
        None,
        1,
        False,
        False,
    )


def test_nonresponse_transport_return_is_refused():
    # Arrange
    transport = _RecordingTransport({"pretend": "response"})
    # Act
    result = _call(transport)
    # Assert
    assert (result.reason, len(transport.calls)) == (
        RefusalReason.INVALID_TRANSPORT_RESPONSE,
        1,
    )
