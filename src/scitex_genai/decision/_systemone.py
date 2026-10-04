"""One-shot Systemone request and strict, non-normalizing response parser."""

from __future__ import annotations

import json
import math
from typing import Mapping

from ._credential_echo import _echoes_credential
from ._transport import _HTTPSinglePost
from ._types import (
    ChoiceDecision,
    DecisionBudget,
    DecisionRefusal,
    DecisionUsage,
    RefusalReason,
    SystemOneResponse,
    SystemOneTarget,
    SystemOneTransport,
)


def _pairs(items):
    obj = {}
    for key, value in items:
        if key in obj:
            raise ValueError("duplicate JSON key")
        obj[key] = value
    return obj


def _nonfinite(value):
    raise ValueError("nonfinite JSON constant")


def _number(value):
    return type(value) in (int, float) and 0 <= value <= 1 and math.isfinite(value)


def _parse(response, *, target, choices, hold_choice, question):
    def refuse(reason):
        return DecisionRefusal(target, reason, response)

    try:
        obj = json.loads(
            response.body.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_nonfinite,
        )
    except (ValueError, RecursionError):
        return refuse(RefusalReason.INVALID_JSON)
    if type(obj) is not dict or set(obj) != {"model", "answers", "usage"}:
        return refuse(RefusalReason.INVALID_ENVELOPE)
    model = obj["model"]
    if type(model) is not str or model not in (
        target.model,
        *target.response_model_aliases,
    ):
        return refuse(RefusalReason.MODEL_MISMATCH)
    answers = obj["answers"]
    if type(answers) is not dict or set(answers) != {question}:
        return refuse(RefusalReason.INVALID_ENVELOPE)
    answer = answers[question]
    if type(answer) is not dict or set(answer) != {
        "type",
        "choice",
        "confidence",
        "probabilities",
    }:
        return refuse(RefusalReason.INVALID_CHOICE)
    probabilities = answer["probabilities"]
    choice = answer["choice"]
    if (
        answer["type"] != "choice"
        or type(choice) is not str
        or choice not in choices
        or type(probabilities) is not dict
        or set(probabilities) != set(choices)
    ):
        return refuse(RefusalReason.INVALID_CHOICE)
    if (
        not _number(answer["confidence"])
        or not all(_number(value) for value in probabilities.values())
        or not math.isclose(
            math.fsum(probabilities.values()), 1, rel_tol=0, abs_tol=1e-6
        )
    ):
        return refuse(RefusalReason.INVALID_PROBABILITY)
    maximum = max(probabilities.values())
    if [key for key, value in probabilities.items() if value == maximum] != [choice]:
        return refuse(RefusalReason.INVALID_CHOICE)
    usage = obj["usage"]
    if (
        type(usage) is not dict
        or set(usage) != {"input_tokens", "output_tokens"}
        or any(type(value) is not int or value < 0 for value in usage.values())
    ):
        return refuse(RefusalReason.INVALID_USAGE)
    return ChoiceDecision(
        target=target,
        reported_model=model,
        choice=choice,
        confidence=answer["confidence"],
        probabilities=probabilities,
        usage=DecisionUsage(**usage),
        is_hold=choice == hold_choice,
        response=response,
    )


def decide(
    state: str,
    *,
    target: SystemOneTarget,
    api_key: str,
    choices: Mapping[str, str],
    instructions: str,
    hold_choice: str,
    budget: DecisionBudget,
    question: str = "decision",
    transport: SystemOneTransport | None = None,
) -> ChoiceDecision | DecisionRefusal:
    """Send exactly one fixed-choice Systemone request, without chat adaptation.

    ``state`` and ``instructions`` are serialized as supplied, with no trim,
    rewriting or added instructions. The caller declares the fixed choices,
    HOLD key, target, credential and I/O budget. Invalid local arguments raise
    ``ValueError`` before any transport call. A transport/response failure
    returns ``DecisionRefusal`` without retry, fallback or invented confidence.

    The default transport performs one POST. An injected transport is called
    once and must honor that contract itself. No key/env/account discovery,
    eligibility check, scientific validation, spend authorization, ZDR policy,
    forwarding or outer process-lifetime management is performed here.
    """
    if type(target) is not SystemOneTarget or type(budget) is not DecisionBudget:
        raise ValueError("explicit target and budget are required")
    if (
        type(api_key) is not str
        or not api_key
        or any(char in api_key for char in "\r\n\x00")
    ):
        raise ValueError("explicit credential is required")
    if type(state) is not str or not state or type(instructions) is not str:
        raise ValueError("state and instructions must be strings; state is nonempty")
    if type(question) is not str or not question:
        raise ValueError("question requires a nonempty name")
    try:
        criteria = dict(choices)
    except (TypeError, ValueError):
        raise ValueError("choices require a fixed mapping") from None
    if (
        not isinstance(choices, Mapping)
        or len(criteria) < 2
        or any(type(key) is not str or not key for key in criteria)
        or any(type(value) is not str or not value for value in criteria.values())
        or type(hold_choice) is not str
        or hold_choice not in criteria
    ):
        raise ValueError("choices require distinct nonempty keys/descriptions and HOLD")
    try:
        api_key.encode("utf-8")
        body = json.dumps(
            {
                "model": target.model,
                "state": state,
                "questions": {
                    question: {
                        "type": "choice",
                        "instructions": instructions,
                        "criteria": criteria,
                    }
                },
            },
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except UnicodeError:
        raise ValueError("request strings must encode as UTF-8") from None
    sender = transport if transport is not None else _HTTPSinglePost()
    try:
        response = sender.post(target=target, body=body, budget=budget, api_key=api_key)
    except Exception:
        # Vendor/transport exception messages may contain credentials or text.
        return DecisionRefusal(target, RefusalReason.TRANSPORT_ERROR)
    if type(response) is not SystemOneResponse:
        return DecisionRefusal(target, RefusalReason.INVALID_TRANSPORT_RESPONSE)
    if _echoes_credential(response, api_key):
        return DecisionRefusal(target, RefusalReason.CREDENTIAL_ECHO)
    if not response.body_complete or len(response.body) > budget.max_response_bytes:
        return DecisionRefusal(target, RefusalReason.RESPONSE_LIMIT, response)
    if response.status != 200:
        return DecisionRefusal(target, RefusalReason.HTTP_STATUS, response)
    return _parse(
        response,
        target=target,
        choices=criteria,
        hold_choice=hold_choice,
        question=question,
    )
