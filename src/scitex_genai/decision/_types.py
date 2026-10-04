"""Nonsecret target identity, bounded I/O, and typed decision outcomes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Protocol
from urllib.parse import urlsplit


def _text(value: str) -> bool:
    return (
        type(value) is str
        and bool(value)
        and not any(char in value for char in "\r\n\x00")
    )


@dataclass(frozen=True)
class SystemOneTarget:
    """Explicit selection; account is a nonsecret caller label, not custody.

    Credentials are passed separately to ``decide`` and never stored here.
    Response model aliases must be explicitly enumerated; no normalization
    or provider/account fallback occurs.
    """

    provider: str
    model: str
    account: str
    endpoint: str
    response_model_aliases: tuple[str, ...] = ()

    def __post_init__(self):
        if not all(
            _text(value)
            for value in (self.provider, self.model, self.account, self.endpoint)
        ):
            raise ValueError("target requires nonempty nonsecret identity strings")
        url = urlsplit(self.endpoint)
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
        ):
            raise ValueError("endpoint requires HTTPS without userinfo/query/fragment")
        aliases = self.response_model_aliases
        if (
            type(aliases) is not tuple
            or not all(_text(name) for name in aliases)
            or len(set(aliases)) != len(aliases)
            or self.model in aliases
        ):
            raise ValueError("response aliases require distinct explicit model names")


@dataclass(frozen=True)
class DecisionBudget:
    """Per-call socket timeout and response-byte limit, not a dollar budget.

    A socket timeout is not a hard elapsed-time or process-lifetime fence.
    Account eligibility, tariffs, spending grants and outer deadlines belong
    to the caller; token counts are reported after the call without pricing.
    """

    timeout_s: float
    max_response_bytes: int

    def __post_init__(self):
        try:
            valid_timeout = (
                type(self.timeout_s) in (int, float)
                and math.isfinite(self.timeout_s)
                and self.timeout_s > 0
            )
        except OverflowError:
            valid_timeout = False
        if (
            not valid_timeout
            or type(self.max_response_bytes) is not int
            or self.max_response_bytes < 1
        ):
            raise ValueError("budget requires positive finite timeout and byte limit")


@dataclass(frozen=True)
class SystemOneResponse:
    """Original response body and HTTP metadata, never a rewritten model.

    ``body_complete=False`` explicitly marks a bounded prefix. Headers are
    the HTTP library's field pairs, not a byte-for-byte raw wire header block.
    Raw data is excluded from repr but available for caller-owned evidence.
    """

    status: int
    body: bytes = field(repr=False)
    headers: tuple[tuple[str, str], ...] = field(default=(), repr=False)
    body_complete: bool = True

    def __post_init__(self):
        if type(self.status) is not int or not 100 <= self.status <= 599:
            raise ValueError("invalid HTTP status")
        if type(self.body) is not bytes or type(self.body_complete) is not bool:
            raise ValueError("invalid response body metadata")
        if type(self.headers) is not tuple or any(
            type(pair) is not tuple
            or len(pair) != 2
            or any(type(value) is not str for value in pair)
            for pair in self.headers
        ):
            raise ValueError("invalid response header pairs")


class SystemOneTransport(Protocol):
    """Injected transports must honor one POST and the supplied I/O budget."""

    def post(
        self,
        *,
        target: SystemOneTarget,
        body: bytes,
        budget: DecisionBudget,
        api_key: str,
    ) -> SystemOneResponse: ...


@dataclass(frozen=True)
class DecisionUsage:
    """Vendor-reported token counts; no cost or entitlement inference."""

    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class ChoiceDecision:
    """Validated fixed-choice distribution; confidence is uncalibrated."""

    target: SystemOneTarget
    reported_model: str
    choice: str
    confidence: float
    probabilities: Mapping[str, float]
    usage: DecisionUsage
    is_hold: bool
    response: SystemOneResponse = field(repr=False)

    def __post_init__(self):
        object.__setattr__(
            self, "probabilities", MappingProxyType(dict(self.probabilities))
        )


class RefusalReason(str, Enum):
    TRANSPORT_ERROR = "transport_error"
    INVALID_TRANSPORT_RESPONSE = "invalid_transport_response"
    HTTP_STATUS = "http_status"
    RESPONSE_LIMIT = "response_limit"
    INVALID_JSON = "invalid_json"
    INVALID_ENVELOPE = "invalid_envelope"
    MODEL_MISMATCH = "model_mismatch"
    INVALID_CHOICE = "invalid_choice"
    INVALID_PROBABILITY = "invalid_probability"
    INVALID_USAGE = "invalid_usage"
    CREDENTIAL_ECHO = "credential_echo"


@dataclass(frozen=True)
class DecisionRefusal:
    """Nonsecret failure, not a synthesized HOLD or probability distribution.

    Transport exception messages, request text and credentials are not copied
    into the result. An available original response is retained separately.
    """

    target: SystemOneTarget
    reason: RefusalReason
    response: SystemOneResponse | None = field(default=None, repr=False)
