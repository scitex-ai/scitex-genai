# Fixed-choice Systemone decisions

`scitex_genai.decision.decide` is a separate synchronous API for one Systemone
decision. Existing `GenAI` text and streaming calls remain unchanged. Importing
the decision package does not initialize those provider SDKs.

```python
from scitex_genai.decision import (
    ChoiceDecision, DecisionBudget, SystemOneTarget, decide,
)

# The caller supplies an approved endpoint, account label and credential.
target = SystemOneTarget(
    provider="approved-provider",
    model="approved-model",
    account="approved-account-label",
    endpoint="https://approved-provider.example/v1/systemone",
)
result = decide(
    original_text,
    target=target,
    api_key=approved_credential,
    choices={"figrecipe": "Figure creation", "hold": "Need more information"},
    instructions=original_instructions,
    hold_choice="hold",
    budget=DecisionBudget(timeout_s=5, max_response_bytes=65_536),
)
```

The wire format is `model`, unedited `state`, and one entry in `questions`
containing `type: "choice"`, unedited `instructions`, and the supplied fixed
`criteria`. The question name defaults to `decision`; callers can supply a name
such as `lead`. There is no OpenAI chat-envelope conversion.

A `ChoiceDecision` includes the choice, vendor-reported confidence, immutable
probabilities for every supplied key, token counts and `is_hold`. Probabilities
must be finite, within [0, 1], sum to 1 within 1e-6, and have the chosen key as
their unique maximum. Confidence is not calibrated or equated with the maximum
probability. A valid HOLD is a decision, not an error or automatic forwarding.

`result.target.model` is the requested model; `result.reported_model` is the
actual response model. Alternate reported model names are accepted only when
explicitly listed in `response_model_aliases`; neither name nor response is
rewritten. `result.response` retains original body bytes and HTTP status/header
pairs. Duplicate JSON keys, nonfinite numbers, unexpected response shape,
missing usage or invalid distributions are refused. Raw body/headers are
excluded from repr; callers own their retention and privacy policy.

Invalid local arguments raise `ValueError` before transport. Other failures
return `DecisionRefusal` with a nonsecret reason and an available original
response. A refusal has no synthesized choice, probabilities or confidence.
An oversized response records only a bounded prefix with `body_complete=False`.
Credentials are separate call arguments and are absent from target/outcome
records; transport exception text is not copied into refusals. An exact
credential echo in raw body bytes, HTTP header fields or decoded JSON
keys/strings produces `credential_echo` with the entire response withheld.
No redacted or rewritten response is represented as original evidence. This
finite check is not a universal secret/arbitrary-encoding detector; the caller
still owns response evidence retention and privacy.

The default HTTPS transport makes one POST with no redirects, retries, proxy
environment lookup, provider/account discovery, failover or paid fallback. An
injected `SystemOneTransport` is invoked once and must honor the same contract.
The explicit budget limits socket timeout and response bytes. It does not
enforce a hard elapsed deadline, dollar limit, tariff, free-model eligibility
or subscription entitlement. Token counts are actual reported counts, not a
cost estimate. A free SKU does not imply a Go subscription acceptance.

SAC or another caller owns credential custody, target/spending eligibility,
ZDR/exposure admission, outer deadlines/process cleanup, handshakes, leases,
dispatch and forwarding. This primitive supplies no production activation,
scientific validation, native artifact completion or authenticated-human
approval. Offline tests use explicitly injected transports and make no model
requests.
