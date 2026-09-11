# External-provider model firewall

Paid provider credentials belong at the SciTeX GenAI egress boundary, not in
agent containers. A harness-level model setting is not a security or spending
control: a built-in provider can be discovered from an environment variable,
and an interactive `/model` command can replace the configured model.

The external-provider gateway separates the two credentials:

```text
agent container                       trusted gateway host
┌────────────────────┐                ┌─────────────────────────┐
│ local gateway token│ ── request ──> │ exact model allowlist   │
│ no vendor API key  │                │ request/run budget gate │
└────────────────────┘                │ vendor API key          │
                                      └───────────┬─────────────┘
                                                  │ canonical model only
                                                  ▼
                                         paid provider API
```

An example Flash-only DeepSeek gateway configuration is:

```yaml
gateway:
  host: 0.0.0.0
  port: 18775
  inference_timeout_s: 1800
  inference_capacity_per_upstream: 2
  inference_max_queue_size: 8
  external_provider:
    provider: deepseek
    upstream: https://api.deepseek.com
    upstream_auth_token_env: DEEPSEEK_API_KEY
    canonical_model: deepseek-flash
    model_aliases:
      - deepseek-v4-flash
    anthropic_path_prefix: /anthropic
    max_tokens_per_request: 16384
    max_requests_per_run: 100
    max_input_tokens_per_run: 5000000
    max_output_tokens_per_run: 200000
    max_total_tokens_per_run: 5200000
    max_estimated_usd_per_run: 1.0
    input_usd_per_million_tokens: 1.0
    output_usd_per_million_tokens: 1.0
```

Set the two secrets only on their respective sides:

- The gateway host receives `DEEPSEEK_API_KEY` through its service environment.
- SAC resolves the normal SciTeX gateway token and injects that token into the
  container under a neutral name. It must not inject `DEEPSEEK_API_KEY`.

The `1.0` price rows above are an intentionally conservative example, not a
claim about the current DeepSeek tariff. The price fields are explicit
deployment data because provider prices change.
A non-zero `max_estimated_usd_per_run` is meaningful only when both current
price fields are configured. Input reservation uses request bytes plus a
protocol allowance, rather than a chars-per-token guess, so it errs toward an
early refusal when the vendor tokenizer is unavailable.

`GET /v1/models` is synthetic and lists only the canonical model. Compatibility
aliases are accepted on requests but normalized before outbound HTTP. Any
other name, including `deepseek-v4-pro`, receives HTTP 400 before an upstream
connection is opened. The local bearer token is removed, and only the vendor
bearer token is attached to the outbound request.

`GET /health` reports payload-free counters: request and token totals,
estimated cost, runs observed, the last model label reported by the provider,
and response-model mismatches. No prompt, completion, raw run ID, or credential
is retained. Streaming requests force provider usage reporting when the
OpenAI-compatible route supports it; missing usage is conservatively charged
as the full reservation.

The optional `anthropic_path_prefix` lets one root serve both protocols:
OpenAI `/v1/chat/completions` remains unchanged, while Anthropic `/v1/messages`
becomes `/anthropic/v1/messages` upstream. It is empty for providers that use
the same route root for both protocols.

Budgets are per run within one gateway incarnation. A production-wide or
calendar-period spending authority still belongs in the SciTeX Postgres store;
the in-process gate intentionally does not create a second local database.
