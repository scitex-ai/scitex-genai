# Isolated SGLang A/B replay

`scitex-genai-sglang-ab` replays exact JSON request bodies at fixed arrival
offsets and records streaming TTFT, TPOT, end-to-end latency, token usage,
cache fields, scheduler fields, and deterministic request IDs as JSONL. It has
no default endpoint and refuses to send requests without an explicit canary
acknowledgement.

Create a scenario manifest (keep large prompt bodies out of Git):

```json
{
  "schema_version": 1,
  "name": "cold-then-warm",
  "seed": 42,
  "configuration": {"scheduler": "fcfs", "kv_cache_dtype": "fp8_e4m3"},
  "metadata": {"warmup": "warm-hub-once", "engine_restart": true},
  "requests": [
    {
      "id": "scholar-cold",
      "arrival_ms": 0,
      "cache_state": "cold",
      "expected_prompt_tokens": 740000,
      "body": {"model": "qwen", "messages": [], "stream": true,
               "stream_options": {"include_usage": true}, "seed": 42,
               "max_tokens": 64}
    },
    {
      "id": "hub-warm",
      "arrival_ms": 1000,
      "cache_state": "warm",
      "expected_prompt_tokens": 425000,
      "body": {"model": "qwen", "messages": [], "stream": true,
               "stream_options": {"include_usage": true}, "seed": 42,
               "max_tokens": 64}
    }
  ]
}
```

Populate `messages` with the already-tokenized-and-verified A/B corpus and use
the identical file for every configuration. The command fails if reported
prompt tokens differ from `expected_prompt_tokens`. Every request in a
multi-request replay must declare `cache_state` explicitly. Two requests whose
state is `cold` or `unknown` are conservatively treated as a crash probe.

Run only against an isolated canary:

```bash
scitex-genai-sglang-ab \
  --scenario /secure/replays/cold-then-warm.json \
  --endpoint http://CANARY_HOST:30000/v1/chat/completions \
  --run-id fcfs-r01 \
  --output results/fcfs-r01.jsonl \
  --i-understand-this-sends-load-to-an-isolated-canary
```

Set `SCITEX_SGLANG_BENCHMARK_API_KEY` only if the canary requires a bearer
token. Rows remain in declared arrival order even when requests finish in a
different order. Vendor telemetry found in SSE `usage`, `meta_info`, cache,
and scheduler fields is retained; generated text is intentionally omitted.

## Cold + cold is a crash probe

Two concurrent cold long-context requests can exhaust or destabilize an
engine. Such a manifest must label every relevant request with
`"cache_state": "cold"`, declare `"risk_class": "crash-probe"` and
`"target_scope": "dedicated-canary"`, and the command requires both safety
acknowledgements:

```bash
scitex-genai-sglang-ab \
  --scenario /secure/replays/cold-cold.json \
  --endpoint http://DEDICATED_CANARY/v1/chat/completions \
  --run-id cold-cold-r01 \
  --i-understand-this-sends-load-to-an-isolated-canary \
  --i-understand-this-may-crash-the-isolated-canary
```

Never point a crash probe at production. The loader rejects an unmarked
cold+cold scenario, and the runner rejects a marked one without the second
acknowledgement.
