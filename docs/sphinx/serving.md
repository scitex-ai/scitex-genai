# Serving local models

`scitex-genai` owns model-engine configuration, launch, supervision, and the
single-port inference gateway. Model settings live in
`~/.scitex/genai/models.d/<key>.conf`; start from the tracked
`examples/serve/qwen38-27b-sglang.conf` profiles.

## Session-aware SGLang caching

Cache locality has three separate parts:

1. Compiled kernels are stored below
   `serve.cache_root/<engine-key>-cache`, so an engine restart on the same node
   can reuse compilation artifacts.
2. Gateway stickiness sends later requests for an explicit session back to
   the same backend. Stickiness cannot prevent eviction within that backend.
3. SGLang's session-aware `UnifiedRadixCache` associates reusable KV leaves
   with a top-level `session_id`. It evicts unreferenced KV before referenced
   KV when possible.

The launcher enables the third layer with
`SGLANG_ENABLE_UNIFIED_RADIX_TREE=1` and
`--enable-session-radix-cache`. Before starting the engine, it validates that
the pinned image implements both controls and the metrics flag; an older image
is refused instead of silently starting without session retention. The
canonical Qwen profiles enable metrics for cache/eviction observation on the
next controlled server activation.

Clients should send a stable `X-SciTeX-Session-ID` header. The compatibility
headers `session_id` and `x-session-id` are also accepted, in that order after
the canonical header. The gateway replaces the raw value with a bounded,
domain-separated SHA-256 key and removes all raw session headers. On the
SGLang OpenAI chat and Responses routes, it adds the opaque key as the
request's top-level `session_id`. If no explicit identity is present,
body-derived affinity may still keep a request on one backend, but it does not
register an SGLang session.

Session references are **soft protection, not pinned memory**. Referenced KV
may still be evicted when unreferenced KV is insufficient. Each request must
also continue to carry its complete intended prompt; `session_id` does not
reconstruct conversation history. See SGLang's
[session-aware radix cache documentation](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/session_radix_cache.mdx).

The pinned SGLang build propagates top-level `session_id` through OpenAI chat
completions and Responses requests. Its native Anthropic Messages request
model currently ignores that extension. `/v1/messages` remains wire-compatible
and backend-sticky, but does not receive intra-backend session protection until
the pinned SGLang Anthropic adapter propagates the field. Prefer the OpenAI
chat or Responses route when session-aware retention is required.

Explicit session closure is not currently sent by the gateway. Consequently,
references remain active for the lifetime of the engine process; they remain
evictable under pressure. A future lifecycle integration should call SGLang's
`/close_session` when the owning agent session ends.

No configuration change restarts a live engine. These settings take effect on
the next normal or controlled server activation.
