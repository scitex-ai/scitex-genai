# TP=1 cold-prefill admission

The controlled H100 TP=1 measurements on 2026-09-14 found 694,720 device-KV
tokens. Concurrent cold requests did not overlap: 512k concurrency 2 completed
strictly serially (TTFT 291.863 s and 583.891 s), while a 640k immediate replay
with 638,976 cached tokens had 1.717 s TTFT. After one 256k pressure request,
the same 640k prefix retained only 434,176 device-cached tokens; host and storage
restored zero, and TTFT rose to 240.946 s. The runtime logged both a Mamba-file
fetch failure and a discarded hybrid prefetch.

For a measured TP=1 deployment, configure both values together:

```yaml
gateway:
  inference_cold_prefill_limit_per_upstream: 1
  inference_cold_prefill_min_tokens: 256000
```

The guard applies only to a long first turn carrying an explicit stable session
identity. Continuations and unclassified requests are not claimed to be cold:
the gateway has no authoritative pre-admission cache-residency lookup. Both
settings default to unset, so this PR does not change the TP=2 production path.

`SGLangCacheTierValidator` is fail-closed. Device capacity becomes usable when
the startup log reports `max_total_num_tokens`. Merely allocating host or file
HiCache does not make it usable; an actual positive tier hit validates the tier.
A hybrid KV+Mamba restore failure revokes host and storage validation. Operators
must not add configured host/storage tokens to admission capacity while either
tier is unverified or failed.
