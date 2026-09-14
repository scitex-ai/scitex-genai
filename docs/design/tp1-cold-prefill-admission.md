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
  inference_cold_prefill_min_tokens: 128000
```

The threshold applies to predicted **uncached prefill work**, not full prompt
length, turn count, or a claim that a continuation is resident.  A compatible
lineage uses the previous response's actual input and cached-token report plus
new prompt growth.  Thus a 640k request with about 640k cached remains eligible
to coexist with other hot continuations, while large unknown/cold prefills are
limited.  A missing cache report and an engine-generation or lineage mismatch
remain unknown and conservatively budget the full prompt.  Existing request
count and total-input-token caps remain independent safety ceilings.
The example's 128k threshold is deliberately below the observed 140,138-token
uncached remainder; it is guidance, not a package default or deployment change.

The live schema-v2 evidence that motivated this distinction was not KV
exhaustion: SGLang reported 2 running / 1 waiting and token usage about 0.63.
A 243,434-token request with only 103,296 cached waited behind roughly 677k of
cold work and reached first output after 263.346 s.  In contrast, a hot ~640k
continuation with ~640k cached reached first output in about 4 s.

Cache evidence also expires after 300 seconds.  The live failure mode was
compounding: a 642,616-token request queued 362.803 s at the gateway, then
reached SGLang with only 160,640 cached (13,184 device + 147,456 host, storage
zero).  Its TTFT was 217.480 s and total latency 583.051 s, implying about 482k
uncached work despite having previously been hot.  Historical evidence older
than the bound is unknown; a queued hot ticket crossing the same bound is
reclassified as unknown/full-prefill before it can be admitted.  Separately, a
141,198-token request with 140,480 cached still took 254.907 s total under
concurrent decode load, so this guard does not claim to solve decode contention.

Queue fairness is still bounded by the existing age and bypass rules.  Once a
large uncached prefill can make progress, aged work receives the next fitting
slot; cancelled waiters release their queue ownership.  Both cold-prefill
settings default to unset, so merging this mechanism alone changes no deployed
profile and makes no SGLang, TP, or context-window change.

`SGLangCacheTierValidator` is fail-closed. Device capacity becomes usable when
the startup log reports `max_total_num_tokens`. Merely allocating host or file
HiCache does not make it usable; an actual positive tier hit validates the tier.
A hybrid KV+Mamba restore failure revokes host and storage validation. Operators
must not add configured host/storage tokens to admission capacity while either
tier is unverified or failed.
