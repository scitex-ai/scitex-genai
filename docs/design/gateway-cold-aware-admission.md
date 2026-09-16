# Gateway cold-aware admission foundation

Status: active implementation proposed, 2026-09-17. This document does not
authorize a production rollout.

The gateway now combines a stable session ID, sticky upstream identity, an
authoritative SGLang engine generation, request-lineage digests, and the prior
response's cache report. Prompt length alone never proves a hit. A compatible
lineage on the same engine generation is hot only when its predicted uncached
suffix is within the configured boundary; a new or incompatible lineage is
conservatively cold.

The required future SAC/Hermes client contract is one opaque
`X-SciTeX-Session-ID` value that remains stable for the conversation lifetime.
The gateway already hashes that header before sticky routing or journal
correlation, accepts the legacy `session_id` and `x-session-id` spellings, and
falls back to its existing prompt-derived affinity when no header exists. This
gateway support is not evidence that the current Hermes client sends it. No
request content or raw session identity is added to admission metrics.

Successful relayed responses expose the deterministic state:

```text
X-SciTeX-Admission-Mode: active
X-SciTeX-Cache-Residency: hot|cold
X-SciTeX-Session-Key: <12-character opaque key, or none>
```

Active mode remains opt-in. Known-hot work may pass queued cold work at an
available-slot boundary. Bypasses are counted, and aged ordinary work receives
the next fitting slot, preventing an endless hot stream from starving cold
work. Session affinity remains authoritative: cache priority never migrates a
conversation to another engine.

Activation requires generation-bearing engine metrics and cache reporting on
the exact sticky upstream. Missing generation or malformed historical cache
evidence fails before dispatch. No compatible history is a safe cold result,
not a guessed hit. With multiple upstreams, each retains independent evidence
because their L1/L2/L3 pages differ. The gateway can reorder work waiting at
its boundary but cannot preempt a cold prefill already admitted to SGLang.

L1 remains the HBM radix/KV cache, L2 is bounded host HiCache, and any L3
storage cache remains engine-owned. A future signal must report the actual
tier and generation; the gateway must not infer either from latency. The
current engine KV precision is already FP8 E4M3 and is unaffected here.
