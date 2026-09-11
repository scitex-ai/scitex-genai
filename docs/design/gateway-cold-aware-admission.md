# Gateway cold-aware admission foundation

Status: observe-only, 2026-09-12. This change is not a production rollout.

SAC's Hermes compiler selects OpenAI Chat Completions or Responses. Inspection
of the compiler found no configured inference-session header and no exact
pre-admission HBM, host, or storage cache-residency result. A stable session ID
would preserve routing identity, but even it and a previous hit would not prove
that pages remain resident in the selected SGLang process. Prompt length does
not prove a miss. The gateway therefore reports residency as `unknown` and
does not enable cache-priority scheduling.

The required future SAC/Hermes client contract is one opaque
`X-SciTeX-Session-ID` value that remains stable for the conversation lifetime.
The gateway already hashes that header before sticky routing or journal
correlation, accepts the legacy `session_id` and `x-session-id` spellings, and
falls back to its existing prompt-derived affinity when no header exists. This
gateway support is not evidence that the current Hermes client sends it. No
request content or raw session identity is added to admission metrics.

Successful relayed responses expose the deterministic state:

```text
X-SciTeX-Admission-Mode: observe-only
X-SciTeX-Cache-Residency: unknown
X-SciTeX-Session-Key: <12-character opaque key, or none>
```

`AdmissionController` is disabled by default. Its future enabled mode accepts
only an authoritative `hot`, `cold`, or `unknown` classification. Known-hot
work may pass queued cold/unknown work at an available-slot boundary. Once the
oldest cold/unknown waiter reaches `max_cold_wait_s`, it receives the next
progressing slot. The controller exposes content-free queue, wait, class, and
overtake counters through `snapshot()`.

Activation requires an engine-owned lookup against the exact upstream and
cache generation selected by sticky routing. With multiple upstreams, each
must have independent admission/residency state because their L1/L2/L3 pages
differ. With one upstream, a gateway can reorder queued work but cannot
preempt a cold prefill already admitted to SGLang; that requires engine
preemption/time slicing or separate capacity.

L1 remains the HBM radix/KV cache, L2 is bounded host HiCache, and any L3
storage cache remains engine-owned. A future signal must report the actual
tier and generation; the gateway must not infer either from latency. The
current engine KV precision is already FP8 E4M3 and is unaffected here.
