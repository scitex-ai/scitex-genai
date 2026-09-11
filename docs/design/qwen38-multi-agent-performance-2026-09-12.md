# Qwen3.8-27B multi-agent performance baseline

Date: 2026-09-12 UTC

This record separates observations from proposed experiments. The production
engine was not reconfigured and no synthetic load was sent while collecting
the baseline.

## Observed deployment

- Engine: SGLang `0.0.0.dev1+g4ccff141d.d20260907`
- Model: `Qwen3.8-27B-FP8`
- GPUs: two NVIDIA H100 80 GB devices
- Topology: one engine with tensor parallelism 2
- Context ceiling: 1,000,000 tokens with YaRN factor 4
- Weight format: FP8
- KV-cache format: `fp8_e4m3`
- GPU memory fraction: 0.85
- KV capacity: 2,180,158 logical tokens
- Maximum engine requests: 8
- Gateway admission: 2 requests and approximately 1,100,000 input tokens
- Prefix cache: radix cache and session radix cache enabled
- Scheduler: FCFS
- Chunked prefill: 32,768 tokens
- Dynamic chunking: disabled
- Mixed chunk: disabled
- Hierarchical cache: disabled
- CPU and storage KV offload: disabled
- Speculative decoding: EAGLE, 3 steps, 4 draft tokens

The KV capacity corresponds to approximately 33.27 GiB of target KV and
2.08 GiB of draft KV per tensor-parallel rank. The process reported about
10.22 GiB available per GPU after initialization and CUDA graph capture.

## Observed cache behavior

Across the current process, approximately 92% of prompt tokens were served as
device-cache hits. Warm continuations included:

| Cached tokens | Newly computed tokens |
| ---: | ---: |
| 424,832 | 477 |
| 423,296 | 1,551 |
| 431,182 | 790 |

Therefore ordinary continuation does not recompute the whole conversation
while its exact prefix remains resident.

The cache also recorded approximately 3.0 million evicted tokens. A cold
Scholar request entered with zero cached tokens and roughly 708,000 pending
tokens. SGLang processed it in 32,768-token chunks. Chunk throughput declined
from roughly 8,300 tokens/s near the beginning to roughly 1,400--2,300
tokens/s at long positions.

At the same time, a Hub request with roughly 425,000 cached tokens waited. The
Scholar and Hub requests completed after approximately 304 and 302 seconds.
The Hub prefix was cache-hot; its latency came from waiting behind the cold
prefill.

## Scheduler finding

In the pinned SGLang source, LPM sorts only the waiting queue by matched-prefix
length. Once a cold request becomes `chunked_req`, the scheduler admits that
request's next chunk before iterating the newly sorted waiting queue. The
current 32,768-token chunk consumes the complete prefill-token budget.

Consequences for the observed arrival order:

- Changing FCFS to LPM alone would not allow a cache-hot Hub request to jump
  ahead of a cold Scholar request that has already started chunked prefill.
- Priority scheduling can retract lower-priority decode requests, but does not
  preempt the active `chunked_req` in this build.
- This build permits priority scheduling with FCFS or LOF, not with LPM.
- Mixed chunking can protect decode requests that are already running, but
  does not by itself improve TTFT for a newly queued request.
- Dynamic chunking requires pipeline parallelism and is not applicable to the
  current TP=2, PP=1 engine.
- Overlap scheduling overlaps CPU scheduling with GPU execution; it is not a
  request-fairness mechanism.

## Current bottleneck

The observed failure chain is:

1. Several 400k--700k conversations compete for a 2.18M-token HBM cache.
2. An idle prefix is evicted, or the engine restarts and loses all HBM cache.
3. The next turn performs a very large cold prefill.
4. Chunked prefill advances, but the active chunked request retains scheduler
   precedence.
5. Cache-hot interactive requests wait behind it.

The dominant problem is therefore not normal decode speed and not a general
failure of prefix caching. It is the combination of finite cache residency,
cold long-context prefill, and head-of-line blocking.

## Passive throughput observations

These are production log samples, not a controlled benchmark:

| Concurrent decode requests | Median aggregate output | Approx. per request |
| ---: | ---: | ---: |
| 1 | 146 tokens/s | 146 tokens/s |
| 2 | 197 tokens/s | 99 tokens/s |
| 4 | 233 tokens/s | 58 tokens/s |

Eight concurrent decode requests were not observed. The current gateway admits
only two, so four- and eight-agent end-to-end results must be measured in an
isolated canary rather than inferred from these samples.

## Tiered-cache canary

The pinned engine supports:

1. L1: the existing HBM radix/KV cache.
2. L2: pinned host-memory HiCache.
3. L3: an optional storage backend, including a file backend.

The first canary must test L2 independently before adding L3. The current
128-GB job is too small for an unconstrained ratio-based allocation. Use an
explicit, bounded HiCache size and record the actual KV/Mamba split reported
at startup. A 256-GB host-memory allocation is preferred for the full
experiment.

L2 is process-local and does not survive an engine restart. Use
`write_through`: the pinned implementation's hybrid Mamba state is not safely
covered by `write_back`. A conservative L2-only canary keeps the current page
size and adds only:

```text
--enable-hierarchical-cache
--hicache-size 32
--hicache-io-backend kernel
--hicache-mem-layout page_first
--hicache-write-policy write_through
```

`hicache-size` is decimal GB per TP rank and is divided between full KV and
Mamba pools for this hybrid model. Capacity must therefore be taken from the
startup log, not calculated as though all 32 GB were ordinary KV pages.

For L3, node-local NVMe under `/tmp` is fast and currently has ample capacity,
but does not survive allocation loss. Shared project storage can survive an
engine or allocation restart, but is nearly full and must use a strict maximum
size and free-space watermark. Do not enable an unbounded file cache.

The storage directory must include the model checkpoint, SGLang build, TP,
KV dtype, RoPE profile, and page size. The file backend does not encode all of
these compatibility facts in its own keys, so reusing a directory after any
of them changes can return stale incompatible pages. The full L3 canary uses
page size 64 to avoid creating approximately one file per token, and caps the
file cache explicitly. It must not be promoted until L2-only testing passes.

## Required A/B measurements

Use the same tokenized request bodies, arrival order, output limits, and random
seed for every run. Restart the canary engine between configurations and run a
documented warm-up sequence.

Scenarios:

1. One warm 425k continuation.
2. One cold 740k request.
3. Cold 740k first, then warm 425k after prefill begins.
4. Both requests waiting before admission.
5. Already-decoding Hub followed by cold Scholar.
6. Cache pressure from 425k, 664k, and 405k sessions, followed by reuse.

Configurations:

1. Current FCFS baseline.
2. LPM, to measure only waiting-queue reordering.
3. FCFS with mixed chunking, to measure decode protection.
4. A smaller prefill-token budget, to measure interactive admission latency.
5. L1 plus bounded L2 HiCache.
6. L1 plus bounded L2 and L3, only after L2 passes correctness and latency
   checks.

LPM is a diagnostic for scenario 4, not a proposed fix for scenario 3. For
scenario 3 the canary is expected to confirm that LPM cannot preempt the active
chunked prefill. Priority scheduling has the same limitation. Mixed chunking
must be compared with both EAGLE-off/mixed-off and EAGLE-off/mixed-on controls,
because this pinned build does not permit mixed chunking with speculative
decoding enabled.

Record per request:

- tokenizer-exact prompt tokens;
- device-, host-, and storage-hit tokens;
- newly computed prompt tokens;
- queue time and time to first token;
- inter-token latency and end-to-end duration;
- output tokens/s;
- running and waiting requests;
- HBM, host RAM, storage, and cache utilization;
- eviction, preemption, and starvation events.

Do not promote a configuration merely because aggregate throughput increases.
It must materially reduce interactive TTFT without causing incorrect output,
OOM, unbounded storage growth, or starvation of cold work.

For the simultaneous-waiting LPM experiment, require at least 95% of the warm
prefix to hit cache, Hub TTFT at most 10 seconds and at least 90% below the
FCFS result, Scholar end-to-end time no more than 10% worse, and a starvation
guard proving the cold request begins within 30 seconds after the last of ten
cache-hot arrivals. Repeat each measured cell at least five times after three
warm-ups.

## Pending capacity

The additional two-H100 lease, job `30409720`, was pending for priority at the
time of this record, with an estimated start of 2026-09-15 22:30 UTC. Until a
separate lease is available, do not run the replay against the production
engine.

A dedicated full-tier canary lease, job `30459210`, requests two H100 GPUs,
256 GB host RAM, and one day. It was pending for priority when submitted. The
128-GB job can run scheduler and conservative L2-only experiments; the full
L2/L3 1M-context matrix must wait for the 256-GB canary.
