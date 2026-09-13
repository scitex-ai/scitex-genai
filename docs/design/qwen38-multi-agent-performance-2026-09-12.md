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

## Live promotion evidence

The guarded profile was promoted on 2026-09-12 after the production TP=2
engine passed these checks on two H100 GPUs:

- Live `server_args` reported LPM scheduling, 8,192-token prefill chunks,
  FP8 E4M3 KV, cache reporting, and the EAGLE 3/1/4 tuple.
- L2 exposed 1,263,680 ordinary-KV tokens per TP rank and accepted 370,688
  tokens during the observed continuation.
- L3 reopened 15,345,057,792 bytes and 6,556 entries per TP rank after an
  engine restart.
- The first resumed long request after that restart loaded 379,776 tokens
  from storage and computed 2,112 prompt tokens: 99.45% measured storage
  reuse for that request.
- The gateway returned HTTP 200 with `reachable=true` and `ready=true`; the
  observed generation gauge was 137 tokens/s before the controlled restart.

These measurements establish working persistence and reuse for the observed
request. They do not establish eight-agent concurrency or starvation bounds;
those remain separate admission tests.

## Pending capacity

The additional two-H100 lease, job `30409720`, was pending for priority at the
time of this record, with an estimated start of 2026-09-15 22:30 UTC. Until a
separate lease is available, do not run the replay against the production
engine.

A dedicated full-tier canary lease, job `30459210`, requests two H100 GPUs,
256 GB host RAM, and one day. It was pending for priority when submitted. The
128-GB job can run scheduler and conservative L2-only experiments; the full
L2/L3 1M-context matrix must wait for the 256-GB canary.

## Controlled TP=1 capacity run, 2026-09-14

The previously split job `30409720` exposed one otherwise idle H100 on
`spartan-gpgpu013`. A dedicated overlap step launched the pinned TP=1 canary
there. It did not send synthetic load to the production engine. The committed
fixture, profile, and matrix are in
[PR 82](https://github.com/scitex-ai/scitex-genai/pull/82); the raw result root
is
`/data/scratch/projects/punim0264/ywatanabe/canary-results/tp1-context-20260914T030500Z/`.

The canary kept the production model, FP8 weights, FP8 E4M3 KV, LPM,
8,192-token prefill chunks, 32,768 maximum prefill tokens, EAGLE 3/1/4, and
three cache tiers. Only tensor parallelism changed from two to one. Startup
reported:

- configured `context_len`: 1,000,000 tokens;
- actual device-KV capacity: 694,720 tokens;
- host HiCache capacity: 604,480 ordinary-KV tokens;
- model weights: 28.72 GiB;
- device KV: 21.20 GiB;
- HBM in use during measured rows: approximately 70,058--70,060 MiB.

Therefore the one-million-token setting is an API ceiling, not a promise that
one TP=1 H100 can hold such a request. The unsafe 750k row was replaced with a
640k single-request edge row after startup disclosed the actual capacity.

### Cold rows

Every request used an exact input-token list and generated 32 tokens. Prefixes
were distinct. The 128 cached tokens in the first row came from the endpoint
validation request and are negligible but retained in the record rather than
silently relabelled as cold.

| Prompt | Concurrent | Cached/request | TTFT (s) | Queue (s) | Max waiting | Max active KV | Mean GPU | Prompt tok/s | Decode tok/s/request | Output tok/s including prefill |
| ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| 256k | 1 | 128 | 83.406 | 0.886 | 0 | 254,080 | unavailable | 3,058.7 | 110.55 | 0.382 |
| 256k | 2 | 0, 0 | 83.150, 163.631 | 0.665, 80.507 | 1 | 507,904 | unavailable | 3,123.0 | 0.40, 101.56 | 0.390 |
| 256k | 4 | 0 each | 82.229--325.046 | 0.002--241.931 | 3 | 512,128 | 99.63% | 3,146.7 | 0.40--95.66 | 0.393 |
| 512k | 1 | 0 | 293.221 | 1.199 | 0 | 512,064 | 99.21% | 1,743.4 | 69.74 | 0.109 |
| 512k | 2 | 0, 0 | 291.863, 583.891 | 0.002, 292.223 | 1 | 512,064 | 99.73% | 1,752.7 | 88.24, 88.64 | 0.110 |
| 640k | 1 | 0 | 446.833 | 1.472 | 0 | 640,064 | 99.48% | 1,431.1 | 84.69 | 0.072 |

The four-request 256k row reached only 448 free device-KV tokens and 604,160
of 604,480 host-cache tokens. SGLang exposed one running request and three
waiting requests. The 512k pair was also strictly serialized. No request
failed and health remained HTTP 200 after every row, but concurrency increased
latency rather than useful prefill throughput. Long-position prefill also
became materially slower: aggregate prompt throughput fell from approximately
3,100 tokens/s at 256k to 1,431 tokens/s at 640k.

Decode was not uniformly protected while another cold prefill ran. In the
256k two-request row, one request's TPOT was 10.2 ms while the other's was
2.606 s. In the four-request row two requests had approximately 10.8 ms TPOT
and two had approximately 2.6 s TPOT. The long prefill therefore delayed both
admission and already-started output in this fixed EAGLE/non-mixed-chunk
configuration.

### Cache survival rows

An immediate replay of the 640k request hit 638,976 tokens in device cache and
reduced TTFT from 446.833 seconds to 1.717 seconds. This proves that exact
resident-prefix reuse is effective.

After other prefixes created pressure:

- the old 256k prefix reported zero cached tokens and TTFT 81.691 seconds;
- the 640k prefix retained 434,176 device tokens, but reported zero host and
  zero storage hits; recomputing the missing 205,824 long-position tokens took
  TTFT to 240.946 seconds;
- lifetime canary counters ended at 1,073,280 device-hit tokens, zero host-hit
  tokens, and zero storage-hit tokens, despite 3.55 million storage backup
  tokens and a 28 GiB file cache;
- the engine logged `Failed to fetch ... .mamba from HiCacheFile storage` and
  then `HiCache hybrid prefetch discarded ... completed=6464 requested=6464`
  for the pressured 640k replay.

Thus L2/L3 capacity cannot yet be counted as usable admission capacity for
this hybrid model. The file backend accepted writes, but the measured replay
did not restore a usable full hybrid prefix when its Mamba artifact was
missing.

### Correlated production incident

The gateway journal provides a real-agent counterpart. For FigRecipe
conversation `8e9ef1e0`, the 2026-09-13 16:54:54 UTC turn had 758,767 input
tokens and 757,504 device-cached tokens. At 17:00:48 UTC, the next observed
turn had 760,118 input tokens but only 103,296 cached: 8,192 device, 95,104
host, and zero storage. Its TTFT was 315.896 seconds and total gateway time was
341.400 seconds. Hub and UI requests then entered with queue times 330.711 and
319.867 seconds. This is observed eviction between adjacent turns, not a
general cache-hit-rate inference.

### Measured admission conclusion

For this TP=1 profile, the responsive envelope is one cold long-context
prefill at a time. Four agents may remain resumable, but four simultaneous
cold 256k turns are not an interactive workload: their TTFTs are serialized
out to 325 seconds. A single resident 640k continuation is fast, but one 256k
pressure request was enough to make the next 640k turn recompute 205,824
long-position tokens.

The smallest evidence-backed rules are:

1. Admit at most one cold prefill per TP=1 engine.
2. Do not admit from configured `max_model_len`; use actual prompt tokens,
   measured resident-prefix tokens, and the 694,720-token device capacity.
3. Keep device-resident interactive continuations sticky.
4. Treat host and storage cache as experimental until a validator proves that
   both KV and Mamba artifacts restore and the response reports tier hits.
5. Do not route 750k--1M agents to TP=1. TP=2 remains required for that actual
   context range; a two-replica TP=1 topology is only a candidate for shorter
   agents and needs a separate controlled comparison.

These results do not establish a general four-agent limit. They establish a
one-cold-prefill limit for one H100 under the exact pinned configuration and
show why admission must distinguish cache-hot continuations from cold turns.
