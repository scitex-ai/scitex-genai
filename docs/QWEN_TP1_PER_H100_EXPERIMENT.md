# Qwen TP=1 per H100 experiment

## Read-only TP=2 baseline

Captured at 2026-09-11 12:42-12:45 UTC from job 30024585 using only
`srun --overlap` node-local observations and serialized requests. No service or
topology was changed.

- Allocation: node `spartan-gpgpu177`, 16 CPUs, 128 GiB RAM, two H100 80GB
  GPUs; job time limit seven days.
- Image: `sglang-qwen38-cu12-amd64-45e39d4c5bcf.sif`, source commit
  `4ccff141dbe992794f9da6c3aa23535b4f72000d`, SGLang
  `0.0.0.dev1+g4ccff141d.d20260907`.
- Topology: TP=2, 1,000,000-token context, `max-running-requests=8`, FP8 KV,
  32,768-token chunked prefill, overlap scheduling, and EAGLE 3-step/4-draft.
- Idle KV state: 0 running, 0 waiting, 0 used tokens, and a
  `max_total_num_tokens` capacity of 2,180,158.
- Idle GPU state: 0% utilization; 79,511 MiB and 78,608 MiB allocated. The
  difference includes model/runtime allocation, not active KV tokens.

Each probe used `/generate`, exact repeated token ID 42, one output token, and a
unique `cache_salt`. SGLang confirmed the exact prompt count and zero cached
tokens. The server returned to zero running/waiting and zero used tokens after
each probe.

| Input tokens | Wall time | Peak KV used | Peak GPU utilization |
| ---: | ---: | ---: | ---: |
| 65,536 | 4.847 s | 65,536 | 100% / 100% |
| 131,072 | 13.915 s | 131,072 | 100% / 100% |
| 262,144 | 44.454 s | 262,144 | 100% / 100% |

An initial 65,536-token harness check also completed, but exposed that the HTTP
response can precede the `/v1/loads` idle transition briefly. It was excluded
from the table; the final harness explicitly waits for idle between probes.

## Proposed TP=1 pair comparison

The experiment holds aggregate concurrency constant: two independent TP=1
replicas, one per H100, each with `max-running-requests=4`. All other material
model, KV, chunking, parser, and EAGLE flags match the TP=2 baseline. Context is
capped at 400,000 tokens with `mem-fraction-static=0.75`: the earlier TP=1 run
allocated 432,564 KV tokens and completed an approximately 270K cold request at
that memory setting. This experiment does not advertise or claim 1M per replica.
It records each replica's KV capacity, repeats the exact cold 64K/128K/256K
series, and runs the disconnect-during-cold-prefill acceptance test against
each patched replica.

The launcher at
`containers/sglang-cancel-safe/run_tp1_pair_experiment.sh` refuses to run
outside Slurm, unless exactly two GPUs are visible, or when either GPU has 2
GiB or more allocated. Therefore it cannot start alongside the current TP=2
server. Run it only in a separate clean two-H100 allocation, or after explicit
approval to replace the current server in a controlled window:

```bash
srun --overlap --ntasks=1 --gres=gpu:2 \
  containers/sglang-cancel-safe/run_tp1_pair_experiment.sh \
  /scratch/$USER/images/sglang-qwen38-cancel-safe-4ccff141.sif \
  /data/scratch/projects/punim0264/$USER/hf/Qwen3.8-27B-FP8
```

Do not point the production tunnel or gateway at experiment ports. Promotion
and rollback remain separate operator decisions documented in the image
README.

## 2026-09-11 transition result

No TP=1 performance result was accepted. The first launch failed before model
loading because the writable per-replica cache directory was not bind-mounted.
After rollback, an explicit no-model write/read preflight passed for both paths.
The second launch then failed at argument validation because the launcher did
not propagate `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` for the 400K YaRN
context cap. Per the two-attempt stop rule, the experiment was rolled back and
not retried. The original TP=2 image, argv, version, 2,180,158-token KV capacity,
and port 8768 health were reverified.

The launcher now includes both offline corrections (explicit writable bind and
the YaRN override), but it has not received an end-to-end GPU acceptance run.
Before any future attempt, add a no-model environment preflight that verifies
both `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN` and cache writability from the
same launch environment, then execute the full test in a separate allocation.
