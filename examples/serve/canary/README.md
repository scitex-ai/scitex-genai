# Qwen3.8 Scheduler Canary Matrix

These profiles are unmeasured, isolated canary inputs. They do not change the
production profile. Run one profile at a time in a separate two-H100 lease and
send load only with `scitex-genai-sglang-ab`'s explicit canary acknowledgement.
The launcher refuses these profiles unless it observes a SLURM step with the
declared purpose, exactly two H100 GPUs, at least 128 GB host RAM, no existing
GPU compute process, and no local bind conflict on the declared ports. The SSH
tunnel separately uses `ExitOnForwardFailure=yes`, so a remote reverse-port
conflict fails instead of silently publishing the wrong engine. The launcher
also verifies the container and model-manifest digests and records an
incarnation manifest.

Launch through the dedicated lease's overlap step with
`SCITEX_GENAI_CANARY_PURPOSE=qwen38-scheduler`. The launcher refuses direct
booking for a canary profile; resource acquisition remains the responsibility
of `scitex-hpc`.

The matrix fixes every observed production parameter except the named
scheduler variable. The baseline uses FP8 weights, FP8 E4M3 KV cache, TP=2,
1,000,000-token context, a 32,768-token prefill budget, session radix caching,
and checkpoint EAGLE. No profile is labelled faster until its replay result is
recorded.

| Profile | Policy | Chunk | EAGLE | Mixed chunk | Role |
|---|---:|---:|---:|---:|---|
| `fcfs-eagle-c32768` | FCFS | 32,768 | yes | no | production-shape baseline |
| `lpm-eagle-c32768` | LPM | 32,768 | yes | no | negative control |
| `fcfs-eagle-c8192` | FCFS | 8,192 | yes | no | smaller-chunk candidate |
| `fcfs-eagle-c4096` | FCFS | 4,096 | yes | no | smaller-chunk candidate |
| `fcfs-noeagle-mixed-c8192` | FCFS | 8,192 | no | yes | three-dimension compatibility control |

The mixed-chunk row is not a one-variable performance comparison. It changes
three dimensions together: chunk size, EAGLE, and mixed-chunk mode. The pinned
SGLang build disables mixed chunk whenever EAGLE is enabled, so this row can
only test compatibility and already-decoding-request protection. It cannot
attribute any difference to mixed chunk alone.

Every profile exports the same JIT/cache directories under the manifest's
version-pinned namespace. The namespace includes both the SGLang commit and
the container digest prefix. Run the profiles sequentially on the same
dedicated canary lease; do not clear that namespace between profiles. This
makes compilation warm-up identical instead of charging the first profile for
a cold JIT cache.

Prime that namespace with one unmeasured baseline launch and successful
readiness probe before collecting any replay row. Preserve its startup log
with the canary incarnation manifest. Restart the baseline for its measured
row, then run every other profile against the same namespace. A result without
that warm-up evidence is invalid.

Warm-up evidence is deliberately procedural. The current launcher can observe
engine readiness, but it has no trustworthy signal distinguishing an
unmeasured warm-up incarnation from a measured incarnation. It therefore does
not create or infer a warm-up marker.

The primary replay is cold + warm. A cold + cold replay is a separate expected
crash-probe risk, not an ordinary throughput case. The replay harness requires
dedicated-canary metadata and an additional crash-probe acknowledgement; do
not direct it at the production gateway or engine.

## Exact pinned-source basis

The image contains SGLang commit
`4ccff141dbe992794f9da6c3aa23535b4f72000d`. At that commit:

- `chunked_prefill_size` is the per-request chunk bound, while
  `max_prefill_tokens` is the prefill-batch budget.
- The scheduler sorts the waiting queue, constructs `PrefillAdder` with both
  independent values, and adds an existing `chunked_req` before walking the
  sorted waiting queue. LPM therefore cannot overtake a cold request once that
  request becomes `chunked_req`; it remains a required negative control.
- The EAGLE argument hook disables mixed chunk at startup. The mixed-chunk
  control therefore omits every speculative-decoding argument instead of
  pretending the combination is active.
- Dynamic chunking is enabled by the scheduler only when pipeline parallelism
  is greater than one, so it is not included in this TP=2, PP=1 matrix.

Source links are pinned, not `main`:

- [server arguments](https://github.com/sgl-project/sglang/blob/4ccff141dbe992794f9da6c3aa23535b4f72000d/python/sglang/srt/server_args.py#L771-L813)
- [prefill scheduling order](https://github.com/sgl-project/sglang/blob/4ccff141dbe992794f9da6c3aa23535b4f72000d/python/sglang/srt/managers/scheduler.py#L3012-L3102)
- [EAGLE mixed-chunk handling](https://github.com/sgl-project/sglang/blob/4ccff141dbe992794f9da6c3aa23535b4f72000d/python/sglang/srt/arg_groups/speculative_hook.py#L499-L530)

The JSON manifest is the machine-readable source of the intended differences.
Its schema and dry-render tests prevent silent drift between the manifest,
profile files, and generated SGLang argv.
