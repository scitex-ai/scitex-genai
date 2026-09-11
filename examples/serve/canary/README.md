# Qwen3.8 Scheduler Canary Matrix

These profiles are unmeasured, isolated canary inputs. They do not change the
production profile. Run one profile at a time in a separate two-H100 lease and
send load only with `scitex-genai-sglang-ab`'s explicit canary acknowledgement.

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
| `fcfs-noeagle-mixed-c8192` | FCFS | 8,192 | no | yes | mixed-chunk control |

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

- [server arguments](https://github.com/sgl-project/sglang/blob/4ccff141dbe992794f9da6c3aa23535b4f72000d/python/sglang/srt/server_args.py#L802-L839)
- [prefill scheduling order](https://github.com/sgl-project/sglang/blob/4ccff141dbe992794f9da6c3aa23535b4f72000d/python/sglang/srt/managers/scheduler.py#L3238-L3310)
- [EAGLE mixed-chunk handling](https://github.com/sgl-project/sglang/blob/4ccff141dbe992794f9da6c3aa23535b4f72000d/python/sglang/srt/arg_groups/speculative_hook.py#L543-L576)

The JSON manifest is the machine-readable source of the intended differences.
Its schema and dry-render tests prevent silent drift between the manifest,
profile files, and generated SGLang argv.
