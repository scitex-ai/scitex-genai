# SGLang HiCache canary profiles

These profiles reproduce the measured Qwen3.8-27B FP8 deployment while adding
one cache tier at a time. The recipe declares a stable purpose and minimum
resources; a SLURM job ID is runtime state and is never stored in the profile.
Neither profile may be added to the production gateway upstream pool.

| Profile | Purpose | Cache tiers | Required resources |
| --- | --- | --- | --- |
| `qwen38-27b-sglang-hicache-l2-canary.conf` | `qwen38-hicache-l2` | L1 HBM + L2 pinned RAM | 2× H100, 128 GB RAM |
| `qwen38-27b-sglang-hicache-l3-canary.conf` | `qwen38-hicache-l3` | L1 + L2 + bounded L3 file | 2× H100, 256 GB RAM |

Both retain the observed TP=2, 1,000,000-token context ceiling, FP8 weights,
`fp8_e4m3` KV cache, 0.85 static GPU-memory fraction, 32,768-token prefill
chunks, session radix cache, and EAGLE settings.

## Immutable engine and model identity

The real-launch guard hashes the SIF and requires digest
`b742f112f8403417c781216e9d4cf9d7eff49f2d6127e682805c40225a74cab2`.
The container preflight additionally requires SGLang
`0.0.0.dev1+g4ccff141d.d20260907`.

The model guard hashes `config.json`, tokenizer configuration and vocabulary,
the chat template, checkpoint index, and the checkpoint CRC manifest. It also
streams every `.safetensors` shard through CRC32 and requires the manifest to
name every shard exactly once. Their
canonical manifest is checked in at
`examples/serve/manifests/qwen38-27b-fp8.sha256`; its digest is
`ae63fb8baffb044e4d0ee476a03283de640690e50fe3282c95996d9dc016a01c`.
That digest is also part of the L3 directory name, preventing incompatible
model/tokenizer/checkpoint state from sharing stored KV pages.

## Safety and sizing

`--hicache-size` and file-backend `max_size` are decimal GB **per TP rank**.
The L2 profile requests 32 GB/rank, or at most 64 GB across TP=2. The L3
profile requests 48 GB/rank of host cache and 32 GB/rank of file cache, or at
most 96 GB RAM and 64 GB NVMe across TP=2. Qwen3.8 is hybrid, so SGLang divides
host capacity between regular KV pages and Mamba/GDN state. Record the actual
split and token capacity from startup logs.

Both profiles use `write_through`. Do not substitute `write_back`: the pinned
implementation does not safely persist all branching hybrid-model state that
way. L2 is process-local and is lost when the engine exits.

The optional L3 profile stores files on node-local `/tmp`. The file cache keeps
100 GB free and evicts at 90%. It survives an engine restart inside the same
allocation, not allocation loss. Page size 64 avoids approximately one file per
token but changes cache granularity, so L3 remains experimental after L2 passes.

## Inspect without starting anything

```bash
scitex-genai-serve qwen38-27b-sglang-hicache-l2-canary \
  --models-dir /path/to/canary-models.d --dry-run
```

Dry-run does not require a lease. A real launch requires a dedicated idle lease:
exactly two H100 GPUs, sufficient RAM, an `srun` step, no existing GPU compute
process, and unused engine/sidecar ports. GPU visibility is inherited from
`srun`; profiles never replace `CUDA_VISIBLE_DEVICES` with assumed indices.
Never reuse a lease whose hold body
already starts an inference engine.

For L2, run this exact step after resolving `CANARY_JOB_ID` from the lease store:

```bash
SCITEX_GENAI_CANARY_PURPOSE=qwen38-hicache-l2 \
srun --overlap --jobid="${CANARY_JOB_ID:?}" --nodes=1 --ntasks=1 \
  --gpus=2 --cpus-per-task=16 --mem=128G \
  scitex-genai-serve qwen38-27b-sglang-hicache-l2-canary \
  --models-dir /path/to/canary-models.d
```

For L3, the exact step is:

```bash
SCITEX_GENAI_CANARY_PURPOSE=qwen38-hicache-l3 \
srun --overlap --jobid="${CANARY_JOB_ID:?}" --nodes=1 --ntasks=1 \
  --gpus=2 --cpus-per-task=16 --mem=256G \
  scitex-genai-serve qwen38-27b-sglang-hicache-l3-canary \
  --models-dir /path/to/canary-models.d
```

At launch, the validator resolves the current job, step, node, GPU inventory,
RAM, occupied GPU processes, ports, image digest, and model manifest. It
publishes that run-specific incarnation through `scitex_dev.store.host_store()`
to the central PostgreSQL state store on port 55432. Failure to publish refuses
the launch; there is no node-local JSON fallback. The record identity is
`slurm-<job>-step-<step>-<node>`: runtime state, not recipe.

## Promotion gate

Test L2 before L3. For each tier, use the fixed bodies and measurements in
`SGLANG_AB_BENCHMARK.md`: cold and repeated prompts, forced L1 eviction, cache
hit tokens per tier, queue time, TTFT, TPOT, throughput, HBM/RAM/storage usage,
and output correctness. For L3, additionally force L2 pressure, restart only
the engine, and prove a storage hit after restart. Reject any configuration
that causes incorrect output, OOM, unbounded storage growth, or cold-request
starvation.
