# Production Qwen TP=1 profiles

The tracked `qwen-tp1-512k.conf` and `qwen-tp1-256k.conf` examples turn the
measured one-H100 Qwen3.8 configuration into ordinary `scitex-genai-serve`
profiles. They are not canary fixtures and need no manual tmux wrapper. The
canonical supervisor owns the SGLang process, health-gated reverse tunnel,
LiteLLM sidecar, restarts, logs, and stable per-profile JIT cache.

Both profiles retain the merged HPC facts: one H100, TP=1, Qwen3.8-27B FP8
weights, FP8 E4M3 KV, the pinned `4ccff141` SGLang build and SIF, YaRN factor
4, FlashInfer, LPM, 8,192-token prefill chunks, the checkpoint-trained MTP
EAGLE 3/1/4 tuple, session-aware UnifiedRadixCache, metrics, and cache reports.
They deliberately omit HiCache: the measured hybrid-model replay did not
validate host or storage restoration, so those tiers are not production
capacity.

| Profile key | Context ceiling | Observed `max_total_num_tokens` | Engine | LiteLLM | Reverse tunnel | Gateway label | Gateway budget |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `qwen-tp1-512k` | 512,000 | 565,263 | 28769 | 24004 | 18774 | `qwen-tp1-512k` | 500,000 |
| `qwen-tp1-256k` | 256,000 | 572,149 | 28770 | 24005 | 18775 | `qwen-tp1-256k` | 250,000 |

The observed totals are engine-resident token capacities, not gateway
budgets. The lower gateway values preserve output and estimation headroom;
do not copy 565,263 or 572,149 into `token_capacity`.

## Install and launch

Copy the selected tracked profile into the user configuration tree without
editing its key:

```console
install -m 0600 examples/serve/qwen-tp1-512k.conf \
  ~/.scitex/genai/models.d/qwen-tp1-512k.conf
scitex-genai-serve qwen-tp1-512k --dry-run
```

Book each profile as its own persistent one-H100 lease. This keeps GPU
ownership explicit even if the two leases land on different nodes:

```console
scitex-genai-serve launch qwen-tp1-512k \
  --lease qwen-tp1-512k --host hpc --gpus h100:1 \
  --partition gpu-h100 --time 7-00:00:00
scitex-genai-serve launch qwen-tp1-256k \
  --lease qwen-tp1-256k --host hpc --gpus h100:1 \
  --partition gpu-h100 --time 7-00:00:00
```

`scitex-genai-serve launch` renders the persistent lease body from these
profiles. Inside the allocation, each step runs `scitex-genai-serve <key>`;
that supervisor waits for `/health` before exposing the reverse tunnel and
re-establishes the tunnel after a disconnect. Do not add an independent tmux
engine or SSH-forward script around it.

## Gateway members

After both supervised tunnel endpoints are independently healthy, configure
the gateway with the conservative budgets:

```yaml
gateway:
  inference_upstreams:
    - label: qwen-tp1-512k
      url: http://127.0.0.1:18774
      token_capacity: 500000
    - label: qwen-tp1-256k
      url: http://127.0.0.1:18775
      token_capacity: 250000
  inference_cache_report_enabled: true
  cache_admission:
    mode: active
    hot_max_uncached_tokens: 32768
    cold_prefill_limit_per_upstream: 1
    max_hot_bypasses: 4
    starvation_age_s: 30.0
    evidence_max_age_s: 300.0
```

The 32k boundary is on predicted **uncached** tokens. Live feedback after the
two-TP1 rollout observed cache-backed continuations with about 0.3k--15.6k
uncached tokens and a cold lineage break with about 100.3k uncached tokens.
Active mode requires generation-bearing SGLang metrics and cache reports; it
fails before dispatch if that evidence is unavailable. New or incompatible
lineages are conservatively cold, while compatible device/host/storage-backed
lineages may receive hot priority. Four bypasses and 30-second aging bound
starvation without moving a session away from its sticky home.

Configuration files do not restart services. Use the gateway's documented
drained rollout only in a separate, deliberate deployment after checking the
new tunnel health. A missing tunnel is not spare capacity and must not be
listed as an active member.
