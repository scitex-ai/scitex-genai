---
description: |
  [TOPIC] Environment Variables
  [DETAILS] Per-provider API keys, the self-hosted SCITEX_GENAI_* endpoint
  vars, and the SCITEX_GENAI_BACKEND dispatch switch read by GenAI() at call
  time.
tags: [scitex-genai-env-vars]
---

# scitex-genai — Environment Variables

Resolution is lazy (at first call, not import): a var is only needed when you
actually invoke the path that reads it.

## Provider API keys

| Variable | Purpose | Default | Type |
|---|---|---|---|
| `OPENAI_API_KEY` | OpenAI (GPT / o-series). | (unset) | string |
| `ANTHROPIC_API_KEY` | Anthropic (Claude). | (unset) | string |
| `GOOGLE_API_KEY` | Google (Gemini). | (unset) | string |
| `GROQ_API_KEY` | Groq. | (unset) | string |
| `DEEPSEEK_API_KEY` | DeepSeek. | (unset) | string |
| `PERPLEXITY_API_KEY` | Perplexity. | (unset) | string |
| `LLAMA_API_KEY` | Local / self-hosted Llama. | (unset) | string |

## Self-hosted / OpenAI-compatible endpoint

| Variable | Purpose | Default | Type |
|---|---|---|---|
| `SCITEX_GENAI_BASE_URL` | Base URL of a self-hosted OpenAI-compatible endpoint (e.g. a vLLM model behind a LiteLLM proxy). | (unset) | string (URL) |
| `SCITEX_GENAI_API_KEY` | API key for that endpoint. | (unset) | string |

These two are read **only on the self-hosted passthrough path** — an
unknown/local model name (e.g. `qwen36-35b-a3b`) or an explicit `base_url`.
Known provider models keep using their own `*_API_KEY` above and are
unaffected.

```python
# env injected (SCITEX_GENAI_BASE_URL + SCITEX_GENAI_API_KEY set):
GenAI(model="qwen36-35b-a3b")
# or explicit (explicit args win over env):
GenAI(model="qwen36-35b-a3b", base_url="http://host:4000/v1", api_key="sk-...")
```

## Gateway authentication

| Variable | Purpose | Default | Type |
|---|---|---|---|
| `SCITEX_GENAI_GATEWAY_API_KEY` | The key clients present to the gateway, and the one the gateway validates. | (unset) | string |

**This variable has a file home, and that is the point.** It resolves from the
environment first, then from `~/.scitex/genai/secrets` — beside
`~/.scitex/genai/config.yaml`, so what a gateway *is* and what *opens* it live
together.

```
# ~/.scitex/genai/secrets      (mode 0600, NAME=value, no `export`)
SCITEX_GENAI_GATEWAY_API_KEY=<64 hex characters>
```

Why the file exists: a shell profile is readable only by a process that gets a
login shell. The gateway's unit used to arrange one for itself; nothing else
did. Measured 2026-09-05 to -07, one gateway was healthy for 33 hours
(`/health` 200, no restarts) and served **zero** completions — 444 consecutive
`401`s — because every client resolved the variable to the empty string and
presented an empty bearer token. A file any process can read removes that
class of failure.

**Only the gateway mints a key.** `scitex-genai-gateway` and
`scitex-genai-gateway install-unit` create one when nothing resolves; a client
refuses and names the file instead. That asymmetry is deliberate: a key minted
on a host with no gateway opens nothing while looking like a configured
system, which is worse than the missing file it would replace.

`install-unit` captures whatever key currently resolves — including one that
exists only as an `export` line in your profile — into the file *before*
writing the unit, so migrating does not change the key your clients already
present.

## Dispatch backend

| Variable | Purpose | Default | Type |
|---|---|---|---|
| `SCITEX_GENAI_BACKEND` | Dispatch backend for `GenAI()`. `litellm` routes ANY provider through the single litellm-backed handler (experimental, opt-in). | `default` (per-provider classes) | `default` \| `litellm` |

Explicit `GenAI(backend=...)` wins over the env var; `backend="default"`
forces the classic dispatch even when the env is set. Unknown values raise
`ValueError` (typos fail loudly).

## Notes

- Namespaced on purpose. Do **not** use `OPENAI_BASE_URL`: the openai SDK
  auto-reads it and would silently redirect real `gpt-*` traffic to the
  self-hosted proxy.
- Precedence: explicit `GenAI(...)` args > `SCITEX_GENAI_*` env > the
  per-provider SDK default.

## Audit

```bash
rg -ho 'SCITEX_[A-Z0-9_]+' src/ | sort -u
```
