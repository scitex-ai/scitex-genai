---
description: |
  [TOPIC] Environment Variables
  [DETAILS] Per-provider API keys and the self-hosted SCITEX_GENAI_* endpoint
  vars read by GenAI() at call time.
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
