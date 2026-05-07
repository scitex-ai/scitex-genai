---
description: |
  [TOPIC] Installation
  [DETAILS] `pip install scitex-genai`. Modality-organised generative-AI provider abstraction. Core install is provider-agnostic; install the matching provider extra (`openai`, `anthropic`, `google`, `groq`) for each LLM you call. Or install `scitex-genai[all]` for every provider.
tags: [scitex-genai-installation]
---

# Installation

## Standard

```bash
pip install scitex-genai
```

Pulls `scitex-config>=0.3.0` (for env-var resolution) and core glue.
The provider SDKs are **opt-in extras** — installing each only when you
need it keeps cold-start light.

## Provider extras

```bash
pip install scitex-genai[openai]      # openai SDK
pip install scitex-genai[anthropic]   # anthropic SDK
pip install scitex-genai[google]      # google-genai SDK (Gemini)
pip install scitex-genai[groq]        # groq SDK
pip install scitex-genai[all]         # every provider above
```

Reserved-but-not-implemented modalities (`agent`, `image`, `audio`,
`video`, `embed`, `multimodal`) have no extras yet — installing them
errors with `NotImplementedError` on attribute access.

## API keys

`scitex-genai` reads provider keys from environment variables:

| Provider     | Env variable                  |
| ------------ | ----------------------------- |
| OpenAI       | `OPENAI_API_KEY`              |
| Anthropic    | `ANTHROPIC_API_KEY`           |
| Google       | `GOOGLE_API_KEY`              |
| Groq         | `GROQ_API_KEY`                |
| DeepSeek     | `DEEPSEEK_API_KEY`            |
| Perplexity   | `PERPLEXITY_API_KEY`          |
| Llama        | `LLAMA_API_KEY`               |

Resolution is lazy at first call (not at import) — a missing key only
errors when you actually invoke that provider.

## Verify

```bash
python -c "import scitex_genai; print(scitex_genai.__version__)"
```

Should print `0.1.0` or later.

## Umbrella shim

The `scitex` umbrella re-exports this package as `scitex.genai`:

```python
import scitex
ai = scitex.genai.GenAI(model="gpt-4o-mini")
```

Both surfaces share the same API.
