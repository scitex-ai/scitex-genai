---
description: |
  [TOPIC] Installation
  [DETAILS] `pip install scitex-genai`. Modality-organised generative-AI provider abstraction. Provider SDKs (OpenAI, Anthropic, Google, Groq) are included in the core install. Or install `scitex-genai[all]` for everything.
tags: [scitex-genai-installation]
---

# Installation

## Standard

```bash
pip install scitex-genai
```

Pulls `scitex-dev>=0.11.7` (for env-var resolution, audit, and path
management) and the major provider SDKs (`openai`, `anthropic`,
`google-genai`, `groq`) as core dependencies.

## Optional extras

```bash
pip install scitex-genai[agent]     # + claude-agent-sdk (forthcoming `agent` submodule)
pip install scitex-genai[litellm]   # + litellm router (preview)
pip install scitex-genai[ollama]    # + local ollama
pip install scitex-genai[all]       # agent + litellm + ollama
```

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

Should print `0.1.1` or later.

## Umbrella shim

The `scitex` umbrella re-exports this package as `scitex.genai`:

```python
import scitex
ai = scitex.genai.GenAI(model="gpt-4o-mini")
```

Both surfaces share the same API.
