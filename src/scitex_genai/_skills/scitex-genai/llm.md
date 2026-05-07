---
description: Unified LLM provider factory `GenAI` — same API across OpenAI / Anthropic / Google / Groq / DeepSeek / Perplexity / Llama, with cost tracking and conversation history.
---

# scitex_genai.llm

Single entrypoint for text-completion-style LLMs. Every provider goes through the same `GenAI` factory, returning a client whose `.complete()` and `.stream()` methods have identical signatures regardless of backend.

## Quick reference

```python
from scitex_genai import GenAI

ai = GenAI(provider="openai", model="gpt-4o")
print(ai.complete("Explain neural networks in one sentence."))
print(ai.get_cost_summary())     # cumulative cost since instantiation
```

## Supported providers

| Provider     | `provider=`     | Notes                                        |
| ------------ | --------------- | -------------------------------------------- |
| OpenAI       | `"openai"`      | Requires `OPENAI_API_KEY`.                   |
| Anthropic    | `"anthropic"`   | Requires `ANTHROPIC_API_KEY`.                |
| Google       | `"google"`      | Gemini family. Requires `GOOGLE_API_KEY`.    |
| Groq         | `"groq"`        | Fast OSS-model inference. `GROQ_API_KEY`.    |
| DeepSeek     | `"deepseek"`    | OpenAI-compatible API.                       |
| Perplexity   | `"perplexity"`  | Search-augmented generation.                 |
| Llama        | `"llama"`       | Local / self-hosted Llama servers.           |

Provider classes themselves (`Anthropic`, `OpenAI`, …) live in
`scitex_genai.llm` and inherit from `BaseGenAI`. Use `GenAI(...)` —
the factory — for all normal application code.

## Cost tracking

Every call updates an internal accumulator. `get_cost_summary()` returns a
human-readable string with prompt / completion token counts and the
estimated USD spend per provider in the session.

## Conversation history

Each `GenAI` instance carries a conversation buffer; consecutive
`.complete()` calls re-send the running history. Reset with
`ai.clear_history()`.

## What's coming

- **litellm backend.** Move dispatch under [litellm](https://github.com/BerriAI/litellm) for one OpenAI-compatible interface, lazy provider imports, free streaming/retry/cost-tracking, and out-of-the-box Ollama support (just `model="ollama/llama3"`).
- **Native Ollama path.** Direct `scitex_genai.llm` integration for users who don't want litellm.

## Related

- Reserved sibling submodules: `scitex_genai.agent`, `.image`, `.audio`,
  `.video`, `.embed`, `.multimodal`. Importable but raise
  `NotImplementedError` on attribute access until features land.
- For ML / classification / training utilities (factored out of the same
  legacy `scitex.ai`), see [`scitex-ml`](https://github.com/ywatanabe1989/scitex-ml).
