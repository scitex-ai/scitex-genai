---
description: Unified LLM provider factory `GenAI` — same API across OpenAI / Anthropic / Google / Groq / DeepSeek / Perplexity / Llama, with cost tracking and conversation history.
---

# scitex_genai.llm

Single entrypoint for text-completion-style LLMs. Every provider goes through the same `GenAI(model=...)` factory, which infers the provider from the model name and returns a callable client. The instance's `__call__(prompt)` returns the response text; `.cost` and `.history` track cumulative spend and conversation buffer.

## Quick reference

```python
from scitex_genai import GenAI

ai = GenAI(model="gpt-4o-mini")
print(ai("Explain neural networks in one sentence."))
print("cost USD:", ai.cost)     # cumulative cost since instantiation
```

## Supported providers

The provider is inferred from the model name via `scitex_genai.llm._PARAMS.MODELS`.

| Provider     | API-key env             | Notes                                        |
| ------------ | ----------------------- | -------------------------------------------- |
| OpenAI       | `OPENAI_API_KEY`        | GPT-4 / GPT-4o / o-series.                   |
| Anthropic    | `ANTHROPIC_API_KEY`     | Claude family.                               |
| Google       | `GOOGLE_API_KEY`        | Gemini family.                               |
| Groq         | `GROQ_API_KEY`          | Fast OSS-model inference.                    |
| DeepSeek     | `DEEPSEEK_API_KEY`      | OpenAI-compatible API.                       |
| Perplexity   | `PERPLEXITY_API_KEY`    | Search-augmented generation.                 |
| Llama        | (none — local)          | Local / self-hosted Llama servers.           |

Provider classes themselves (`Anthropic`, `OpenAI`, …) live in
`scitex_genai.llm` and inherit from `BaseGenAI`. Use `GenAI(...)` —
the factory — for all normal application code.

## Cost tracking

Every call updates internal token counters; `ai.cost` exposes the
cumulative USD estimate. `ai.input_tokens` / `ai.output_tokens` give the
raw counts.

## Conversation history

Each `GenAI` instance carries a conversation buffer (`ai.history`);
consecutive `ai(prompt)` calls re-send the running history (controlled
by `n_keep` at construction). Reset with `ai.reset()`.

## What's coming

- **litellm backend.** Move dispatch under [litellm](https://github.com/BerriAI/litellm) for one OpenAI-compatible interface, lazy provider imports, free streaming/retry/cost-tracking, and out-of-the-box Ollama support (just `model="ollama/llama3"`).
- **Native Ollama path.** Direct `scitex_genai.llm` integration for users who don't want litellm.

## Related

- Reserved sibling submodules: `scitex_genai.agent`, `.image`, `.audio`,
  `.video`, `.embed`, `.multimodal`. Importable but raise
  `NotImplementedError` on attribute access until features land.
- For ML / classification / training utilities (factored out of the same
  legacy `scitex.ai`), see [`scitex-ml`](https://github.com/ywatanabe1989/scitex-ml).
