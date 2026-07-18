---
description: |
  [TOPIC] LLM Submodule
  [DETAILS] Unified LLM provider factory `GenAI` — same API across OpenAI /
  Anthropic / Google / Groq / DeepSeek / Perplexity / Llama, with cost tracking
  and conversation history. Provider is inferred from the model name via the
  internal `_PARAMS.MODELS` table; provider SDKs are opt-in extras to keep
  cold-start light. Opt-in litellm-backed routing (backend="litellm" or
  SCITEX_GENAI_BACKEND=litellm) consolidates the dispatch layer; the default
  stays the per-provider classes for now.
tags: [scitex-genai-llm]
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
| Self-hosted  | `SCITEX_GENAI_API_KEY`  | OpenAI-compatible endpoint via `SCITEX_GENAI_BASE_URL` (e.g. vLLM behind LiteLLM). |

Provider classes themselves (`Anthropic`, `OpenAI`, …) live in
`scitex_genai.llm` and inherit from `BaseGenAI`. Use `GenAI(...)` —
the factory — for all normal application code.

For a self-hosted, OpenAI-compatible model, pass an unknown model name
plus `base_url` (and `api_key`), or set `SCITEX_GENAI_BASE_URL` +
`SCITEX_GENAI_API_KEY` and just give the model name:

```python
GenAI(model="qwen36-35b-a3b", base_url="http://host:4000/v1", api_key="sk-...")
```

The factory skips the MODELS lookup for unknown names when a `base_url`
(or explicit `provider`) is given; explicit args win over the env
fallback, which engages only on this passthrough path.

## Cost tracking

Every call updates internal token counters; `ai.cost` exposes the
cumulative USD estimate. `ai.input_tokens` / `ai.output_tokens` give the
raw counts.

## Conversation history

Each `GenAI` instance carries a conversation buffer (`ai.history`);
consecutive `ai(prompt)` calls re-send the running history (controlled
by `n_keep` at construction). Reset with `ai.reset()`.

## litellm backend (landed — experimental, opt-in)

Dispatch can route through [litellm](https://github.com/BerriAI/litellm):
ONE OpenAI-compatible code path serves every provider and self-hosted
endpoints, instead of one handler class per provider SDK. Opt in per call
or fleet-wide via env:

```python
GenAI(model="claude-3-5-haiku-20241022", backend="litellm")  # per call
# or: export SCITEX_GENAI_BACKEND=litellm                    # fleet-wide
```

- The **default backend is unchanged** (per-provider classes); flipping the
  default is a later step once the litellm path is battle-tested.
- Explicit `backend=` wins over `SCITEX_GENAI_BACKEND`; pass
  `backend="default"` to force the classic dispatch under that env.
- Same instance contract: `ai(prompt)`, `.cost`, `.history`, `.reset()`,
  `.stream`, `.input_tokens` / `.output_tokens`.
- Provider selection uses litellm model-string prefixes internally
  (`anthropic/claude-*`, `gemini/gemini-*`, `groq/...`, `deepseek/...`,
  `perplexity/...`); `ai.model` stays the plain name you passed.
- Self-hosted works through both backends: an unknown model + `base_url`
  maps to litellm's `openai/<model>` OpenAI-compatible convention.
- litellm is imported lazily — the default backend does not pay its
  import cost.

## What's coming

- **Flip the default to litellm** and demote per-provider SDKs to optional
  extras (lazy provider imports, lighter cold start, out-of-the-box Ollama
  via `model="ollama/llama3"`).
- **Native Ollama path.** Direct `scitex_genai.llm` integration for users who don't want litellm.

## Related

- Reserved sibling submodules: `scitex_genai.agent`, `.image`, `.audio`,
  `.video`, `.embed`, `.multimodal`. Importable but raise
  `NotImplementedError` on attribute access until features land.
- For ML / classification / training utilities (factored out of the same
  legacy `scitex.ai`), see [`scitex-ml`](https://github.com/ywatanabe1989/scitex-ml).
