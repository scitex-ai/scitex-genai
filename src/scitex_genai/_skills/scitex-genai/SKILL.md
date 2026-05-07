---
name: scitex-genai
description: Modality-organised generative-AI provider abstraction — unified LLM factory (OpenAI / Anthropic / Google / Groq / DeepSeek / Perplexity / Llama) today, with reserved namespaces for agent / image / audio / video / embed / multimodal as the field fragments by modality.
primary_interface: python
interfaces: {python: 3, cli: 0, mcp: 0, skills: 1, hook: 0, http: 0}
---

# scitex-genai

Generative-AI utilities for scientific research. Standalone, factored out
of the legacy `scitex.ai._gen_ai` (and briefly of `scitex-ai`). The
umbrella `scitex-python` exposes this package as `scitex.genai`.

## Layout (modality-based)

| Submodule              | Status     | Notes                                          |
| ---------------------- | ---------- | ---------------------------------------------- |
| `scitex_genai.llm`     | implemented | Unified provider factory `GenAI`. Litellm-backed in a follow-up. |
| `scitex_genai.agent`   | reserved   | Wrapper over `claude-agent-sdk` and friends.   |
| `scitex_genai.image`   | reserved   | Image generation / editing.                    |
| `scitex_genai.audio`   | reserved   | TTS / STT / music.                             |
| `scitex_genai.video`   | reserved   | Video generation.                              |
| `scitex_genai.embed`   | reserved   | Embeddings.                                    |
| `scitex_genai.multimodal` | reserved | Any-to-any unified models.                    |

Reserved namespaces import successfully but raise `NotImplementedError`
on attribute access — import paths are stable as features land.

## Sub-skills

* [llm.md](llm.md) — `GenAI` factory, supported providers, cost tracking, planned litellm backend.

## Quick Reference

```python
import scitex_genai

ai = scitex_genai.GenAI(provider="openai", model="gpt-4o")
print(ai.complete("Explain neural networks in one sentence."))
print(ai.get_cost_summary())
```

## Umbrella access

```python
import scitex
scitex.genai.GenAI  # same object as scitex_genai.GenAI
```

## Optional extras

| Extra        | Purpose                                                    |
| ------------ | ---------------------------------------------------------- |
| `[agent]`    | `claude-agent-sdk` for the (forthcoming) `agent` submodule |
| `[litellm]`  | `litellm` router (future `llm/` backend)                   |
| `[ollama]`   | Local `ollama` provider                                    |
| `[all]`      | Everything available today                                 |
