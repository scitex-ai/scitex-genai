---
name: scitex-genai
description: |
  [WHAT] Modality-organised generative-AI provider abstraction — one `GenAI` factory
  routes prompts to OpenAI / Anthropic / Google / Groq / DeepSeek / Perplexity /
  Llama by inferring the provider from the model name. Cost tracking and
  conversation history come for free on every instance. Reserved namespaces for
  `agent` / `image` / `audio` / `video` / `embed` / `multimodal` so import paths
  stay stable as the field fragments by modality.
  [WHEN] Calling LLMs from research code; switching providers without per-provider
  glue; budget-tracking grant-funded experiments; building agents that need
  consistent client semantics across vendors.
  [HOW] `pip install scitex-genai` (provider SDKs included in core). Read
  `01_installation.md` for the per-provider env-var matrix, `02_quick-start.md`
  for one-call examples, `03_python-api.md` for the `GenAI` class signature, and
  `10_llm.md` for the full provider-routing table.
tags: [scitex-genai]
primary_interface: python
interfaces: {python: 3, cli: 1, mcp: 0, skills: 1, hook: 0, http: 1}
---

# scitex-genai

Generative-AI utilities for scientific research. Standalone, factored out
of the legacy `scitex.ai._gen_ai` (and briefly of `scitex-ai`). The
umbrella `scitex-python` exposes this package as `scitex.genai`.

## Layout (modality-based)

| Submodule                    | Status      | Notes                                                            |
| ---------------------------- | ----------- | ---------------------------------------------------------------- |
| `scitex_genai.llm`           | implemented | Unified provider factory `GenAI`. Opt-in litellm backend via `backend="litellm"`. |
| `scitex_genai.gateway`       | implemented | Anthropic Messages endpoint backed by Codex subscription accounts. |
| `scitex_genai.agent`         | reserved    | Wrapper over `claude-agent-sdk` and friends.                     |
| `scitex_genai.image`         | reserved    | Image generation / editing.                                      |
| `scitex_genai.audio`         | reserved    | TTS / STT / music.                                               |
| `scitex_genai.video`         | reserved    | Video generation.                                                |
| `scitex_genai.embed`         | reserved    | Embeddings.                                                      |
| `scitex_genai.multimodal`    | reserved    | Any-to-any unified models.                                       |

Reserved namespaces import successfully but raise `NotImplementedError`
on attribute access — import paths are stable as features land.

## Sub-skills

- [01_installation.md](01_installation.md) — `pip install`, provider extras, env-var matrix.
- [02_quick-start.md](02_quick-start.md) — three-line LLM call, provider switching, conversation history.
- [03_python-api.md](03_python-api.md) — `GenAI` class signature, instance state, errors.
- [04_cli-reference.md](04_cli-reference.md) — gateway command, options, and environment requirements.
- [06_http-api.md](06_http-api.md) — Claude Code-compatible endpoint, Codex OAuth accounts, quota-aware routing.
- [10_llm.md](10_llm.md) — full provider table, cost tracking internals, opt-in litellm backend.
- [20_env-vars.md](20_env-vars.md) — provider API keys, self-hosted `SCITEX_GENAI_*` endpoint vars, `SCITEX_GENAI_BACKEND` switch.

## Quick Reference

```python
import scitex_genai

ai = scitex_genai.GenAI(model="gpt-4o-mini")
print(ai("Explain neural networks in one sentence."))
print("cost USD:", ai.cost)
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
| `[litellm]`  | No-op alias (litellm is now a core dep for `backend="litellm"`) |
| `[ollama]`   | Local `ollama` provider                                    |
| `[gateway]`  | FastAPI, HTTPX, and Uvicorn for the model gateway          |
| `[all]`      | Everything available today                                 |
