# SciTeX GenAI (`scitex-genai`)

<p align="center">
  <a href="https://scitex.ai">
    <img src="docs/scitex-logo-blue-cropped.png" alt="SciTeX" width="400">
  </a>
</p>

<p align="center">
  <a href="https://scitex-genai.readthedocs.io/">Full Documentation</a> · <code>pip install scitex-genai</code>
</p>

---

## Overview

`scitex-genai` is the standalone home of generative-AI utilities that
previously lived inside `scitex.ai._gen_ai` (and briefly inside
`scitex-ai`). The package is organised by **modality** so the namespace
ages well as the field fragments:

| Submodule                  | Status     | Notes                                          |
| -------------------------- | ---------- | ---------------------------------------------- |
| `scitex_genai.llm`         | ✅ shipped | Unified provider factory (OpenAI, Anthropic, Google, Groq, DeepSeek, Perplexity, Llama). Litellm-backed in a follow-up. |
| `scitex_genai.agent`       | 🔒 reserved | Wrapper over `claude-agent-sdk` and friends. |
| `scitex_genai.image`       | 🔒 reserved | Image generation / editing.                  |
| `scitex_genai.audio`       | 🔒 reserved | TTS / STT / music.                           |
| `scitex_genai.video`       | 🔒 reserved | Video generation.                            |
| `scitex_genai.embed`       | 🔒 reserved | Embeddings.                                   |
| `scitex_genai.multimodal`  | 🔒 reserved | Any-to-any unified models.                    |

Reserved namespaces import successfully but raise `NotImplementedError` on
attribute access — they exist so import paths are stable as features land.

## Installation

```bash
pip install scitex-genai            # core (LLM providers)
pip install scitex-genai[agent]     # + claude-agent-sdk
pip install scitex-genai[litellm]   # + litellm router (preview)
pip install scitex-genai[ollama]    # + local ollama
pip install scitex-genai[all]       # everything available today
```

## Quick start

```python
from scitex_genai import GenAI

ai = GenAI(provider="openai", model="gpt-4o")
print(ai.complete("Explain neural networks in one sentence."))
print(ai.get_cost_summary())
```

## Roadmap

1. **litellm backend for `llm`** — collapse per-provider adapters into one
   OpenAI-compatible interface, lazy provider imports, drop eager SDK deps.
2. **`agent`** — thin wrapper around `claude-agent-sdk` (already battle-tested
   in `scitex-agent-container`), with room for other backends.
3. **Modality submodules** — `image` (DALL-E / SD / Flux), `audio`
   (ElevenLabs / Whisper / Suno), `video` (Sora / Veo / Runway).
4. **`embed` + `multimodal`** — once stable embedding/multimodal APIs settle.

## License

AGPL-3.0-only. See [LICENSE](LICENSE).
