---
description: |
  [TOPIC] scitex-genai — Environment Variables (SCITEX_GENAI_* surface)
  [DETAILS] Environment variables scitex-genai reads at runtime. Cloud-provider API keys are NOT listed here — they follow each provider's canonical env (ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY, DEEPSEEK_API_KEY, PERPLEXITY_API_KEY, LLAMA_API_KEY). This leaf documents the scitex-genai-owned `SCITEX_GENAI_*` knobs that route to LOCAL providers (vLLM today; ollama / litellm future).
tags: [scitex-genai-env-vars]
---

# scitex-genai — Environment Variables

scitex-genai-owned env vars follow the `SCITEX_GENAI_<MODULE>_<KNOB>` convention. Provider API keys keep their canonical names (e.g. `OPENAI_API_KEY`) and are NOT in scope here — see `01_installation.md` for that matrix.

## vLLM provider (`provider="vLLM"` in `_PARAMS.MODELS`)

| Var | Default | Effect |
|---|---|---|
| `SCITEX_GENAI_VLLM_BASE_URL` | `http://127.0.0.1:8765/v1` | Base URL of the running vLLM server. Must end in `/v1` (the OpenAI API root). Override for a remote box or a custom port. |
| `SCITEX_GENAI_VLLM_API_KEY`  | placeholder `"EMPTY"` | Sent in the `Authorization` header. vLLM IGNORES the value by default; set only if a reverse proxy / gateway in front of vLLM enforces a token. The openai SDK rejects empty strings, hence the placeholder. |

Resolution order (highest priority first):

1. Explicit keyword arg on `VLLM(...)` (`base_url=`, `api_key=`).
2. The env var listed above.
3. The default value listed above.

## Usage

```python
from scitex_genai.llm import GenAI

ai = GenAI(model="qwen36-35b-fp8")                  # local vLLM @ 127.0.0.1:8765
```

```bash
# Aim at a remote vLLM box
export SCITEX_GENAI_VLLM_BASE_URL=http://gpu-host:8000/v1
export SCITEX_GENAI_VLLM_API_KEY=optional-token    # only if proxy demands it
```

See also `.env.example` at the repo root for a copy-paste template.

<!-- EOF -->
