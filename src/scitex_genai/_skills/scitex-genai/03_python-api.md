---
description: |
  [TOPIC] Python API
  [DETAILS] Public surface of `scitex_genai` — the `GenAI` factory class (model→provider inference, `__call__` for completion, `.cost` for spend, `.history` for conversation buffer), the modality submodules (`llm` implemented; `agent`/`image`/`audio`/`video`/`embed`/`multimodal` reserved), version metadata, and the umbrella shim path `scitex.genai`.
tags: [scitex-genai-python-api]
---

# Python API

## Top-level

```python
import scitex_genai
scitex_genai.__version__       # str — package version (PEP 440)
scitex_genai.GenAI             # class — provider-agnostic LLM factory
```

`GenAI` is lazy-loaded via `__getattr__` so `import scitex_genai` does
not pull any provider SDK. The provider SDK is only imported when you
construct an instance with a model from that provider.

## `GenAI` class

```python
ai = GenAI(
    model: str,              # required — e.g. "gpt-4o-mini"
    api_key: str | None = None,    # optional — defaults to env var
    system: str | None = None,     # optional — system prompt
    temperature: float | None = None,
    n_keep: int | None = None,     # how many history turns to keep
    seed: int | None = None,
    stream: bool = False,
)
```

The provider is inferred from `model` via the lookup table in
`scitex_genai.llm._PARAMS.MODELS`. Unknown models raise `ValueError`.

### Calling

```python
response: str = ai(prompt: str)
```

Synchronous, returns the completion text. Streaming via `stream=True`
yields chunks as a generator.

### Instance state

| Attribute     | Type                       | Meaning                                       |
| ------------- | -------------------------- | --------------------------------------------- |
| `ai.cost`     | `float`                    | Cumulative USD since instantiation.           |
| `ai.history`  | `list[dict[str, str]]`     | Conversation buffer; each turn `{role, content}`. |
| `ai.model`    | `str`                      | The model name passed in.                     |
| `ai.provider` | `str`                      | Inferred provider name.                       |

`ai.history.clear()` resets context without rebuilding the instance.

### Errors

- `ValueError` — unknown model name (no provider can claim it).
- `RuntimeError` — provider SDK not installed (e.g. requested
  `gpt-4o-mini` without `pip install scitex-genai[openai]`).
- `KeyError` — required env var missing at first call.

API errors from the underlying provider SDK propagate unchanged.

## Modality submodules

| Submodule                    | Status        | Public surface                         |
| ---------------------------- | ------------- | -------------------------------------- |
| `scitex_genai.llm`           | implemented   | `GenAI` (re-exported at top-level)     |
| `scitex_genai.agent`         | reserved      | raises `NotImplementedError`           |
| `scitex_genai.image`         | reserved      | raises `NotImplementedError`           |
| `scitex_genai.audio`         | reserved      | raises `NotImplementedError`           |
| `scitex_genai.video`         | reserved      | raises `NotImplementedError`           |
| `scitex_genai.embed`         | reserved      | raises `NotImplementedError`           |
| `scitex_genai.multimodal`    | reserved      | raises `NotImplementedError`           |

Reserved modules import successfully (so `from scitex_genai import
agent` works), but attribute access raises until the modality lands.
This keeps the public namespace stable as features arrive.

## Umbrella shim

The `scitex` umbrella package re-exports everything as `scitex.genai`:

```python
import scitex

ai = scitex.genai.GenAI(model="gpt-4o-mini")  # identical to GenAI()
```

Both `scitex_genai.GenAI` and `scitex.genai.GenAI` resolve to the same
class — pick the import style that matches the rest of your codebase.

## Stable, opinionated, minimal

The public surface is intentionally narrow:

- **One class** (`GenAI`) for all LLM providers — no per-provider client.
- **Two state attributes** (`.cost`, `.history`) — no other side channels.
- **Provider inference from model name** — no enum / config / factory dance.

When new modalities land (`image.generate`, `audio.tts`, …) they will
follow the same shape: one entry-point class, lazy provider loading,
inferred routing. The package surface won't accumulate adapters.
