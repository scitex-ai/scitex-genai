---
description: |
  [TOPIC] Quick Start
  [DETAILS] Three-line LLM call across any provider. Cost tracking and conversation history come for free on every instance. Provider is inferred from the model name — no per-provider client class to remember.
tags: [scitex-genai-quick-start]
---

# Quick Start

## One LLM call

```python
from scitex_genai import GenAI

ai = GenAI(model="gpt-4o-mini")
print(ai("Explain neural networks in one sentence."))
print("cost USD:", ai.cost)
```

That's it. The provider (OpenAI here) is inferred from the model
prefix; `ANTHROPIC_API_KEY=...` would similarly route `claude-*` models
to Anthropic.

## Switch providers without changing code

```python
from scitex_genai import GenAI

prompt = "Write a haiku about gradient descent."
for model in ("gpt-4o-mini", "claude-haiku-4-5", "gemini-2.0-flash"):
    ai = GenAI(model=model)
    print(model, "→", ai(prompt))
```

Each call resolves the provider, validates the key is present, and
returns the response text.

## Conversation history

Every `GenAI` instance buffers turns. Subsequent calls send the full
history as context:

```python
ai = GenAI(model="gpt-4o-mini")
ai("My name is Alice.")
ai("What's my name?")     # → "Your name is Alice."
print(ai.history)         # list of {role, content} dicts
ai.history.clear()        # reset
```

## Cost tracking

`ai.cost` is cumulative dollars since instantiation. Useful for
research-grant accounting:

```python
ai = GenAI(model="claude-opus-4-5")
for paper in papers:
    ai(f"Summarise: {paper.abstract}")
print(f"Total spend: ${ai.cost:.4f}")
```

## Inside a `@stx.session`

If you're already in a SciTeX session (the recommended way to run
research scripts), `GenAI` plays nicely — the env-var resolution
honours your `CONFIG`-based key store:

```python
import scitex

@scitex.session
def main(model: str = "gpt-4o-mini"):
    ai = scitex.genai.GenAI(model=model)
    answer = ai("What's 2+2?")
    return {"answer": answer, "cost": ai.cost}
```

## Where to next

- [`03_python-api.md`](03_python-api.md) — `GenAI` class signature, history shape, cost units, error handling
- [`10_llm.md`](10_llm.md) — full provider matrix, model-name → provider table, advanced usage
