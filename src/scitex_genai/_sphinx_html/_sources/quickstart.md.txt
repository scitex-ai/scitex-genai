# Quickstart

## Single completion

```python
from scitex_genai import GenAI

ai = GenAI(provider="openai", model="gpt-4o")
print(ai.complete("Explain neural networks in one sentence."))
print(ai.get_cost_summary())
```

## Switch providers without changing the call

```python
ai = GenAI(provider="anthropic", model="claude-sonnet-4-6")
ai.complete("Same call, different backend.")
```

The `GenAI` factory dispatches to the matching provider class inside
`scitex_genai.llm` (`OpenAI`, `Anthropic`, `Google`, `Groq`, `DeepSeek`,
`Perplexity`, `Llama`). Cost tracking and conversation history are
provider-agnostic.

For a runnable walk-through see [`examples/01_genai.ipynb`](https://github.com/ywatanabe1989/scitex-genai/blob/main/examples/01_genai.ipynb).
