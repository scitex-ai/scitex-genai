# Quickstart

## Single completion

```python
from scitex_genai import GenAI

ai = GenAI(model="gpt-4o-mini")
print(ai("Explain neural networks in one sentence."))
print("cost USD:", ai.cost)
```

## Switch providers without changing the call

```python
ai = GenAI(model="claude-sonnet-4-6")
ai("Same call, different backend.")
```

The `GenAI` factory dispatches to the matching provider class inside
`scitex_genai.llm` (`OpenAI`, `Anthropic`, `Google`, `Groq`, `DeepSeek`,
`Perplexity`, `Llama`). Cost tracking and conversation history are
provider-agnostic.

For a runnable walk-through see [`examples/01_genai.ipynb`](https://github.com/ywatanabe1989/scitex-genai/blob/main/examples/01_genai.ipynb).
