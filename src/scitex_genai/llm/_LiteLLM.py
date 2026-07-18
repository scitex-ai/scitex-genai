#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex_genai/llm/_LiteLLM.py
# ----------------------------------------
import os

__FILE__ = "./src/scitex_genai/llm/_LiteLLM.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""litellm-backed dispatch handler.

One OpenAI-compatible code path for every provider (SSoT/DRY): instead of one
handler class per provider SDK, `litellm.completion(...)` translates
OpenAI-format requests to each vendor's API. Providers are selected via
litellm's model-string prefixes (``anthropic/claude-*``, ``gemini/gemini-*``,
...); self-hosted OpenAI-compatible endpoints use the ``openai/`` prefix plus
``base_url``.

Opt-in via ``GenAI(..., backend="litellm")`` or ``SCITEX_GENAI_BACKEND=litellm``
— the default dispatch still uses the per-provider classes.

litellm itself is imported lazily (in ``_init_client``): its import cost is
non-trivial and the default backend must not pay it.
"""

from ._BaseGenAI import BaseGenAI
from ._PARAMS import MODELS

# scitex-genai provider name -> litellm model-string prefix.
# https://docs.litellm.ai/docs/providers
# "OpenAI" maps to the empty prefix: plain `gpt-*` / `o*` names are already
# litellm's OpenAI convention. "Llama" (local llama servers) routes through
# ollama, litellm's local-model convention.
LITELLM_PROVIDER_PREFIXES = {
    "OpenAI": "",
    "Anthropic": "anthropic/",
    "Google": "gemini/",
    "Groq": "groq/",
    "DeepSeek": "deepseek/",
    "Perplexity": "perplexity/",
    "Llama": "ollama/",
}


def to_litellm_model(model, provider=None, base_url=None):
    """Map a scitex-genai (model, provider) pair to a litellm model string.

    Pure function — no I/O, no env reads — so the mapping is testable on its
    own.

    Parameters
    ----------
    model : str
        Model name as passed to ``GenAI`` (e.g. ``claude-3-5-haiku-20241022``).
    provider : str, optional
        scitex-genai provider name (``OpenAI``, ``Anthropic``, ...). Ignored
        when ``base_url`` is given.
    base_url : str, optional
        Self-hosted / OpenAI-compatible endpoint. When set, litellm needs the
        ``openai/`` prefix to know the wire protocol (it cannot infer a
        provider from an arbitrary local model name such as ``qwen36-35b-a3b``).

    Returns
    -------
    str
        litellm model string, e.g. ``anthropic/claude-3-5-haiku-20241022``,
        ``gemini/gemini-2.5-pro``, or ``openai/qwen36-35b-a3b``.
    """
    if base_url:
        return model if model.startswith("openai/") else f"openai/{model}"
    prefix = LITELLM_PROVIDER_PREFIXES.get(provider, "")
    if prefix and not model.startswith(prefix):
        return f"{prefix}{model}"
    return model


def _resolve_provider(model):
    """Resolve the scitex-genai provider for a model from the MODELS table."""
    rows = MODELS[MODELS.name == model]
    if len(rows):
        return rows.provider.iloc[0]
    return None


def _default_api_key(provider):
    """Read the provider's conventional API-key env var (may be None)."""
    rows = MODELS[MODELS.provider == provider]
    if len(rows):
        return os.getenv(rows.api_key_env.iloc[0])
    return None


class LiteLLM(BaseGenAI):
    """Provider-agnostic handler dispatching through ``litellm.completion``.

    Same instance contract as every other handler: ``ai(prompt)``, ``ai.cost``,
    ``ai.history``, ``ai.reset()``, ``ai.stream``, ``ai.input_tokens`` /
    ``ai.output_tokens``.
    """

    def __init__(
        self,
        system_setting="",
        model="",
        api_key=None,
        stream=False,
        seed=None,
        n_keep=1,
        temperature=1.0,
        chat_history=None,
        max_tokens=4096,
        base_url=None,
        provider=None,
    ):
        if provider is None:
            provider = "OpenAI" if base_url else _resolve_provider(model)
        if api_key is None and not base_url:
            api_key = _default_api_key(provider)
        # `self.model` keeps the plain name (history/cost/verify contract);
        # the litellm prefix lives only on the wire-facing model string.
        self.litellm_model = to_litellm_model(
            model, provider=provider, base_url=base_url
        )

        super().__init__(
            system_setting=system_setting,
            model=model,
            api_key=api_key or "",
            stream=stream,
            seed=seed,
            n_keep=n_keep,
            temperature=temperature,
            provider=provider or "OpenAI",
            chat_history=chat_history,
            max_tokens=max_tokens if max_tokens is not None else 4096,
            base_url=base_url,
        )

    def _init_client(self):
        # Lazy import: litellm is heavy and only this backend needs it.
        import litellm

        return litellm

    def _completion_kwargs(self, stream):
        kwargs = dict(
            model=self.litellm_model,
            messages=self.history,
            api_key=self.api_key or None,
            base_url=self.base_url,
            seed=self.seed,
            stream=stream,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if stream:
            kwargs["stream_options"] = {"include_usage": True}
        return kwargs

    def _track_usage(self, response):
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.output_tokens += getattr(usage, "completion_tokens", 0) or 0

    def _api_call_static(self):
        output = self.client.completion(**self._completion_kwargs(stream=False))
        self._track_usage(output)
        return output.choices[0].message.content

    def _api_call_stream(self):
        stream = self.client.completion(**self._completion_kwargs(stream=True))
        buffer = ""
        for chunk in stream:
            if not chunk:
                continue
            self._track_usage(chunk)
            try:
                current_text = chunk.choices[0].delta.content
            except (AttributeError, IndexError):
                current_text = None
            if current_text:
                buffer += current_text
                # Yield complete sentences or words.
                if any(char in ".!?\n " for char in current_text):
                    yield buffer
                    buffer = ""
        if buffer:
            yield buffer

    def _api_format_history(self, history):
        # litellm accepts OpenAI-format messages (same shape as _OpenAI.py).
        formatted_history = []
        for msg in history:
            if isinstance(msg["content"], list):
                content = []
                for item in msg["content"]:
                    if item["type"] == "text":
                        content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "_image":
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{item['_image']}"
                                },
                            }
                        )
                formatted_msg = {"role": msg["role"], "content": content}
            else:
                formatted_msg = {
                    "role": msg["role"],
                    "content": msg["content"],
                }
            formatted_history.append(formatted_msg)
        return formatted_history


# EOF
