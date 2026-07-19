#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-03 11:57:10 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/ai/_gen_ai/_genai_factory.py
# ----------------------------------------
import os

__FILE__ = "./src/scitex/ai/_gen_ai/_genai_factory.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import random

from ._Anthropic import Anthropic
from ._DeepSeek import DeepSeek
from ._Google import Google
from ._Groq import Groq
from ._LiteLLM import LiteLLM
from ._Llama import Llama
from ._OpenAI import OpenAI
from ._PARAMS import MODELS
from ._Perplexity import Perplexity

_KNOWN_BACKENDS = ("default", "litellm")


def _resolve_backend(backend):
    """Resolve the dispatch backend: explicit arg > SCITEX_GENAI_BACKEND env.

    Returns "default" or "litellm"; raises ValueError on anything else so
    typos fail loudly instead of silently using the default dispatch.
    """
    if backend is None:
        backend = os.getenv("SCITEX_GENAI_BACKEND")
    backend = (backend or "default").strip().lower()
    if backend not in _KNOWN_BACKENDS:
        raise ValueError(
            f"Unknown backend {backend!r}. Choose from: {list(_KNOWN_BACKENDS)}"
        )
    return backend


def genai_factory(
    model="gpt-3.5-turbo",
    stream=False,
    api_key=None,
    seed=None,
    temperature=1.0,
    n_keep=1,
    chat_history=None,
    max_tokens=4096,
    base_url=None,
    provider=None,
    backend=None,
):
    """Factory function to create an instance of an AI model handler.

    backend : {"default", "litellm"}, optional
        "litellm" routes ANY provider through the single litellm-backed
        handler (one OpenAI-compatible code path). Defaults to the
        SCITEX_GENAI_BACKEND env var, then to the per-provider classes.
    """
    backend = _resolve_backend(backend)
    AVAILABLE_MODELS = MODELS.name.tolist()

    if model in AVAILABLE_MODELS:
        # Known model: resolve provider from the MODELS table (today's behavior).
        provider = MODELS[MODELS.name == model].provider.iloc[0]
    else:
        # Unknown model: only allowed when targeting a self-hosted /
        # OpenAI-compatible endpoint via base_url, or an explicit provider.
        # Fall back to fleet-injected env for endpoint + key (explicit args
        # win); engages only on this passthrough path, so known provider
        # models keep using their own *_API_KEY.
        if base_url is None:
            base_url = os.getenv("SCITEX_GENAI_BASE_URL")
        if api_key is None:
            api_key = os.getenv("SCITEX_GENAI_API_KEY")
        if not base_url and not provider:
            raise ValueError(
                f'Model "{model}" is not available. Please choose from:{MODELS.name.tolist()}'
            )
        # Default to an OpenAI-compatible passthrough; skip the MODELS lookup.
        if provider is None:
            provider = "OpenAI"

    # Select a random API key from the list
    if isinstance(api_key, (list, tuple)):
        api_key = random.choice(api_key)

    kwargs = dict(
        model=model,
        stream=stream,
        api_key=api_key,
        seed=seed,
        temperature=temperature,
        n_keep=n_keep,
        chat_history=chat_history,
        max_tokens=max_tokens,
    )

    # Opt-in litellm backend: ONE OpenAI-compatible code path for every
    # provider and for self-hosted endpoints. The default dispatch below is
    # unchanged (safe rollout; flipping the default is a later step).
    if backend == "litellm":
        return LiteLLM(**kwargs, base_url=base_url, provider=provider)

    # model_class = globals()[provider]
    model_class = {
        "OpenAI": OpenAI,
        "Anthropic": Anthropic,
        "Google": Google,
        "Llama": Llama,
        "Perplexity": Perplexity,
        "DeepSeek": DeepSeek,
        "Groq": Groq,
    }[provider]

    # Only the OpenAI(-compatible) handler accepts base_url; other handler
    # constructors do not, so add it conditionally.
    if provider == "OpenAI":
        kwargs["base_url"] = base_url

    return model_class(**kwargs)


# EOF
