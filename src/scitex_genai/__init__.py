#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_genai/__init__.py
# ----------------------------------------
from __future__ import annotations

"""SciTeX GenAI — generative-AI provider abstraction across modalities.

Factored out of `scitex.ai._gen_ai` (which lived in the umbrella scitex-python
and was first lifted to `scitex-ai`). Top-level layout is modality-based so
the namespace ages well as the field fragments by modality:

    scitex_genai.llm          # text  (litellm-backed in future; native today)
    scitex_genai.agent        # agentic loops (claude-agent-sdk, ...)
    scitex_genai.image        # image generation / editing
    scitex_genai.audio        # TTS / STT / music
    scitex_genai.video        # video generation
    scitex_genai.embed        # embeddings
    scitex_genai.multimodal   # any-to-any unified models

Only `llm` is implemented today; the rest are reserved namespaces with
stub modules so import paths are stable as features land.
"""

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("scitex-genai")
except _PackageNotFoundError:
    __version__ = "0.0.0+local"


def __getattr__(name):
    if name == "GenAI":
        from .llm import GenAI

        return GenAI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["__version__", "GenAI"]
