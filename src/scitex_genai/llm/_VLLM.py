#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex_genai/llm/_VLLM.py

"""
1. Functionality:
   - Implements an OpenAI-compatible client for a LOCAL vLLM server
2. Input:
   - Text prompts; chat-completion history
3. Output:
   - Generated responses (streaming or static)
4. Prerequisites:
   - A running vLLM server speaking the OpenAI HTTP API (default
     `http://127.0.0.1:8765/v1`; override via `SCITEX_GENAI_VLLM_BASE_URL`)
   - The `openai` Python SDK (already a scitex-genai dependency)
   - No real API key is required — vLLM ignores the `Authorization`
     header. The OpenAI SDK rejects empty keys, so a placeholder `EMPTY`
     is used.

Why this exists
---------------
The OpenAI Python SDK speaks vLLM's HTTP API natively when pointed at a
custom `base_url`. That keeps every call shape (chat.completions.create,
streaming chunks, message format) identical to the cloud providers.

`_DeepSeek.py` already uses this same pattern with a hardcoded
`base_url=https://api.deepseek.com/beta`. This module is its sibling,
specialised for a local vLLM endpoint whose URL is operator-configurable
via `SCITEX_GENAI_VLLM_BASE_URL` so a Qwen (or any other vLLM-served)
model can be addressed by name through the `GenAI()` factory.
"""

import os
from typing import Generator

from openai import OpenAI as _OpenAI

from ._BaseGenAI import BaseGenAI

DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8765/v1"
_VLLM_PLACEHOLDER_KEY = (
    "EMPTY"  # vLLM ignores the value; OpenAI SDK requires non-empty.
)


class VLLM(BaseGenAI):
    """OpenAI-compatible client for a local vLLM server.

    Pass any model name registered in `_PARAMS.MODELS` under
    `provider="vLLM"`. The model id is forwarded verbatim to vLLM's
    `/v1/chat/completions` (must match the `--served-model-name` the
    server was launched with).
    """

    def __init__(
        self,
        system_setting="",
        model="qwen36-35b-fp8",
        api_key="",
        stream=False,
        seed=None,
        n_keep=1,
        temperature=1.0,
        chat_history=None,
        max_tokens=4096,
        base_url=None,
    ):
        # vLLM ignores the API key — use the operator override, fall
        # back to the env var, then the placeholder.
        resolved_key = (
            api_key
            or os.environ.get("SCITEX_GENAI_VLLM_API_KEY", "")
            or _VLLM_PLACEHOLDER_KEY
        )
        # base_url precedence: explicit arg > env > module default.
        self._base_url = (
            base_url
            or os.environ.get("SCITEX_GENAI_VLLM_BASE_URL")
            or DEFAULT_VLLM_BASE_URL
        )
        super().__init__(
            system_setting=system_setting,
            model=model,
            api_key=resolved_key,
            stream=stream,
            n_keep=n_keep,
            temperature=temperature,
            provider="vLLM",
            chat_history=chat_history,
            max_tokens=max_tokens,
        )

    @property
    def base_url(self) -> str:
        return self._base_url

    def _init_client(self):
        return _OpenAI(api_key=self.api_key, base_url=self._base_url)

    def _api_call_static(self) -> str:
        kwargs = dict(
            model=self.model,
            messages=self.history,
            seed=self.seed,
            stream=False,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        output = self.client.chat.completions.create(**kwargs)
        if getattr(output, "usage", None):
            self.input_tokens += output.usage.prompt_tokens
            self.output_tokens += output.usage.completion_tokens
        return output.choices[0].message.content

    def _api_call_stream(self) -> Generator[str, None, None]:
        kwargs = dict(
            model=self.model,
            messages=self.history,
            max_tokens=self.max_tokens,
            n=1,
            stream=self.stream,
            seed=self.seed,
            temperature=self.temperature,
        )
        stream = self.client.chat.completions.create(**kwargs)
        buffer = ""
        for chunk in stream:
            if not chunk:
                continue
            usage = getattr(chunk, "usage", None)
            if usage:
                try:
                    self.input_tokens += usage.prompt_tokens
                except Exception:
                    pass
                try:
                    self.output_tokens += usage.completion_tokens
                except Exception:
                    pass
            try:
                current_text = chunk.choices[0].delta.content
            except Exception:
                continue
            if not current_text:
                continue
            buffer += current_text
            if any(char in ".!?\n " for char in current_text):
                yield buffer
                buffer = ""
        if buffer:
            yield buffer


# EOF
