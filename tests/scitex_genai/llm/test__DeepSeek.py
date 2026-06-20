#!/usr/bin/env python3
# Timestamp: "2026-05-19 (rewritten for PA-307 / STX-TQ001-007)"
# File: ./tests/scitex_genai/llm/test__DeepSeek.py
# ----------------------------------------

"""Tests for scitex_genai.llm._DeepSeek.

Rewritten to comply with PA-307 (no unittest.mock, no monkeypatch fixture).
The previous version patched openai.OpenAI / requests and asserted on
mock kwargs.

What survives: init parameter contract via real init with a fake key.
DeepSeek uses the openai SDK shape and accepts a fake key at construction;
real validation happens only on remote call.
"""

import os

import pytest

from scitex_genai.llm import DeepSeek
from scitex_genai.llm._PARAMS import MODELS as _MODELS

_VALID_MODEL = _MODELS[_MODELS.provider == "DeepSeek"].name.tolist()[0]


def test_init_preserves_explicit_api_key():
    # Arrange
    # Act
    ai = DeepSeek(model=_VALID_MODEL, api_key="explicit-key")
    # Assert
    assert ai.api_key == "explicit-key"


def test_init_sets_provider_to_deepseek():
    # Arrange
    # Act
    ai = DeepSeek(model=_VALID_MODEL, api_key="k")
    # Assert
    assert ai.provider == "DeepSeek"


def test_init_preserves_model_name():
    # Arrange
    # Act
    ai = DeepSeek(model=_VALID_MODEL, api_key="k")
    # Assert
    assert ai.model == _VALID_MODEL


def test_init_preserves_stream_flag():
    # Arrange
    # Act
    ai = DeepSeek(model=_VALID_MODEL, api_key="k", stream=True)
    # Assert
    assert ai.stream is True


def test_init_preserves_temperature():
    # Arrange
    # Act
    ai = DeepSeek(model=_VALID_MODEL, api_key="k", temperature=0.3)
    # Assert
    assert ai.temperature == 0.3


def test_init_preserves_n_keep():
    # Arrange
    # Act
    ai = DeepSeek(model=_VALID_MODEL, api_key="k", n_keep=5)
    # Assert
    assert ai.n_keep == 5


def test_init_preserves_max_tokens():
    # Arrange
    # Act
    ai = DeepSeek(model=_VALID_MODEL, api_key="k", max_tokens=2_048)
    # Assert
    assert ai.max_tokens == 2_048


def test_init_default_max_tokens_is_4096():
    # Arrange
    # Act
    ai = DeepSeek(model=_VALID_MODEL, api_key="k")
    # Assert
    assert ai.max_tokens == 4_096


def test_init_default_model_is_deepseek_chat():
    # Arrange
    # Act
    ai = DeepSeek(api_key="k")
    # Assert
    assert ai.model == "deepseek-chat"


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])

# EOF
