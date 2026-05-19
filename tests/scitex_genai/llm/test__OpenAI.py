#!/usr/bin/env python3
# Timestamp: "2026-05-19 (rewritten for PA-307 / STX-TQ001-007)"
# File: ./tests/scitex_genai/llm/test__OpenAI.py
# ----------------------------------------

"""Tests for scitex_genai.llm._OpenAI.

Rewritten to comply with PA-307 (no unittest.mock, no monkeypatch fixture).
The previous version patched openai.OpenAI and asserted on mock kwargs.

What survives: init parameter contract, max_tokens auto-selection by model
prefix, and passed_model preservation. SDK-call paths are intentionally
omitted because the previous mock-based tests for them only verified that
the mock got the kwargs the test built.
"""

import os

import pytest

from scitex_genai.llm import OpenAI
from scitex_genai.llm._PARAMS import MODELS as _MODELS

_VALID_MODEL = _MODELS[_MODELS.provider == "OpenAI"].name.tolist()[0]


@pytest.fixture
def openai_env():
    """Provide a fake OPENAI_API_KEY for the duration of a test."""
    # Arrange
    saved = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "sk-fake-from-env"
    # Act
    yield
    # Assert
    if saved is None:
        os.environ.pop("OPENAI_API_KEY", None)
    else:
        os.environ["OPENAI_API_KEY"] = saved


def test_init_preserves_explicit_api_key(openai_env):
    # Arrange
    # Act
    ai = OpenAI(model=_VALID_MODEL, api_key="explicit-key")
    # Assert
    assert ai.api_key == "explicit-key"


def test_init_sets_provider_to_openai(openai_env):
    # Arrange
    # Act
    ai = OpenAI(model=_VALID_MODEL, api_key="k")
    # Assert
    assert ai.provider == "OpenAI"


def test_init_preserves_passed_model_field(openai_env):
    # Arrange
    # Act
    ai = OpenAI(model="o3", api_key="k")
    # Assert
    assert ai.passed_model == "o3"


def test_init_preserves_stream_flag(openai_env):
    # Arrange
    # Act
    ai = OpenAI(model=_VALID_MODEL, api_key="k", stream=True)
    # Assert
    assert ai.stream is True


def test_init_preserves_temperature(openai_env):
    # Arrange
    # Act
    ai = OpenAI(model=_VALID_MODEL, api_key="k", temperature=0.5)
    # Assert
    assert ai.temperature == 0.5


def test_init_preserves_seed(openai_env):
    # Arrange
    # Act
    ai = OpenAI(model=_VALID_MODEL, api_key="k", seed=42)
    # Assert
    assert ai.seed == 42


def test_init_preserves_n_keep(openai_env):
    # Arrange
    # Act
    ai = OpenAI(model=_VALID_MODEL, api_key="k", n_keep=5)
    # Assert
    assert ai.n_keep == 5


def test_gpt_4_turbo_gets_128k_max_tokens(openai_env):
    # Arrange
    # Act
    ai = OpenAI(model="gpt-4-turbo", api_key="k")
    # Assert
    assert ai.max_tokens == 128_000


def test_gpt_4_gets_8k_max_tokens(openai_env):
    # Arrange
    # Act
    ai = OpenAI(model="gpt-4", api_key="k")
    # Assert
    assert ai.max_tokens == 8_192


def test_gpt_3_5_turbo_16k_gets_16k_max_tokens(openai_env):
    # Arrange
    # Act
    ai = OpenAI(model="gpt-3.5-turbo-16k", api_key="k")
    # Assert
    assert ai.max_tokens == 16_384


def test_gpt_3_5_gets_4k_max_tokens(openai_env):
    # Arrange
    # Act
    ai = OpenAI(model="gpt-3.5-turbo", api_key="k")
    # Assert
    assert ai.max_tokens == 4_096


def test_unknown_model_gets_4k_default_max_tokens(openai_env):
    # Arrange
    # Act
    ai = OpenAI(model="o3", api_key="k")
    # Assert
    assert ai.max_tokens == 4_096


def test_explicit_max_tokens_overrides_model_default(openai_env):
    # Arrange
    # Act
    ai = OpenAI(model="gpt-4", api_key="k", max_tokens=2_048)
    # Assert
    assert ai.max_tokens == 2_048


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])

# EOF
