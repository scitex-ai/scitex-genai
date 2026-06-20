#!/usr/bin/env python3
# Timestamp: "2026-05-19 (rewritten for PA-307 / STX-TQ001-007)"
# File: ./tests/scitex_genai/llm/test__Anthropic.py
# ----------------------------------------

"""Tests for scitex_genai.llm._Anthropic.

Rewritten to comply with PA-307 (no unittest.mock, no monkeypatch fixture).
The previous version patched the anthropic SDK module and asserted on mock
call args — zero production signal. Per PA-307: delete or rewrite.

What survives here:
  - Init parameter contract (model, api_key, temperature, etc.) — real init
    with a fake key, since the Anthropic SDK only contacts the remote on
    actual API call, not at construction.
  - _api_format_history: pure-Python transformation of the history dict
    shape, exercised against real input.
  - max_tokens override for the sonnet model variant.

The streaming + static API call paths are NOT tested here because they
talk to the live Anthropic SDK; the previous mock-based tests for them
were green-bar theater (they verified the mock got the kwargs the test
itself crafted).
"""

import os

import pytest

from scitex_genai.llm import Anthropic
from scitex_genai.llm._PARAMS import MODELS as _MODELS

_VALID_MODEL = _MODELS[_MODELS.provider == "Anthropic"].name.tolist()[0]


@pytest.fixture
def anthropic_env():
    """Provide a fake ANTHROPIC_API_KEY for the duration of a test."""
    # Arrange
    saved = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = "sk-fake-from-env"
    # Act
    yield
    # Assert
    if saved is None:
        os.environ.pop("ANTHROPIC_API_KEY", None)
    else:
        os.environ["ANTHROPIC_API_KEY"] = saved


@pytest.fixture
def no_anthropic_env():
    """Ensure ANTHROPIC_API_KEY is absent for the test."""
    # Arrange
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    # Act
    yield
    # Assert
    if saved is not None:
        os.environ["ANTHROPIC_API_KEY"] = saved


def test_init_uses_api_key_from_environment(anthropic_env):
    # Arrange
    # Act
    ai = Anthropic(model=_VALID_MODEL)
    # Assert
    assert ai.api_key == "sk-fake-from-env"


def test_init_uses_explicit_api_key_over_environment(anthropic_env):
    # Arrange
    # Act
    ai = Anthropic(api_key="explicit-key", model=_VALID_MODEL)
    # Assert
    assert ai.api_key == "explicit-key"


def test_init_without_api_key_raises_value_error(no_anthropic_env):
    # Arrange
    # Act
    # Assert
    with pytest.raises(
        ValueError, match="ANTHROPIC_API_KEY environment variable not set"
    ):
        Anthropic(model=_VALID_MODEL)


def test_init_sets_provider_to_anthropic(anthropic_env):
    # Arrange
    # Act
    ai = Anthropic(model=_VALID_MODEL)
    # Assert
    assert ai.provider == "Anthropic"


def test_init_preserves_model_name(anthropic_env):
    # Arrange
    # Act
    ai = Anthropic(model=_VALID_MODEL)
    # Assert
    assert ai.model == _VALID_MODEL


def test_sonnet_model_gets_128k_max_tokens(anthropic_env):
    # Arrange
    # Act
    ai = Anthropic(model="claude-3-7-sonnet-2025-0219")
    # Assert
    assert ai.max_tokens == 128_000


def test_stream_parameter_is_preserved(anthropic_env):
    # Arrange
    # Act
    ai = Anthropic(model=_VALID_MODEL, stream=True)
    # Assert
    assert ai.stream is True


def test_n_keep_parameter_is_preserved(anthropic_env):
    # Arrange
    # Act
    ai = Anthropic(model=_VALID_MODEL, n_keep=5)
    # Assert
    assert ai.n_keep == 5


def test_temperature_parameter_is_preserved(anthropic_env):
    # Arrange
    # Act
    ai = Anthropic(model=_VALID_MODEL, temperature=0.5)
    # Assert
    assert ai.temperature == 0.5


def test_api_format_history_keeps_text_message_shape(anthropic_env):
    # Arrange
    ai = Anthropic(model=_VALID_MODEL)
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    # Act
    formatted = ai._api_format_history(history)
    # Assert
    assert formatted == history


def test_api_format_history_translates_internal_image_marker(anthropic_env):
    # Arrange
    ai = Anthropic(model=_VALID_MODEL)
    history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's this?"},
                {"type": "_image", "_image": "base64data"},
            ],
        }
    ]
    # Act
    formatted = ai._api_format_history(history)
    # Assert
    assert formatted == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's this?"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "base64data",
                    },
                },
            ],
        }
    ]


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])

# EOF
