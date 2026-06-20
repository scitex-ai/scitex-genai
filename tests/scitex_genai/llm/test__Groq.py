#!/usr/bin/env python3
# Timestamp: "2026-05-19 (rewritten for PA-307 / STX-TQ001-007)"
# File: ./tests/scitex_genai/llm/test__Groq.py
# ----------------------------------------

"""Tests for scitex_genai.llm._Groq.

Rewritten to comply with PA-307 (no unittest.mock, no monkeypatch fixture).
The previous version patched the groq SDK and asserted on mock kwargs.

What survives: init parameter contract, max_tokens clamping (Groq caps
max_tokens at 8000), and the missing-key error path. SDK-call paths
are omitted — the previous mock-based tests were green-bar theater.
"""

import os

import pytest

from scitex_genai.llm import Groq
from scitex_genai.llm._PARAMS import MODELS as _MODELS

_VALID_MODEL = _MODELS[_MODELS.provider == "Groq"].name.tolist()[0]


@pytest.fixture
def groq_env():
    """Provide a fake GROQ_API_KEY for the duration of a test."""
    # Arrange
    saved = os.environ.get("GROQ_API_KEY")
    os.environ["GROQ_API_KEY"] = "sk-fake-groq-from-env"
    # Act
    yield
    # Assert
    if saved is None:
        os.environ.pop("GROQ_API_KEY", None)
    else:
        os.environ["GROQ_API_KEY"] = saved


@pytest.fixture
def no_groq_env():
    """Ensure GROQ_API_KEY is absent."""
    # Arrange
    saved = os.environ.pop("GROQ_API_KEY", None)
    # Act
    yield
    # Assert
    if saved is not None:
        os.environ["GROQ_API_KEY"] = saved


def test_init_uses_api_key_from_environment(groq_env):
    # Arrange
    # Act
    ai = Groq(model=_VALID_MODEL)
    # Assert
    assert ai.api_key == "sk-fake-groq-from-env"


def test_init_uses_explicit_api_key_over_environment(groq_env):
    # Arrange
    # Act
    ai = Groq(api_key="explicit-key", model=_VALID_MODEL)
    # Assert
    assert ai.api_key == "explicit-key"


def test_init_without_api_key_raises_value_error(no_groq_env):
    # Arrange
    # Act
    # Assert
    with pytest.raises(ValueError, match="GROQ_API_KEY environment variable not set"):
        Groq(model=_VALID_MODEL)


def test_init_sets_provider_to_groq(groq_env):
    # Arrange
    # Act
    ai = Groq(model=_VALID_MODEL)
    # Assert
    assert ai.provider == "Groq"


def test_init_preserves_model_name(groq_env):
    # Arrange
    # Act
    ai = Groq(model=_VALID_MODEL)
    # Assert
    assert ai.model == _VALID_MODEL


def test_init_preserves_stream_flag(groq_env):
    # Arrange
    # Act
    ai = Groq(model=_VALID_MODEL, stream=True)
    # Assert
    assert ai.stream is True


def test_init_preserves_temperature(groq_env):
    # Arrange
    # Act
    ai = Groq(model=_VALID_MODEL, temperature=0.3)
    # Assert
    assert ai.temperature == 0.3


def test_init_preserves_seed(groq_env):
    # Arrange
    # Act
    ai = Groq(model=_VALID_MODEL, seed=42)
    # Assert
    assert ai.seed == 42


def test_init_preserves_n_keep(groq_env):
    # Arrange
    # Act
    ai = Groq(model=_VALID_MODEL, n_keep=5)
    # Assert
    assert ai.n_keep == 5


def test_max_tokens_clamped_to_8000_when_too_large(groq_env):
    # Arrange
    # Act
    ai = Groq(model=_VALID_MODEL, max_tokens=20_000)
    # Assert
    assert ai.max_tokens == 8_000


def test_max_tokens_preserved_when_within_cap(groq_env):
    # Arrange
    # Act
    ai = Groq(model=_VALID_MODEL, max_tokens=4_000)
    # Assert
    assert ai.max_tokens == 4_000


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])

# EOF
