#!/usr/bin/env python3
# Timestamp: "2026-05-19 (rewritten for PA-307 / STX-TQ001-007)"
# File: ./tests/scitex_genai/llm/test__Perplexity.py
# ----------------------------------------

"""Tests for scitex_genai.llm._Perplexity.

Rewritten to comply with PA-307 (no unittest.mock, no monkeypatch fixture).
The previous version patched the openai SDK and asserted on mock kwargs.

What survives: init parameter contract via real init with a fake key, plus
max_tokens auto-selection based on "128k" substring in the model name.
SDK-call paths are intentionally omitted because the previous mock-based
tests for them only verified that the mock got the kwargs the test built.
"""

import os

import pytest

from scitex_genai.llm import Perplexity
from scitex_genai.llm._PARAMS import MODELS as _MODELS

_VALID_MODEL = _MODELS[_MODELS.provider == "Perplexity"].name.tolist()[0]


@pytest.fixture
def perplexity_env():
    """Provide a fake PERPLEXITY_API_KEY for the duration of a test."""
    # Arrange
    saved = os.environ.get("PERPLEXITY_API_KEY")
    os.environ["PERPLEXITY_API_KEY"] = "pplx-fake-from-env"
    # Act
    yield
    # Assert
    if saved is None:
        os.environ.pop("PERPLEXITY_API_KEY", None)
    else:
        os.environ["PERPLEXITY_API_KEY"] = saved


@pytest.fixture
def no_perplexity_env():
    """Ensure PERPLEXITY_API_KEY is absent."""
    # Arrange
    saved = os.environ.pop("PERPLEXITY_API_KEY", None)
    # Act
    yield
    # Assert
    if saved is not None:
        os.environ["PERPLEXITY_API_KEY"] = saved


def test_init_uses_api_key_from_environment(perplexity_env):
    # Arrange
    # Act
    ai = Perplexity(model=_VALID_MODEL)
    # Assert
    assert ai.api_key == "pplx-fake-from-env"


def test_init_uses_explicit_api_key_over_environment(perplexity_env):
    # Arrange
    # Act
    ai = Perplexity(api_key="explicit-key", model=_VALID_MODEL)
    # Assert
    assert ai.api_key == "explicit-key"


def test_init_without_api_key_raises_value_error(no_perplexity_env):
    # Arrange
    # Act
    # Assert
    with pytest.raises(
        ValueError, match="PERPLEXITY_API_KEY environment variable not set"
    ):
        Perplexity(model=_VALID_MODEL)


def test_init_sets_provider_to_perplexity(perplexity_env):
    # Arrange
    # Act
    ai = Perplexity(model=_VALID_MODEL)
    # Assert
    assert ai.provider == "Perplexity"


def test_init_preserves_model_name(perplexity_env):
    # Arrange
    # Act
    ai = Perplexity(model=_VALID_MODEL)
    # Assert
    assert ai.model == _VALID_MODEL


def test_init_preserves_stream_flag(perplexity_env):
    # Arrange
    # Act
    ai = Perplexity(model=_VALID_MODEL, stream=True)
    # Assert
    assert ai.stream is True


def test_init_preserves_temperature(perplexity_env):
    # Arrange
    # Act
    ai = Perplexity(model=_VALID_MODEL, temperature=0.5)
    # Assert
    assert ai.temperature == 0.5


def test_init_preserves_seed(perplexity_env):
    # Arrange
    # Act
    ai = Perplexity(model=_VALID_MODEL, seed=42)
    # Assert
    assert ai.seed == 42


def test_init_preserves_n_keep(perplexity_env):
    # Arrange
    # Act
    ai = Perplexity(model=_VALID_MODEL, n_keep=5)
    # Assert
    assert ai.n_keep == 5


def test_128k_in_model_name_yields_128k_max_tokens(perplexity_env):
    # Arrange
    # Act
    ai = Perplexity(model="llama-3.1-sonar-small-128k-online")
    # Assert
    assert ai.max_tokens == 128_000


def test_non_128k_model_yields_32k_max_tokens(perplexity_env):
    # Arrange
    # Act
    ai = Perplexity(model="some-other-model")
    # Assert
    assert ai.max_tokens == 32_000


def test_explicit_max_tokens_overrides_model_default(perplexity_env):
    # Arrange
    # Act
    ai = Perplexity(model=_VALID_MODEL, max_tokens=2_048)
    # Assert
    assert ai.max_tokens == 2_048


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])

# EOF
