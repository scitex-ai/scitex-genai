#!/usr/bin/env python3
# Timestamp: "2026-05-19 (rewritten for PA-307 / STX-TQ001-007)"
# File: ./tests/scitex_genai/llm/test__Google.py
# ----------------------------------------

"""Tests for scitex_genai.llm._Google.

Rewritten to comply with PA-307 (no unittest.mock, no monkeypatch fixture).
The previous version patched google.generativeai and asserted on mock kwargs.

What survives: init parameter contract via real init with a fake API key.
"""

import os

import pytest

from scitex_genai.llm import Google
from scitex_genai.llm._PARAMS import MODELS as _MODELS

_VALID_MODEL = _MODELS[_MODELS.provider == "Google"].name.tolist()[0]


@pytest.fixture
def google_env():
    """Provide a fake GOOGLE_API_KEY for the duration of a test."""
    # Arrange
    saved = os.environ.get("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = "sk-fake-from-env"
    # Act
    yield
    # Assert
    if saved is None:
        os.environ.pop("GOOGLE_API_KEY", None)
    else:
        os.environ["GOOGLE_API_KEY"] = saved


@pytest.fixture
def no_google_env():
    """Ensure GOOGLE_API_KEY is absent."""
    # Arrange
    saved = os.environ.pop("GOOGLE_API_KEY", None)
    # Act
    yield
    # Assert
    if saved is not None:
        os.environ["GOOGLE_API_KEY"] = saved


def test_init_uses_api_key_from_environment(google_env):
    # Arrange
    # Act
    # Note: api_key=None forces fallback to os.getenv, which the fixture
    # has overwritten. Without this, the default-arg form of __init__
    # would have frozen the real env value at import time.
    ai = Google(api_key=None, model=_VALID_MODEL)
    # Assert
    assert ai.api_key == "sk-fake-from-env"


def test_init_uses_explicit_api_key_over_environment(google_env):
    # Arrange
    # Act
    ai = Google(api_key="explicit-key", model=_VALID_MODEL)
    # Assert
    assert ai.api_key == "explicit-key"


def test_init_without_api_key_raises_value_error(no_google_env):
    # Arrange
    # Act
    # Assert
    # api_key=None forces the runtime fallback path; otherwise the
    # default-arg form would resolve at import time, before the fixture.
    with pytest.raises(ValueError, match="GOOGLE_API_KEY environment variable not set"):
        Google(api_key=None, model=_VALID_MODEL)


def test_init_sets_provider_to_google(google_env):
    # Arrange
    # Act
    ai = Google(model=_VALID_MODEL)
    # Assert
    assert ai.provider == "Google"


def test_init_preserves_model_name(google_env):
    # Arrange
    # Act
    ai = Google(model=_VALID_MODEL)
    # Assert
    assert ai.model == _VALID_MODEL


def test_init_preserves_stream_flag(google_env):
    # Arrange
    # Act
    ai = Google(model=_VALID_MODEL, stream=True)
    # Assert
    assert ai.stream is True


def test_init_preserves_temperature(google_env):
    # Arrange
    # Act
    ai = Google(model=_VALID_MODEL, temperature=0.5)
    # Assert
    assert ai.temperature == 0.5


def test_init_preserves_seed(google_env):
    # Arrange
    # Act
    ai = Google(model=_VALID_MODEL, seed=42)
    # Assert
    assert ai.seed == 42


def test_init_preserves_n_keep(google_env):
    # Arrange
    # Act
    ai = Google(model=_VALID_MODEL, n_keep=5)
    # Assert
    assert ai.n_keep == 5


def test_init_preserves_max_tokens(google_env):
    # Arrange
    # Act
    ai = Google(model=_VALID_MODEL, max_tokens=2_048)
    # Assert
    assert ai.max_tokens == 2_048


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])

# EOF
