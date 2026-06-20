#!/usr/bin/env python3
# Timestamp: "2026-05-19 (rewritten for PA-307 / STX-TQ001-007)"
# File: ./tests/scitex_genai/llm/test__genai_factory.py
# ----------------------------------------

"""Tests for scitex_genai.llm._genai_factory.

Rewritten to comply with PA-307 (no unittest.mock, no monkeypatch fixture).

The previous version patched MODELS, every provider class, and random.choice,
then asserted that the patched classes were called with arrays of kwargs. That
verified only that the test rewrote the world correctly — zero production
signal. Per PA-307: design signal -> delete those tests.

The replacement tests exercise the *real* dispatch logic against the *real*
MODELS table, instantiating each provider with a fake API key (the SDKs only
validate keys on the actual remote call, not at construction time). What
survives is the dispatch contract and the error path.
"""

import pytest

pd = pytest.importorskip("pandas")

from scitex_genai.llm._genai_factory import genai_factory
from scitex_genai.llm._PARAMS import MODELS as _REAL_MODELS


# Provider -> first model name shipped for that provider.
# Computed at import so failures highlight a missing provider, not a typo.
def _first_model_for(provider: str) -> str:
    rows = _REAL_MODELS[_REAL_MODELS.provider == provider].name.tolist()
    if not rows:
        pytest.skip(f"No models registered for provider {provider!r}")
    return rows[0]


@pytest.fixture
def fake_api_keys():
    """Set fake API keys for all providers; restore on teardown."""
    # Arrange
    import os

    keys = {
        "ANTHROPIC_API_KEY": "sk-fake",
        "OPENAI_API_KEY": "sk-fake",
        "GOOGLE_API_KEY": "sk-fake",
        "GROQ_API_KEY": "sk-fake",
        "DEEPSEEK_API_KEY": "sk-fake",
        "PERPLEXITY_API_KEY": "sk-fake",
    }
    saved = {k: os.environ.get(k) for k in keys}
    os.environ.update(keys)
    # Act
    yield
    # Assert
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_factory_dispatches_anthropic_provider(fake_api_keys):
    # Arrange
    model = _first_model_for("Anthropic")
    # Act
    instance = genai_factory(model=model, api_key="fake-key")
    # Assert
    assert type(instance).__name__ == "Anthropic"


def test_factory_dispatches_openai_provider(fake_api_keys):
    # Arrange
    model = _first_model_for("OpenAI")
    # Act
    instance = genai_factory(model=model, api_key="fake-key")
    # Assert
    assert type(instance).__name__ == "OpenAI"


def test_factory_dispatches_google_provider(fake_api_keys):
    # Arrange
    model = _first_model_for("Google")
    # Act
    instance = genai_factory(model=model, api_key="fake-key")
    # Assert
    assert type(instance).__name__ == "Google"


def test_factory_dispatches_groq_provider(fake_api_keys):
    # Arrange
    model = _first_model_for("Groq")
    # Act
    instance = genai_factory(model=model, api_key="fake-key")
    # Assert
    assert type(instance).__name__ == "Groq"


def test_factory_dispatches_deepseek_provider(fake_api_keys):
    # Arrange
    model = _first_model_for("DeepSeek")
    # Act
    instance = genai_factory(model=model, api_key="fake-key")
    # Assert
    assert type(instance).__name__ == "DeepSeek"


def test_factory_dispatches_perplexity_provider(fake_api_keys):
    # Arrange
    model = _first_model_for("Perplexity")
    # Act
    instance = genai_factory(model=model, api_key="fake-key")
    # Assert
    assert type(instance).__name__ == "Perplexity"


def test_factory_rejects_unknown_model_with_value_error():
    # Arrange
    # Act
    # Assert
    with pytest.raises(ValueError, match='Model "not-a-real-model" is not available'):
        genai_factory(model="not-a-real-model")


def test_factory_passes_api_key_through_to_instance(fake_api_keys):
    # Arrange
    model = _first_model_for("Anthropic")
    # Act
    instance = genai_factory(model=model, api_key="my-explicit-key")
    # Assert
    assert instance.api_key == "my-explicit-key"


def test_factory_passes_temperature_through_to_instance(fake_api_keys):
    # Arrange
    model = _first_model_for("Anthropic")
    # Act
    instance = genai_factory(model=model, api_key="fake-key", temperature=0.3)
    # Assert
    assert instance.temperature == 0.3


def test_factory_passes_max_tokens_through_to_instance(fake_api_keys):
    # Arrange
    model = _first_model_for("Anthropic")
    # Act
    instance = genai_factory(model=model, api_key="fake-key", max_tokens=1_024)
    # Assert
    assert instance.max_tokens == 1_024


def test_factory_passes_stream_flag_through_to_instance(fake_api_keys):
    # Arrange
    model = _first_model_for("Anthropic")
    # Act
    instance = genai_factory(model=model, api_key="fake-key", stream=True)
    # Assert
    assert instance.stream is True


def test_factory_picks_one_key_when_api_key_is_list(fake_api_keys):
    # Arrange
    model = _first_model_for("Anthropic")
    candidates = ["alpha-key", "beta-key", "gamma-key"]
    # Act
    instance = genai_factory(model=model, api_key=candidates)
    # Assert
    assert instance.api_key in candidates


def test_factory_picks_one_key_when_api_key_is_tuple(fake_api_keys):
    # Arrange
    model = _first_model_for("Anthropic")
    candidates = ("alpha-key", "beta-key", "gamma-key")
    # Act
    instance = genai_factory(model=model, api_key=candidates)
    # Assert
    assert instance.api_key in candidates


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# EOF
