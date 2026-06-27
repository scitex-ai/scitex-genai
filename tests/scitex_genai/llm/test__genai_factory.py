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


def test_factory_self_hosted_base_url_constructs_openai_handler():
    # Arrange
    # An unknown model name plus a base_url targets a self-hosted,
    # OpenAI-compatible endpoint (e.g. a vLLM model behind a LiteLLM proxy).
    base_url = "http://localhost:1/v1"
    # Act
    instance = genai_factory(
        model="some-local-model",
        base_url=base_url,
        api_key="sk-x",
    )
    # Assert
    assert type(instance).__name__ == "OpenAI"
    assert instance.base_url == base_url
    # The openai SDK normalizes base_url; it must still reflect the host we passed.
    assert str(instance.client.base_url).rstrip("/") == base_url.rstrip("/")


def test_factory_self_hosted_unknown_model_does_not_raise():
    # Arrange
    # Act
    instance = genai_factory(
        model="qwen36-35b-a3b",
        base_url="http://some-host:4000/v1",
        api_key="sk-clew-local",
    )
    # Assert
    assert instance.base_url == "http://some-host:4000/v1"


@pytest.fixture
def self_hosted_env():
    """Set the fleet-injected self-hosted env vars; restore on teardown."""
    # Arrange
    import os

    env = {
        "SCITEX_GENAI_BASE_URL": "http://env-host:4000/v1",
        "SCITEX_GENAI_API_KEY": "sk-env-local",
    }
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    # Act
    yield env
    # Assert
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_factory_self_hosted_reads_base_url_and_key_from_env(self_hosted_env):
    # Arrange
    # With SCITEX_GENAI_BASE_URL/API_KEY injected (e.g. fleet-wide), an unknown
    # model needs no explicit base_url/api_key — the passthrough path reads env.
    # Act
    instance = genai_factory(model="qwen36-35b-a3b")
    # Assert
    assert type(instance).__name__ == "OpenAI"
    assert instance.base_url == self_hosted_env["SCITEX_GENAI_BASE_URL"]
    assert instance.api_key == self_hosted_env["SCITEX_GENAI_API_KEY"]


def test_factory_explicit_args_override_self_hosted_env(self_hosted_env):
    # Arrange
    # Explicit args must win over the injected env fallback.
    # Act
    instance = genai_factory(
        model="qwen36-35b-a3b",
        base_url="http://explicit-host:4000/v1",
        api_key="sk-explicit",
    )
    # Assert
    assert instance.base_url == "http://explicit-host:4000/v1"
    assert instance.api_key == "sk-explicit"


def test_factory_unknown_model_without_base_url_or_provider_raises():
    # Arrange
    # Regression guard: an unknown model with neither base_url nor an explicit
    # provider must still raise, exactly as before this feature.
    # Act
    # Assert
    with pytest.raises(ValueError, match='Model "not-a-real-model" is not available'):
        genai_factory(model="not-a-real-model")


def test_factory_known_model_still_resolves_without_base_url(fake_api_keys):
    # Arrange
    # Regression guard: a real registered OpenAI model resolves exactly as
    # before, with no base_url threaded through.
    model = _first_model_for("OpenAI")
    # Act
    instance = genai_factory(model=model, api_key="fake-key")
    # Assert
    assert type(instance).__name__ == "OpenAI"
    assert instance.base_url is None


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
