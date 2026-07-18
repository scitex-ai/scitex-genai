#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./tests/scitex_genai/llm/test__LiteLLM.py
# ----------------------------------------

"""Tests for scitex_genai.llm._LiteLLM (litellm dispatch backend).

Wiring-level only: no network calls, no mocks (PA-307). The pure model-string
mapping helper is tested directly; handler construction is exercised through
the real factory with fake API keys (SDKs validate keys on the remote call,
not at construction time).
"""

import pytest

pd = pytest.importorskip("pandas")

from scitex_genai.llm._genai_factory import genai_factory
from scitex_genai.llm._LiteLLM import LiteLLM, to_litellm_model
from scitex_genai.llm._PARAMS import MODELS as _REAL_MODELS


def _first_model_for(provider: str) -> str:
    rows = _REAL_MODELS[_REAL_MODELS.provider == provider].name.tolist()
    if not rows:
        pytest.skip(f"No models registered for provider {provider!r}")
    return rows[0]


@pytest.fixture(autouse=True)
def isolate_scitex_genai_env():
    """Clear fleet-injected SCITEX_GENAI_* vars for every test in this module.

    Agent containers inject SCITEX_GENAI_BASE_URL / SCITEX_GENAI_API_KEY
    fleet-wide, which would otherwise leak a base_url (and a backend) into
    these tests. Fixtures that need them set them explicitly on top of this.
    """
    import os

    names = (
        "SCITEX_GENAI_BASE_URL",
        "SCITEX_GENAI_API_KEY",
        "SCITEX_GENAI_BACKEND",
    )
    saved = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ.pop(name, None)
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


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


# ----------------------------------------
# Pure provider-prefix mapping (litellm model-string conventions)
# ----------------------------------------


def test_to_litellm_model_openai_stays_plain():
    # Arrange
    # Act
    mapped = to_litellm_model("gpt-4o-mini", provider="OpenAI")
    # Assert
    assert mapped == "gpt-4o-mini"


def test_to_litellm_model_anthropic_gets_prefix():
    # Arrange
    # Act
    mapped = to_litellm_model("claude-3-5-haiku-20241022", provider="Anthropic")
    # Assert
    assert mapped == "anthropic/claude-3-5-haiku-20241022"


def test_to_litellm_model_google_maps_to_gemini_prefix():
    # Arrange
    # Act
    mapped = to_litellm_model("gemini-2.5-flash", provider="Google")
    # Assert
    assert mapped == "gemini/gemini-2.5-flash"


def test_to_litellm_model_groq_gets_prefix():
    # Arrange
    # Act
    mapped = to_litellm_model("llama-3.3-70b-versatile", provider="Groq")
    # Assert
    assert mapped == "groq/llama-3.3-70b-versatile"


def test_to_litellm_model_deepseek_gets_prefix():
    # Arrange
    # Act
    mapped = to_litellm_model("deepseek-chat", provider="DeepSeek")
    # Assert
    assert mapped == "deepseek/deepseek-chat"


def test_to_litellm_model_perplexity_gets_prefix():
    # Arrange
    # Act
    mapped = to_litellm_model(
        "llama-3.1-sonar-small-128k-online", provider="Perplexity"
    )
    # Assert
    assert mapped == "perplexity/llama-3.1-sonar-small-128k-online"


def test_to_litellm_model_prefix_is_idempotent():
    # Arrange: an already-prefixed name must not be double-prefixed.
    # Act
    mapped = to_litellm_model("anthropic/claude-3-5-haiku-20241022", provider="Anthropic")
    # Assert
    assert mapped == "anthropic/claude-3-5-haiku-20241022"


# Self-hosted / OpenAI-compatible: litellm cannot infer a provider from an
# arbitrary local model name, so the wire string uses the openai/ prefix
# (litellm's documented convention for OpenAI-compatible endpoints).
def test_to_litellm_model_base_url_uses_openai_compatible_prefix():
    # Arrange
    # Act
    mapped = to_litellm_model("qwen36-35b-a3b", base_url="http://host:4000/v1")
    # Assert
    assert mapped == "openai/qwen36-35b-a3b"


def test_to_litellm_model_base_url_wins_over_provider():
    # Arrange: base_url means OpenAI-compatible wire protocol regardless of
    # any provider hint.
    # Act
    mapped = to_litellm_model(
        "qwen36-35b-a3b", provider="Anthropic", base_url="http://host:4000/v1"
    )
    # Assert
    assert mapped == "openai/qwen36-35b-a3b"


# ----------------------------------------
# Factory routing: backend="litellm" (explicit argument)
# ----------------------------------------


def test_factory_backend_litellm_returns_litellm_handler(fake_api_keys):
    # Arrange
    model = _first_model_for("Anthropic")
    # Act
    instance = genai_factory(model=model, api_key="fake-key", backend="litellm")
    # Assert
    assert type(instance) is LiteLLM


def test_factory_backend_litellm_maps_model_string(fake_api_keys):
    # Arrange
    model = _first_model_for("Anthropic")
    # Act
    instance = genai_factory(model=model, api_key="fake-key", backend="litellm")
    # Assert
    assert instance.litellm_model == f"anthropic/{model}"


def test_factory_backend_litellm_keeps_plain_model_attribute(fake_api_keys):
    # Arrange: `.model` is part of the public contract (cost/history); the
    # litellm prefix must stay internal to the wire-facing string.
    model = _first_model_for("Anthropic")
    # Act
    instance = genai_factory(model=model, api_key="fake-key", backend="litellm")
    # Assert
    assert instance.model == model


def test_factory_backend_litellm_works_for_openai_models(fake_api_keys):
    # Arrange
    model = _first_model_for("OpenAI")
    # Act
    instance = genai_factory(model=model, api_key="fake-key", backend="litellm")
    # Assert
    assert type(instance) is LiteLLM


def test_factory_backend_litellm_zero_cost_before_any_call(fake_api_keys):
    # Arrange
    model = _first_model_for("Anthropic")
    # Act
    instance = genai_factory(model=model, api_key="fake-key", backend="litellm")
    # Assert
    assert instance.cost == 0.0


def test_factory_rejects_unknown_backend():
    # Arrange
    # Act
    # Assert
    with pytest.raises(ValueError, match="Unknown backend"):
        genai_factory(model="gpt-4o-mini", api_key="fake-key", backend="not-a-backend")


# ----------------------------------------
# Factory routing: SCITEX_GENAI_BACKEND env var
# ----------------------------------------


@pytest.fixture
def litellm_backend_env():
    """Set SCITEX_GENAI_BACKEND=litellm; restore on teardown."""
    # Arrange
    import os

    saved = os.environ.get("SCITEX_GENAI_BACKEND")
    os.environ["SCITEX_GENAI_BACKEND"] = "litellm"
    # Act
    yield
    # Assert
    if saved is None:
        os.environ.pop("SCITEX_GENAI_BACKEND", None)
    else:
        os.environ["SCITEX_GENAI_BACKEND"] = saved


def test_factory_env_backend_litellm_returns_litellm_handler(
    fake_api_keys, litellm_backend_env
):
    # Arrange
    model = _first_model_for("Anthropic")
    # Act
    instance = genai_factory(model=model, api_key="fake-key")
    # Assert
    assert type(instance) is LiteLLM


def test_factory_explicit_backend_default_overrides_env(
    fake_api_keys, litellm_backend_env
):
    # Arrange: explicit args must win over the env fallback.
    model = _first_model_for("Anthropic")
    # Act
    instance = genai_factory(model=model, api_key="fake-key", backend="default")
    # Assert
    assert type(instance).__name__ == "Anthropic"


# ----------------------------------------
# Self-hosted passthrough through the litellm backend
# ----------------------------------------

_SELF_HOSTED_BASE_URL = "http://localhost:1/v1"


def _self_hosted_litellm_instance():
    return genai_factory(
        model="qwen36-35b-a3b",
        base_url=_SELF_HOSTED_BASE_URL,
        api_key="sk-x",
        backend="litellm",
    )


def test_factory_self_hosted_litellm_returns_litellm_handler():
    # Arrange
    # Act
    instance = _self_hosted_litellm_instance()
    # Assert
    assert type(instance) is LiteLLM


def test_factory_self_hosted_litellm_sets_base_url():
    # Arrange
    # Act
    instance = _self_hosted_litellm_instance()
    # Assert
    assert instance.base_url == _SELF_HOSTED_BASE_URL


def test_factory_self_hosted_litellm_maps_openai_compatible_model():
    # Arrange
    # Act
    instance = _self_hosted_litellm_instance()
    # Assert
    assert instance.litellm_model == "openai/qwen36-35b-a3b"


def test_factory_self_hosted_litellm_keeps_plain_model_attribute():
    # Arrange
    # Act
    instance = _self_hosted_litellm_instance()
    # Assert
    assert instance.model == "qwen36-35b-a3b"


def test_factory_self_hosted_litellm_passes_api_key():
    # Arrange
    # Act
    instance = _self_hosted_litellm_instance()
    # Assert
    assert instance.api_key == "sk-x"


# ----------------------------------------
# Instance contract (same as every other handler)
# ----------------------------------------


def test_litellm_handler_client_is_litellm_module(fake_api_keys):
    # Arrange
    model = _first_model_for("Anthropic")
    litellm = pytest.importorskip("litellm")
    # Act
    instance = genai_factory(model=model, api_key="fake-key", backend="litellm")
    # Assert
    assert instance.client is litellm


def test_litellm_handler_reset_clears_history(fake_api_keys):
    # Arrange
    model = _first_model_for("Anthropic")
    instance = genai_factory(model=model, api_key="fake-key", backend="litellm")
    instance.update_history("user", "hello")
    # Act
    instance.reset()
    # Assert
    assert instance.history == []


def test_litellm_handler_stream_flag_passes_through(fake_api_keys):
    # Arrange
    model = _first_model_for("Anthropic")
    # Act
    instance = genai_factory(
        model=model, api_key="fake-key", backend="litellm", stream=True
    )
    # Assert
    assert instance.stream is True


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# EOF
