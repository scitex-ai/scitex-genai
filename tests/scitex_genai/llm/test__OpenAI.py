#!/usr/bin/env python3
# File: tests/scitex_genai/llm/test__OpenAI.py

"""Tests for scitex_genai.llm._OpenAI.OpenAI.

These tests exercise the OpenAI provider class without mocks. The
BaseGenAI constructor catches exceptions when verify_model() or
_init_client() fails, so we can instantiate the class with a placeholder
api_key in CI and inspect attributes / list_models / class hierarchy
without hitting the OpenAI API.
"""

import pytest

from scitex_genai.llm._BaseGenAI import BaseGenAI
from scitex_genai.llm._OpenAI import OpenAI


@pytest.fixture
def openai_models():
    """Models advertised as OpenAI-provided in the real MODELS table."""
    return OpenAI.list_models(provider="OpenAI")


@pytest.fixture
def first_openai_model_name(openai_models):
    """The first OpenAI model name from the live MODELS table."""
    if not openai_models:
        pytest.skip("No OpenAI models present in MODELS table")
    return openai_models[0]


@pytest.fixture
def openai_instance_with_dummy_key(first_openai_model_name):
    """Instantiate OpenAI with a placeholder key (constructor swallows API errors)."""
    return OpenAI(
        model=first_openai_model_name,
        api_key="sk-test-dummy-key-1234",
        stream=False,
        temperature=0.7,
    )


def test_openai_class_inherits_from_basegenai():
    """``OpenAI`` is a concrete subclass of ``BaseGenAI``."""
    # Arrange
    cls = OpenAI

    # Act
    is_subclass = issubclass(cls, BaseGenAI)

    # Assert
    assert is_subclass is True


def test_openai_list_models_returns_nonempty_list(openai_models):
    """``OpenAI.list_models(provider='OpenAI')`` returns at least one model."""
    # Arrange
    models = openai_models

    # Act
    count = len(models)

    # Assert
    assert count > 0


def test_openai_list_models_returns_list_of_strings(openai_models):
    """Every element in ``list_models`` is a non-empty string."""
    # Arrange
    models = openai_models

    # Act
    all_strings = all(isinstance(m, str) and len(m) > 0 for m in models)

    # Assert
    assert all_strings is True


def test_openai_instance_sets_provider_to_openai(openai_instance_with_dummy_key):
    """An OpenAI instance records its provider as ``"OpenAI"``."""
    # Arrange
    instance = openai_instance_with_dummy_key

    # Act
    provider = instance.provider

    # Assert
    assert provider == "OpenAI"


def test_openai_instance_stores_temperature(openai_instance_with_dummy_key):
    """An OpenAI instance stores the temperature it was constructed with."""
    # Arrange
    instance = openai_instance_with_dummy_key

    # Act
    temp = instance.temperature

    # Assert
    assert temp == 0.7


def test_openai_instance_stores_stream_flag(openai_instance_with_dummy_key):
    """An OpenAI instance stores the stream flag it was constructed with."""
    # Arrange
    instance = openai_instance_with_dummy_key

    # Act
    stream = instance.stream

    # Assert
    assert stream is False


def test_openai_instance_masked_api_key_contains_stars(
    openai_instance_with_dummy_key,
):
    """The masked_api_key property masks the middle of the key with stars."""
    # Arrange
    instance = openai_instance_with_dummy_key

    # Act
    masked = instance.masked_api_key

    # Assert
    assert "****" in masked


def test_openai_gpt4_turbo_model_string_yields_128k_max_tokens():
    """A model name containing ``gpt-4-turbo`` defaults max_tokens to 128_000."""
    # Arrange
    instance = OpenAI(model="gpt-4-turbo-2024-04-09", api_key="sk-dummy")

    # Act
    max_tokens = instance.max_tokens

    # Assert
    assert max_tokens == 128_000


def test_openai_gpt4_model_string_yields_8192_max_tokens():
    """A model name containing ``gpt-4`` (but not turbo) defaults max_tokens to 8_192."""
    # Arrange
    instance = OpenAI(model="gpt-4-0613", api_key="sk-dummy")

    # Act
    max_tokens = instance.max_tokens

    # Assert
    assert max_tokens == 8_192


def test_openai_gpt35_turbo_16k_model_yields_16384_max_tokens():
    """A model name containing ``gpt-3.5-turbo-16k`` defaults max_tokens to 16_384."""
    # Arrange
    instance = OpenAI(model="gpt-3.5-turbo-16k", api_key="sk-dummy")

    # Act
    max_tokens = instance.max_tokens

    # Assert
    assert max_tokens == 16_384


def test_openai_unknown_model_string_yields_4096_max_tokens():
    """An unknown / fallback model name defaults max_tokens to 4_096."""
    # Arrange
    instance = OpenAI(model="completely-unknown-model", api_key="sk-dummy")

    # Act
    max_tokens = instance.max_tokens

    # Assert
    assert max_tokens == 4_096


def test_openai_explicit_max_tokens_argument_is_respected():
    """When the caller passes max_tokens explicitly, it overrides the default."""
    # Arrange
    instance = OpenAI(model="gpt-4", api_key="sk-dummy", max_tokens=2_048)

    # Act
    max_tokens = instance.max_tokens

    # Assert
    assert max_tokens == 2_048


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])
