#!/usr/bin/env python3
# File: ./tests/scitex_genai/llm/test__VLLM.py
# ----------------------------------------

"""Tests for scitex_genai.llm._VLLM (OpenAI-compatible local vLLM provider).

No mocks (PA-307): every test constructs a real `VLLM` instance with the
placeholder API key vLLM ignores. `_init_client` builds an `openai.OpenAI`
client object — that's a pure constructor call, no network — so offline
runs in CI work without a live vLLM server.
"""

import os

import pytest

from scitex_genai.llm import VLLM, GenAI
from scitex_genai.llm._PARAMS import MODELS as _MODELS
from scitex_genai.llm._VLLM import DEFAULT_VLLM_BASE_URL

_VLLM_MODEL = "qwen36-35b-fp8"


def test_qwen36_35b_fp8_is_registered_in_models():
    # Arrange
    # Act
    available = _MODELS.name.tolist()
    # Assert
    assert _VLLM_MODEL in available


def test_qwen36_35b_fp8_row_uses_vllm_provider():
    # Arrange
    row = _MODELS[_MODELS.name == _VLLM_MODEL].iloc[0]
    # Act
    provider = row.provider
    # Assert
    assert provider == "vLLM"


def test_genai_factory_builds_vllm_instance_for_qwen():
    # Arrange
    # Act
    ai = GenAI(model=_VLLM_MODEL)
    # Assert
    assert isinstance(ai, VLLM)


def test_vllm_init_records_provider_as_vllm():
    # Arrange
    # Act
    ai = VLLM(model=_VLLM_MODEL)
    # Assert
    assert ai.provider == "vLLM"


def test_vllm_init_preserves_qwen_model_name():
    # Arrange
    # Act
    ai = VLLM(model=_VLLM_MODEL)
    # Assert
    assert ai.model == _VLLM_MODEL


def test_vllm_default_base_url_is_local_8765():
    # Arrange
    saved = os.environ.pop("SCITEX_GENAI_VLLM_BASE_URL", None)
    try:
        # Act
        ai = VLLM(model=_VLLM_MODEL)
        # Assert
        assert ai.base_url == DEFAULT_VLLM_BASE_URL
    finally:
        if saved is not None:
            os.environ["SCITEX_GENAI_VLLM_BASE_URL"] = saved


def test_vllm_base_url_honours_env_override():
    # Arrange
    override = "http://10.0.0.42:9000/v1"
    saved = os.environ.get("SCITEX_GENAI_VLLM_BASE_URL")
    os.environ["SCITEX_GENAI_VLLM_BASE_URL"] = override
    try:
        # Act
        ai = VLLM(model=_VLLM_MODEL)
        # Assert
        assert ai.base_url == override
    finally:
        if saved is None:
            os.environ.pop("SCITEX_GENAI_VLLM_BASE_URL", None)
        else:
            os.environ["SCITEX_GENAI_VLLM_BASE_URL"] = saved


def test_vllm_base_url_explicit_arg_beats_env():
    # Arrange
    explicit = "http://192.168.1.5:8000/v1"
    saved = os.environ.get("SCITEX_GENAI_VLLM_BASE_URL")
    os.environ["SCITEX_GENAI_VLLM_BASE_URL"] = "http://ignored:1/v1"
    try:
        # Act
        ai = VLLM(model=_VLLM_MODEL, base_url=explicit)
        # Assert
        assert ai.base_url == explicit
    finally:
        if saved is None:
            os.environ.pop("SCITEX_GENAI_VLLM_BASE_URL", None)
        else:
            os.environ["SCITEX_GENAI_VLLM_BASE_URL"] = saved


def test_vllm_substitutes_empty_placeholder_when_no_key():
    # Arrange
    saved = os.environ.pop("SCITEX_GENAI_VLLM_API_KEY", None)
    try:
        # Act
        ai = VLLM(model=_VLLM_MODEL)
        # Assert
        assert ai.api_key == "EMPTY"
    finally:
        if saved is not None:
            os.environ["SCITEX_GENAI_VLLM_API_KEY"] = saved


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])

# EOF
