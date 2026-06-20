#!/usr/bin/env python3
# Timestamp: "2026-05-19 (rewritten for PA-307 / STX-TQ001-007)"
# File: ./tests/scitex_genai/llm/test__BaseGenAI.py
# ----------------------------------------

"""Tests for scitex_genai.llm._BaseGenAI.

Rewritten to comply with PA-307 (no unittest.mock, no monkeypatch fixture).

Approach:
  - Use a hand-rolled FakeGenAI concrete subclass that records its own
    calls; production code talks to this real collaborator.
  - Use real MODELS entries from scitex_genai.llm._PARAMS so verify_model
    succeeds against actual production data, not a hand-shaped fake.
  - For abstract-class concerns (init, history shaping, streaming helpers)
    the tests touch only the BaseGenAI pure-Python logic — no SDK calls.
"""

from typing import Any, Generator, List

import pytest

pd = pytest.importorskip("pandas")

from scitex_genai.llm import BaseGenAI
from scitex_genai.llm._PARAMS import MODELS as _REAL_MODELS

# A real model + provider pair that ships in MODELS, used as the
# "valid" axis of verify_model tests. Picked at module load time so
# the assertion is stable across MODELS edits as long as the row exists.
_REAL_ROW = _REAL_MODELS.iloc[0].to_dict()
REAL_MODEL_NAME = _REAL_ROW["name"]
REAL_MODEL_PROVIDER = _REAL_ROW["provider"]


class FakeGenAI(BaseGenAI):
    """Hand-rolled concrete subclass — no mocks, records its own calls."""

    def __init__(self, *args, **kwargs):
        self.calls: List[tuple] = []
        self._static_response = kwargs.pop("_static_response", "Test response")
        self._stream_chunks = kwargs.pop(
            "_stream_chunks", ["Test", " ", "stream", " ", "response"]
        )
        super().__init__(*args, **kwargs)

    def _init_client(self) -> Any:
        self.calls.append(("_init_client",))
        return object()  # opaque sentinel — not a Mock

    def _api_call_static(self) -> str:
        self.calls.append(("_api_call_static",))
        return self._static_response

    def _api_call_stream(self) -> Generator[str, None, None]:
        self.calls.append(("_api_call_stream",))
        for chunk in self._stream_chunks:
            yield chunk


@pytest.fixture
def gen_ai():
    """A FakeGenAI instance built against a real production MODELS row."""
    # Arrange
    # Act
    # Assert
    return FakeGenAI(
        model=REAL_MODEL_NAME,
        api_key="test-key-1234",
        provider=REAL_MODEL_PROVIDER,
    )


def test_initialization_preserves_system_setting():
    # Arrange
    # Act
    ai = FakeGenAI(
        system_setting="You are a helpful assistant",
        model=REAL_MODEL_NAME,
        api_key="test-key",
        provider=REAL_MODEL_PROVIDER,
    )
    # Assert
    assert ai.system_setting == "You are a helpful assistant"


def test_initialization_preserves_model_name():
    # Arrange
    # Act
    ai = FakeGenAI(model=REAL_MODEL_NAME, api_key="k", provider=REAL_MODEL_PROVIDER)
    # Assert
    assert ai.model == REAL_MODEL_NAME


def test_initialization_preserves_api_key():
    # Arrange
    # Act
    ai = FakeGenAI(
        model=REAL_MODEL_NAME, api_key="abcd1234", provider=REAL_MODEL_PROVIDER
    )
    # Assert
    assert ai.api_key == "abcd1234"


def test_initialization_preserves_stream_flag():
    # Arrange
    # Act
    ai = FakeGenAI(
        model=REAL_MODEL_NAME,
        api_key="k",
        provider=REAL_MODEL_PROVIDER,
        stream=True,
    )
    # Assert
    assert ai.stream is True


def test_initialization_preserves_seed():
    # Arrange
    # Act
    ai = FakeGenAI(
        model=REAL_MODEL_NAME, api_key="k", provider=REAL_MODEL_PROVIDER, seed=42
    )
    # Assert
    assert ai.seed == 42


def test_initialization_preserves_n_keep():
    # Arrange
    # Act
    ai = FakeGenAI(
        model=REAL_MODEL_NAME, api_key="k", provider=REAL_MODEL_PROVIDER, n_keep=5
    )
    # Assert
    assert ai.n_keep == 5


def test_initialization_preserves_temperature():
    # Arrange
    # Act
    ai = FakeGenAI(
        model=REAL_MODEL_NAME,
        api_key="k",
        provider=REAL_MODEL_PROVIDER,
        temperature=0.7,
    )
    # Assert
    assert ai.temperature == 0.7


def test_initialization_preserves_max_tokens():
    # Arrange
    # Act
    ai = FakeGenAI(
        model=REAL_MODEL_NAME,
        api_key="k",
        provider=REAL_MODEL_PROVIDER,
        max_tokens=2_048,
    )
    # Assert
    assert ai.max_tokens == 2_048


def test_initialization_preserves_provider():
    # Arrange
    # Act
    ai = FakeGenAI(model=REAL_MODEL_NAME, api_key="k", provider=REAL_MODEL_PROVIDER)
    # Assert
    assert ai.provider == REAL_MODEL_PROVIDER


def test_initialization_calls_init_client_once():
    # Arrange
    # Act
    ai = FakeGenAI(model=REAL_MODEL_NAME, api_key="k", provider=REAL_MODEL_PROVIDER)
    # Assert
    assert ai.calls == [("_init_client",)]


def test_masked_api_key_obscures_middle(gen_ai):
    # Arrange
    # Act
    masked = gen_ai.masked_api_key
    # Assert
    assert masked == "test****1234"


def test_list_models_all_returns_real_production_names(capsys):
    # Arrange
    expected = _REAL_MODELS.name.tolist()
    # Act
    result = BaseGenAI.list_models()
    # Assert
    assert result == expected


def test_list_models_filtered_by_provider_subset(capsys):
    # Arrange
    expected = _REAL_MODELS[
        _REAL_MODELS.api_key_env.str.contains(REAL_MODEL_PROVIDER, case=False)
    ].name.tolist()
    # Act
    result = BaseGenAI.list_models(provider=REAL_MODEL_PROVIDER)
    # Assert
    assert result == expected


def test_reset_with_no_system_setting_empties_history(gen_ai):
    # Arrange
    gen_ai.history = [{"role": "user", "content": "test"}]
    # Act
    gen_ai.reset()
    # Assert
    assert gen_ai.history == []


def test_reset_with_system_setting_seeds_history(gen_ai):
    # Arrange
    # Act
    gen_ai.reset("New system setting")
    # Assert
    assert gen_ai.history == [{"role": "system", "content": "New system setting"}]


def test_update_history_appends_user_message(gen_ai):
    # Arrange
    gen_ai.n_keep = 10
    # Act
    gen_ai.update_history("user", "Hello")
    # Assert
    assert gen_ai.history == [{"role": "user", "content": "Hello"}]


def test_ensure_alternative_history_merges_consecutive_same_role(gen_ai):
    # Arrange
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "Hi again"},
    ]
    # Act
    result = gen_ai._ensure_alternative_history(history)
    # Assert
    assert result == [{"role": "user", "content": "Hello\n\nHi again"}]


def test_ensure_start_from_user_strips_leading_non_user():
    # Arrange
    history = [
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Hello"},
    ]
    # Act
    result = BaseGenAI._ensure_start_from_user(history)
    # Assert
    assert result == [{"role": "user", "content": "Hello"}]


def test_call_static_mode_returns_concrete_response(gen_ai):
    # Arrange
    gen_ai.stream = False
    gen_ai.n_keep = 10
    # Act
    result = gen_ai("Test prompt")
    # Assert
    assert result == "Test response"


def test_call_stream_mode_concatenates_chunks(gen_ai):
    # Arrange
    gen_ai.stream = True
    # Act
    result = gen_ai("Test prompt")
    # Assert
    assert result == "Test stream response"


def test_call_with_empty_prompt_returns_none(gen_ai, capsys):
    # Arrange
    # Act
    result = gen_ai("")
    # Assert
    assert result is None


def test_verify_model_with_valid_model_stores_no_error():
    # Arrange
    # Act
    ai = FakeGenAI(model=REAL_MODEL_NAME, api_key="k", provider=REAL_MODEL_PROVIDER)
    # Assert
    assert ai._error_messages == []


def test_verify_model_with_invalid_model_stores_error():
    # Arrange
    # Act
    ai = FakeGenAI(
        model="definitely-not-a-real-model-name",
        api_key="k",
        provider=REAL_MODEL_PROVIDER,
    )
    # Assert
    assert any("not supported" in m for m in ai._error_messages)


def test_to_stream_single_string_yields_one_chunk():
    # Arrange
    # Act
    chunks = list(BaseGenAI._to_stream("Hello world"))
    # Assert
    assert chunks == ["Hello world"]


def test_to_stream_list_yields_per_item():
    # Arrange
    # Act
    chunks = list(BaseGenAI._to_stream(["Hello", " ", "world"]))
    # Assert
    assert chunks == ["Hello", " ", "world"]


def test_n_keep_limits_history_length(gen_ai):
    # Arrange
    gen_ai.n_keep = 3
    # Act
    for i in range(5):
        gen_ai.update_history("user", f"Message {i}")
    # Assert
    assert len(gen_ai.history) <= gen_ai.n_keep


def test_gen_error_returns_concatenated_messages(gen_ai):
    # Arrange
    gen_ai._error_messages.append("Test error")
    # Act
    error_flag, error_obj = gen_ai.gen_error(return_stream=False)
    # Assert
    assert (error_flag, error_obj) == (True, "Test error")


def test_abstract_methods_require_concrete_implementation():
    # Arrange
    class IncompleteGenAI(BaseGenAI):
        pass

    # Act
    # Assert
    with pytest.raises(TypeError):
        IncompleteGenAI()


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# EOF
