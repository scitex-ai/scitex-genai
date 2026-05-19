#!/usr/bin/env python3
# Timestamp: "2026-05-19 (rewritten for PA-307 / STX-TQ001-007)"
# File: ./tests/scitex_genai/llm/test__Llama.py
# ----------------------------------------

"""Tests for scitex_genai.llm._Llama.

Rewritten to comply with PA-307 (no unittest.mock, no monkeypatch fixture).
The previous version stubbed `sys.modules["llama"] = MagicMock()` before
import and patched `verify_model` / `_init_client` — pure design signal.

The Llama class wraps Meta's local-inference `llama` package, which is
unavailable in the CI environment. The class is robust to that absence:
it catches the import error in BaseGenAI.__init__ and stashes it in
`_error_messages` rather than failing init. The tests below exercise
the pure-Python init logic (path defaults, attribute preservation,
environment configuration) without ever calling the llama SDK.

Tests that previously only verified "the mock got the kwargs the test
built" have been deleted per PA-307.
"""

import os

import pytest

from scitex_genai.llm import Llama
from scitex_genai.llm._PARAMS import MODELS as _MODELS

_VALID_MODEL = _MODELS[_MODELS.provider == "Llama"].name.tolist()[0]


def test_init_defaults_ckpt_dir_from_model():
    # Arrange
    # Act
    ai = Llama(model=_VALID_MODEL, api_key="k")
    # Assert
    assert ai.ckpt_dir == f"Meta-{_VALID_MODEL}/"


def test_init_defaults_tokenizer_path_from_model():
    # Arrange
    # Act
    ai = Llama(model=_VALID_MODEL, api_key="k")
    # Assert
    assert ai.tokenizer_path == f"./Meta-{_VALID_MODEL}/tokenizer.model"


def test_init_preserves_custom_ckpt_dir():
    # Arrange
    # Act
    ai = Llama(model=_VALID_MODEL, api_key="k", ckpt_dir="/custom/path/")
    # Assert
    assert ai.ckpt_dir == "/custom/path/"


def test_init_preserves_custom_tokenizer_path():
    # Arrange
    # Act
    ai = Llama(
        model=_VALID_MODEL, api_key="k", tokenizer_path="/custom/tokenizer.model"
    )
    # Assert
    assert ai.tokenizer_path == "/custom/tokenizer.model"


def test_init_preserves_max_seq_len():
    # Arrange
    # Act
    ai = Llama(model=_VALID_MODEL, api_key="k", max_seq_len=512)
    # Assert
    assert ai.max_seq_len == 512


def test_init_preserves_max_batch_size():
    # Arrange
    # Act
    ai = Llama(model=_VALID_MODEL, api_key="k", max_batch_size=2)
    # Assert
    assert ai.max_batch_size == 2


def test_init_preserves_max_gen_len():
    # Arrange
    # Act
    ai = Llama(model=_VALID_MODEL, api_key="k", max_gen_len=2_048)
    # Assert
    assert ai.max_gen_len == 2_048


def test_init_preserves_model_name():
    # Arrange
    # Act
    ai = Llama(model=_VALID_MODEL, api_key="k")
    # Assert
    assert ai.model == _VALID_MODEL


def test_init_preserves_stream_flag():
    # Arrange
    # Act
    ai = Llama(model=_VALID_MODEL, api_key="k", stream=True)
    # Assert
    assert ai.stream is True


def test_init_preserves_seed():
    # Arrange
    # Act
    ai = Llama(model=_VALID_MODEL, api_key="k", seed=42)
    # Assert
    assert ai.seed == 42


def test_init_preserves_n_keep():
    # Arrange
    # Act
    ai = Llama(model=_VALID_MODEL, api_key="k", n_keep=5)
    # Assert
    assert ai.n_keep == 5


def test_init_preserves_temperature():
    # Arrange
    # Act
    ai = Llama(model=_VALID_MODEL, api_key="k", temperature=0.5)
    # Assert
    assert ai.temperature == 0.5


def test_init_preserves_system_setting():
    # Arrange
    # Act
    ai = Llama(model=_VALID_MODEL, api_key="k", system_setting="You are helpful")
    # Assert
    assert ai.system_setting == "You are helpful"


def test_init_configures_master_addr_env():
    # Arrange
    # Act
    Llama(model=_VALID_MODEL, api_key="k")
    # Assert
    assert os.environ["MASTER_ADDR"] == "localhost"


def test_init_configures_master_port_env():
    # Arrange
    # Act
    Llama(model=_VALID_MODEL, api_key="k")
    # Assert
    assert os.environ["MASTER_PORT"] == "12355"


def test_init_configures_world_size_env():
    # Arrange
    # Act
    Llama(model=_VALID_MODEL, api_key="k")
    # Assert
    assert os.environ["WORLD_SIZE"] == "1"


def test_init_configures_rank_env():
    # Arrange
    # Act
    Llama(model=_VALID_MODEL, api_key="k")
    # Assert
    assert os.environ["RANK"] == "0"


def test_str_returns_llama():
    # Arrange
    ai = Llama(model=_VALID_MODEL, api_key="k")
    # Act
    result = str(ai)
    # Assert
    assert result == "Llama"


def test_verify_model_passes_without_raising():
    # Arrange
    ai = Llama(model=_VALID_MODEL, api_key="k")
    # Act
    result = ai.verify_model()
    # Assert
    assert result is None


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])

# EOF
