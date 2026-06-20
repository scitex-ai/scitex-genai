#!/usr/bin/env python3
# Timestamp: "2026-05-19 (rewritten for PA-307 / STX-TQ001-007)"
# File: ./tests/scitex_genai/llm/test__calc_cost.py
# ----------------------------------------

"""Tests for scitex_genai.llm._calc_cost.

Rewritten to comply with PA-307 (no unittest.mock, no monkeypatch fixture).
The previous version patched MODELS with a hand-shaped DataFrame. The
replacement uses the real MODELS table from _PARAMS, computing the
expected cost from real production pricing for each tested model.

calc_cost is pure pandas arithmetic — testing it against real pricing
data catches accidental column-shape changes or arithmetic regressions
that mock-based tests would miss.
"""

import os

import pytest

pd = pytest.importorskip("pandas")

from scitex_genai.llm import calc_cost
from scitex_genai.llm._PARAMS import MODELS as _REAL_MODELS


def _row(model: str):
    """Return the (input_cost, output_cost) tuple for a real model."""
    # Arrange
    # Act
    # Assert
    rows = _REAL_MODELS[_REAL_MODELS["name"] == model]
    return rows.iloc[0]["input_cost"], rows.iloc[0]["output_cost"]


# Pick a few real models to anchor the tests; if MODELS edits drop one
# of these, the test should fail loudly rather than silently mock around it.
_REAL_MODEL_PRIMARY = _REAL_MODELS.iloc[0]["name"]
_REAL_MODEL_OPENAI = _REAL_MODELS[_REAL_MODELS["provider"] == "OpenAI"].iloc[0]["name"]
_REAL_MODEL_ANTHROPIC = _REAL_MODELS[_REAL_MODELS["provider"] == "Anthropic"].iloc[0][
    "name"
]


def test_calc_cost_primary_model_matches_pricing_table():
    # Arrange
    in_tok, out_tok = 1_000, 500
    in_cost, out_cost = _row(_REAL_MODEL_PRIMARY)
    expected = (in_tok * in_cost + out_tok * out_cost) / 1_000_000
    # Act
    cost = calc_cost(_REAL_MODEL_PRIMARY, in_tok, out_tok)
    # Assert
    assert cost == pytest.approx(expected)


def test_calc_cost_openai_model_matches_pricing_table():
    # Arrange
    in_tok, out_tok = 5_000, 2_000
    in_cost, out_cost = _row(_REAL_MODEL_OPENAI)
    expected = (in_tok * in_cost + out_tok * out_cost) / 1_000_000
    # Act
    cost = calc_cost(_REAL_MODEL_OPENAI, in_tok, out_tok)
    # Assert
    assert cost == pytest.approx(expected)


def test_calc_cost_anthropic_model_matches_pricing_table():
    # Arrange
    in_tok, out_tok = 2_000, 1_000
    in_cost, out_cost = _row(_REAL_MODEL_ANTHROPIC)
    expected = (in_tok * in_cost + out_tok * out_cost) / 1_000_000
    # Act
    cost = calc_cost(_REAL_MODEL_ANTHROPIC, in_tok, out_tok)
    # Assert
    assert cost == pytest.approx(expected)


def test_calc_cost_zero_tokens_returns_zero():
    # Arrange
    # Act
    cost = calc_cost(_REAL_MODEL_PRIMARY, 0, 0)
    # Assert
    assert cost == 0.0


def test_calc_cost_only_input_tokens_charges_input_rate():
    # Arrange
    in_cost, _ = _row(_REAL_MODEL_PRIMARY)
    expected = (1_000 * in_cost) / 1_000_000
    # Act
    cost = calc_cost(_REAL_MODEL_PRIMARY, 1_000, 0)
    # Assert
    assert cost == pytest.approx(expected)


def test_calc_cost_only_output_tokens_charges_output_rate():
    # Arrange
    _, out_cost = _row(_REAL_MODEL_PRIMARY)
    expected = (1_000 * out_cost) / 1_000_000
    # Act
    cost = calc_cost(_REAL_MODEL_PRIMARY, 0, 1_000)
    # Assert
    assert cost == pytest.approx(expected)


def test_calc_cost_invalid_model_raises_value_error():
    # Arrange
    # Act
    # Assert
    with pytest.raises(
        ValueError, match="Model 'invalid-model' not found in pricing table"
    ):
        calc_cost("invalid-model", 100, 100)


def test_calc_cost_scales_linearly_with_input_tokens():
    # Arrange
    cost_1k = calc_cost(_REAL_MODEL_PRIMARY, 1_000, 0)
    # Act
    cost_10k = calc_cost(_REAL_MODEL_PRIMARY, 10_000, 0)
    # Assert
    assert cost_10k == pytest.approx(cost_1k * 10)


def test_calc_cost_scales_linearly_with_output_tokens():
    # Arrange
    cost_1k = calc_cost(_REAL_MODEL_PRIMARY, 0, 1_000)
    # Act
    cost_10k = calc_cost(_REAL_MODEL_PRIMARY, 0, 10_000)
    # Assert
    assert cost_10k == pytest.approx(cost_1k * 10)


def test_calc_cost_return_value_is_a_float():
    # Arrange
    # Act
    cost = calc_cost(_REAL_MODEL_PRIMARY, 100, 100)
    # Assert
    assert isinstance(cost, float)


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])

# EOF
