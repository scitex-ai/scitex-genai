#!/usr/bin/env python3
# Timestamp: "2025-06-13 23:03:53 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/SciTeX-Code/tests/scitex/ai/llm/test__calc_cost.py
# ----------------------------------------
import os

__FILE__ = "./tests/scitex/ai/llm/test__calc_cost.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
# Time-stamp: "2025-06-01 13:55:00 (ywatanabe)"

"""Tests for scitex_genai.llm._calc_cost module.

Each test exercises one observable cost-calculation behaviour with
explicit Arrange / Act / Assert markers and a single assertion. A
deterministic in-memory pricing DataFrame is passed via the new
``models=`` keyword so the unit tests are independent of the live
MODELS table maintained in source.
"""

import pytest

pd = pytest.importorskip("pandas")

from scitex_genai.llm import calc_cost


@pytest.fixture
def fake_models_df():
    """Deterministic pricing table covering OpenAI / Anthropic / free-tier rows."""
    return pd.DataFrame(
        {
            "name": ["gpt-4", "gpt-3.5-turbo", "claude-3-opus", "free-model"],
            "input_cost": [30.00, 0.50, 15.00, 0.00],
            "output_cost": [60.00, 1.50, 75.00, 0.00],
            "provider": ["OpenAI", "OpenAI", "Anthropic", "Test"],
        }
    )


class TestCalcCost:
    """Test suite for calc_cost function."""

    def test_calc_cost_gpt4_returns_six_cents_for_1000_in_500_out(self, fake_models_df):
        """GPT-4 at 1000 input + 500 output tokens costs exactly $0.06."""
        # Arrange
        model = "gpt-4"
        input_tokens, output_tokens = 1_000, 500

        # Act
        cost = calc_cost(model, input_tokens, output_tokens, models=fake_models_df)

        # Assert
        assert cost == 0.06

    def test_calc_cost_gpt35_turbo_returns_pricing_formula_result(self, fake_models_df):
        """GPT-3.5-turbo cost matches the (in*0.50 + out*1.50)/1M formula."""
        # Arrange
        model = "gpt-3.5-turbo"
        input_tokens, output_tokens = 5_000, 2_000
        expected = (input_tokens * 0.50 + output_tokens * 1.50) / 1_000_000

        # Act
        cost = calc_cost(model, input_tokens, output_tokens, models=fake_models_df)

        # Assert
        assert cost == expected

    def test_calc_cost_claude_opus_returns_pricing_formula_result(self, fake_models_df):
        """Claude-3-opus cost matches the (in*15 + out*75)/1M formula."""
        # Arrange
        model = "claude-3-opus"
        input_tokens, output_tokens = 2_000, 1_000
        expected = (input_tokens * 15.00 + output_tokens * 75.00) / 1_000_000

        # Act
        cost = calc_cost(model, input_tokens, output_tokens, models=fake_models_df)

        # Assert
        assert cost == expected

    def test_calc_cost_zero_tokens_returns_zero(self, fake_models_df):
        """Zero input + zero output tokens cost $0."""
        # Arrange
        model = "gpt-4"

        # Act
        cost = calc_cost(model, 0, 0, models=fake_models_df)

        # Assert
        assert cost == 0.0

    def test_calc_cost_only_input_tokens_uses_input_pricing(self, fake_models_df):
        """With zero output tokens, cost equals input_tokens * input_cost / 1M."""
        # Arrange
        model = "gpt-4"
        input_tokens = 1_000
        expected = (input_tokens * 30.00) / 1_000_000

        # Act
        cost = calc_cost(model, input_tokens, 0, models=fake_models_df)

        # Assert
        assert cost == expected

    def test_calc_cost_only_output_tokens_uses_output_pricing(self, fake_models_df):
        """With zero input tokens, cost equals output_tokens * output_cost / 1M."""
        # Arrange
        model = "gpt-4"
        output_tokens = 1_000
        expected = (output_tokens * 60.00) / 1_000_000

        # Act
        cost = calc_cost(model, 0, output_tokens, models=fake_models_df)

        # Assert
        assert cost == expected

    def test_calc_cost_free_model_returns_zero_for_any_token_count(
        self, fake_models_df
    ):
        """A model with zero input/output costs always returns 0."""
        # Arrange
        model = "free-model"

        # Act
        cost = calc_cost(model, 10_000, 10_000, models=fake_models_df)

        # Assert
        assert cost == 0.0

    def test_calc_cost_invalid_model_raises_value_error(self, fake_models_df):
        """Unknown model name raises ValueError with a clear message."""
        # Arrange
        bogus_model = "invalid-model"
        do_call = lambda: calc_cost(bogus_model, 100, 100, models=fake_models_df)

        # Act
        # (Act-and-assert is fused inside the pytest.raises block below.)
        outcome = do_call

        # Assert
        with pytest.raises(
            ValueError,
            match="Model 'invalid-model' not found in pricing table",
        ):
            outcome()

    def test_calc_cost_large_token_counts_returns_expected_dollar_amount(
        self, fake_models_df
    ):
        """1M GPT-4 input + 500k output costs $60."""
        # Arrange
        model = "gpt-4"

        # Act
        cost = calc_cost(model, 1_000_000, 500_000, models=fake_models_df)

        # Assert
        assert cost == 60.0

    @pytest.mark.parametrize(
        "input_tokens,output_tokens",
        [
            (100, 50),
            (1_000, 500),
            (10_000, 5_000),
            (100_000, 50_000),
        ],
    )
    def test_calc_cost_various_token_counts_match_formula(
        self, fake_models_df, input_tokens, output_tokens
    ):
        """Cost matches (in*0.50 + out*1.50)/1M across a range of magnitudes."""
        # Arrange
        model = "gpt-3.5-turbo"
        expected = (input_tokens * 0.50 + output_tokens * 1.50) / 1_000_000

        # Act
        cost = calc_cost(model, input_tokens, output_tokens, models=fake_models_df)

        # Assert
        assert cost == expected

    def test_calc_cost_precision_within_floating_point_tolerance(self, fake_models_df):
        """Cost for awkward token counts is within 1e-10 of the analytic value."""
        # Arrange
        model = "gpt-4"
        input_tokens, output_tokens = 333, 777
        expected = (input_tokens * 30.00 + output_tokens * 60.00) / 1_000_000

        # Act
        cost = calc_cost(model, input_tokens, output_tokens, models=fake_models_df)

        # Assert
        assert abs(cost - expected) < 1e-10

    def test_calc_cost_with_real_models_dataframe_returns_float(self):
        """Integration: real MODELS table yields a float for a known model."""
        # Arrange
        from scitex_genai.llm import MODELS

        known_model = (
            "gpt-3.5-turbo"
            if "gpt-3.5-turbo" in MODELS["name"].values
            else MODELS["name"].iloc[0]
        )

        # Act
        cost = calc_cost(known_model, 1_000, 500)

        # Assert
        assert isinstance(cost, float)

    def test_calc_cost_with_real_models_dataframe_returns_nonnegative_value(self):
        """Integration: real MODELS table yields a non-negative cost for a known model."""
        # Arrange
        from scitex_genai.llm import MODELS

        known_model = (
            "gpt-3.5-turbo"
            if "gpt-3.5-turbo" in MODELS["name"].values
            else MODELS["name"].iloc[0]
        )

        # Act
        cost = calc_cost(known_model, 1_000, 500)

        # Assert
        assert cost >= 0

    def test_calc_cost_return_type_is_float(self, fake_models_df):
        """calc_cost always returns a Python float."""
        # Arrange
        model = "gpt-4"

        # Act
        cost = calc_cost(model, 100, 100, models=fake_models_df)

        # Assert
        assert isinstance(cost, float)


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])
