"""Unit proofs for conservative external-provider usage settlement."""

from __future__ import annotations

import pytest

from scitex_genai.gateway._external import ExternalProviderPolicy
from scitex_genai.gateway._external_usage import ExternalUsageLedger


@pytest.mark.asyncio
async def test_each_missing_usage_dimension_keeps_its_own_reservation() -> None:
    # Arrange
    policy = ExternalProviderPolicy(provider="deepseek", upstream_api_key="secret")
    ledger = ExternalUsageLedger(policy)
    await ledger.reserve("run", estimated_input=30, output=20)
    # Act
    await ledger.settle(
        "run",
        reserved_input=30,
        reserved_output=20,
        input_tokens=None,
        output_tokens=7,
        reported_model="deepseek-flash",
    )
    # Assert
    assert (
        ledger.total.input_tokens,
        ledger.total.output_tokens,
        ledger.total.responses_without_usage,
    ) == (30, 7, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reported_input", "reported_output", "expected_input", "expected_output"),
    [
        ("bogus", 7, 30, 7),
        (5, "bogus", 5, 20),
        (-1, 7, 30, 7),
        (5, -1, 5, 20),
        (True, 7, 30, 7),
        (5, False, 5, 20),
        (float("nan"), 7, 30, 7),
        (5, float("inf"), 5, 20),
    ],
)
async def test_invalid_usage_dimension_keeps_only_its_own_reservation(
    reported_input: object,
    reported_output: object,
    expected_input: int,
    expected_output: int,
) -> None:
    # Arrange
    policy = ExternalProviderPolicy(provider="deepseek", upstream_api_key="secret")
    ledger = ExternalUsageLedger(policy)
    await ledger.reserve("run", estimated_input=30, output=20)
    # Act
    await ledger.settle(
        "run",
        reserved_input=30,
        reserved_output=20,
        input_tokens=reported_input,
        output_tokens=reported_output,
        reported_model="deepseek-flash",
    )
    # Assert
    assert (
        ledger.total.input_tokens,
        ledger.total.output_tokens,
        ledger.total.responses_without_usage,
        ledger.total.reserved_input_tokens,
        ledger.total.reserved_output_tokens,
    ) == (expected_input, expected_output, 1, 0, 0)
