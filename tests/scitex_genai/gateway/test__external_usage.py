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
