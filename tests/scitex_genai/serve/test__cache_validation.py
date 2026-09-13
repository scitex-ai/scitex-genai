import pytest

from scitex_genai.serve._cache_validation import SGLangCacheTierValidator


@pytest.mark.parametrize(
    ("tier", "field", "expected"),
    [
        ("device", "usable_tokens", 694720),
        ("host", "configured_tokens", 604480),
        ("host", "usable_tokens", 0),
        ("storage", "usable_tokens", 0),
    ],
)
def test_offload_capacity_is_fail_closed_until_observed(
    tier: str, field: str, expected: int
) -> None:
    # Arrange
    validator = SGLangCacheTierValidator()
    validator.observe_log_line("max_total_num_tokens=694720")
    validator.observe_log_line("HiCache kv host pool (604480 tokens)")
    validator.observe_log_line("Creating storage backend 'file'")

    # Act
    observed = validator.snapshot()[tier][field]

    # Assert
    assert observed == expected


@pytest.mark.parametrize("tier", ["host", "storage"])
def test_hybrid_restore_failure_revokes_offload_tiers(tier: str) -> None:
    # Arrange
    validator = SGLangCacheTierValidator()
    validator.observe_log_line("HiCache kv host pool (604480 tokens)")
    validator.observe_cache_hit(host_tokens=95104, storage_tokens=8192)

    # Act
    validator.observe_log_line("Failed to fetch abc.mamba from HiCacheFile storage.")
    snapshot = validator.snapshot()

    # Assert
    assert (snapshot[tier]["status"], snapshot[tier]["usable_tokens"]) == ("failed", 0)
