from scitex_genai.serve._cache_validation import SGLangCacheTierValidator


def test_offload_capacity_is_fail_closed_until_observed() -> None:
    validator = SGLangCacheTierValidator()
    validator.observe_log_line("max_total_num_tokens=694720")
    validator.observe_log_line("HiCache kv host pool (604480 tokens)")
    validator.observe_log_line("Creating storage backend 'file'")

    snapshot = validator.snapshot()
    assert snapshot["device"]["usable_tokens"] == 694720
    assert snapshot["host"]["configured_tokens"] == 604480
    assert snapshot["host"]["usable_tokens"] == 0
    assert snapshot["storage"]["usable_tokens"] == 0


def test_hybrid_restore_failure_revokes_offload_tiers() -> None:
    validator = SGLangCacheTierValidator()
    validator.observe_log_line("HiCache kv host pool (604480 tokens)")
    validator.observe_cache_hit(host_tokens=95104, storage_tokens=8192)
    assert validator.snapshot()["host"]["status"] == "validated"

    validator.observe_log_line("Failed to fetch abc.mamba from HiCacheFile storage.")
    snapshot = validator.snapshot()
    assert snapshot["host"]["status"] == "failed"
    assert snapshot["storage"]["status"] == "failed"
    assert snapshot["host"]["usable_tokens"] == 0
