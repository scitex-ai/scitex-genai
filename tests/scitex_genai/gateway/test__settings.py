"""Settings resolve direct -> config file -> environment -> default, from a real file.

No mocks (PA-306): every case writes a real YAML file under ``tmp_path`` and
the environment is edited and restored by a fixture, not patched.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from scitex_config import get_scitex_dir

from scitex_genai.gateway._inference import (
    DEFAULT_CAPACITY_PER_UPSTREAM,
    DEFAULT_MAX_QUEUE_SIZE,
    DEFAULT_TIMEOUT_S,
    TIMEOUT_ENV,
)
from scitex_genai.gateway._settings import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    SCITEX_TIMEOUT_ENV,
    default_admission_history_path,
    default_config_path,
    load_settings,
)

ENV_KEYS = (
    "HOIST_UPSTREAM",
    SCITEX_TIMEOUT_ENV,
    TIMEOUT_ENV,
    "SCITEX_GATEWAY_HOST",
    "SCITEX_GATEWAY_PORT",
    "SCITEX_GATEWAY_INFERENCE_UPSTREAMS",
    "SCITEX_GATEWAY_INFERENCE_CAPACITY_PER_UPSTREAM",
    "SCITEX_GATEWAY_INFERENCE_MAX_QUEUE_SIZE",
    "SCITEX_GATEWAY_INFERENCE_TOKEN_CAPACITY_PER_UPSTREAM",
    "SCITEX_GATEWAY_INFERENCE_CONTINUATION_QOS_ENABLED",
    "SCITEX_GATEWAY_INFERENCE_CONTINUATION_QOS_MAX_RETRIES",
    "SCITEX_GATEWAY_INFERENCE_CONTINUATION_QOS_MIN_PREEMPT_TOKENS",
    "SCITEX_GATEWAY_INFERENCE_CACHE_REPORT_ENABLED",
)


@pytest.fixture
def clean_env() -> Iterator[dict[str, str]]:
    """Start every case with none of the gateway's environment names set."""
    saved = {key: os.environ.pop(key) for key in ENV_KEYS if key in os.environ}
    try:
        yield saved
    finally:
        for key in ENV_KEYS:
            os.environ.pop(key, None)
        os.environ.update(saved)


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def _raised(call) -> BaseException | None:
    """What ``call()`` raised, or None -- so a refusal is one plain assertion."""
    try:
        call()
    except Exception as exc:  # noqa: BLE001 -- the test names the type it expects
        return exc
    return None


FULL = (
    "gateway:\n"
    "  host: 0.0.0.0\n"
    "  port: 18772\n"
    "  inference_upstreams:\n"
    "    - label: qwen-tp2\n"
    "      url: http://127.0.0.1:18773\n"
    "      token_capacity: 1600000\n"
    "    - label: qwen-tp1\n"
    "      url: http://127.0.0.1:18774\n"
    "      token_capacity: 563215\n"
)


def test_a_missing_file_gives_the_package_defaults(tmp_path: Path, clean_env):
    # Arrange
    absent = tmp_path / "none.yaml"

    # Act
    settings = load_settings(absent)

    # Assert
    assert (
        settings.host,
        settings.port,
        settings.source,
        settings.inference_timeout_s,
        settings.inference_capacity_per_upstream,
        settings.inference_max_queue_size,
        settings.inference_continuation_qos_enabled,
        settings.inference_continuation_qos_max_retries,
        settings.inference_continuation_qos_min_preempt_tokens,
        settings.inference_cache_report_enabled,
    ) == (
        DEFAULT_HOST,
        DEFAULT_PORT,
        None,
        DEFAULT_TIMEOUT_S,
        DEFAULT_CAPACITY_PER_UPSTREAM,
        DEFAULT_MAX_QUEUE_SIZE,
        False,
        1,
        0,
        False,
    )


def test_cache_report_is_an_explicit_configured_sglang_capability(
    tmp_path: Path, clean_env
) -> None:
    # Arrange
    path = _write(
        tmp_path / "config.yaml", FULL + "  inference_cache_report_enabled: true\n"
    )

    # Act
    configured = load_settings(path)
    overridden = load_settings(path, inference_cache_report_enabled=False)

    # Assert
    assert (
        configured.inference_cache_report_enabled,
        overridden.inference_cache_report_enabled,
    ) == (True, False)


def test_the_file_supplies_host_port_and_upstreams(tmp_path: Path, clean_env):
    # Arrange
    path = _write(tmp_path / "config.yaml", FULL)

    # Act
    settings = load_settings(path)

    # Assert
    assert (
        settings.host,
        settings.port,
        [
            (item.label, item.url, item.token_capacity)
            for item in settings.inference_upstreams
        ],
    ) == (
        "0.0.0.0",
        18772,
        [
            ("qwen-tp2", "http://127.0.0.1:18773", 1_600_000),
            ("qwen-tp1", "http://127.0.0.1:18774", 563_215),
        ],
    )


def test_source_names_the_file_that_was_read(tmp_path: Path, clean_env):
    # Arrange
    path = _write(tmp_path / "config.yaml", FULL)

    # Act
    settings = load_settings(path)

    # Assert
    assert settings.source == path


def test_direct_values_beat_the_file(tmp_path: Path, clean_env):
    # Arrange
    path = _write(tmp_path / "config.yaml", FULL)

    # Act
    settings = load_settings(path, host="127.0.0.2", port=1)

    # Assert
    assert (settings.host, settings.port) == (
        "127.0.0.2",
        1,
    )


@pytest.mark.parametrize(
    "environment", ["HOIST_UPSTREAM", "SCITEX_GATEWAY_INFERENCE_UPSTREAMS"]
)
def test_retired_upstream_environment_refuses_with_migration_guidance(
    tmp_path: Path, clean_env, environment: str
):
    # Arrange
    path = _write(tmp_path / "config.yaml", "gateway:\n  port: 18772\n")
    os.environ[environment] = "http://from-env"

    # Act
    error = _raised(lambda: load_settings(path))

    # Assert
    assert isinstance(error, ValueError) and "label, url, and token_capacity" in str(
        error
    )


@pytest.mark.parametrize("configured", ["null", "''"])
def test_present_non_list_upstream_value_is_refused(
    tmp_path: Path, clean_env, configured: str
):
    # Arrange
    path = _write(
        tmp_path / "config.yaml",
        f"gateway:\n  inference_upstreams: {configured}\n",
    )

    # Act
    error = _raised(lambda: load_settings(path))

    # Assert
    assert isinstance(error, ValueError) and "must be a list of mappings" in str(error)


def test_a_comma_string_in_the_file_is_refused_with_migration_guidance(
    tmp_path: Path, clean_env
):
    # Arrange
    path = _write(
        tmp_path / "config.yaml",
        'gateway:\n  inference_upstreams: "http://a, http://b"\n',
    )

    # Act
    error = _raised(lambda: load_settings(path))

    # Assert
    assert (type(error), "list of mappings" in str(error)) == (ValueError, True)


@pytest.mark.parametrize(
    "rows, message",
    [
        ("    - label: tp1\n      url: http://tp1\n", "missing: token_capacity"),
        (
            "    - label: tp1\n      url: http://\n      token_capacity: 10\n",
            "must be an http(s) base URL",
        ),
        (
            "    - label: tp1\n      url: http://tp1\n      token_capacity: true\n",
            "token_capacity must be an integer",
        ),
        (
            "    - label: tp1\n"
            "      url: http://secret@tp1/path?token=x#fragment\n"
            "      token_capacity: 10\n",
            "without credentials, query, or fragment",
        ),
        (
            "    - label: duplicate\n"
            "      url: http://tp1\n"
            "      token_capacity: 10\n"
            "    - label: duplicate\n"
            "      url: http://tp2\n"
            "      token_capacity: 20\n",
            "labels must be unique",
        ),
        (
            "    - label: tp1\n"
            "      url: http://same\n"
            "      token_capacity: 10\n"
            "    - label: tp2\n"
            "      url: http://same\n"
            "      token_capacity: 20\n",
            "urls must be unique",
        ),
    ],
)
def test_structured_upstream_schema_rejects_ambiguous_members(
    tmp_path: Path, clean_env, rows: str, message: str
) -> None:
    # Arrange
    path = _write(tmp_path / "config.yaml", "gateway:\n  inference_upstreams:\n" + rows)

    # Act
    error = _raised(lambda: load_settings(path))

    # Assert
    assert (type(error), message in str(error)) == (ValueError, True)


@pytest.mark.parametrize(
    "text", ["gateway:\n  port: 0\n", "gateway:\n  host: '0.0.0.0 x'\n"]
)
def test_a_bad_host_or_port_in_the_file_is_refused(
    tmp_path: Path, clean_env, text: str
):
    # Arrange
    path = _write(tmp_path / "config.yaml", text)

    # Act
    raised = _raised(lambda: load_settings(path))

    # Assert
    assert isinstance(raised, ValueError)


def test_the_file_supplies_an_explicit_inference_timeout(tmp_path: Path, clean_env):
    # Arrange
    path = _write(
        tmp_path / "config.yaml",
        "gateway:\n  inference_timeout_s: 1800\n",
    )

    # Act
    settings = load_settings(path)

    # Assert
    assert settings.inference_timeout_s == 1800.0


def test_file_and_direct_values_resolve_admission_bounds(tmp_path: Path, clean_env):
    # Arrange
    path = _write(
        tmp_path / "config.yaml",
        "gateway:\n"
        "  inference_capacity_per_upstream: 3\n"
        "  inference_max_queue_size: 9\n",
    )

    # Act
    from_file = load_settings(path)
    direct = load_settings(
        path, inference_capacity_per_upstream=4, inference_max_queue_size=10
    )

    # Assert
    assert (
        from_file.inference_capacity_per_upstream,
        from_file.inference_max_queue_size,
        direct.inference_capacity_per_upstream,
        direct.inference_max_queue_size,
    ) == (3, 9, 4, 10)


def test_retired_global_token_capacity_has_migration_guidance(
    tmp_path: Path, clean_env
):
    # Arrange
    path = _write(
        tmp_path / "config.yaml",
        "gateway:\n  inference_token_capacity_per_upstream: 1600000\n",
    )

    # Act
    refused = _raised(lambda: load_settings(path))

    # Assert
    assert isinstance(refused, ValueError) and "label, url, and token_capacity" in str(
        refused
    )


def test_continuation_qos_is_opt_in_and_resolves_bounds(tmp_path: Path, clean_env):
    # Arrange
    path = _write(
        tmp_path / "config.yaml",
        "gateway:\n"
        "  inference_continuation_qos_enabled: true\n"
        "  inference_continuation_qos_max_retries: 2\n"
        "  inference_continuation_qos_min_preempt_tokens: 400000\n",
    )

    # Act
    settings = load_settings(path)

    # Assert
    assert (
        settings.inference_continuation_qos_enabled,
        settings.inference_continuation_qos_max_retries,
        settings.inference_continuation_qos_min_preempt_tokens,
    ) == (True, 2, 400_000)


@pytest.mark.parametrize(
    "field,value",
    [
        ("inference_capacity_per_upstream", 0),
        ("inference_capacity_per_upstream", -1),
        ("inference_max_queue_size", -1),
        ("inference_max_queue_size", 1.5),
        ("inference_continuation_qos_max_retries", -1),
        ("inference_continuation_qos_min_preempt_tokens", -1),
    ],
)
def test_invalid_admission_bounds_are_refused(tmp_path: Path, clean_env, field, value):
    # Arrange
    path = _write(tmp_path / "config.yaml", f"gateway:\n  {field}: {value}\n")

    # Act
    # Assert
    with pytest.raises(ValueError):
        load_settings(path)


def test_a_direct_timeout_beats_the_file(tmp_path: Path, clean_env):
    # Arrange
    path = _write(
        tmp_path / "config.yaml",
        "gateway:\n  inference_timeout_s: 1800\n",
    )

    # Act
    settings = load_settings(path, inference_timeout_s=2400)

    # Assert
    assert settings.inference_timeout_s == 2400.0


def test_the_timeout_file_value_beats_the_legacy_environment(tmp_path: Path, clean_env):
    # Arrange
    path = _write(
        tmp_path / "config.yaml",
        "gateway:\n  inference_timeout_s: 1800\n",
    )
    os.environ[TIMEOUT_ENV] = "900"
    os.environ[SCITEX_TIMEOUT_ENV] = "600"

    # Act
    settings = load_settings(path)

    # Assert
    assert settings.inference_timeout_s == 1800.0


def test_legacy_timeout_environment_remains_a_fallback(tmp_path: Path, clean_env):
    # Arrange
    os.environ[TIMEOUT_ENV] = "1200"

    # Act
    settings = load_settings(tmp_path / "none.yaml")

    # Assert
    assert settings.inference_timeout_s == 1200.0


def test_external_provider_policy_is_loaded_from_config(tmp_path: Path, clean_env):
    # Arrange
    path = _write(
        tmp_path / "config.yaml",
        "gateway:\n"
        "  external_provider:\n"
        "    provider: deepseek\n"
        "    upstream: https://api.deepseek.com/\n"
        "    upstream_auth_token_env: DEEPSEEK_API_KEY\n"
        "    canonical_model: deepseek-flash\n"
        "    model_aliases: [deepseek-v4-flash]\n"
        "    anthropic_path_prefix: /anthropic\n"
        "    max_requests_per_run: 12\n"
        "    max_tokens_per_request: 4096\n",
    )
    # Act
    settings = load_settings(path)
    external = settings.external_provider
    # Assert
    assert external is not None and (
        external.upstream,
        external.canonical_model,
        external.model_aliases,
        external.anthropic_path_prefix,
        external.max_requests_per_run,
    ) == (
        "https://api.deepseek.com",
        "deepseek-flash",
        ("deepseek-v4-flash",),
        "/anthropic",
        12,
    )


def test_external_provider_and_local_inference_are_mutually_exclusive(
    tmp_path: Path, clean_env
):
    # Arrange
    path = _write(
        tmp_path / "config.yaml",
        "gateway:\n"
        "  inference_upstreams:\n"
        "    - label: local\n"
        "      url: http://local\n"
        "      token_capacity: 1000\n"
        "  external_provider:\n"
        "    provider: deepseek\n"
        "    upstream: https://api.deepseek.com\n"
        "    upstream_auth_token_env: DEEPSEEK_API_KEY\n"
        "    canonical_model: deepseek-flash\n",
    )
    # Act
    # Assert
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_settings(path)


def test_external_provider_typo_is_refused(tmp_path: Path, clean_env):
    # Arrange
    path = _write(
        tmp_path / "config.yaml",
        "gateway:\n"
        "  external_provider:\n"
        "    provider: deepseek\n"
        "    upstream: https://api.deepseek.com\n"
        "    upstream_auth_token_env: DEEPSEEK_API_KEY\n"
        "    canonical_model: deepseek-flash\n"
        "    max_request_per_run: 12\n",
    )
    # Act
    # Assert
    with pytest.raises(ValueError, match="unknown keys: max_request_per_run"):
        load_settings(path)


def test_namespaced_timeout_environment_remains_a_fallback(tmp_path: Path, clean_env):
    # Arrange
    os.environ[SCITEX_TIMEOUT_ENV] = "900"

    # Act
    settings = load_settings(tmp_path / "none.yaml")

    # Assert
    assert settings.inference_timeout_s == 900.0


def test_legacy_timeout_environment_beats_the_implicit_namespaced_one(
    tmp_path: Path, clean_env
):
    # Arrange
    os.environ[TIMEOUT_ENV] = "1200"
    os.environ[SCITEX_TIMEOUT_ENV] = "900"

    # Act
    settings = load_settings(tmp_path / "none.yaml")

    # Assert
    assert settings.inference_timeout_s == 1200.0


def test_direct_timeout_beats_both_environment_names(tmp_path: Path, clean_env):
    # Arrange
    os.environ[TIMEOUT_ENV] = "1200"
    os.environ[SCITEX_TIMEOUT_ENV] = "900"

    # Act
    settings = load_settings(
        tmp_path / "none.yaml",
        inference_timeout_s=1800,
    )

    # Assert
    assert settings.inference_timeout_s == 1800.0


@pytest.mark.parametrize("value", [0, -1, "nan", "inf", "not-a-number"])
def test_a_non_positive_or_non_finite_timeout_is_refused(
    tmp_path: Path, clean_env, value
):
    # Arrange
    path = _write(
        tmp_path / "config.yaml",
        f"gateway:\n  inference_timeout_s: {value}\n",
    )

    # Act
    raised = _raised(lambda: load_settings(path))

    # Assert
    assert isinstance(raised, ValueError)


def test_the_default_path_is_under_the_scitex_dir():
    # Arrange
    scitex_dir = Path(get_scitex_dir())

    # Act
    path = default_config_path()

    # Assert
    assert path == scitex_dir / "genai" / "config.yaml"


def test_admission_history_handoff_sits_in_the_genai_runtime_directory():
    # Arrange
    scitex_dir = Path(get_scitex_dir())

    # Act
    path = default_admission_history_path()

    # Assert
    assert path == scitex_dir / "genai" / "runtime" / "admission-history.json"


def test_cold_prefill_guard_requires_an_explicit_limit_and_threshold(tmp_path: Path):
    # Arrange
    path = _write(
        tmp_path / "config.yaml",
        "gateway:\n"
        "  inference_cold_prefill_limit_per_upstream: 1\n"
        "  inference_cold_prefill_min_tokens: 256000\n",
    )

    # Act
    settings = load_settings(path)

    # Assert
    assert (
        settings.inference_cold_prefill_limit_per_upstream,
        settings.inference_cold_prefill_min_tokens,
    ) == (1, 256000)


def test_cold_prefill_guard_rejects_an_incomplete_pair(tmp_path: Path):
    # Arrange
    incomplete = _write(
        tmp_path / "incomplete.yaml",
        "gateway:\n  inference_cold_prefill_limit_per_upstream: 1\n",
    )

    # Act
    raised = _raised(lambda: load_settings(incomplete))

    # Assert
    assert (
        str(raised)
        == "cold prefill limit and minimum tokens must be configured together"
    )
