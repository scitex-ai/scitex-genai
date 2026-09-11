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
    UPSTREAM_ENV,
)
from scitex_genai.gateway._settings import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    SCITEX_TIMEOUT_ENV,
    default_config_path,
    load_settings,
)

ENV_KEYS = (
    UPSTREAM_ENV,
    SCITEX_TIMEOUT_ENV,
    TIMEOUT_ENV,
    "SCITEX_GATEWAY_HOST",
    "SCITEX_GATEWAY_PORT",
    "SCITEX_GATEWAY_INFERENCE_UPSTREAMS",
    "SCITEX_GATEWAY_INFERENCE_CAPACITY_PER_UPSTREAM",
    "SCITEX_GATEWAY_INFERENCE_MAX_QUEUE_SIZE",
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
    "    - http://127.0.0.1:18773\n"
    "    - http://127.0.0.1:18774\n"
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
        settings.inference_upstream,
        settings.source,
        settings.inference_timeout_s,
        settings.inference_capacity_per_upstream,
        settings.inference_max_queue_size,
    ) == (
        DEFAULT_HOST,
        DEFAULT_PORT,
        "",
        None,
        DEFAULT_TIMEOUT_S,
        DEFAULT_CAPACITY_PER_UPSTREAM,
        DEFAULT_MAX_QUEUE_SIZE,
    )


def test_the_file_supplies_host_port_and_upstreams(tmp_path: Path, clean_env):
    # Arrange
    path = _write(tmp_path / "config.yaml", FULL)

    # Act
    settings = load_settings(path)

    # Assert
    assert (settings.host, settings.port, settings.inference_upstream) == (
        "0.0.0.0",
        18772,
        "http://127.0.0.1:18773,http://127.0.0.1:18774",
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
    settings = load_settings(
        path, host="127.0.0.2", port=1, inference_upstream="http://z"
    )

    # Assert
    assert (settings.host, settings.port, settings.inference_upstream) == (
        "127.0.0.2",
        1,
        "http://z",
    )


def test_the_file_beats_the_environment(tmp_path: Path, clean_env):
    # Arrange
    path = _write(tmp_path / "config.yaml", FULL)
    os.environ[UPSTREAM_ENV] = "http://from-env"

    # Act
    settings = load_settings(path)

    # Assert
    assert (
        settings.inference_upstream == "http://127.0.0.1:18773,http://127.0.0.1:18774"
    )


def test_the_environment_fills_in_when_the_file_is_silent(tmp_path: Path, clean_env):
    # Arrange
    path = _write(tmp_path / "config.yaml", "gateway:\n  port: 18772\n")
    os.environ[UPSTREAM_ENV] = "http://from-env"

    # Act
    settings = load_settings(path)

    # Assert
    assert settings.inference_upstream == "http://from-env"


def test_a_comma_string_in_the_file_is_accepted_too(tmp_path: Path, clean_env):
    # Arrange
    path = _write(
        tmp_path / "config.yaml",
        'gateway:\n  inference_upstreams: "http://a, http://b"\n',
    )

    # Act
    settings = load_settings(path)

    # Assert
    assert settings.inference_upstream == "http://a,http://b"


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


@pytest.mark.parametrize(
    "field,value",
    [
        ("inference_capacity_per_upstream", 0),
        ("inference_capacity_per_upstream", -1),
        ("inference_max_queue_size", -1),
        ("inference_max_queue_size", 1.5),
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
    path = _write(
        tmp_path / "config.yaml",
        "gateway:\n"
        "  external_provider:\n"
        "    provider: deepseek\n"
        "    upstream: https://api.deepseek.com/anthropic/\n"
        "    upstream_auth_token_env: DEEPSEEK_API_KEY\n"
        "    canonical_model: deepseek-flash\n"
        "    model_aliases: [deepseek-v4-flash]\n"
        "    max_requests_per_run: 12\n"
        "    max_tokens_per_request: 4096\n",
    )
    settings = load_settings(path)
    assert settings.external_provider is not None
    assert settings.external_provider.upstream == "https://api.deepseek.com/anthropic"
    assert settings.external_provider.canonical_model == "deepseek-flash"
    assert settings.external_provider.model_aliases == ("deepseek-v4-flash",)
    assert settings.external_provider.max_requests_per_run == 12


def test_external_provider_and_local_inference_are_mutually_exclusive(
    tmp_path: Path, clean_env
):
    path = _write(
        tmp_path / "config.yaml",
        "gateway:\n"
        "  inference_upstreams: [http://local]\n"
        "  external_provider:\n"
        "    provider: deepseek\n"
        "    upstream: https://api.deepseek.com\n"
        "    upstream_auth_token_env: DEEPSEEK_API_KEY\n"
        "    canonical_model: deepseek-flash\n",
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
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
