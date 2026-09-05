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

from scitex_genai.gateway._inference import UPSTREAM_ENV
from scitex_genai.gateway._settings import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    default_config_path,
    load_settings,
)

ENV_KEYS = (
    UPSTREAM_ENV,
    "SCITEX_GATEWAY_HOST",
    "SCITEX_GATEWAY_PORT",
    "SCITEX_GATEWAY_INFERENCE_UPSTREAMS",
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
    ) == (
        DEFAULT_HOST,
        DEFAULT_PORT,
        "",
        None,
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


def test_the_default_path_is_under_the_scitex_dir():
    # Arrange
    scitex_dir = Path(get_scitex_dir())

    # Act
    path = default_config_path()

    # Assert
    assert path == scitex_dir / "genai" / "config.yaml"
