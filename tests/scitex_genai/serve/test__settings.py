"""The ``serve:`` section is read from a real file and refuses what it cannot use."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from scitex_genai.serve._settings import load_serve_settings

KEY_ENV = "SCITEX_SERVE_LITELLM_MASTER_KEY"
FULL = (
    "serve:\n"
    "  base: /scratch/site/me\n"
    "  cache_root: /local\n"
    "  cuda_home: /apps/cuda\n"
    "  bastion: bastion.example.org\n"
    "  bastion_user: me\n"
    "  proxy_command: /home/me/bin/cloudflared access ssh --hostname bastion.example.org\n"
    "  litellm_master_key: sk-local\n"
)


@pytest.fixture
def clean_env() -> Iterator[None]:
    saved = os.environ.pop(KEY_ENV, None)
    try:
        yield
    finally:
        os.environ.pop(KEY_ENV, None)
        if saved is not None:
            os.environ[KEY_ENV] = saved


def _raised(call) -> BaseException | None:
    try:
        call()
    except Exception as exc:  # noqa: BLE001 -- the test names the type it expects
        return exc
    return None


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def test_explicit_values_are_read(tmp_path: Path, clean_env):
    # Arrange
    path = _write(tmp_path, FULL)

    # Act
    settings = load_serve_settings(path)

    # Assert
    assert (
        settings.base,
        settings.cache_root,
        settings.bastion,
        settings.bastion_user,
    ) == (
        Path("/scratch/site/me"),
        Path("/local"),
        "bastion.example.org",
        "me",
    )


def test_logs_and_binaries_default_under_base(tmp_path: Path, clean_env):
    # Arrange
    path = _write(tmp_path, FULL)

    # Act
    settings = load_serve_settings(path)

    # Assert
    assert (settings.logs, settings.vllm_bin, settings.litellm_bin) == (
        Path("/scratch/site/me/serve-logs"),
        Path("/scratch/site/me/vllm-venv/bin/vllm"),
        Path("/scratch/site/me/vllm-venv/bin/litellm"),
    )


def test_the_master_key_can_come_from_the_environment(tmp_path: Path, clean_env):
    # Arrange
    path = _write(tmp_path, FULL.replace("  litellm_master_key: sk-local\n", ""))
    os.environ[KEY_ENV] = "sk-from-env"

    # Act
    settings = load_serve_settings(path)

    # Assert
    assert settings.litellm_master_key == "sk-from-env"


def test_source_names_the_file(tmp_path: Path, clean_env):
    # Arrange
    path = _write(tmp_path, FULL)

    # Act
    settings = load_serve_settings(path)

    # Assert
    assert settings.source == path


def test_a_missing_file_is_refused_by_path(tmp_path: Path, clean_env):
    # Arrange
    absent = tmp_path / "none.yaml"

    # Act
    raised = _raised(lambda: load_serve_settings(absent))

    # Assert
    assert str(absent) in str(raised)


def test_a_missing_base_is_refused(tmp_path: Path, clean_env):
    # Arrange
    path = _write(tmp_path, "serve:\n  bastion: b\n  litellm_master_key: k\n")

    # Act
    raised = _raised(lambda: load_serve_settings(path))

    # Assert
    assert "serve.base" in str(raised)


def test_a_missing_master_key_is_refused(tmp_path: Path, clean_env):
    # Arrange
    path = _write(tmp_path, FULL.replace("  litellm_master_key: sk-local\n", ""))

    # Act
    raised = _raised(lambda: load_serve_settings(path))

    # Assert
    assert "litellm_master_key" in str(raised)


def test_a_relative_base_is_refused(tmp_path: Path, clean_env):
    # Arrange
    path = _write(tmp_path, FULL.replace("/scratch/site/me", "relative/site"))

    # Act
    raised = _raised(lambda: load_serve_settings(path))

    # Assert
    assert isinstance(raised, ValueError)
