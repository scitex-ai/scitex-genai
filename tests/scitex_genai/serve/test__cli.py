"""``scitex-genai-serve`` reads real files, prints the launch on --dry-run, starts nothing."""

from __future__ import annotations

from pathlib import Path

import pytest

from scitex_genai.serve._cli import main

SETTINGS = (
    "serve:\n"
    "  base: {base}\n"
    "  cache_root: {base}/cache\n"
    "  bastion: bastion.example.org\n"
    "  bastion_user: me\n"
    "  litellm_master_key: sk-local\n"
)
CONF = (
    "MODEL_PATH=/weights/model-a\n"
    "SERVED_NAME=model-a\n"
    "VLLM_PORT=8768\n"
    "LITELLM_PORT=4003\n"
    "TUNNEL_PORT=18773\n"
    "MAX_MODEL_LEN=1024\n"
)


def _site(tmp_path: Path) -> list[str]:
    config = tmp_path / "config.yaml"
    config.write_text(SETTINGS.format(base=tmp_path / "base"))
    models = tmp_path / "models.d"
    models.mkdir()
    (models / "model-a.conf").write_text(CONF)
    return ["--config", str(config), "--models-dir", str(models)]


def test_dry_run_prints_the_engine_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    # Arrange
    argv = ["model-a", "--dry-run", *_site(tmp_path)]

    # Act
    main(argv)

    # Assert
    assert "--served-model-name model-a" in capsys.readouterr().out


def test_dry_run_exits_zero(tmp_path: Path):
    # Arrange
    argv = ["model-a", "--dry-run", *_site(tmp_path)]

    # Act
    rc = main(argv)

    # Assert
    assert rc == 0


def test_list_prints_the_keys(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    # Arrange
    argv = ["--list", *_site(tmp_path)]

    # Act
    main(argv)

    # Assert
    assert capsys.readouterr().out.split() == ["model-a"]


def test_an_unknown_key_is_usage_error_naming_the_available(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    # Arrange
    argv = ["model-z", "--dry-run", *_site(tmp_path)]

    # Act
    rc = main(argv)

    # Assert
    assert (rc, "available: model-a" in capsys.readouterr().err) == (2, True)


def test_no_key_is_a_usage_error(tmp_path: Path):
    # Arrange
    argv = _site(tmp_path)

    # Act
    rc = main(argv)

    # Assert
    assert rc == 2
