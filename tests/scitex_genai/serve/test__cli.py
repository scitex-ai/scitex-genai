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


# ---------------------------------------------------------------------------
# `launch`: the fleet-side verb. --dry-run renders the hold body from real
# settings + conf files and books nothing; booking itself is exercised in
# test__launch.py through the injectable `book`.
# ---------------------------------------------------------------------------
LAUNCH_ARGS = ["--lease", "h100-pair", "--host", "spartan", "--gpus", "2", "--dry-run"]


def test_launch_dry_run_prints_one_step_per_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    # Arrange
    site = _site(tmp_path)
    (tmp_path / "models.d" / "model-b.conf").write_text(
        CONF.replace("8768", "8769").replace("4003", "4004").replace("18773", "18774")
    )
    argv = ["launch", "model-a", "model-b", *LAUNCH_ARGS, *site]

    # Act
    main(argv)

    # Assert
    assert [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("_step ")
    ] == [
        "_step model-a &",
        "_step model-b &",
    ]


def test_launch_dry_run_exits_zero_and_books_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    # Arrange
    argv = ["launch", "model-a", *LAUNCH_ARGS, *_site(tmp_path)]

    # Act
    rc = main(argv)

    # Assert
    assert (rc, "booked" in capsys.readouterr().out) == (0, False)


def test_launch_refuses_an_unknown_key_naming_the_available(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    # Arrange
    argv = ["launch", "model-z", *LAUNCH_ARGS, *_site(tmp_path)]

    # Act
    rc = main(argv)

    # Assert
    assert (rc, "available: model-a" in capsys.readouterr().err) == (2, True)


def test_launch_requires_a_lease_name(tmp_path: Path):
    # Arrange
    argv = [
        "launch",
        "model-a",
        "--host",
        "spartan",
        "--gpus",
        "2",
        "--dry-run",
        *_site(tmp_path),
    ]

    # Act
    raised = _raised(lambda: main(argv))

    # Assert
    assert isinstance(raised, SystemExit)


def _raised(call) -> BaseException | None:
    try:
        call()
    except BaseException as exc:  # noqa: BLE001 -- argparse exits with SystemExit
        return exc
    return None
