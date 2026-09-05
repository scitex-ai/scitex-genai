"""The gateway CLI keeps its serve form and gains ``install-unit``.

The serve path is not started here (it binds a port and runs forever); what is
proved is the parse, and that ``install-unit`` writes a real file and returns
without touching a server.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scitex_genai.gateway._cli import INSTALL_UNIT, build_parser, main
from scitex_genai.gateway._unit import UNIT_NAME, render_unit

UPSTREAM = "http://127.0.0.1:18773,http://127.0.0.1:18774"
INSTALL_ARGS = [
    INSTALL_UNIT,
    "--host",
    "0.0.0.0",
    "--port",
    "18772",
    "--inference-upstream",
    UPSTREAM,
]


def test_no_subcommand_is_the_serve_form():
    # Arrange
    parser = build_parser()

    # Act
    args = parser.parse_args(["--host", "0.0.0.0", "--port", "18772"])

    # Assert
    assert args.command is None


def test_serve_flags_default_to_unset_so_the_settings_file_decides():
    # Arrange
    parser = build_parser()

    # Act
    args = parser.parse_args([])

    # Assert
    assert (args.config, args.host, args.port, args.inference_upstream) == (
        None,
        None,
        None,
        None,
    )


def test_install_unit_is_recognised():
    # Arrange
    parser = build_parser()

    # Act
    args = parser.parse_args([INSTALL_UNIT])

    # Assert
    assert args.command == INSTALL_UNIT


def test_install_unit_takes_the_settings_flags():
    # Arrange
    parser = build_parser()

    # Act
    args = parser.parse_args(INSTALL_ARGS)

    # Assert
    assert (args.host, args.port, args.inference_upstream) == (
        "0.0.0.0",
        18772,
        UPSTREAM,
    )


def test_install_unit_takes_a_unit_dir():
    # Arrange
    parser = build_parser()

    # Act
    args = parser.parse_args([INSTALL_UNIT, "--unit-dir", "/somewhere/units"])

    # Assert
    assert args.unit_dir == Path("/somewhere/units")


def test_install_unit_takes_no_enable():
    # Arrange
    parser = build_parser()

    # Act
    args = parser.parse_args([INSTALL_UNIT, "--no-enable"])

    # Assert
    assert args.no_enable is True


def test_main_install_unit_writes_the_unit_without_starting_a_server(tmp_path: Path):
    # Arrange
    argv = [*INSTALL_ARGS, "--unit-dir", str(tmp_path), "--no-enable"]

    # Act
    main(argv)

    # Assert
    assert (tmp_path / UNIT_NAME).read_text() == render_unit(
        host="0.0.0.0", port=18772, upstream=UPSTREAM
    )


def test_main_install_unit_reports_the_path_and_the_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    # Arrange
    argv = [INSTALL_UNIT, "--unit-dir", str(tmp_path), "--no-enable"]

    # Act
    main(argv)

    # Assert
    assert (str(tmp_path / UNIT_NAME) in capsys.readouterr().out, "written only") == (
        True,
        "written only",
    )
