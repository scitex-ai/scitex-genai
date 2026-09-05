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


def test_no_subcommand_is_the_serve_form():
    # Arrange
    parser = build_parser()

    # Act
    args = parser.parse_args(["--host", "0.0.0.0", "--port", "18772"])

    # Assert
    assert args.command is None
    assert (args.host, args.port) == ("0.0.0.0", 18772)


def test_install_unit_takes_the_same_serve_flags_plus_its_own():
    # Arrange
    parser = build_parser()

    # Act
    args = parser.parse_args(
        [
            INSTALL_UNIT,
            "--host",
            "0.0.0.0",
            "--port",
            "18772",
            "--inference-upstream",
            "http://127.0.0.1:18773,http://127.0.0.1:18774",
            "--unit-dir",
            "/somewhere/units",
            "--no-enable",
        ]
    )

    # Assert
    assert args.command == INSTALL_UNIT
    assert (args.host, args.port) == ("0.0.0.0", 18772)
    assert args.inference_upstream == "http://127.0.0.1:18773,http://127.0.0.1:18774"
    assert args.unit_dir == Path("/somewhere/units")
    assert args.no_enable is True


def test_main_install_unit_writes_the_unit_and_returns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    # Arrange
    argv = [
        INSTALL_UNIT,
        "--host",
        "0.0.0.0",
        "--port",
        "18772",
        "--inference-upstream",
        "http://127.0.0.1:18773",
        "--unit-dir",
        str(tmp_path),
        "--no-enable",
    ]

    # Act
    main(argv)

    # Assert
    written = tmp_path / UNIT_NAME
    assert written.read_text() == render_unit(
        host="0.0.0.0", port=18772, upstream="http://127.0.0.1:18773"
    )
    out = capsys.readouterr().out
    assert str(written) in out
    assert "written only" in out
