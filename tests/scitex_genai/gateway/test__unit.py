"""``install-unit`` writes exactly the unit systemd will run, and nothing else.

No mocks (PA-306): the write is a real write under ``tmp_path`` and the
``systemctl`` step is exercised through a recording runner -- a plain callable
handed in as the dependency the function already takes.
"""

from __future__ import annotations

import configparser
import shlex
import sys
from pathlib import Path

import pytest

from scitex_genai.gateway._unit import (
    DEFAULT_UNIT_DIR,
    MODULE,
    UNIT_NAME,
    gateway_command,
    install_unit,
    render_unit,
)

UPSTREAM = "http://127.0.0.1:18773, http://127.0.0.1:18774"


def _sections(text: str) -> configparser.ConfigParser:
    """systemd units are INI-shaped; parse them the way a reader would."""
    parsed = configparser.ConfigParser(interpolation=None, strict=False)
    parsed.optionxform = str  # type: ignore[method-assign]
    parsed.read_string(text)
    return parsed


def _exec_argv(text: str) -> list[str]:
    """The argv bash receives from ExecStart, then the argv it execs."""
    exec_start = _sections(text)["Service"]["ExecStart"]
    shell = shlex.split(exec_start)
    assert shell[:2] == ["/bin/bash", "-lc"]
    return shlex.split(shell[2])


def test_execstart_execs_this_interpreter_under_a_login_shell():
    # Arrange
    text = render_unit(host="0.0.0.0", port=18772, upstream=UPSTREAM)

    # Act
    argv = _exec_argv(text)

    # Assert
    assert argv[0] == "exec"
    assert argv[1:4] == [sys.executable, "-m", MODULE]
    assert argv[4:8] == ["--host", "0.0.0.0", "--port", "18772"]
    assert argv[8:] == [
        "--inference-upstream",
        "http://127.0.0.1:18773,http://127.0.0.1:18774",
    ]


def test_no_upstream_means_the_codex_backend_and_no_flag():
    # Arrange / Act
    argv = _exec_argv(render_unit(host="127.0.0.1", port=8765))

    # Assert
    assert "--inference-upstream" not in argv
    assert argv[-2:] == ["--port", "8765"]


def test_unit_is_supervised_and_wanted_at_login():
    # Arrange / Act
    parsed = _sections(render_unit(host="0.0.0.0", port=18772, upstream=UPSTREAM))

    # Assert
    assert parsed["Service"]["Restart"] == "always"
    assert parsed["Service"]["Type"] == "simple"
    assert parsed["Unit"]["Wants"] == "network-online.target"
    assert parsed["Install"]["WantedBy"] == "default.target"
    assert "0.0.0.0:18772" in parsed["Unit"]["Description"]


def test_interpreter_can_be_pinned_explicitly():
    # Arrange / Act
    argv = gateway_command(host="0.0.0.0", port=1, interpreter="/opt/venv/bin/python")

    # Assert
    assert argv[:3] == ["/opt/venv/bin/python", "-m", MODULE]


@pytest.mark.parametrize(
    "host, port",
    [("", 18772), ("0.0.0.0 evil", 18772), ("0.0.0.0", 0), ("0.0.0.0", 65536)],
)
def test_a_host_with_whitespace_or_a_port_out_of_range_is_refused(host, port):
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        render_unit(host=host, port=port)


def test_install_writes_the_rendered_text_and_skips_systemctl_when_asked(
    tmp_path: Path,
):
    # Arrange
    calls: list[list[str]] = []
    unit_dir = tmp_path / "systemd" / "user"

    # Act
    path = install_unit(
        host="0.0.0.0",
        port=18772,
        upstream=UPSTREAM,
        unit_dir=unit_dir,
        enable=False,
        runner=lambda argv: calls.append(list(argv)),
    )

    # Assert
    assert path == unit_dir / UNIT_NAME
    assert path.read_text() == render_unit(
        host="0.0.0.0", port=18772, upstream=UPSTREAM
    )
    assert calls == []


def test_install_reloads_the_user_manager_then_enables_now(tmp_path: Path):
    # Arrange
    calls: list[list[str]] = []

    # Act
    install_unit(
        host="0.0.0.0",
        port=18772,
        upstream=UPSTREAM,
        unit_dir=tmp_path,
        runner=lambda argv: calls.append(list(argv)),
    )

    # Assert
    assert calls == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", "--now", UNIT_NAME],
    ]


def test_a_second_install_overwrites_in_place_and_leaves_nothing_else(
    tmp_path: Path,
):
    # Arrange
    install_unit(host="0.0.0.0", port=18772, unit_dir=tmp_path, enable=False)

    # Act
    path = install_unit(host="0.0.0.0", port=18790, unit_dir=tmp_path, enable=False)

    # Assert
    assert [p.name for p in tmp_path.iterdir()] == [UNIT_NAME]
    assert "--port 18790" in path.read_text()
    assert "18772" not in path.read_text()


def test_default_unit_dir_is_the_user_manager_directory():
    # Arrange / Act / Assert
    assert DEFAULT_UNIT_DIR == Path.home() / ".config" / "systemd" / "user"
