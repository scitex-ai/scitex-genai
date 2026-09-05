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


def _shell_argv(text: str) -> list[str]:
    """The argv systemd hands to the shell."""
    return shlex.split(_sections(text)["Service"]["ExecStart"])


def _exec_argv(text: str) -> list[str]:
    """The argv the shell execs."""
    return shlex.split(_shell_argv(text)[2])


def _record(calls: list[list[str]]):
    return lambda argv: calls.append(list(argv))


def _raised(call) -> BaseException | None:
    """What ``call()`` raised, or None -- so a refusal is one plain assertion."""
    try:
        call()
    except Exception as exc:  # noqa: BLE001 -- the test names the type it expects
        return exc
    return None


def test_execstart_runs_under_a_login_shell():
    # Arrange
    text = render_unit()

    # Act
    argv = _shell_argv(text)

    # Assert
    assert argv[:2] == ["/bin/bash", "-lc"]


def test_with_no_flags_the_unit_execs_only_this_interpreter_and_the_module():
    # Arrange
    text = render_unit()

    # Act
    argv = _exec_argv(text)

    # Assert
    assert argv == ["exec", sys.executable, "-m", MODULE]


def test_given_settings_are_baked_into_execstart():
    # Arrange
    text = render_unit(host="0.0.0.0", port=18772, upstream=UPSTREAM)

    # Act
    argv = _exec_argv(text)

    # Assert
    assert argv[4:] == [
        "--host",
        "0.0.0.0",
        "--port",
        "18772",
        "--inference-upstream",
        "http://127.0.0.1:18773,http://127.0.0.1:18774",
    ]


def test_a_config_path_is_baked_into_execstart():
    # Arrange
    text = render_unit(config="/srv/genai/config.yaml")

    # Act
    argv = _exec_argv(text)

    # Assert
    assert argv[4:] == ["--config", "/srv/genai/config.yaml"]


def test_the_description_names_the_settings_file():
    # Arrange
    text = render_unit(config="/srv/genai/config.yaml")

    # Act
    description = _sections(text)["Unit"]["Description"]

    # Assert
    assert "/srv/genai/config.yaml" in description


def test_unit_restarts_always():
    # Arrange
    text = render_unit()

    # Act
    parsed = _sections(text)

    # Assert
    assert parsed["Service"]["Restart"] == "always"


def test_unit_is_wanted_at_login():
    # Arrange
    text = render_unit()

    # Act
    parsed = _sections(text)

    # Assert
    assert parsed["Install"]["WantedBy"] == "default.target"


def test_interpreter_can_be_pinned_explicitly():
    # Arrange
    interpreter = "/opt/venv/bin/python"

    # Act
    argv = gateway_command(interpreter=interpreter)

    # Assert
    assert argv[0] == interpreter


@pytest.mark.parametrize(
    "host, port",
    [("", 18772), ("0.0.0.0 evil", 18772), ("0.0.0.0", 0), ("0.0.0.0", 65536)],
)
def test_a_host_with_whitespace_or_a_port_out_of_range_is_refused(host, port):
    # Arrange
    given = {"host": host, "port": port}

    # Act
    raised = _raised(lambda: render_unit(**given))

    # Assert
    assert isinstance(raised, ValueError)


def test_install_returns_the_unit_path(tmp_path: Path):
    # Arrange
    unit_dir = tmp_path / "systemd" / "user"

    # Act
    path = install_unit(unit_dir=unit_dir, enable=False)

    # Assert
    assert path == unit_dir / UNIT_NAME


def test_install_writes_the_rendered_text(tmp_path: Path):
    # Arrange
    expected = render_unit(host="0.0.0.0", port=18772, upstream=UPSTREAM)

    # Act
    path = install_unit(
        host="0.0.0.0", port=18772, upstream=UPSTREAM, unit_dir=tmp_path, enable=False
    )

    # Assert
    assert path.read_text() == expected


def test_install_skips_systemctl_when_asked(tmp_path: Path):
    # Arrange
    calls: list[list[str]] = []

    # Act
    install_unit(unit_dir=tmp_path, enable=False, runner=_record(calls))

    # Assert
    assert calls == []


def test_install_reloads_the_user_manager_then_enables_now(tmp_path: Path):
    # Arrange
    calls: list[list[str]] = []

    # Act
    install_unit(unit_dir=tmp_path, runner=_record(calls))

    # Assert
    assert calls == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", "--now", UNIT_NAME],
    ]


def test_a_second_install_leaves_exactly_one_file(tmp_path: Path):
    # Arrange
    install_unit(port=18772, unit_dir=tmp_path, enable=False)

    # Act
    install_unit(port=18790, unit_dir=tmp_path, enable=False)

    # Assert
    assert [entry.name for entry in tmp_path.iterdir()] == [UNIT_NAME]


def test_a_second_install_carries_the_new_settings(tmp_path: Path):
    # Arrange
    install_unit(port=18772, unit_dir=tmp_path, enable=False)

    # Act
    path = install_unit(port=18790, unit_dir=tmp_path, enable=False)

    # Assert
    assert _exec_argv(path.read_text())[4:] == ["--port", "18790"]


def test_default_unit_dir_is_the_user_manager_directory():
    # Arrange
    home = Path.home()

    # Act
    expected = home / ".config" / "systemd" / "user"

    # Assert
    assert DEFAULT_UNIT_DIR == expected
