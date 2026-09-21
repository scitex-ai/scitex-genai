"""Smoke: gateway ``install-unit --no-enable`` writes a real unit (PS-211).

The default systemd user dir is never touched — ``--unit-dir`` points at a
tmp dir. ``--no-enable`` skips ``daemon-reload`` / ``enable --now`` so no
systemd bus is needed. The first-install path mints the gateway key into
the fake ``$SCITEX_DIR`` from the smoke conftest, never the real home.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scitex_genai.gateway._unit import UNIT_NAME

pytestmark = pytest.mark.smoke


@pytest.fixture
def installed_unit(tmp_path: Path) -> tuple[subprocess.CompletedProcess, Path]:
    unit_dir = tmp_path / "systemd"
    argv = [
        sys.executable,
        "-m",
        "scitex_genai.gateway._cli",
        "install-unit",
        "--host",
        "127.0.0.1",
        "--port",
        "18772",
        "--unit-dir",
        str(unit_dir),
        "--no-enable",
    ]
    completed = subprocess.run(
        argv, capture_output=True, text=True, timeout=30, env=dict(os.environ)
    )
    return completed, unit_dir / UNIT_NAME


def test_install_unit_command_exits_successfully(
    installed_unit: tuple[subprocess.CompletedProcess, Path],
) -> None:
    # Arrange
    completed, _unit = installed_unit

    # Act
    code = completed.returncode

    # Assert
    assert code == 0


def test_install_unit_writes_the_unit_file(
    installed_unit: tuple[subprocess.CompletedProcess, Path],
) -> None:
    # Arrange
    _completed, unit = installed_unit

    # Act
    written = unit.is_file()

    # Assert
    assert written is True


def test_installed_unit_embeds_exec_start_with_port(
    installed_unit: tuple[subprocess.CompletedProcess, Path],
) -> None:
    # Arrange
    _completed, unit = installed_unit

    # Act
    text = unit.read_text()

    # Assert
    assert ("ExecStart=" in text, "--port" in text, "18772" in text) == (
        True,
        True,
        True,
    )


@pytest.fixture
def second_install_attempt(
    tmp_path: Path,
) -> tuple[subprocess.CompletedProcess, subprocess.CompletedProcess]:
    # Arrange — a first install mints the key into the fake home; pointing
    # the second install at a FRESH home with no resolving key exercises the
    # replace-guard: minting there would rotate every client's key (401s).
    unit_dir = tmp_path / "systemd"
    install = [
        sys.executable,
        "-m",
        "scitex_genai.gateway._cli",
        "install-unit",
        "--unit-dir",
        str(unit_dir),
        "--no-enable",
    ]
    env = dict(os.environ)
    # Blank (not deleted): scitex-config loads dotenv on every resolution,
    # so a deleted var is repopulated from the developer's real .env while
    # a blank one still reads as unset after strip().
    env["SCITEX_GENAI_GATEWAY_API_KEY"] = " "
    first = subprocess.run(
        install, capture_output=True, text=True, timeout=30, env=env
    )
    fresh_home = tmp_path / "fresh-home"
    env["SCITEX_DIR"] = str(fresh_home)
    second = subprocess.run(
        install, capture_output=True, text=True, timeout=30, env=env
    )
    return first, second


def test_install_unit_first_install_exits_successfully(
    second_install_attempt: tuple[
        subprocess.CompletedProcess, subprocess.CompletedProcess
    ],
) -> None:
    # Arrange
    first, _second = second_install_attempt

    # Act
    code = first.returncode

    # Assert
    assert code == 0


def test_install_unit_second_install_refuses_without_resolving_shell(
    second_install_attempt: tuple[
        subprocess.CompletedProcess, subprocess.CompletedProcess
    ],
) -> None:
    # Arrange
    _first, second = second_install_attempt

    # Act
    outcome = (second.returncode != 0, "401" in (second.stdout + second.stderr))

    # Assert
    assert outcome == (True, True)


def test_sglang_ab_refuses_load_without_the_canary_ack() -> None:
    # Arrange — no --endpoint, no ack: argparse must fail before any request
    # could be built, so no network is possible by construction.
    argv = [sys.executable, "-m", "scitex_genai.benchmark._sglang_ab"]

    # Act
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=30)

    # Assert
    assert (completed.returncode != 0, "endpoint" in completed.stderr) == (True, True)
