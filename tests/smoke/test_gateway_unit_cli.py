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


def test_install_unit_writes_the_unit_file(tmp_path: Path) -> None:
    # Arrange
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

    # Act
    completed = subprocess.run(
        argv, capture_output=True, text=True, timeout=30, env=dict(os.environ)
    )

    # Assert
    unit = unit_dir / UNIT_NAME
    assert (completed.returncode, unit.is_file()) == (0, True)
    text = unit.read_text()
    assert ("ExecStart=" in text, "--port" in text, "18772" in text) == (
        True,
        True,
        True,
    )


def test_install_unit_refuses_a_second_key_without_a_resolving_shell(
    tmp_path: Path,
) -> None:
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

    # Act
    fresh_home = tmp_path / "fresh-home"
    env["SCITEX_DIR"] = str(fresh_home)
    second = subprocess.run(
        install, capture_output=True, text=True, timeout=30, env=env
    )

    # Assert
    assert first.returncode == 0
    assert (second.returncode != 0, "401" in (second.stdout + second.stderr)) == (
        True,
        True,
    )


def test_sglang_ab_refuses_load_without_the_canary_ack() -> None:
    # Arrange — no --endpoint, no ack: argparse must fail before any request
    # could be built, so no network is possible by construction.
    argv = [sys.executable, "-m", "scitex_genai.benchmark._sglang_ab"]

    # Act
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=30)

    # Assert
    assert (completed.returncode != 0, "endpoint" in completed.stderr) == (True, True)
