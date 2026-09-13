from __future__ import annotations

import subprocess
from pathlib import Path


FIXTURE = (
    Path(__file__).parents[3]
    / "examples"
    / "serve"
    / "canary"
    / "run-tp1-context-canary-in-step.sh"
)


def test_canary_step_fixture_is_valid_shell():
    # Arrange / Act
    proc = subprocess.run(
        ["bash", "-n", str(FIXTURE)], capture_output=True, text=True, check=False
    )

    # Assert
    assert (proc.returncode, proc.stderr) == (0, "")


def test_canary_step_fixture_has_one_store_and_fails_closed():
    # Arrange / Act
    text = FIXTURE.read_text()

    # Assert
    assert "ExitOnForwardFailure=yes" in text
    assert "SCITEX_STORE_DSN=" in text
    assert "scitex-primary:55432" in text
    assert "STORE_PORT=${SCITEX_GENAI_CANARY_STORE_PORT:-55432}" in text
    assert "sqlite" not in text.lower()
    assert "SCITEX_GENAI_CANARY_STORE_SSH:?" in text
