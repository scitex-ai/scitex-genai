"""Smoke test for examples/example_genai.py — runs the script and checks exit 0."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("scitex_genai")

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "example_genai.py"


def test_example_genai_runs_example_is_file():
    # Arrange
    # Act
    # Assert
    # Arrange
    # Act
    # Assert
    assert EXAMPLE.is_file(), f"missing example: {EXAMPLE}"


def test_example_genai_runs_r_returncode_equals_n_0():
    # Arrange
    # Arrange
    # Act
    r = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    # Act
    # Assert
    # Assert
    assert r.returncode == 0, r.stderr


