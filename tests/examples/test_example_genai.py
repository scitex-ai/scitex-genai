"""Smoke test for examples/example_genai.py — runs the script and checks exit 0."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("scitex_genai")

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "example_genai.py"


@pytest.fixture
def example_script_path():
    """Provide the example script path, skipping if absent."""
    if not EXAMPLE.is_file():
        pytest.skip(f"missing example: {EXAMPLE}")
    return EXAMPLE


def test_example_genai_script_exits_with_zero_status(example_script_path):
    """Running examples/example_genai.py exits 0 (script skips API calls when key is unset)."""
    # Arrange
    cmd = [sys.executable, str(example_script_path)]

    # Act
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )

    # Assert
    assert proc.returncode == 0, proc.stderr
