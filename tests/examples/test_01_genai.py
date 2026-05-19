"""Smoke test for examples/01_genai.ipynb via jupyter nbconvert --execute.

Per PS505 (SciTeX audit): notebook smoke tests must invoke
``jupyter nbconvert --execute`` or ``pytest --nbval[-lax]``. The
notebook itself skips live API calls when OPENAI_API_KEY is unset, so
this passes offline as well.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("nbformat")
pytest.importorskip("nbconvert")
pytest.importorskip("scitex_genai")

NOTEBOOK = Path(__file__).resolve().parents[2] / "examples" / "01_genai.ipynb"


def test_notebook_executes_notebook_is_file(tmp_path):
    # Arrange
    # Act
    # Assert
    # Arrange
    # Act
    # Assert
    assert NOTEBOOK.is_file(), f"missing notebook: {NOTEBOOK}"


def test_notebook_executes_proc_returncode_equals_n_0(tmp_path):
    # Arrange
    # Arrange
    target = tmp_path / NOTEBOOK.name
    shutil.copy(NOTEBOOK, target)
    # Act
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=180",
            str(target),
        ],
        capture_output=True,
        text=True,
        timeout=240,
    )
    # Act
    # Assert
    # Assert
    assert proc.returncode == 0, (
        f"nbconvert failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )


