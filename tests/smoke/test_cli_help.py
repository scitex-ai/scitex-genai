"""Smoke: every CLI launches and answers ``--help`` (PS-211).

Subprocess-driven (``sys.executable -m ...``) so this proves the installed
entry points resolve — an in-process ``build_parser()`` call would not.
Hermetic: no network, no credentials, no writes outside tmp dirs.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.smoke

CLIS = [
    "scitex_genai.gateway._cli",
    "scitex_genai.serve._cli",
    "scitex_genai.benchmark._sglang_ab",
]


@pytest.mark.parametrize("module", CLIS)
def test_cli_module_answers_help(module: str) -> None:
    # Arrange
    argv = [sys.executable, "-m", module, "--help"]

    # Act
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=30)

    # Assert
    assert (completed.returncode, "usage:" in completed.stdout) == (0, True)
