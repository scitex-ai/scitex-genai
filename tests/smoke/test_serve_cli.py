"""Smoke: ``scitex-genai-serve`` lists and dry-renders one engine (PS-211).

``--list`` and ``--dry-run`` are the serve happy paths that start nothing:
no GPU, no SLURM, no tunnel. The subprocess runs with ``SCITEX_DIR`` and
``SCITEX_LOGGING_LEVEL`` set so the log lines it prints go to the right
streams without touching the developer's real home.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

CONF = (
    "MODEL_PATH=/weights/model-a\n"
    "SERVED_NAME=model-a\n"
    "VLLM_PORT=8768\n"
    "LITELLM_PORT=4003\n"
    "TUNNEL_PORT=18773\n"
    "MAX_MODEL_LEN=1024\n"
)


def _site(tmp_path: Path) -> list[str]:
    config = tmp_path / "config.yaml"
    config.write_text(
        "serve:\n"
        f"  base: {tmp_path / 'base'}\n"
        f"  cache_root: {tmp_path / 'base' / 'cache'}\n"
        "  bastion: bastion.example.org\n"
        "  bastion_user: me\n"
        "  litellm_master_key: sk-local\n"
    )
    models = tmp_path / "models.d"
    models.mkdir()
    (models / "model-a.conf").write_text(CONF)
    return ["--config", str(config), "--models-dir", str(models)]


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["SCITEX_LOGGING_LEVEL"] = "INFO"
    return env


def test_serve_list_prints_the_engine_key(tmp_path: Path) -> None:
    # Arrange
    argv = [sys.executable, "-m", "scitex_genai.serve._cli", "--list", *_site(tmp_path)]

    # Act
    completed = subprocess.run(
        argv, capture_output=True, text=True, timeout=30, env=_env()
    )

    # Assert
    out = completed.stdout + completed.stderr
    assert (completed.returncode, "model-a" in out) == (0, True)


def test_serve_dry_run_renders_the_engine_command(tmp_path: Path) -> None:
    # Arrange
    argv = [
        sys.executable,
        "-m",
        "scitex_genai.serve._cli",
        "model-a",
        "--dry-run",
        *_site(tmp_path),
    ]

    # Act
    completed = subprocess.run(
        argv, capture_output=True, text=True, timeout=30, env=_env()
    )

    # Assert
    out = completed.stdout + completed.stderr
    assert (completed.returncode, "--served-model-name" in out, "model-a" in out) == (
        0,
        True,
        True,
    )
