"""Smoke-test isolation: every CLI subprocess gets a fake home.

``install-unit`` legitimately mints a gateway key into
``$SCITEX_DIR/genai/secrets`` and ``scitex-logging`` writes a runtime log
under ``$SCITEX_DIR`` — both must land in a tmp dir, never the developer's
real ``~/.scitex``. The key variable is set BLANK (not deleted):
``scitex-config`` loads dotenv on every path resolution, so a deleted var
gets repopulated from the developer's real ``.env`` while a blank one
still reads as unset after ``strip()``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scitex_genai.gateway._secrets import GATEWAY_KEY_ENV


@pytest.fixture(autouse=True)
def _isolated_scitex_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SCITEX_DIR", str(tmp_path / "scitex-home"))
    monkeypatch.setenv(GATEWAY_KEY_ENV, " ")
