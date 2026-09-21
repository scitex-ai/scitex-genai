"""Smoke-test isolation: every CLI subprocess gets a fake home.

``install-unit`` legitimately mints a gateway key into
``$SCITEX_DIR/genai/secrets`` and ``scitex-logging`` writes a runtime log
under ``$SCITEX_DIR`` — both must land in a tmp dir, never the developer's
real ``~/.scitex``. The key variable is set BLANK (not deleted):
``scitex-config`` loads dotenv on every path resolution, so a deleted var
gets repopulated from the developer's real ``.env`` while a blank one
still reads as unset after ``strip()``.

No monkeypatch (PA-306): explicit save/restore with try/finally.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from scitex_genai.gateway._secrets import GATEWAY_KEY_ENV


@pytest.fixture(autouse=True)
def _isolated_scitex_home(tmp_path: Path) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in ("SCITEX_DIR", GATEWAY_KEY_ENV)}
    os.environ["SCITEX_DIR"] = str(tmp_path / "scitex-home")
    os.environ[GATEWAY_KEY_ENV] = " "
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
