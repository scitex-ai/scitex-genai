"""E2E fixtures: fake home only — no blanket skip gate.

Per the layered-testing leaf, e2e runs on every PR. The only legitimate
skips are per-test, keyed to a missing subsystem (missing binary, missing
import, missing GPU). These tests need none: they drive real subsystems
that always exist on a dev runner (loopback sockets, real files).
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
    monkeypatch.delenv(GATEWAY_KEY_ENV, raising=False)
