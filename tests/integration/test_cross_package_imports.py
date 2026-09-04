"""Runtime cross-package import gate (auto-generated).

This test imports every cross-package module that 'scitex-genai' references
in its source tree. Two outcomes:

- Module installed AND import succeeds -> test PASSES.
- Module installed BUT import fails -> test FAILS loudly.
- Module NOT installed (peer standalone absent in the CI env) ->
  test is SKIPPED via pytest.importorskip. The umbrella's CI
  (which installs every peer) catches cross-package renames.
"""

import importlib

import pytest

# ===== AUTO-GENERATED: cross-package imports =====
CROSS_PACKAGE_IMPORTS = [
    "scitex",
    "scitex_dev",
    "scitex_io",
    "scitex_str",
]
# ===== END AUTO-GENERATED =====


@pytest.mark.parametrize("module_name", CROSS_PACKAGE_IMPORTS)
def test_cross_package_import_resolves_to_real_module(module_name):
    """Importing scitex-genai's declared cross-package dependency must succeed."""
    # Arrange -- skip on the ROOT distribution only (a lean install where
    # the peer is genuinely absent must SKIP, not fail); two statements on
    # purpose so the root/full-path distinction stays visible (PS-140 s2).
    root = module_name.split(".")[0]
    pytest.importorskip(root)
    # Act -- hard-import the FULL dotted path, so a renamed submodule
    # FAILS instead of being skipped as an absence.
    mod = importlib.import_module(module_name)
    # Assert
    assert mod.__name__ == module_name
