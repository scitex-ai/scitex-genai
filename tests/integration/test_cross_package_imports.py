"""Runtime cross-package import gate (auto-generated).

This test imports every cross-package module that 'scitex-genai' references
in its source tree. Two outcomes:

- Module installed AND import succeeds -> test PASSES.
- Module installed BUT import fails -> test FAILS loudly.
- Module NOT installed (peer standalone absent in the CI env) ->
  test is SKIPPED via pytest.importorskip. The umbrella's CI
  (which installs every peer) catches cross-package renames.
"""

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
    # Arrange
    # Act
    mod = pytest.importorskip(module_name)
    # Assert
    assert mod.__name__ == module_name
