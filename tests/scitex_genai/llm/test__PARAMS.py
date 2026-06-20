"""Auto-generated smoke test for scitex_genai.llm._PARAMS.

Replaces the prior placeholder-only stub (audit-project PS206). The
test imports the target module — if the import fails, the test
fails. Renames, broken peer deps, or missing optional deps all
surface here as red, not as a silent skip.

If a module legitimately requires an optional dep, that dep should
be lazy-imported inside the function bodies — not at module top.
"""

import importlib


def test_params_module_exposes_models_dataframe():
    """Smoke: target module imports and exposes a populated MODELS table."""
    # Arrange
    # Act
    mod = importlib.import_module("scitex_genai.llm._PARAMS")
    # Assert
    assert len(mod.MODELS) > 0
