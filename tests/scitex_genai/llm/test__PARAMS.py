"""Smoke test for scitex_genai.llm._PARAMS.

Replaces the prior placeholder-only stub (audit-project PS206). The
test imports the target module and asserts that the imported object is
a real module exposing the expected MODELS table — if the import fails
or MODELS is missing/empty, the test fails loudly. Renames, broken
peer deps, or missing optional deps all surface here as red, not as a
silent skip.

If a module legitimately requires an optional dep, that dep should
be lazy-imported inside the function bodies — not at module top.
"""

import importlib
import types


def test_params_module_exposes_nonempty_models_table():
    """Importing scitex_genai.llm._PARAMS yields a module with a non-empty MODELS DataFrame."""
    # Arrange
    target_module_name = "scitex_genai.llm._PARAMS"

    # Act
    module = importlib.import_module(target_module_name)

    # Assert
    assert (
        isinstance(module, types.ModuleType)
        and hasattr(module, "MODELS")
        and len(module.MODELS) > 0
    )
