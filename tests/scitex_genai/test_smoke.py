"""Smoke tests for scitex_genai — verify package and submodule imports."""

from __future__ import annotations

import importlib

import pytest


def test_import_scitex_genai():
    import scitex_genai  # noqa: F401


def test_genai_lazy_attr():
    pytest.importorskip("anthropic")
    pytest.importorskip("openai")
    import scitex_genai

    assert hasattr(scitex_genai, "GenAI")


def test_modality_submodules_importable():
    """Every modality namespace must import; reserved ones may be stubs."""
    submods = ["llm", "agent", "image", "audio", "video", "embed", "multimodal"]
    failures = []
    for name in submods:
        try:
            importlib.import_module(f"scitex_genai.{name}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not failures, "Submodule imports failed:\n  " + "\n  ".join(failures)


def test_reserved_modules_raise_on_attr_access():
    """Reserved stubs should be importable but raise NotImplementedError on use."""
    import scitex_genai.agent as agent

    with pytest.raises(NotImplementedError):
        agent.SomeClass  # noqa: B018
