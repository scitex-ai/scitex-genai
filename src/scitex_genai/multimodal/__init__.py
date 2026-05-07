"""scitex_genai.multimodal — reserved namespace (not yet implemented).

Tracking issue: see scitex-genai roadmap. Importing this module is safe;
attribute access raises NotImplementedError to make the gap explicit.
"""
from __future__ import annotations


def __getattr__(name: str):
    raise NotImplementedError(
        f"scitex_genai.multimodal.{name} is not implemented yet. "
        "This namespace is reserved; track scitex-genai milestones."
    )


__all__: list[str] = []
