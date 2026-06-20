"""scitex_genai.agent — reserved namespace (not yet implemented).

Tracking issue: see scitex-genai roadmap. Importing this module is safe;
attribute access for any non-dunder name raises NotImplementedError to
make the gap explicit. Dunder lookups (`__sphinx_mock__`, `__path__`,
`__all__`, …) raise AttributeError instead so introspection tooling
(Sphinx, pickle, IPython tab-complete) sees a normal "no such attribute"
rather than a feature-gap error.
"""
from __future__ import annotations


def __getattr__(name: str):
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    raise NotImplementedError(
        f"scitex_genai.agent.{name} is not implemented yet. "
        "This namespace is reserved; track scitex-genai milestones."
    )


__all__: list[str] = []
