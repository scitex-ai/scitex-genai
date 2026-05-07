"""Sphinx configuration for scitex-genai documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------

project = "SciTeX GenAI"
copyright = "2024-2026, Yusuke Watanabe"
author = "Yusuke Watanabe"

try:
    from importlib.metadata import version as _get_version

    release = _get_version("scitex-genai")
except Exception:
    release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx_rtd_theme",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Provider SDKs are eagerly imported by llm/* today — mock them so RTD
# builds without network or API keys.
autodoc_mock_imports = [
    "openai",
    "anthropic",
    "google",
    "google.genai",
    "groq",
    "litellm",
    "ollama",
    "claude_agent_sdk",
]

autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "prev_next_buttons_location": "bottom",
}

html_static_path = ["_static"]
html_title = f"{project} v{release}"
html_short_title = project

html_context = {
    "display_github": True,
    "github_user": "ywatanabe1989",
    "github_repo": "scitex-genai",
    "github_version": "main",
    "conf_py_path": "/docs/sphinx/",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_admonition",
    "tasklist",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
