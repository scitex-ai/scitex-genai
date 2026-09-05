"""Serve a local model engine from the user's settings -- the mechanism, not the site.

Operator ruling 2026-09-05: model serving is scitex-genai's concern; scitex-hpc
owns only the allocation and the tunnel; our settings (which models, which
ports, which scratch, which bastion) live in ``~/.scitex/genai/`` config files;
state goes to the Postgres store. This package is the mechanism half:

- ``_conf``      one ``<KEY>.conf`` from ``~/.scitex/genai/models.d`` -> ``EngineConf``
- ``_settings``  the ``serve:`` section of ``~/.scitex/genai/config.yaml`` -> ``ServeSettings``
- ``_render``    both -> ``Launch`` (env, argvs, config text, log paths), pure

The runner and the CLI that execute a ``Launch`` inside a held allocation
follow in their own change; nothing here starts a process.
"""

from ._conf import (
    EngineConf,
    default_models_dir,
    list_engines,
    load_engine,
    parse_engine_conf,
)
from ._render import Launch, render
from ._settings import ServeSettings, load_serve_settings

__all__ = [
    "EngineConf",
    "Launch",
    "ServeSettings",
    "default_models_dir",
    "list_engines",
    "load_engine",
    "load_serve_settings",
    "parse_engine_conf",
    "render",
]
