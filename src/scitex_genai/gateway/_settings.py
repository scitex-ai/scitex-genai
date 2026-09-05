"""The gateway's settings come from the user's config tree, never from the package.

OPERATOR RULING (Telegram, 2026-09-05): a package carries the reusable, public
mechanism; settings particular to one deployment -- which address, which port,
which upstreams -- live in that user's config tree under ``~/.scitex/genai/``;
anything that is STATE goes to the Postgres store; and where a scitex-dev
primitive exists it is used rather than re-invented.

So this module reads ONE file, ``~/.scitex/genai/config.yaml``, through
scitex-config's ``ScitexConfig`` (the ecosystem's YAML + environment cascade)
and answers with a fixed dataclass. Precedence is the primitive's own:
direct (command line) -> config file -> environment -> default.
``HOIST_UPSTREAM`` stays the environment name for the upstream list because
the relay's users already export it::

    # ~/.scitex/genai/config.yaml
    gateway:
      host: 0.0.0.0
      port: 18772
      inference_upstreams:
        - http://127.0.0.1:18773
        - http://127.0.0.1:18774

A missing file is not an error: the package must run for someone who has no
config yet, on the defaults the command line always had (127.0.0.1:8765, the
Codex backend). ``source`` says which case applied.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scitex_config import ScitexConfig, get_scitex_dir

from ._inference import UPSTREAM_ENV, parse_upstreams

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
KEY_HOST = "gateway.host"
KEY_PORT = "gateway.port"
KEY_UPSTREAMS = "gateway.inference_upstreams"


def default_config_path() -> Path:
    """``$SCITEX_DIR/genai/config.yaml`` -- ``~/.scitex/genai/config.yaml`` normally."""
    return Path(get_scitex_dir()) / "genai" / "config.yaml"


def check_host(host: Any) -> str:
    """A bare address or name: non-empty, no whitespace."""
    text = str(host)
    if not text or any(ch.isspace() for ch in text):
        raise ValueError(f"host must be a bare address, got {host!r}")
    return text


def check_port(port: Any) -> int:
    """A TCP port in 1..65535."""
    number = int(port)
    if not 0 < number < 65536:
        raise ValueError(f"port must be within 1..65535, got {port!r}")
    return number


def upstream_string(value: Any) -> str:
    """The comma-separated form the server takes; a list, a string or nothing."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = ",".join(str(item) for item in value)
    return ",".join(parse_upstreams(str(value)))


@dataclass(frozen=True)
class GatewaySettings:
    """One gateway, fully described. ``inference_upstream`` empty = Codex backend."""

    host: str
    port: int
    inference_upstream: str
    source: Path | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "host", check_host(self.host))
        object.__setattr__(self, "port", check_port(self.port))
        object.__setattr__(
            self, "inference_upstream", upstream_string(self.inference_upstream)
        )


def load_settings(
    config_path: Path | str | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
    inference_upstream: str | None = None,
) -> GatewaySettings:
    """Resolve the gateway's settings: direct -> config file -> environment -> default."""
    path = Path(config_path) if config_path is not None else default_config_path()
    present = path.is_file()
    config = ScitexConfig(config_path=path if present else None)
    upstream = config.resolve(
        KEY_UPSTREAMS, direct_val=inference_upstream, default=None
    )
    if upstream is None:
        upstream = os.getenv(UPSTREAM_ENV, "")
    return GatewaySettings(
        host=config.resolve(KEY_HOST, direct_val=host, default=DEFAULT_HOST),
        port=config.resolve(KEY_PORT, direct_val=port, default=DEFAULT_PORT, type=int),
        inference_upstream=upstream,
        source=path if present else None,
    )
