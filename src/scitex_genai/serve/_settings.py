"""The ``serve:`` section of ``~/.scitex/genai/config.yaml``: where the site keeps things.

Everything an engine launch needs that is about the SITE rather than the
MODEL -- the scratch base, the log directory, the node-local cache root, the
vLLM and LiteLLM binaries, the bastion the reverse tunnel climbs, the LiteLLM
master key -- lives here, read through scitex-config (direct -> file ->
environment -> default), and never in the package. Unlike the gateway there
are no usable defaults for the base or the bastion, so a missing section is
an error that names the file, not a silent launch into the wrong place.

    # ~/.scitex/genai/config.yaml
    serve:
      base: /scratch/<project>/<user>        # weights venv, generated files
      logs: /scratch/<project>/<user>/serve-logs
      cache_root: /tmp                       # NODE-LOCAL; keyed by engine, never by job
      vllm_bin: /scratch/.../vllm-venv/bin/vllm
      litellm_bin: /scratch/.../vllm-venv/bin/litellm
      cuda_home: /apps/.../CUDA/12.8.0       # optional
      bastion: bastion.example.org           # the -R target
      bastion_user: me
      proxy_command: "~/bin/cloudflared access ssh --hostname bastion.example.org"  # optional
      litellm_master_key: sk-local           # or SCITEX_SERVE_LITELLM_MASTER_KEY
"""

from __future__ import annotations

import getpass
from dataclasses import dataclass
from pathlib import Path

from scitex_config import ScitexConfig

from ..gateway._settings import default_config_path

SECTION = "serve"


@dataclass(frozen=True)
class ServeSettings:
    """Site-level facts for launching engines; validated at construction."""

    base: Path
    logs: Path
    cache_root: Path
    vllm_bin: Path
    litellm_bin: Path
    bastion: str
    bastion_user: str
    litellm_master_key: str
    cuda_home: Path | None = None
    proxy_command: str = ""
    source: Path | None = None

    def __post_init__(self) -> None:
        for name in ("base", "logs", "cache_root", "vllm_bin", "litellm_bin"):
            value = getattr(self, name)
            if not Path(value).is_absolute():
                raise ValueError(
                    f"serve.{name} must be an absolute path, got {str(value)!r}"
                )
        if self.cuda_home is not None and not Path(self.cuda_home).is_absolute():
            raise ValueError(
                f"serve.cuda_home must be absolute, got {str(self.cuda_home)!r}"
            )
        for name in ("bastion", "bastion_user", "litellm_master_key"):
            value = getattr(self, name)
            if not value or any(ch.isspace() for ch in value):
                raise ValueError(f"serve.{name} must be one non-empty token")


def load_serve_settings(config_path: Path | str | None = None) -> ServeSettings:
    """Read the ``serve:`` section; refuse, naming the file, when it is absent."""
    path = Path(config_path) if config_path is not None else default_config_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"serve needs {path} with a `{SECTION}:` section; it does not exist"
        )
    config = ScitexConfig(config_path=path)

    def get(key: str, default=None):
        return config.resolve(f"{SECTION}.{key}", default=default)

    base = get("base")
    if base is None:
        raise ValueError(f"{path}: `{SECTION}.base` is required")
    base_path = Path(str(base))
    return ServeSettings(
        base=base_path,
        logs=Path(str(get("logs", base_path / "serve-logs"))),
        cache_root=Path(str(get("cache_root", "/tmp"))),
        vllm_bin=Path(str(get("vllm_bin", base_path / "vllm-venv" / "bin" / "vllm"))),
        litellm_bin=Path(
            str(get("litellm_bin", base_path / "vllm-venv" / "bin" / "litellm"))
        ),
        bastion=str(get("bastion") or ""),
        bastion_user=str(get("bastion_user", getpass.getuser())),
        litellm_master_key=str(get("litellm_master_key") or ""),
        cuda_home=Path(str(get("cuda_home"))) if get("cuda_home") else None,
        proxy_command=str(get("proxy_command", "") or ""),
        source=path,
    )
