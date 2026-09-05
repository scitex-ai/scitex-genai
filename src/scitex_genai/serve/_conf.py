"""One served engine, read from the user's config tree and validated once.

WHY THIS SHAPE. Every locally hosted model was started by a script that
``source``d a ``<KEY>.conf`` of ``NAME=value`` lines and trusted whatever came
out: a missing port surfaced as a bash "unset" one line before launch, a typo
in ``MAX_MODEL_LEN`` reached vLLM as a string, and the ``export`` lines that
must reach the engine (``CUDA_VISIBLE_DEVICES``, ``VLLM_USE_DEEP_GEMM``) were
indistinguishable from the ones that must not. Those confs are the USER'S
settings (operator ruling 2026-09-05: settings live under ``~/.scitex/<pkg>/``,
the package ships the mechanism), so they stay exactly where and what they
are -- ``~/.scitex/genai/models.d/<KEY>.conf`` in the same ``NAME=value``
shape -- and this module is the one reader: scitex-config's ``parse_src_file``
(the ecosystem's bash-style parser) for the values, a fixed dataclass with a
validator for the meaning, and the ``export`` set kept apart so the launcher
knows which names to hand to the child process.

    # ~/.scitex/genai/models.d/<KEY>.conf
    export CUDA_VISIBLE_DEVICES=1        # reaches the engine
    export VLLM_USE_DEEP_GEMM=1          # reaches the engine
    MODEL_PATH=/path/to/weights          # required
    SERVED_NAME=my-model                 # required
    VLLM_PORT=8768                       # required
    LITELLM_PORT=4003                    # required
    TUNNEL_PORT=18773                    # required
    MAX_MODEL_LEN=1048576                # required
    TP=1  GPU_MEM_UTIL=0.92  MAX_NUM_SEQS=8            # defaults
    EXTRA_VLLM_ARGS="--enable-prefix-caching ..."      # shell-split

Nothing here names a model, a site or a host.
"""

from __future__ import annotations

import shlex
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from scitex_config import get_scitex_dir, parse_src_file

REQUIRED = (
    "MODEL_PATH",
    "SERVED_NAME",
    "VLLM_PORT",
    "LITELLM_PORT",
    "TUNNEL_PORT",
    "MAX_MODEL_LEN",
)
KNOWN = REQUIRED + ("TP", "GPU_MEM_UTIL", "MAX_NUM_SEQS", "EXTRA_VLLM_ARGS")
CONF_SUFFIX = ".conf"


def default_models_dir() -> Path:
    """``$SCITEX_DIR/genai/models.d`` -- ``~/.scitex/genai/models.d`` normally."""
    return Path(get_scitex_dir()) / "genai" / "models.d"


def _port(name: str, value: object) -> int:
    number = int(str(value))
    if not 0 < number < 65536:
        raise ValueError(f"{name} must be within 1..65535, got {value!r}")
    return number


def exported_names(text: str) -> frozenset[str]:
    """The names an ``export NAME=...`` line makes visible to a child process."""
    names = set()
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("export ") and "=" in line:
            names.add(line[len("export ") :].split("=", 1)[0].strip())
    return frozenset(names)


@dataclass(frozen=True)
class EngineConf:
    """Everything the launcher needs for ONE engine; validated at construction."""

    key: str
    model_path: Path
    served_name: str
    vllm_port: int
    litellm_port: int
    tunnel_port: int
    max_model_len: int
    tp: int = 1
    gpu_mem_util: float = 0.92
    max_num_seqs: int = 8
    extra_vllm_args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    source: Path | None = None

    def __post_init__(self) -> None:
        if not self.key or "/" in self.key or self.key.startswith("."):
            raise ValueError(f"key must be a bare name, got {self.key!r}")
        if not self.served_name or any(ch.isspace() for ch in self.served_name):
            raise ValueError(f"SERVED_NAME must be one token, got {self.served_name!r}")
        if not Path(self.model_path).is_absolute():
            raise ValueError(
                f"MODEL_PATH must be absolute, got {str(self.model_path)!r}"
            )
        ports = {
            "VLLM_PORT": _port("VLLM_PORT", self.vllm_port),
            "LITELLM_PORT": _port("LITELLM_PORT", self.litellm_port),
            "TUNNEL_PORT": _port("TUNNEL_PORT", self.tunnel_port),
        }
        if len(set(ports.values())) != 3:
            raise ValueError(f"the three ports must differ, got {ports}")
        if self.tp < 1:
            raise ValueError(f"TP must be >= 1, got {self.tp}")
        if not 0 < self.gpu_mem_util <= 1:
            raise ValueError(
                f"GPU_MEM_UTIL must be within (0, 1], got {self.gpu_mem_util}"
            )
        if self.max_model_len < 1:
            raise ValueError(f"MAX_MODEL_LEN must be >= 1, got {self.max_model_len}")
        if self.max_num_seqs < 1:
            raise ValueError(f"MAX_NUM_SEQS must be >= 1, got {self.max_num_seqs}")


def _values(text: str, source: Path | None) -> dict[str, str]:
    """``parse_src_file`` takes a path; feed it the file, or the text via a temp file."""
    if source is not None:
        return parse_src_file(Path(source))
    with tempfile.NamedTemporaryFile("w", suffix=CONF_SUFFIX, delete=False) as handle:
        handle.write(text)
        temp = Path(handle.name)
    try:
        return parse_src_file(temp)
    finally:
        temp.unlink(missing_ok=True)


def parse_engine_conf(key: str, text: str, source: Path | None = None) -> EngineConf:
    """Build an :class:`EngineConf` from the text of a ``<KEY>.conf``."""
    values = _values(text, source)
    missing = [name for name in REQUIRED if not values.get(name)]
    if missing:
        where = str(source) if source is not None else f"{key}{CONF_SUFFIX}"
        raise ValueError(f"{where}: unset: {', '.join(missing)}")
    exported = exported_names(text)
    env = {
        name: values[name]
        for name in sorted(exported)
        if name in values and name not in KNOWN
    }
    return EngineConf(
        key=key,
        model_path=Path(values["MODEL_PATH"]),
        served_name=values["SERVED_NAME"],
        vllm_port=_port("VLLM_PORT", values["VLLM_PORT"]),
        litellm_port=_port("LITELLM_PORT", values["LITELLM_PORT"]),
        tunnel_port=_port("TUNNEL_PORT", values["TUNNEL_PORT"]),
        max_model_len=int(values["MAX_MODEL_LEN"]),
        tp=int(values.get("TP") or 1),
        gpu_mem_util=float(values.get("GPU_MEM_UTIL") or 0.92),
        max_num_seqs=int(values.get("MAX_NUM_SEQS") or 8),
        extra_vllm_args=tuple(shlex.split(values.get("EXTRA_VLLM_ARGS") or "")),
        env=env,
        source=source,
    )


def list_engines(models_dir: Path | None = None) -> list[str]:
    """The keys that have a conf, sorted."""
    directory = Path(models_dir) if models_dir is not None else default_models_dir()
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob(f"*{CONF_SUFFIX}") if p.is_file())


def load_engine(key: str, models_dir: Path | None = None) -> EngineConf:
    """Read and validate ``<models_dir>/<key>.conf``."""
    directory = Path(models_dir) if models_dir is not None else default_models_dir()
    path = directory / f"{key}{CONF_SUFFIX}"
    if not path.is_file():
        available = ", ".join(list_engines(directory)) or "none"
        raise FileNotFoundError(f"no engine conf {path}; available: {available}")
    return parse_engine_conf(key, path.read_text(), source=path)
