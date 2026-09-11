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

import json
import math
import os
import re
import shlex
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from scitex_config import get_scitex_dir, parse_src_file

REQUIRED = (
    "MODEL_PATH",
    "SERVED_NAME",
    "LITELLM_PORT",
    "TUNNEL_PORT",
    "MAX_MODEL_LEN",
)
KNOWN = REQUIRED + (
    "ENGINE",
    "ENGINE_PORT",
    "VLLM_PORT",
    "SGLANG_IMAGE",
    "TP",
    "GPU_MEM_UTIL",
    "MAX_NUM_SEQS",
    "EXTRA_VLLM_ARGS",
    "EXTRA_SGLANG_ARGS",
    "CANARY_ONLY",
    "CANARY_PURPOSE",
    "REQUIRED_GPU_COUNT",
    "REQUIRED_GPU_MODEL",
    "REQUIRED_HOST_MEMORY_GB",
    "SGLANG_IMAGE_SHA256",
    "MODEL_MANIFEST_FILES",
    "MODEL_MANIFEST_SHA256",
)
CONF_SUFFIX = ".conf"
_REF = re.compile(
    r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}|\$(?P<bare>[A-Za-z_][A-Za-z0-9_]*)"
)


def expand(value: str, env: dict[str, str] | None = None) -> str:
    """The two shell forms a conf uses: ``${NAME:-default}`` and ``$NAME`` / ``${NAME}``.

    A conf line like ``export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-1}`` is
    how an operator makes a flag overridable from the environment; sourcing
    expanded it, so this reader must too, or the engine receives the literal
    text ``${VLLM_USE_DEEP_GEMM:-1}`` as its setting.
    """
    source = os.environ if env is None else env

    def sub(match: re.Match[str]) -> str:
        name = match.group("name") or match.group("bare")
        current = source.get(name)
        if current:
            return current
        return match.group("default") or ""

    return _REF.sub(sub, value)


def default_models_dir() -> Path:
    """``$SCITEX_DIR/genai/models.d`` -- ``~/.scitex/genai/models.d`` normally."""
    return Path(get_scitex_dir()) / "genai" / "models.d"


def _port(name: str, value: object) -> int:
    number = int(str(value))
    if not 0 < number < 65536:
        raise ValueError(f"{name} must be within 1..65535, got {value!r}")
    return number


def _boolean(name: str, value: str | None) -> bool:
    if value in (None, "", "0"):
        return False
    if value == "1":
        return True
    raise ValueError(f"{name} must be 0 or 1, got {value!r}")


def exported_names(text: str) -> frozenset[str]:
    """The names an ``export NAME=...`` line makes visible to a child process."""
    names = set()
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("export ") and "=" in line:
            names.add(line[len("export ") :].split("=", 1)[0].strip())
    return frozenset(names)


def _flag_value(args: tuple[str, ...], flag: str) -> str | None:
    if flag not in args:
        return None
    index = args.index(flag) + 1
    return args[index] if index < len(args) else None


def _validate_hicache(args: tuple[str, ...], env: dict[str, str]) -> None:
    if "--enable-hierarchical-cache" not in args:
        return
    size = _positive_number(_flag_value(args, "--hicache-size"))
    if size is None:
        raise ValueError(
            "HiCache requires an explicit positive --hicache-size per TP rank"
        )
    if _flag_value(args, "--hicache-write-policy") != "write_through":
        raise ValueError("HiCache requires --hicache-write-policy write_through")
    if _flag_value(args, "--hicache-mem-layout") != "page_first":
        raise ValueError("HiCache requires --hicache-mem-layout page_first")
    if _flag_value(args, "--hicache-io-backend") != "kernel":
        raise ValueError("HiCache requires --hicache-io-backend kernel")
    storage = _flag_value(args, "--hicache-storage-backend")
    if storage is None:
        return
    if storage != "file":
        raise ValueError(
            "the reproducible HiCache profile supports only the file L3 backend"
        )
    directory = env.get("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR", "")
    if not directory.startswith("/"):
        raise ValueError("file HiCache requires an absolute storage directory")
    page_size = _positive_integer(_flag_value(args, "--page-size"))
    if page_size is None or page_size < 2:
        raise ValueError(
            "file HiCache requires an explicit --page-size greater than one"
        )
    raw_guard = _flag_value(args, "--hicache-storage-backend-extra-config")
    try:
        guard = json.loads(raw_guard or "")
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("file HiCache requires valid bounded storage JSON") from exc
    required = {"max_size", "min_free_space", "eviction_ratio"}
    if not required.issubset(guard):
        raise ValueError(
            "file HiCache storage JSON requires max_size, min_free_space, and eviction_ratio"
        )
    for name in ("max_size", "min_free_space"):
        if not _positive_storage_size(guard[name]):
            raise ValueError(f"file HiCache {name} must be a positive storage size")
    ratio = _positive_number(guard["eviction_ratio"])
    if ratio is None or ratio > 1:
        raise ValueError("file HiCache eviction_ratio must be within (0, 1]")


def _positive_number(value: object) -> float | None:
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _positive_integer(value: object) -> int | None:
    try:
        number = int(str(value))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _positive_storage_size(value: object) -> bool:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGT]?)", str(value))
    return bool(match and float(match.group(1)) > 0)


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
    engine: str = "vllm"
    sglang_image: Path | None = None
    tp: int = 1
    gpu_mem_util: float = 0.92
    max_num_seqs: int = 8
    extra_vllm_args: tuple[str, ...] = ()
    extra_sglang_args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    source: Path | None = None
    canary_only: bool = False
    canary_purpose: str | None = None
    required_gpu_count: int | None = None
    required_gpu_model: str | None = None
    required_host_memory_gb: int | None = None
    sglang_image_sha256: str | None = None
    model_manifest_files: tuple[str, ...] = ()
    model_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.key or "/" in self.key or self.key.startswith("."):
            raise ValueError(f"key must be a bare name, got {self.key!r}")
        if not self.served_name or any(ch.isspace() for ch in self.served_name):
            raise ValueError(f"SERVED_NAME must be one token, got {self.served_name!r}")
        if not Path(self.model_path).is_absolute():
            raise ValueError(
                f"MODEL_PATH must be absolute, got {str(self.model_path)!r}"
            )
        if self.engine not in {"vllm", "sglang"}:
            raise ValueError(f"ENGINE must be vllm or sglang, got {self.engine!r}")
        if self.engine == "sglang" and self.sglang_image is None:
            raise ValueError("SGLANG_IMAGE is required when ENGINE=sglang")
        if self.sglang_image is not None and not Path(self.sglang_image).is_absolute():
            raise ValueError(
                f"SGLANG_IMAGE must be absolute, got {str(self.sglang_image)!r}"
            )
        if (
            self.engine == "sglang"
            and "--disable-radix-cache" in self.extra_sglang_args
        ):
            raise ValueError(
                "--disable-radix-cache conflicts with session-aware radix caching"
            )
        unified = self.env.get("SGLANG_ENABLE_UNIFIED_RADIX_TREE")
        if self.engine == "sglang" and unified not in (None, "1"):
            raise ValueError(
                "SGLANG_ENABLE_UNIFIED_RADIX_TREE must be 1 for session-aware "
                f"radix caching, got {unified!r}"
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
        canary_fields = (
            self.canary_purpose,
            self.required_gpu_count,
            self.required_gpu_model,
            self.required_host_memory_gb,
            self.model_manifest_files,
            self.model_manifest_sha256,
        )
        if self.canary_only and not all(canary_fields):
            raise ValueError(
                "CANARY_ONLY requires purpose, GPU count/model, host-memory minimum, "
                "and model manifest files/digest"
            )
        if not self.canary_only and any(canary_fields):
            raise ValueError(
                "canary resource and manifest fields require CANARY_ONLY=1"
            )
        if self.canary_only and self.required_gpu_count < 1:
            raise ValueError(f"{self.key} requires a positive GPU count")
        if self.canary_only and self.tp != self.required_gpu_count:
            raise ValueError(
                f"{self.key} TP={self.tp} must equal its required GPU count "
                f"{self.required_gpu_count}"
            )
        if self.canary_only and not self.required_gpu_model.strip():
            raise ValueError(f"{self.key} requires a GPU model")
        if self.canary_only and self.required_host_memory_gb < 1:
            raise ValueError(f"{self.key} requires a positive host-memory minimum")
        if self.sglang_image_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.sglang_image_sha256
        ):
            raise ValueError("SGLANG_IMAGE_SHA256 must be 64 lowercase hex characters")
        if (
            self.canary_only
            and self.engine == "sglang"
            and not self.sglang_image_sha256
        ):
            raise ValueError("a canary SGLang profile requires SGLANG_IMAGE_SHA256")
        if self.model_manifest_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.model_manifest_sha256
        ):
            raise ValueError(
                "MODEL_MANIFEST_SHA256 must be 64 lowercase hex characters"
            )
        if any(
            not name or Path(name).is_absolute() or ".." in Path(name).parts
            for name in self.model_manifest_files
        ):
            raise ValueError("MODEL_MANIFEST_FILES must be relative model paths")
        if self.canary_only and not self.served_name.endswith("-canary"):
            raise ValueError(
                "a canary profile requires a SERVED_NAME ending in -canary"
            )
        if (
            self.canary_only
            and self.engine == "sglang"
            and not self.env.get("SCITEX_GENAI_EXPECTED_SGLANG_VERSION")
        ):
            raise ValueError(
                "a canary SGLang profile requires SCITEX_GENAI_EXPECTED_SGLANG_VERSION"
            )
        if self.engine == "sglang":
            _validate_hicache(self.extra_sglang_args, self.env)

    @property
    def engine_port(self) -> int:
        """Engine listen port; ``vllm_port`` remains the compatible field name."""
        return self.vllm_port


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


def parse_engine_conf(
    key: str,
    text: str,
    source: Path | None = None,
    env: dict[str, str] | None = None,
) -> EngineConf:
    """Build an :class:`EngineConf` from the text of a ``<KEY>.conf``.

    ``env`` is what ``${NAME:-default}`` references resolve against; the
    process environment when None, exactly as sourcing would.
    """
    # Expand the shell references on the RAW text, before the ecosystem parser
    # sees it: that parser resolves \$NAME against the process environment and
    # knows nothing of ${NAME:-default}, so an unset name would come back empty.
    expanded = "\n".join(expand(line, env) for line in text.splitlines())
    values = _values(expanded, None)
    missing = [name for name in REQUIRED if not values.get(name)]
    if not (values.get("ENGINE_PORT") or values.get("VLLM_PORT")):
        missing.append("ENGINE_PORT (or legacy VLLM_PORT)")
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
        vllm_port=_port(
            "ENGINE_PORT", values.get("ENGINE_PORT") or values["VLLM_PORT"]
        ),
        litellm_port=_port("LITELLM_PORT", values["LITELLM_PORT"]),
        tunnel_port=_port("TUNNEL_PORT", values["TUNNEL_PORT"]),
        max_model_len=int(values["MAX_MODEL_LEN"]),
        engine=(values.get("ENGINE") or "vllm").lower(),
        sglang_image=(
            Path(values["SGLANG_IMAGE"]) if values.get("SGLANG_IMAGE") else None
        ),
        tp=int(values.get("TP") or 1),
        gpu_mem_util=float(values.get("GPU_MEM_UTIL") or 0.92),
        max_num_seqs=int(values.get("MAX_NUM_SEQS") or 8),
        extra_vllm_args=tuple(shlex.split(values.get("EXTRA_VLLM_ARGS") or "")),
        extra_sglang_args=tuple(shlex.split(values.get("EXTRA_SGLANG_ARGS") or "")),
        env=env,
        source=source,
        canary_only=_boolean("CANARY_ONLY", values.get("CANARY_ONLY")),
        canary_purpose=values.get("CANARY_PURPOSE") or None,
        required_gpu_count=(
            int(values["REQUIRED_GPU_COUNT"])
            if values.get("REQUIRED_GPU_COUNT")
            else None
        ),
        required_gpu_model=values.get("REQUIRED_GPU_MODEL") or None,
        required_host_memory_gb=(
            int(values["REQUIRED_HOST_MEMORY_GB"])
            if values.get("REQUIRED_HOST_MEMORY_GB")
            else None
        ),
        sglang_image_sha256=values.get("SGLANG_IMAGE_SHA256") or None,
        model_manifest_files=tuple(
            shlex.split(values.get("MODEL_MANIFEST_FILES") or "")
        ),
        model_manifest_sha256=values.get("MODEL_MANIFEST_SHA256") or None,
    )


def list_engines(models_dir: Path | None = None) -> list[str]:
    """The keys that have a conf, sorted."""
    directory = Path(models_dir) if models_dir is not None else default_models_dir()
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob(f"*{CONF_SUFFIX}") if p.is_file())


def load_engine(
    key: str, models_dir: Path | None = None, env: dict[str, str] | None = None
) -> EngineConf:
    """Read and validate ``<models_dir>/<key>.conf``."""
    directory = Path(models_dir) if models_dir is not None else default_models_dir()
    path = directory / f"{key}{CONF_SUFFIX}"
    if not path.is_file():
        available = ", ".join(list_engines(directory)) or "none"
        raise FileNotFoundError(f"no engine conf {path}; available: {available}")
    return parse_engine_conf(key, path.read_text(), source=path, env=env)
