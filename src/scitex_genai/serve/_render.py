"""Render ONE engine launch from its conf and the site settings -- pure, testable.

This is the launch that ``serve-model.sh`` performed by string substitution,
ported line for line and made inspectable: the child environment, the vLLM
argv, the LiteLLM sidecar config and argv, and the reverse-tunnel argv. Nothing
is executed here; ``_run`` does that, so every choice below can be asserted
against without a GPU.

THREE THINGS CARRIED OVER FROM THE SCRIPT, EACH PAID FOR:
- ``ControlMaster=no`` / ``ControlPath=none`` on the tunnel: without them
  ``ssh -R`` attaches to an existing mux master, exits 0 in two seconds, and
  the forward's lifetime belongs to something other than this launch.
- The tunnel forwards to vLLM's OWN port, not LiteLLM's: vLLM speaks the
  Anthropic ``/v1/messages`` shape natively, and the gateway on the other end
  hoists the one thing it rejects.
- An explicit ``model_name`` entry in the LiteLLM config next to the ``"*"``
  fallback: LiteLLM did not route the bare wildcard on either path (400 "LLM
  Provider NOT provided").

ONE THING DELIBERATELY CHANGED: the node-local cache is ``<cache_root>/<key>``,
keyed by ENGINE and never by job id. Keying it by ``$SLURM_JOB_ID`` guaranteed
a cold FlashInfer JIT (2 h 27 m measured) on every resubmit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ._conf import EngineConf
from ._settings import ServeSettings

LOCAL = "127.0.0.1"
LITELLM_DUMMY_KEY = "dummy-local-vllm"
CACHE_SUBDIRS = {
    "HOME": "home",
    "XDG_CACHE_HOME": "xdg",
    "HF_HOME": "hf",
    "TORCH_HOME": "torch",
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
    "VLLM_CACHE_ROOT": "vllm",
    "FLASHINFER_WORKSPACE_BASE": "flashinfer",
    "SGLANG_CACHE_DIR": "sglang",
    "SGLANG_JIT_CACHE_DIR": "sglang-jit",
    "DG_JIT_CACHE_DIR": "deepgemm-jit",
}
SGLANG_UNIFIED_RADIX_ENV = "SGLANG_ENABLE_UNIFIED_RADIX_TREE"
SGLANG_SESSION_FLAG = "--enable-session-radix-cache"
SGLANG_METRICS_FLAG = "--enable-metrics"


@dataclass(frozen=True)
class Launch:
    """Everything ``_run`` executes for one engine, as data.

    ``env`` is the engine's environment (caches under the engine cache, HOME
    redirected there); ``tunnel_env`` is the UNCHANGED caller environment,
    because ssh needs the real HOME for its keys and known_hosts.
    """

    key: str
    env: dict[str, str]
    tunnel_env: dict[str, str]
    cache_dir: Path
    engine_name: str
    engine_argv: tuple[str, ...]
    engine_preflight_argv: tuple[str, ...] | None
    litellm_config_path: Path
    litellm_config_text: str
    litellm_argv: tuple[str, ...]
    tunnel_argv: tuple[str, ...]
    engine_log: Path
    litellm_log: Path
    tunnel_log: Path
    health_url: str

    @property
    def vllm_argv(self) -> tuple[str, ...]:
        """Compatibility alias for callers written before multiple engines."""
        return self.engine_argv

    @property
    def vllm_log(self) -> Path:
        """Compatibility alias for callers written before multiple engines."""
        return self.engine_log


def cache_dir(settings: ServeSettings, conf: EngineConf) -> Path:
    """Node-local, keyed by engine -- a resubmit finds its JIT cache warm."""
    return Path(settings.cache_root) / f"{conf.key}-cache"


def child_env(
    settings: ServeSettings, conf: EngineConf, base_env: dict[str, str]
) -> dict[str, str]:
    """The environment the engine runs under: caches, CUDA, then the conf's exports."""
    cache = cache_dir(settings, conf)
    env = dict(base_env)
    env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    path = env.get("PATH", "")
    prefix = [str(Path(settings.vllm_bin).parent)]
    if settings.cuda_home is not None:
        env["CUDA_HOME"] = str(settings.cuda_home)
        prefix.insert(0, str(Path(settings.cuda_home) / "bin"))
        ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(
            p for p in (str(Path(settings.cuda_home) / "lib64"), ld) if p
        )
    env["PATH"] = ":".join(p for p in (*prefix, path) if p)
    env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    env["SGLANG_JIT_DEEPGEMM_FAST_WARMUP"] = "1"
    for name, sub in CACHE_SUBDIRS.items():
        env[name] = str(cache / sub)
    env.update(conf.env)
    if conf.engine == "sglang":
        # Session-aware retention is implemented only by UnifiedRadixCache.
        # This is enforced after user exports so a stale profile cannot
        # silently launch the incompatible legacy tree.
        env[SGLANG_UNIFIED_RADIX_ENV] = "1"
    return env


def vllm_argv(settings: ServeSettings, conf: EngineConf) -> tuple[str, ...]:
    return (
        str(settings.vllm_bin),
        "serve",
        str(conf.model_path),
        "--served-model-name",
        conf.served_name,
        "--tensor-parallel-size",
        str(conf.tp),
        "--gpu-memory-utilization",
        str(conf.gpu_mem_util),
        "--max-model-len",
        str(conf.max_model_len),
        "--max-num-seqs",
        str(conf.max_num_seqs),
        *conf.extra_vllm_args,
        "--host",
        LOCAL,
        "--port",
        str(conf.vllm_port),
    )


def _sglang_container_prefix(
    settings: ServeSettings, conf: EngineConf, env: dict[str, str]
) -> tuple[str, ...]:
    """Container prefix shared by capability validation and server launch."""
    assert conf.sglang_image is not None  # validated by EngineConf
    passed_names = tuple(
        dict.fromkeys(
            (
                *(name for name in CACHE_SUBDIRS if name != "HOME"),
                *conf.env,
                "CUDA_VISIBLE_DEVICES",
                "SGLANG_JIT_DEEPGEMM_FAST_WARMUP",
                SGLANG_UNIFIED_RADIX_ENV,
            )
        )
    )
    container_env = tuple(
        item
        for name in passed_names
        if name in env
        for item in ("--env", f"{name}={env[name]}")
    )
    return (
        str(settings.apptainer_bin),
        "exec",
        "--nv",
        "--cleanenv",
        "--bind",
        f"{conf.model_path}:{conf.model_path}:ro",
        *container_env,
        str(conf.sglang_image),
    )


def sglang_preflight_argv(
    settings: ServeSettings, conf: EngineConf, env: dict[str, str]
) -> tuple[str, ...]:
    """Refuse images that cannot provide the required session cache contract."""
    script = (
        "from sglang.srt.server_args import ServerArgs; "
        "from sglang.srt import environ; "
        "fields=ServerArgs.__dataclass_fields__; "
        "required=('enable_session_radix_cache','enable_metrics'); "
        "missing=[name for name in required if name not in fields]; "
        "assert not missing, f'unsupported SGLang image; missing flags: {missing}'; "
        "assert hasattr(environ.envs, 'SGLANG_ENABLE_UNIFIED_RADIX_TREE'), "
        "'unsupported SGLang image; missing SGLANG_ENABLE_UNIFIED_RADIX_TREE'; "
        "import importlib.metadata, os; "
        "expected=os.environ.get('SCITEX_GENAI_EXPECTED_SGLANG_VERSION'); "
        "actual=importlib.metadata.version('sglang'); "
        "assert not expected or actual == expected, "
        "f'unsupported SGLang build: expected {expected}, found {actual}'"
    )
    return (*_sglang_container_prefix(settings, conf, env), "python3", "-c", script)


def sglang_argv(
    settings: ServeSettings, conf: EngineConf, env: dict[str, str]
) -> tuple[str, ...]:
    """Pinned-container SGLang launch with session-aware cache retention."""
    required = tuple(
        flag
        for flag in (SGLANG_SESSION_FLAG, SGLANG_METRICS_FLAG)
        if flag not in conf.extra_sglang_args
    )
    return (
        *_sglang_container_prefix(settings, conf, env),
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(conf.model_path),
        "--served-model-name",
        conf.served_name,
        "--tp-size",
        str(conf.tp),
        "--mem-fraction-static",
        str(conf.gpu_mem_util),
        "--context-length",
        str(conf.max_model_len),
        "--max-running-requests",
        str(conf.max_num_seqs),
        *required,
        *conf.extra_sglang_args,
        "--host",
        LOCAL,
        "--port",
        str(conf.engine_port),
    )


def engine_argv(
    settings: ServeSettings, conf: EngineConf, env: dict[str, str]
) -> tuple[str, ...]:
    if conf.engine == "sglang":
        return sglang_argv(settings, conf, env)
    return vllm_argv(settings, conf)


def litellm_config_text(settings: ServeSettings, conf: EngineConf) -> str:
    """The sidecar config: explicit name plus wildcard, both to this engine."""
    entry = (
        "    litellm_params:\n"
        f"      model: openai/{conf.served_name}\n"
        f"      api_base: http://{LOCAL}:{conf.vllm_port}/v1\n"
        f'      api_key: "{LITELLM_DUMMY_KEY}"\n'
    )
    return (
        f"# Generated by scitex-genai serve for engine {conf.key}; do not edit.\n"
        "model_list:\n"
        f"  - model_name: {conf.served_name}\n"
        f"{entry}"
        '  - model_name: "*"\n'
        f"{entry}"
        "\n"
        "litellm_settings:\n"
        "  drop_params: true\n"
        "  set_verbose: false\n"
        "\n"
        "general_settings:\n"
        f'  master_key: "{settings.litellm_master_key}"\n'
    )


def litellm_argv(
    settings: ServeSettings, conf: EngineConf, config_path: Path
) -> tuple[str, ...]:
    return (
        str(settings.litellm_bin),
        "--config",
        str(config_path),
        "--host",
        "0.0.0.0",
        "--port",
        str(conf.litellm_port),
    )


def tunnel_argv(settings: ServeSettings, conf: EngineConf) -> tuple[str, ...]:
    """``ssh -R <tunnel_port>:127.0.0.1:<vllm_port>`` to the bastion, un-muxed."""
    argv: list[str] = [
        "ssh",
        "-N",
        "-o",
        "BatchMode=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]
    if settings.proxy_command:
        argv += ["-o", f"ProxyCommand={settings.proxy_command}"]
    argv += [
        "-R",
        f"{conf.tunnel_port}:{LOCAL}:{conf.vllm_port}",
        f"{settings.bastion_user}@{settings.bastion}",
    ]
    return tuple(argv)


def render(
    settings: ServeSettings, conf: EngineConf, base_env: dict[str, str]
) -> Launch:
    """The whole launch for one engine, as data."""
    logs = Path(settings.logs)
    litellm_path = Path(settings.base) / f"litellm-{conf.key}.yaml"
    env = child_env(settings, conf, base_env)
    return Launch(
        key=conf.key,
        env=env,
        tunnel_env=dict(base_env),
        cache_dir=cache_dir(settings, conf),
        engine_name=conf.engine,
        engine_argv=engine_argv(settings, conf, env),
        engine_preflight_argv=(
            sglang_preflight_argv(settings, conf, env)
            if conf.engine == "sglang"
            else None
        ),
        litellm_config_path=litellm_path,
        litellm_config_text=litellm_config_text(settings, conf),
        litellm_argv=litellm_argv(settings, conf, litellm_path),
        tunnel_argv=tunnel_argv(settings, conf),
        engine_log=logs / f"{conf.engine}-{conf.key}.log",
        litellm_log=logs / f"litellm-{conf.key}.log",
        tunnel_log=logs / f"tunnel-{conf.key}.log",
        health_url=f"http://{LOCAL}:{conf.vllm_port}/health",
    )
