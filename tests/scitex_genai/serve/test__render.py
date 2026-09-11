"""The rendered launch is the script's launch, made inspectable -- pure data."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
import yaml

from scitex_genai.serve._conf import EngineConf, parse_engine_conf
from scitex_genai.serve._render import CACHE_SUBDIRS, cache_dir, child_env, render
from scitex_genai.serve._settings import ServeSettings

SETTINGS = ServeSettings(
    base=Path("/scratch/site/me"),
    logs=Path("/scratch/site/me/serve-logs"),
    cache_root=Path("/local"),
    vllm_bin=Path("/scratch/site/me/vllm-venv/bin/vllm"),
    litellm_bin=Path("/scratch/site/me/vllm-venv/bin/litellm"),
    bastion="bastion.example.org",
    bastion_user="me",
    litellm_master_key="sk-local",
    cuda_home=Path("/apps/cuda"),
    proxy_command="/home/me/bin/cloudflared access ssh --hostname bastion.example.org",
)
CONF = EngineConf(
    key="model-a",
    model_path=Path("/weights/model-a"),
    served_name="model-a",
    vllm_port=8768,
    litellm_port=4003,
    tunnel_port=18773,
    max_model_len=1048576,
    extra_vllm_args=("--enable-prefix-caching",),
    env={"CUDA_VISIBLE_DEVICES": "1"},
)
SGLANG_CONF = EngineConf(
    key="model-sglang",
    model_path=Path("/weights/model-a"),
    served_name="model-a",
    vllm_port=8769,
    litellm_port=4004,
    tunnel_port=18774,
    max_model_len=1_000_000,
    engine="sglang",
    sglang_image=Path("/images/sglang.sif"),
    tp=2,
    gpu_mem_util=0.85,
    max_num_seqs=8,
    extra_sglang_args=("--kv-cache-dtype", "fp8_e4m3", "--enable-metrics"),
    env={"CUDA_VISIBLE_DEVICES": "0,1"},
)
BASE_ENV = {
    "PATH": "/usr/bin",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "LD_LIBRARY_PATH": "/lib",
}
CANARY_DIR = Path(__file__).parents[3] / "examples" / "serve" / "canary"
CANARY_MANIFEST = json.loads((CANARY_DIR / "qwen38-scheduler-matrix.json").read_text())
CANARY_SCHEMA = json.loads(
    (CANARY_DIR / "qwen38-scheduler-matrix.schema.json").read_text()
)
CANARY_CACHE_NAMESPACE = "/tmp/scitex-genai-canary/qwen38-4ccff141-b742f112"
EXPECTED_CANARY_ENV = {
    "CUDA_VISIBLE_DEVICES": "0,1",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
    "SGLANG_JIT_DEEPGEMM_FAST_WARMUP": "1",
    "SCITEX_GENAI_EXPECTED_SGLANG_VERSION": "0.0.0.dev1+g4ccff141d.d20260907",
    "SGLANG_CACHE_DIR": f"{CANARY_CACHE_NAMESPACE}/sglang",
    "SGLANG_JIT_CACHE_DIR": f"{CANARY_CACHE_NAMESPACE}/sglang-jit",
    "DG_JIT_CACHE_DIR": f"{CANARY_CACHE_NAMESPACE}/deepgemm-jit",
    "FLASHINFER_WORKSPACE_BASE": f"{CANARY_CACHE_NAMESPACE}/flashinfer",
    "TRITON_CACHE_DIR": f"{CANARY_CACHE_NAMESPACE}/triton",
    "TORCHINDUCTOR_CACHE_DIR": f"{CANARY_CACHE_NAMESPACE}/inductor",
}
EXPECTED_YARN = {
    "text_config": {
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "rope_type": "yarn",
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
            "factor": 4.0,
            "original_max_position_embeddings": 262_144,
        }
    }
}


def _canary_profile(profile_id: str) -> dict[str, object]:
    return next(
        item for item in CANARY_MANIFEST["profiles"] if item["id"] == profile_id
    )


def _canary_conf(profile: dict[str, object]):
    path = CANARY_DIR / str(profile["file"])
    return parse_engine_conf(path.stem, path.read_text(), source=path)


def _arg_value(argv: tuple[str, ...], name: str) -> str:
    return argv[argv.index(name) + 1]


_CONTROLLED_VALUE_FLAGS = {
    "--schedule-policy",
    "--chunked-prefill-size",
    "--speculative-algorithm",
    "--speculative-num-steps",
    "--speculative-eagle-topk",
    "--speculative-num-draft-tokens",
}
_CONTROLLED_SWITCH_FLAGS = {"--enable-mixed-chunk"}


def _common_sglang_args(args: tuple[str, ...]) -> tuple[str, ...]:
    common: list[str] = []
    index = 0
    while index < len(args):
        argument = args[index]
        if argument in _CONTROLLED_VALUE_FLAGS:
            index += 2
            continue
        if argument in _CONTROLLED_SWITCH_FLAGS:
            index += 1
            continue
        common.append(argument)
        index += 1
    return tuple(common)


def _profile_invariants(conf: EngineConf) -> tuple[object, ...]:
    return (
        conf.model_path,
        conf.served_name,
        conf.engine,
        conf.sglang_image,
        conf.engine_port,
        conf.litellm_port,
        conf.tunnel_port,
        conf.max_model_len,
        conf.tp,
        conf.gpu_mem_util,
        conf.max_num_seqs,
        conf.env,
        conf.canary_only,
        conf.canary_purpose,
        conf.required_gpu_count,
        conf.required_gpu_model,
        conf.required_host_memory_gb,
        conf.sglang_image_sha256,
        conf.model_manifest_files,
        conf.model_manifest_sha256,
        _common_sglang_args(conf.extra_sglang_args),
    )


def _changed_dimensions(profile: dict[str, object]) -> set[str]:
    baseline = _canary_profile("fcfs-eagle-c32768")
    names = (
        "schedule_policy",
        "chunked_prefill_size",
        "speculation",
        "mixed_chunk",
    )
    return {name for name in names if profile[name] != baseline[name]}


def test_cache_dir_is_keyed_by_engine_never_by_job():
    # Arrange
    settings, conf = SETTINGS, CONF

    # Act
    path = cache_dir(settings, conf)

    # Assert
    assert path == Path("/local/model-a-cache")


def test_every_cache_variable_points_under_the_engine_cache():
    # Arrange
    env = child_env(SETTINGS, CONF, BASE_ENV)

    # Act
    caches = {name: env[name] for name in CACHE_SUBDIRS}

    # Assert
    assert caches == {
        name: f"/local/model-a-cache/{sub}" for name, sub in CACHE_SUBDIRS.items()
    }


def test_hf_transfer_is_dropped_from_the_child():
    # Arrange
    env = child_env(SETTINGS, CONF, BASE_ENV)

    # Act
    present = "HF_HUB_ENABLE_HF_TRANSFER" in env

    # Assert
    assert present is False


def test_path_is_prefixed_with_cuda_then_the_venv():
    # Arrange
    env = child_env(SETTINGS, CONF, BASE_ENV)

    # Act
    path = env["PATH"]

    # Assert
    assert path == "/apps/cuda/bin:/scratch/site/me/vllm-venv/bin:/usr/bin"


def test_ld_library_path_gets_cuda_first():
    # Arrange
    env = child_env(SETTINGS, CONF, BASE_ENV)

    # Act
    value = env["LD_LIBRARY_PATH"]

    # Assert
    assert value == "/apps/cuda/lib64:/lib"


def test_the_confs_exports_reach_the_child():
    # Arrange
    env = child_env(SETTINGS, CONF, BASE_ENV)

    # Act
    visible = env.get("CUDA_VISIBLE_DEVICES")

    # Assert
    assert visible == "1"


def test_vllm_binds_loopback_on_the_confs_port():
    # Arrange
    launch = render(SETTINGS, CONF, BASE_ENV)

    # Act
    tail = launch.vllm_argv[-4:]

    # Assert
    assert tail == ("--host", "127.0.0.1", "--port", "8768")


def test_extra_vllm_args_come_before_the_bind():
    # Arrange
    argv = render(SETTINGS, CONF, BASE_ENV).vllm_argv

    # Act
    order = (argv.index("--enable-prefix-caching"), argv.index("--host"))

    # Assert
    assert order[0] < order[1]


def test_sglang_render_enables_session_cache_and_metrics():
    # Arrange
    launch = render(SETTINGS, SGLANG_CONF, BASE_ENV)

    # Act
    argv = launch.engine_argv
    container_env = {argv[i + 1] for i, arg in enumerate(argv) if arg == "--env"}

    # Assert
    assert (
        argv.count("--enable-session-radix-cache"),
        argv.count("--enable-metrics"),
        "SGLANG_ENABLE_UNIFIED_RADIX_TREE=1" in container_env,
    ) == (1, 1, True)


def test_sglang_preflight_validates_the_exact_required_capabilities():
    # Arrange
    launch = render(SETTINGS, SGLANG_CONF, BASE_ENV)

    # Act
    argv = launch.engine_preflight_argv or ()
    script = argv[argv.index("-c") + 1]

    # Assert
    assert (
        bool(argv),
        "enable_session_radix_cache" in script,
        "enable_metrics" in script,
        "SGLANG_ENABLE_UNIFIED_RADIX_TREE" in script,
        "SCITEX_GENAI_EXPECTED_SGLANG_VERSION" in script,
    ) == (True, True, True, True, True)


def test_sglang_uses_the_pinned_apptainer_image_and_model_bind():
    # Arrange
    conf = SGLANG_CONF

    # Act
    argv = render(SETTINGS, conf, BASE_ENV).engine_argv

    # Assert
    assert (
        argv[:4],
        argv[argv.index("--bind") + 1],
        "/images/sglang.sif" in argv,
    ) == (
        ("/usr/bin/apptainer", "exec", "--nv", "--cleanenv"),
        "/weights/model-a:/weights/model-a:ro",
        True,
    )


def test_canonical_qwen_profile_renders_session_cache_and_metrics():
    # Arrange
    root = Path(__file__).parents[3]
    text = (root / "examples/serve/qwen38-27b-sglang.conf").read_text()
    conf = parse_engine_conf("qwen38-27b-sglang", text)

    # Act
    launch = render(SETTINGS, conf, BASE_ENV)

    # Assert
    assert (
        launch.engine_argv.count("--enable-session-radix-cache"),
        launch.engine_argv.count("--enable-metrics"),
        launch.env["SGLANG_ENABLE_UNIFIED_RADIX_TREE"],
    ) == (1, 1, "1")


def test_scheduler_canary_manifest_conforms_to_its_schema():
    # Arrange
    validator = jsonschema.Draft202012Validator(CANARY_SCHEMA)

    # Act
    errors = sorted(
        validator.iter_errors(CANARY_MANIFEST), key=lambda error: error.json_path
    )

    # Assert
    assert errors == []


def test_scheduler_manifest_declares_every_and_only_canary_conf():
    # Arrange
    declared = {profile["file"] for profile in CANARY_MANIFEST["profiles"]}

    # Act
    present = {path.name for path in CANARY_DIR.glob("*.conf")}

    # Assert
    assert present == declared


def test_scheduler_profile_ids_and_files_are_unique():
    # Arrange
    profiles = CANARY_MANIFEST["profiles"]

    # Act
    unique_counts = (
        len({profile["id"] for profile in profiles}),
        len({profile["file"] for profile in profiles}),
    )

    # Assert
    assert unique_counts == (len(profiles), len(profiles))


@pytest.mark.parametrize(
    "profile_id",
    [profile["id"] for profile in CANARY_MANIFEST["profiles"]],
)
def test_scheduler_profile_dry_renders_declared_shape(profile_id: str):
    # Arrange
    profile = _canary_profile(profile_id)
    conf = _canary_conf(profile)

    # Act
    argv = render(SETTINGS, conf, BASE_ENV).engine_argv
    observed = (
        _arg_value(argv, "--schedule-policy"),
        int(_arg_value(argv, "--chunked-prefill-size")),
        int(_arg_value(argv, "--max-prefill-tokens")),
        "--speculative-algorithm" in argv,
        "--enable-mixed-chunk" in argv,
        argv.count("--enable-session-radix-cache"),
        argv.count("--enable-metrics"),
    )

    # Assert
    assert observed == (
        profile["schedule_policy"],
        profile["chunked_prefill_size"],
        profile["max_prefill_tokens"],
        profile["speculation"] == "eagle",
        profile["mixed_chunk"],
        1,
        1,
    )


@pytest.mark.parametrize(
    "profile_id",
    [profile["id"] for profile in CANARY_MANIFEST["profiles"]],
)
def test_scheduler_profile_keeps_measured_invariants(profile_id: str):
    # Arrange
    profile = _canary_profile(profile_id)

    # Act
    conf = _canary_conf(profile)

    # Assert
    assert (
        conf.engine,
        conf.model_path.as_posix(),
        conf.served_name,
        conf.tp,
        conf.max_model_len,
        conf.gpu_mem_util,
        conf.max_num_seqs,
        conf.sglang_image.as_posix(),
        conf.engine_port,
        conf.litellm_port,
        conf.tunnel_port,
        conf.env,
        _arg_value(conf.extra_sglang_args, "--kv-cache-dtype"),
        _arg_value(conf.extra_sglang_args, "--attention-backend"),
        json.loads(_arg_value(conf.extra_sglang_args, "--json-model-override-args")),
        conf.canary_only,
        conf.canary_purpose,
        conf.required_gpu_count,
        conf.required_gpu_model,
        conf.required_host_memory_gb,
        conf.sglang_image_sha256,
        list(conf.model_manifest_files),
        conf.model_manifest_sha256,
    ) == (
        "sglang",
        "/data/scratch/projects/punim0264/ywatanabe/hf/Qwen3.8-27B-FP8",
        "qwen38-27b-canary",
        2,
        1_000_000,
        0.85,
        8,
        CANARY_MANIFEST["sglang_image"],
        28768,
        24003,
        28773,
        EXPECTED_CANARY_ENV,
        "fp8_e4m3",
        "flashinfer",
        EXPECTED_YARN,
        True,
        CANARY_MANIFEST["runtime_guard"]["purpose"],
        CANARY_MANIFEST["runtime_guard"]["gpu_count"],
        CANARY_MANIFEST["runtime_guard"]["gpu_model"],
        CANARY_MANIFEST["runtime_guard"]["host_memory_gb"],
        CANARY_MANIFEST["sglang_image_sha256"],
        CANARY_MANIFEST["model_manifest_files"],
        CANARY_MANIFEST["model_manifest_sha256"],
    )


@pytest.mark.parametrize(
    "profile_id",
    [profile["id"] for profile in CANARY_MANIFEST["profiles"]],
)
def test_scheduler_profile_exact_diff_matches_declared_dimensions(profile_id: str):
    # Arrange
    profile = _canary_profile(profile_id)

    # Act
    observed = _changed_dimensions(profile)

    # Assert
    assert observed == set(profile["changed_dimensions"])


@pytest.mark.parametrize(
    "profile_id",
    [profile["id"] for profile in CANARY_MANIFEST["profiles"]],
)
def test_scheduler_profile_changes_no_undeclared_engine_invariant(profile_id: str):
    # Arrange
    baseline = _canary_conf(_canary_profile("fcfs-eagle-c32768"))

    # Act
    candidate = _canary_conf(_canary_profile(profile_id))

    # Assert
    assert _profile_invariants(candidate) == _profile_invariants(baseline)


@pytest.mark.parametrize(
    "profile_id",
    [profile["id"] for profile in CANARY_MANIFEST["profiles"]],
)
def test_scheduler_profile_uses_shared_version_pinned_jit_cache(profile_id: str):
    # Arrange
    namespace = CANARY_MANIFEST["warmup"]["cache_namespace"]

    # Act
    env = _canary_conf(_canary_profile(profile_id)).env
    paths = {
        name: env[name]
        for name in (
            "SGLANG_CACHE_DIR",
            "SGLANG_JIT_CACHE_DIR",
            "DG_JIT_CACHE_DIR",
            "FLASHINFER_WORKSPACE_BASE",
            "TRITON_CACHE_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
        )
    }

    # Assert
    assert all(path.startswith(f"{namespace}/") for path in paths.values())


def test_mixed_chunk_control_cannot_silently_enable_eagle():
    # Arrange
    profile = _canary_profile("fcfs-noeagle-mixed-c8192")

    # Act
    args = _canary_conf(profile).extra_sglang_args

    # Assert
    assert (
        "--enable-mixed-chunk" in args,
        any(arg.startswith("--speculative-") for arg in args),
    ) == (True, False)


@pytest.mark.parametrize(
    "profile_id",
    [
        "fcfs-eagle-c32768",
        "lpm-eagle-c32768",
        "fcfs-eagle-c8192",
        "fcfs-eagle-c4096",
    ],
)
def test_eagle_profiles_keep_the_exact_production_speculation_tuple(profile_id: str):
    # Arrange
    args = _canary_conf(_canary_profile(profile_id)).extra_sglang_args

    # Act
    observed = tuple(
        _arg_value(args, flag)
        for flag in (
            "--speculative-algorithm",
            "--speculative-num-steps",
            "--speculative-eagle-topk",
            "--speculative-num-draft-tokens",
        )
    )

    # Assert
    assert observed == ("EAGLE", "3", "1", "4")


def test_lpm_negative_control_changes_only_policy_from_baseline():
    # Arrange
    baseline = list(
        _canary_conf(_canary_profile("fcfs-eagle-c32768")).extra_sglang_args
    )
    lpm = list(_canary_conf(_canary_profile("lpm-eagle-c32768")).extra_sglang_args)

    # Act
    baseline[baseline.index("--schedule-policy") + 1] = "POLICY"
    lpm[lpm.index("--schedule-policy") + 1] = "POLICY"

    # Assert
    assert lpm == baseline


def test_litellm_config_names_the_engine_then_the_wildcard():
    # Arrange
    text = render(SETTINGS, CONF, BASE_ENV).litellm_config_text

    # Act
    names = [entry["model_name"] for entry in yaml.safe_load(text)["model_list"]]

    # Assert
    assert names == ["model-a", "*"]


def test_litellm_config_points_every_entry_at_vllm():
    # Arrange
    text = render(SETTINGS, CONF, BASE_ENV).litellm_config_text

    # Act
    bases = {
        entry["litellm_params"]["api_base"]
        for entry in yaml.safe_load(text)["model_list"]
    }

    # Assert
    assert bases == {"http://127.0.0.1:8768/v1"}


def test_litellm_config_carries_the_master_key():
    # Arrange
    text = render(SETTINGS, CONF, BASE_ENV).litellm_config_text

    # Act
    key = yaml.safe_load(text)["general_settings"]["master_key"]

    # Assert
    assert key == "sk-local"


def test_litellm_config_lands_under_base_named_by_key():
    # Arrange
    launch = render(SETTINGS, CONF, BASE_ENV)

    # Act
    path = launch.litellm_config_path

    # Assert
    assert path == Path("/scratch/site/me/litellm-model-a.yaml")


def test_tunnel_forwards_to_vllm_not_litellm():
    # Arrange
    argv = render(SETTINGS, CONF, BASE_ENV).tunnel_argv

    # Act
    forward = argv[argv.index("-R") + 1]

    # Assert
    assert forward == "18773:127.0.0.1:8768"


def test_tunnel_is_never_multiplexed():
    # Arrange
    argv = render(SETTINGS, CONF, BASE_ENV).tunnel_argv

    # Act
    flags = ("ControlMaster=no" in argv, "ControlPath=none" in argv)

    # Assert
    assert flags == (True, True)


def test_tunnel_uses_the_proxy_command_when_configured():
    # Arrange
    argv = render(SETTINGS, CONF, BASE_ENV).tunnel_argv

    # Act
    proxy = [arg for arg in argv if arg.startswith("ProxyCommand=")]

    # Assert
    assert proxy == [
        "ProxyCommand=/home/me/bin/cloudflared access ssh --hostname bastion.example.org"
    ]


def test_tunnel_omits_the_proxy_command_when_not_configured():
    # Arrange
    plain = ServeSettings(
        base=SETTINGS.base,
        logs=SETTINGS.logs,
        cache_root=SETTINGS.cache_root,
        vllm_bin=SETTINGS.vllm_bin,
        litellm_bin=SETTINGS.litellm_bin,
        bastion=SETTINGS.bastion,
        bastion_user=SETTINGS.bastion_user,
        litellm_master_key=SETTINGS.litellm_master_key,
    )

    # Act
    argv = render(plain, CONF, BASE_ENV).tunnel_argv

    # Assert
    assert not any(arg.startswith("ProxyCommand=") for arg in argv)


def test_tunnel_ends_at_the_bastion_as_the_configured_user():
    # Arrange
    argv = render(SETTINGS, CONF, BASE_ENV).tunnel_argv

    # Act
    target = argv[-1]

    # Assert
    assert target == "me@bastion.example.org"


def test_logs_are_named_by_role_and_key_under_the_logs_dir():
    # Arrange
    launch = render(SETTINGS, CONF, BASE_ENV)

    # Act
    names = (launch.vllm_log, launch.litellm_log, launch.tunnel_log)

    # Assert
    assert names == (
        Path("/scratch/site/me/serve-logs/vllm-model-a.log"),
        Path("/scratch/site/me/serve-logs/litellm-model-a.log"),
        Path("/scratch/site/me/serve-logs/tunnel-model-a.log"),
    )


def test_health_url_is_vllm_on_loopback():
    # Arrange
    launch = render(SETTINGS, CONF, BASE_ENV)

    # Act
    url = launch.health_url

    # Assert
    assert url == "http://127.0.0.1:8768/health"
