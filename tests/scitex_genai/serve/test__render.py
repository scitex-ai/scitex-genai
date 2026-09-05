"""The rendered launch is the script's launch, made inspectable -- pure data."""

from __future__ import annotations

from pathlib import Path

import yaml

from scitex_genai.serve._conf import EngineConf
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
BASE_ENV = {
    "PATH": "/usr/bin",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "LD_LIBRARY_PATH": "/lib",
}


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
