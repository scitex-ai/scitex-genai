"""An engine conf is read exactly as the launcher needs it, from a real file.

No mocks (PA-306): every case writes the conf text to ``tmp_path`` or hands it
to the parser directly; refusals are captured by a helper so each test keeps
one assertion.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from scitex_config import get_scitex_dir

from scitex_genai.serve._conf import (
    EngineConf,
    default_models_dir,
    expand,
    exported_names,
    list_engines,
    load_engine,
    parse_engine_conf,
)

CONF = (
    "# a served engine\n"
    "export CUDA_VISIBLE_DEVICES=1\n"
    "export VLLM_USE_DEEP_GEMM=1\n"
    "PRIVATE_NOTE=not-for-the-child\n"
    "MODEL_PATH=/weights/model-a\n"
    "SERVED_NAME=model-a\n"
    "VLLM_PORT=8768\n"
    "LITELLM_PORT=4003\n"
    "TUNNEL_PORT=18773\n"
    "MAX_MODEL_LEN=1048576\n"
    'EXTRA_VLLM_ARGS="--enable-prefix-caching --kv-cache-dtype fp8"\n'
)


def _raised(call) -> BaseException | None:
    try:
        call()
    except Exception as exc:  # noqa: BLE001 -- the test names the type it expects
        return exc
    return None


def _write(directory: Path, key: str, text: str = CONF) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}.conf"
    path.write_text(text)
    return path


def test_required_fields_are_read():
    # Arrange
    text = CONF

    # Act
    conf = parse_engine_conf("model-a", text)

    # Assert
    assert (
        conf.model_path,
        conf.served_name,
        conf.vllm_port,
        conf.litellm_port,
        conf.tunnel_port,
        conf.max_model_len,
    ) == (Path("/weights/model-a"), "model-a", 8768, 4003, 18773, 1048576)


def test_defaults_apply_when_absent():
    # Arrange
    text = CONF

    # Act
    conf = parse_engine_conf("model-a", text)

    # Assert
    assert (conf.tp, conf.gpu_mem_util, conf.max_num_seqs) == (1, 0.92, 8)


def test_extra_args_are_shell_split():
    # Arrange
    text = CONF

    # Act
    conf = parse_engine_conf("model-a", text)

    # Assert
    assert conf.extra_vllm_args == (
        "--enable-prefix-caching",
        "--kv-cache-dtype",
        "fp8",
    )


def test_sglang_engine_reads_image_port_and_quoted_json_argument():
    # Arrange
    text = (
        CONF.replace("VLLM_PORT=8768", "ENGINE=sglang\nENGINE_PORT=8768")
        + "SGLANG_IMAGE=/images/sglang.sif\n"
        + 'EXTRA_SGLANG_ARGS="--json-model-override-args \'{\\"x\\":1}\'"\n'
    )

    # Act
    conf = parse_engine_conf("model-a", text)

    # Assert
    assert (
        conf.engine,
        conf.engine_port,
        conf.sglang_image,
        conf.extra_sglang_args,
    ) == (
        "sglang",
        8768,
        Path("/images/sglang.sif"),
        ("--json-model-override-args", '{"x":1}'),
    )


def test_sglang_requires_a_pinned_image():
    # Arrange
    text = CONF.replace("VLLM_PORT=8768", "ENGINE=sglang\nENGINE_PORT=8768")

    # Act
    raised = _raised(lambda: parse_engine_conf("model-a", text))

    # Assert
    assert "SGLANG_IMAGE" in str(raised)


@pytest.mark.parametrize(
    "addition",
    [
        'EXTRA_SGLANG_ARGS="--disable-radix-cache"\n',
        "export SGLANG_ENABLE_UNIFIED_RADIX_TREE=0\n",
    ],
)
def test_sglang_refuses_configuration_that_disables_session_cache(addition: str):
    # Arrange
    text = (
        CONF.replace("VLLM_PORT=8768", "ENGINE=sglang\nENGINE_PORT=8768")
        + "SGLANG_IMAGE=/images/sglang.sif\n"
        + addition
    )

    # Act
    raised = _raised(lambda: parse_engine_conf("model-a", text))

    # Assert
    assert "session-aware radix" in str(raised)


def test_exported_names_become_the_child_env():
    # Arrange
    text = CONF

    # Act
    conf = parse_engine_conf("model-a", text)

    # Assert
    assert conf.env == {"CUDA_VISIBLE_DEVICES": "1", "VLLM_USE_DEEP_GEMM": "1"}


def test_an_unexported_name_stays_out_of_the_child_env():
    # Arrange
    names = exported_names(CONF)

    # Act
    leaked = "PRIVATE_NOTE" in names

    # Assert
    assert leaked is False


def test_a_missing_required_field_is_refused_by_name():
    # Arrange
    text = CONF.replace("TUNNEL_PORT=18773\n", "")

    # Act
    raised = _raised(lambda: parse_engine_conf("model-a", text))

    # Assert
    assert "TUNNEL_PORT" in str(raised)


@pytest.mark.parametrize(
    "field, value",
    [
        ("vllm_port", 0),
        ("vllm_port", 4003),
        ("tp", 0),
        ("gpu_mem_util", 1.5),
        ("max_model_len", 0),
        ("model_path", Path("relative/weights")),
        ("served_name", "two words"),
    ],
)
def test_an_invalid_value_is_refused(field, value):
    # Arrange
    given = dict(
        key="model-a",
        model_path=Path("/weights/model-a"),
        served_name="model-a",
        vllm_port=8768,
        litellm_port=4003,
        tunnel_port=18773,
        max_model_len=1024,
    )
    given[field] = value

    # Act
    raised = _raised(lambda: EngineConf(**given))

    # Assert
    assert isinstance(raised, ValueError)


def test_load_engine_records_the_file_it_read(tmp_path: Path):
    # Arrange
    path = _write(tmp_path / "models.d", "model-a")

    # Act
    conf = load_engine("model-a", tmp_path / "models.d")

    # Assert
    assert conf.source == path


def test_load_engine_names_the_available_keys_when_one_is_missing(tmp_path: Path):
    # Arrange
    _write(tmp_path / "models.d", "model-a")

    # Act
    raised = _raised(lambda: load_engine("model-z", tmp_path / "models.d"))

    # Assert
    assert "available: model-a" in str(raised)


def test_list_engines_is_sorted_by_key(tmp_path: Path):
    # Arrange
    _write(tmp_path / "models.d", "model-b")
    _write(tmp_path / "models.d", "model-a")

    # Act
    keys = list_engines(tmp_path / "models.d")

    # Assert
    assert keys == ["model-a", "model-b"]


def test_list_engines_on_a_missing_dir_is_empty(tmp_path: Path):
    # Arrange
    directory = tmp_path / "absent"

    # Act
    keys = list_engines(directory)

    # Assert
    assert keys == []


def test_default_models_dir_is_under_the_scitex_dir():
    # Arrange
    scitex_dir = Path(get_scitex_dir())

    # Act
    path = default_models_dir()

    # Assert
    assert path == scitex_dir / "genai" / "models.d"


def test_a_shell_default_expands_to_its_default_when_unset():
    # Arrange
    text = CONF.replace(
        "export VLLM_USE_DEEP_GEMM=1\n",
        "export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-1}\n",
    )

    # Act
    conf = parse_engine_conf("model-a", text, env={})

    # Assert
    assert conf.env["VLLM_USE_DEEP_GEMM"] == "1"


def test_a_shell_default_yields_to_the_environment():
    # Arrange
    text = CONF.replace(
        "export VLLM_USE_DEEP_GEMM=1\n",
        "export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-1}\n",
    )

    # Act
    conf = parse_engine_conf("model-a", text, env={"VLLM_USE_DEEP_GEMM": "0"})

    # Assert
    assert conf.env["VLLM_USE_DEEP_GEMM"] == "0"


def test_expand_handles_bare_and_braced_references():
    # Arrange
    env = {"BASE": "/scratch/site"}

    # Act
    value = expand("$BASE/hf:${BASE}/x:${MISSING:-d}:${GONE}", env)

    # Assert
    assert value == "/scratch/site/hf:/scratch/site/x:d:"


def test_canonical_l2_hicache_profile_is_bounded_per_rank_and_canary_only():
    # Arrange
    root = Path(__file__).parents[3]
    path = root / "examples/serve/qwen38-27b-sglang-hicache-l2-canary.conf"

    # Act
    conf = parse_engine_conf(path.stem, path.read_text(), source=path)

    # Assert
    assert (
        conf.canary_only,
        conf.canary_purpose,
        conf.required_gpu_count,
        conf.required_gpu_model,
        conf.required_host_memory_gb,
        conf.sglang_image_sha256,
        conf.model_manifest_sha256,
        conf.extra_sglang_args[conf.extra_sglang_args.index("--hicache-size") + 1],
        "--hicache-storage-backend" in conf.extra_sglang_args,
    ) == (
        True,
        "qwen38-hicache-l2",
        2,
        "H100",
        128,
        "b742f112f8403417c781216e9d4cf9d7eff49f2d6127e682805c40225a74cab2",
        "ae63fb8baffb044e4d0ee476a03283de640690e50fe3282c95996d9dc016a01c",
        "32",
        False,
    )


def test_canonical_l3_hicache_profile_has_versioned_bounded_storage():
    # Arrange
    root = Path(__file__).parents[3]
    path = root / "examples/serve/qwen38-27b-sglang-hicache-l3-canary.conf"

    # Act
    conf = parse_engine_conf(path.stem, path.read_text(), source=path)

    # Assert
    assert (
        conf.env["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"],
        conf.extra_sglang_args[conf.extra_sglang_args.index("--page-size") + 1],
        conf.extra_sglang_args[
            conf.extra_sglang_args.index("--hicache-storage-backend-extra-config") + 1
        ],
    ) == (
        "/tmp/hicache-qwen38-4ccff141-tp2-fp8-yarn4-p64-model-ae63fb8baffb044e4d0ee476a03283de640690e50fe3282c95996d9dc016a01c",
        "64",
        '{"max_size":"32G","min_free_space":"100G","eviction_ratio":0.9,"enable_metadata_cache":true,"metadata_ttl":5}',
    )


def test_checked_in_model_manifest_matches_the_profile_digest():
    # Arrange
    root = Path(__file__).parents[3]
    manifest = root / "examples/serve/manifests/qwen38-27b-fp8.sha256"
    profile = root / "examples/serve/qwen38-27b-sglang-hicache-l2-canary.conf"

    # Act
    actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
    expected = parse_engine_conf(
        profile.stem, profile.read_text()
    ).model_manifest_sha256

    # Assert
    assert actual == expected


def test_hicache_refuses_write_back_for_hybrid_state_safety():
    # Arrange
    root = Path(__file__).parents[3]
    path = root / "examples/serve/qwen38-27b-sglang-hicache-l2-canary.conf"
    text = path.read_text().replace("write_through", "write_back")

    # Act
    raised = _raised(lambda: parse_engine_conf(path.stem, text, source=path))

    # Assert
    assert "write_through" in str(raised)


def test_canary_flag_refuses_a_value_that_could_disable_the_guard_by_typo():
    # Arrange
    text = CONF + "CANARY_ONLY=true\n"

    # Act
    raised = _raised(lambda: parse_engine_conf("model-a", text))

    # Assert
    assert "CANARY_ONLY must be 0 or 1" in str(raised)


def test_file_hicache_refuses_an_unbounded_storage_configuration():
    # Arrange
    root = Path(__file__).parents[3]
    path = root / "examples/serve/qwen38-27b-sglang-hicache-l3-canary.conf"
    text = path.read_text().replace("min_free_space", "missing_free_space")

    # Act
    raised = _raised(lambda: parse_engine_conf(path.stem, text, source=path))

    # Assert
    assert "min_free_space" in str(raised)


@pytest.mark.parametrize(
    "name,original,value",
    [
        ("max_size", "32G", "0G"),
        ("max_size", "32G", "many"),
        ("min_free_space", "100G", "-1G"),
        ("min_free_space", "100G", "unknown"),
    ],
)
def test_file_hicache_refuses_non_positive_or_unparseable_storage_sizes(
    name: str, original: str, value: str
):
    # Arrange
    root = Path(__file__).parents[3]
    path = root / "examples/serve/qwen38-27b-sglang-hicache-l3-canary.conf"
    text = path.read_text().replace(
        rf"\"{name}\":\"{original}\"", rf"\"{name}\":\"{value}\""
    )

    # Act
    raised = _raised(lambda: parse_engine_conf(path.stem, text, source=path))

    # Assert
    assert f"{name} must be a positive storage size" in str(raised)


@pytest.mark.parametrize("value", ["zero", "0", "-1"])
def test_hicache_refuses_a_non_positive_or_unparseable_size(value: str):
    # Arrange
    root = Path(__file__).parents[3]
    path = root / "examples/serve/qwen38-27b-sglang-hicache-l2-canary.conf"
    text = path.read_text().replace("--hicache-size 32", f"--hicache-size {value}")

    # Act
    raised = _raised(lambda: parse_engine_conf(path.stem, text, source=path))

    # Assert
    assert "positive --hicache-size" in str(raised)


@pytest.mark.parametrize("value", ["0", "1.1", '"bad"'])
def test_file_hicache_refuses_an_invalid_eviction_ratio(value: str):
    # Arrange
    root = Path(__file__).parents[3]
    path = root / "examples/serve/qwen38-27b-sglang-hicache-l3-canary.conf"
    text = path.read_text().replace(
        r"\"eviction_ratio\":0.9", rf"\"eviction_ratio\":{value}"
    )

    # Act
    raised = _raised(lambda: parse_engine_conf(path.stem, text, source=path))

    # Assert
    assert "eviction_ratio must be within" in str(raised)
