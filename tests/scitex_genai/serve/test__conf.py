"""An engine conf is read exactly as the launcher needs it, from a real file.

No mocks (PA-306): every case writes the conf text to ``tmp_path`` or hands it
to the parser directly; refusals are captured by a helper so each test keeps
one assertion.
"""

from __future__ import annotations

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
