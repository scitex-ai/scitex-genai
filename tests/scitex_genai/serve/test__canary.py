"""Canary admission resolves observed runtime state before any engine starts."""

from __future__ import annotations

import hashlib
import zlib
from pathlib import Path

from scitex_genai.serve._canary import (
    RuntimeObservation,
    _incarnation_schema,
    publish_runtime_manifest,
    validate_runtime,
)
from scitex_genai.serve._conf import EngineConf


def _raised(call) -> BaseException | None:
    try:
        call()
    except Exception as exc:  # noqa: BLE001 -- tests inspect the exact refusal
        return exc
    return None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _conf(tmp_path: Path) -> EngineConf:
    image = tmp_path / "sglang.sif"
    image.write_bytes(b"pinned-image")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_bytes(b"config")
    (model / "model-1.safetensors").write_bytes(b"checkpoint-shard")
    crc = zlib.crc32(b"checkpoint-shard") & 0xFFFFFFFF
    crc_manifest = f"{crc:08x}  model-1.safetensors\n".encode()
    (model / "crc32.txt").write_bytes(crc_manifest)
    entries = (
        f"{_sha256(b'config')}  config.json\n"
        f"{_sha256(crc_manifest)}  crc32.txt\n"
    )
    return EngineConf(
        key="hicache-canary",
        model_path=model,
        served_name="qwen-canary",
        vllm_port=8878,
        litellm_port=4113,
        tunnel_port=18873,
        max_model_len=1_000_000,
        engine="sglang",
        sglang_image=image,
        tp=2,
        env={"SCITEX_GENAI_EXPECTED_SGLANG_VERSION": "pinned"},
        canary_only=True,
        canary_purpose="qwen38-hicache-l2",
        required_gpu_count=2,
        required_gpu_model="H100",
        required_host_memory_gb=128,
        sglang_image_sha256=_sha256(b"pinned-image"),
        model_manifest_files=("config.json", "crc32.txt"),
        model_manifest_sha256=_sha256(entries.encode()),
    )


def _env() -> dict[str, str]:
    return {
        "SLURM_JOB_ID": "runtime-job",
        "SLURM_STEP_ID": "7",
        "SLURMD_NODENAME": "gpu-node",
        "SLURM_STEP_GPUS": "3,4",
        "CUDA_VISIBLE_DEVICES": "3,4",
        "SCITEX_GENAI_CANARY_PURPOSE": "qwen38-hicache-l2",
    }


def _observation(**changes) -> RuntimeObservation:
    values = {
        "cuda_visible_devices": ("3", "4"),
        "gpu_names": ("NVIDIA H100 80GB HBM3", "NVIDIA H100 80GB HBM3"),
        "active_gpu_processes": (),
        "busy_ports": (),
        "host_memory_gb": 128.0,
    }
    values.update(changes)
    return RuntimeObservation(**values)


def test_valid_runtime_resolves_an_incarnation_manifest(tmp_path: Path):
    # Arrange
    conf = _conf(tmp_path)

    # Act
    manifest = validate_runtime(conf, _env(), _observation())

    # Assert
    assert (
        manifest["incarnation_id"],
        manifest["purpose"],
        manifest["resources"]["gpu_names"],
        manifest["engine"]["model_manifest_sha256"],
    ) == (
        "slurm-runtime-job-step-7-gpu-node",
        "qwen38-hicache-l2",
        ("NVIDIA H100 80GB HBM3", "NVIDIA H100 80GB HBM3"),
        conf.model_manifest_sha256,
    )


def test_runtime_refuses_an_engine_already_using_the_gpus(tmp_path: Path):
    # Arrange
    conf = _conf(tmp_path)
    observed = _observation(active_gpu_processes=("1234, python",))

    # Act
    raised = _raised(lambda: validate_runtime(conf, _env(), observed))

    # Assert
    assert "already have compute processes" in str(raised)


def test_runtime_refuses_busy_engine_or_sidecar_ports(tmp_path: Path):
    # Arrange
    conf = _conf(tmp_path)
    observed = _observation(busy_ports=(8878,))

    # Act
    raised = _raised(lambda: validate_runtime(conf, _env(), observed))

    # Assert
    assert "ports are already in use" in str(raised)


def test_runtime_refuses_insufficient_host_memory(tmp_path: Path):
    # Arrange
    conf = _conf(tmp_path)
    observed = _observation(host_memory_gb=127.9)

    # Act
    raised = _raised(lambda: validate_runtime(conf, _env(), observed))

    # Assert
    assert "requires 128 GB host RAM" in str(raised)


def test_runtime_refuses_the_wrong_gpu_inventory(tmp_path: Path):
    # Arrange
    conf = _conf(tmp_path)
    observed = _observation(gpu_names=("NVIDIA A100", "NVIDIA A100"))

    # Act
    raised = _raised(lambda: validate_runtime(conf, _env(), observed))

    # Assert
    assert "requires H100 GPUs" in str(raised)


def test_runtime_refuses_the_wrong_visible_gpu_count(tmp_path: Path):
    # Arrange
    conf = _conf(tmp_path)
    observed = _observation(cuda_visible_devices=("3",))

    # Act
    raised = _raised(lambda: validate_runtime(conf, _env(), observed))

    # Assert
    assert "requires 2 visible GPUs" in str(raised)


def test_runtime_refuses_a_changed_model_artifact(tmp_path: Path):
    # Arrange
    conf = _conf(tmp_path)
    (conf.model_path / "config.json").write_bytes(b"changed")

    # Act
    raised = _raised(lambda: validate_runtime(conf, _env(), _observation()))

    # Assert
    assert "MODEL_MANIFEST_SHA256 mismatch" in str(raised)


def test_runtime_refuses_a_changed_checkpoint_shard(tmp_path: Path):
    # Arrange
    conf = _conf(tmp_path)
    (conf.model_path / "model-1.safetensors").write_bytes(b"changed shard")

    # Act
    raised = _raised(lambda: validate_runtime(conf, _env(), _observation()))

    # Assert
    assert "checkpoint CRC32 mismatch" in str(raised)


def test_runtime_refuses_hicache_without_host_headroom(tmp_path: Path):
    # Arrange
    conf = _conf(tmp_path)
    object.__setattr__(
        conf,
        "extra_sglang_args",
        ("--enable-hierarchical-cache", "--hicache-size", "49"),
    )

    # Act
    raised = _raised(lambda: validate_runtime(conf, _env(), _observation()))

    # Assert
    assert "at least 32 GB host headroom" in str(raised)


def test_runtime_manifest_is_published_to_a_postgres_store(tmp_path: Path):
    # Arrange
    from scitex_dev.store import Store, StoreTarget, WriterPolicy
    from scitex_dev.store.testing import ephemeral_cluster_dsn, ephemeral_schema

    manifest = validate_runtime(_conf(tmp_path), _env(), _observation())

    # Act
    with ephemeral_cluster_dsn() as dsn, ephemeral_schema(dsn) as scoped:
        target = StoreTarget.postgres(
            scoped, pkg="scitex_genai", name="canary_incarnations"
        )
        destination = publish_runtime_manifest(
            manifest,
            target=target,
        )
        with Store(
            target,
            _incarnation_schema(),
            node="reader",
            writer_policy=WriterPolicy.SINGLE_WRITER,
        ) as store:
            written = store.get({"incarnation_id": manifest["incarnation_id"]})

    # Assert
    assert (
        destination.endswith("#slurm-runtime-job-step-7-gpu-node"),
        written.values["manifest"]["purpose"],
    ) == (True, "qwen38-hicache-l2")
