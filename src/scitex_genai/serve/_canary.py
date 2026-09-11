"""Resolve and validate one isolated canary incarnation before GPU launch."""

from __future__ import annotations

import hashlib
import socket
import subprocess
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

from ._conf import EngineConf


@dataclass(frozen=True)
class RuntimeObservation:
    cuda_visible_devices: tuple[str, ...]
    gpu_names: tuple[str, ...]
    active_gpu_processes: tuple[str, ...]
    busy_ports: tuple[int, ...]
    host_memory_gb: float


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_manifest(conf: EngineConf) -> tuple[str, dict[str, str]]:
    entries: dict[str, str] = {}
    for name in conf.model_manifest_files:
        entries[name] = _file_sha256(conf.model_path / name)
    canonical = "".join(f"{digest}  {name}\n" for name, digest in entries.items())
    _verify_checkpoint_crc32(conf.model_path)
    return hashlib.sha256(canonical.encode()).hexdigest(), entries


def _verify_checkpoint_crc32(model_path: Path) -> None:
    manifest = model_path / "crc32.txt"
    expected: dict[str, int] = {}
    for number, raw in enumerate(manifest.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        parts = raw.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"crc32.txt:{number}: expected '<crc32> <path>'")
        digest, name = parts
        relative = Path(name.strip())
        if (
            len(digest) != 8
            or any(ch not in "0123456789abcdefABCDEF" for ch in digest)
            or relative.is_absolute()
            or ".." in relative.parts
            or str(relative) in expected
        ):
            raise ValueError(f"crc32.txt:{number}: invalid or duplicate entry")
        expected[str(relative)] = int(digest, 16)
    shards = {path.name for path in model_path.glob("*.safetensors")}
    declared_shards = {name for name in expected if name.endswith(".safetensors")}
    if not shards or declared_shards != shards:
        raise ValueError("crc32.txt must name every safetensors shard exactly once")
    for name in sorted(declared_shards):
        actual = 0
        with (model_path / name).open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                actual = zlib.crc32(chunk, actual)
        if actual & 0xFFFFFFFF != expected[name]:
            raise ValueError(f"checkpoint CRC32 mismatch: {name}")


def _memory_gb(env: dict[str, str]) -> float:
    raw = env.get("SLURM_MEM_PER_NODE", "")
    if raw.isdigit():
        return int(raw) / 1024
    suffix = raw[-1:].upper()
    try:
        value = float(raw[:-1])
    except ValueError as exc:
        raise ValueError("SLURM_MEM_PER_NODE is missing or invalid") from exc
    factors = {"K": 1 / (1024 * 1024), "M": 1 / 1024, "G": 1, "T": 1024}
    if suffix not in factors:
        raise ValueError("SLURM_MEM_PER_NODE is missing or invalid")
    return value * factors[suffix]


def _gpu_ids(env: dict[str, str]) -> tuple[str, ...]:
    allocated = env.get("SLURM_STEP_GPUS") or env.get("SLURM_JOB_GPUS")
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if not allocated:
        raise ValueError("SLURM_STEP_GPUS is unset; start through the documented srun")
    if not visible:
        raise ValueError("CUDA_VISIBLE_DEVICES is unset; inherit it from srun")
    ids = tuple(part.strip() for part in visible.split(",") if part.strip())
    if not ids:
        raise ValueError("CUDA_VISIBLE_DEVICES does not identify any GPUs")
    return ids


def _nvidia_query(ids: tuple[str, ...], query: str) -> tuple[str, ...]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--id",
            ",".join(ids),
            f"--query-{query}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(line.strip() for line in proc.stdout.splitlines() if line.strip())


def _port_is_busy(port: int) -> bool:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return True
    return False


def observe_runtime(conf: EngineConf, env: dict[str, str]) -> RuntimeObservation:
    ids = _gpu_ids(env)
    try:
        names = _nvidia_query(ids, "gpu=name")
        processes = _nvidia_query(ids, "compute-apps=pid,process_name")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            f"cannot inspect allocated GPUs with nvidia-smi: {exc}"
        ) from exc
    ports = (conf.engine_port, conf.litellm_port, conf.tunnel_port)
    return RuntimeObservation(
        cuda_visible_devices=ids,
        gpu_names=names,
        active_gpu_processes=processes,
        busy_ports=tuple(port for port in ports if _port_is_busy(port)),
        host_memory_gb=_memory_gb(env),
    )


def validate_runtime(
    conf: EngineConf,
    env: dict[str, str],
    observation: RuntimeObservation | None = None,
) -> dict[str, object] | None:
    """Return the resolved incarnation manifest, or refuse an unsafe launch."""
    if not conf.canary_only:
        return None
    job_id = env.get("SLURM_JOB_ID")
    step_id = env.get("SLURM_STEP_ID")
    node = env.get("SLURMD_NODENAME") or env.get("HOSTNAME")
    if not job_id or step_id is None or not node:
        raise ValueError("canary launch requires SLURM_JOB_ID, SLURM_STEP_ID, and node")
    if env.get("SCITEX_GENAI_CANARY_PURPOSE") != conf.canary_purpose:
        raise ValueError(
            f"set SCITEX_GENAI_CANARY_PURPOSE={conf.canary_purpose} on the srun step"
        )
    observed = observation or observe_runtime(conf, env)
    assert conf.required_gpu_count is not None
    assert conf.required_gpu_model is not None
    assert conf.required_host_memory_gb is not None
    if len(observed.cuda_visible_devices) != conf.required_gpu_count:
        raise ValueError(
            f"canary requires {conf.required_gpu_count} visible GPUs; "
            f"CUDA_VISIBLE_DEVICES exposes {observed.cuda_visible_devices}"
        )
    if len(observed.gpu_names) != conf.required_gpu_count:
        raise ValueError(
            f"canary requires {conf.required_gpu_count} GPUs; observed {len(observed.gpu_names)}"
        )
    if any(
        conf.required_gpu_model.lower() not in name.lower()
        for name in observed.gpu_names
    ):
        raise ValueError(
            f"canary requires {conf.required_gpu_model} GPUs; observed {observed.gpu_names}"
        )
    if observed.host_memory_gb < conf.required_host_memory_gb:
        raise ValueError(
            f"canary requires {conf.required_host_memory_gb} GB host RAM; "
            f"observed {observed.host_memory_gb:g} GB"
        )
    hicache_gb = conf.hicache_size_gb_per_rank * conf.tp
    available_gb = min(observed.host_memory_gb, conf.required_host_memory_gb)
    if hicache_gb + 32 > available_gb:
        raise ValueError(
            f"HiCache requests {hicache_gb:g} GB across TP={conf.tp}; "
            f"at least 32 GB host headroom is required within {available_gb:g} GB"
        )
    if observed.active_gpu_processes:
        raise ValueError(
            "allocated GPUs already have compute processes: "
            + ", ".join(observed.active_gpu_processes)
        )
    if observed.busy_ports:
        raise ValueError(f"canary ports are already in use: {observed.busy_ports}")
    assert conf.sglang_image is not None
    assert conf.sglang_image_sha256 is not None
    assert conf.model_manifest_sha256 is not None
    try:
        image_digest = _file_sha256(conf.sglang_image)
        model_digest, model_files = _model_manifest(conf)
    except OSError as exc:
        raise ValueError(f"cannot hash immutable canary artifact: {exc}") from exc
    if image_digest != conf.sglang_image_sha256:
        raise ValueError("SGLANG_IMAGE_SHA256 mismatch")
    if model_digest != conf.model_manifest_sha256:
        raise ValueError("MODEL_MANIFEST_SHA256 mismatch")
    incarnation_id = f"slurm-{job_id}-step-{step_id}-{node}"
    return {
        "incarnation_id": incarnation_id,
        "purpose": conf.canary_purpose,
        "slurm": {"job_id": job_id, "step_id": step_id, "node": node},
        "resources": asdict(observed),
        "engine": {
            "key": conf.key,
            "tp": conf.tp,
            "max_model_len": conf.max_model_len,
            "extra_sglang_args": conf.extra_sglang_args,
            "image_sha256": image_digest,
            "model_manifest_sha256": model_digest,
            "model_files": model_files,
        },
    }


def _incarnation_schema():
    from scitex_dev.store import (
        FieldKind,
        FieldPolicy,
        FieldRole,
        MergeRule,
        Schema,
    )

    def field(kind: FieldKind, role: FieldRole) -> FieldPolicy:
        return FieldPolicy(
            kind=kind,
            role=role,
            required=True,
            merge=MergeRule.IMMUTABLE,
            indexed=role is FieldRole.IDENTITY,
        )

    return Schema.build(
        "genai_canary_incarnations",
        {
            "incarnation_id": field(FieldKind.TEXT, FieldRole.IDENTITY),
            "purpose": field(FieldKind.TEXT, FieldRole.DATA),
            "node": field(FieldKind.TEXT, FieldRole.DATA),
            "manifest": field(FieldKind.JSON, FieldRole.DATA),
        },
    )


def publish_runtime_manifest(
    manifest: dict[str, object], *, target=None
) -> str:
    """Publish one immutable incarnation to the fleet's PostgreSQL store."""
    from scitex_dev.store import NEW_RECORD, Store, WriterPolicy, host_store

    node = str(manifest["slurm"]["node"])
    resolved = target or host_store(pkg="scitex_genai", name="canary_incarnations")
    with Store(
        resolved,
        _incarnation_schema(),
        node=node,
        writer_policy=WriterPolicy.SINGLE_WRITER,
        actor="scitex_genai.serve",
    ) as store:
        store.put(
            {
                "incarnation_id": manifest["incarnation_id"],
                "purpose": manifest["purpose"],
                "node": node,
                "manifest": manifest,
            },
            expected_revision=NEW_RECORD,
        )
    return f"{resolved.describe()}#{manifest['incarnation_id']}"
