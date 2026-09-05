"""The hold body is serve-pair.sh's job as data; booking goes through scitex-hpc.

No mocks (PA-306): the body is asserted as text and by running it through a
real ``bash -n`` parse; booking is exercised with a recording ``book``
callable, the function's own dependency parameter.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scitex_genai.serve._launch import book_serve_lease, render_hold_body
from scitex_genai.serve._settings import ServeSettings

CONF = (
    "MODEL_PATH=/weights/m\nSERVED_NAME=m\nVLLM_PORT=8768\nLITELLM_PORT=4003\n"
    "TUNNEL_PORT=18773\nMAX_MODEL_LEN=1024\n"
)


def _settings(srun: Path | None = Path("/apps/slurm/latest/bin/srun")) -> ServeSettings:
    return ServeSettings(
        base=Path("/scratch/site/me"),
        logs=Path("/scratch/site/me/serve-logs"),
        cache_root=Path("/tmp"),
        vllm_bin=Path("/scratch/site/me/vllm-venv/bin/vllm"),
        litellm_bin=Path("/scratch/site/me/vllm-venv/bin/litellm"),
        bastion="bastion.example.org",
        bastion_user="me",
        litellm_master_key="sk-local",
        serve_bin=Path("/scratch/site/me/vllm-venv/bin/scitex-genai-serve"),
        srun_bin=srun,
    )


def _models_dir(tmp_path: Path, *keys: str) -> Path:
    d = tmp_path / "models.d"
    d.mkdir()
    for key in keys:
        (d / f"{key}.conf").write_text(CONF)
    return d


def _raised(call) -> BaseException | None:
    try:
        call()
    except Exception as exc:  # noqa: BLE001 -- the test names the type it expects
        return exc
    return None


def test_one_backgrounded_step_per_engine():
    # Arrange
    keys = ["engine-a", "engine-b"]

    # Act
    body = render_hold_body(keys, _settings())

    # Assert
    assert [line for line in body.splitlines() if line.startswith("_step ")] == [
        "_step engine-a &",
        "_step engine-b &",
    ]


def test_the_body_ends_by_waiting_for_the_steps():
    # Arrange
    keys = ["engine-a"]

    # Act
    body = render_hold_body(keys, _settings())

    # Assert
    assert body.rstrip().splitlines()[-1] == "wait"


def test_each_step_is_an_overlapping_exact_srun_of_the_serve_command():
    # Arrange
    body = render_hold_body(["engine-a"], _settings())

    # Act
    step_line = next(line for line in body.splitlines() if "--overlap" in line)

    # Assert
    assert (
        step_line.strip()
        == '"$SRUN" --overlap --ntasks=1 --exact bash -c "$SERVE $key"'
    )


def test_the_serve_command_is_the_absolute_console_script():
    # Arrange
    body = render_hold_body(["engine-a"], _settings())

    # Act
    serve_line = next(line for line in body.splitlines() if line.startswith("SERVE="))

    # Assert
    assert serve_line == "SERVE=/scratch/site/me/vllm-venv/bin/scitex-genai-serve"


def test_an_absolute_srun_is_pinned_and_guarded():
    # Arrange
    body = render_hold_body(["engine-a"], _settings())

    # Act
    lines = body.splitlines()

    # Assert
    assert (lines[2], lines[3].startswith('[ -x "$SRUN" ]')) == (
        "SRUN=/apps/slurm/latest/bin/srun",
        True,
    )


def test_without_a_pinned_srun_the_body_checks_path():
    # Arrange
    body = render_hold_body(["engine-a"], _settings(srun=None))

    # Act
    lines = body.splitlines()

    # Assert
    assert (lines[2], lines[3].startswith('command -v "$SRUN"')) == ("SRUN=srun", True)


def test_rc_127_aborts_instead_of_retrying():
    # Arrange
    body = render_hold_body(["engine-a"], _settings())

    # Act
    abort = [line for line in body.splitlines() if "exit 127" in line]

    # Assert
    assert abort == ["      exit 127"]


def test_the_body_installs_no_signal_trap():
    # Arrange
    body = render_hold_body(["engine-a", "engine-b"], _settings())

    # Act
    traps = [line for line in body.splitlines() if line.strip().startswith("trap ")]

    # Assert
    assert traps == []


def test_the_body_parses_as_bash():
    # Arrange
    body = render_hold_body(["engine-a", "engine-b"], _settings())

    # Act
    rc = subprocess.run(
        ["bash", "-n"], input=body, text=True, capture_output=True
    ).returncode

    # Assert
    assert rc == 0


def test_config_and_models_dir_are_passed_to_every_step(tmp_path: Path):
    # Arrange
    models = _models_dir(tmp_path, "engine-a")

    # Act
    body = render_hold_body(
        ["engine-a"], _settings(), models_dir=models, config_path=tmp_path / "c.yaml"
    )
    serve_line = next(line for line in body.splitlines() if line.startswith("SERVE="))

    # Assert
    assert serve_line == (
        f"SERVE='/scratch/site/me/vllm-venv/bin/scitex-genai-serve --config {tmp_path / 'c.yaml'} --models-dir {models}'"
    )


def test_an_unknown_key_is_refused_naming_the_available(tmp_path: Path):
    # Arrange
    models = _models_dir(tmp_path, "engine-a")

    # Act
    raised = _raised(
        lambda: render_hold_body(["engine-z"], _settings(), models_dir=models)
    )

    # Assert
    assert "available: engine-a" in str(raised)


@pytest.mark.parametrize("keys", [[], ["a", "a"], ["a b"], ["x/y"]])
def test_bad_key_lists_are_refused(keys):
    # Arrange
    settings = _settings()

    # Act
    raised = _raised(lambda: render_hold_body(keys, settings))

    # Assert
    assert isinstance(raised, ValueError)


def test_booking_is_persistent_with_the_rendered_body():
    # Arrange
    calls: list[tuple] = []

    def book(config, **kw):
        calls.append((config, kw))
        return "lease"

    # Act
    book_serve_lease(
        ["engine-a", "engine-b"],
        _settings(),
        name="h100-pair",
        project="serve",
        host="spartan",
        gpus="2",
        partition="gpu-h100",
        time="7-00:00:00",
        book=book,
    )

    # Assert
    assert calls[0][1] == {
        "persistent": True,
        "hold_body": render_hold_body(["engine-a", "engine-b"], _settings()),
    }


def test_booking_shapes_the_job_config():
    # Arrange
    calls: list[tuple] = []

    def book(config, **kw):
        calls.append((config, kw))
        return "lease"

    # Act
    book_serve_lease(
        ["engine-a"],
        _settings(),
        name="h100-pair",
        project="serve",
        host="spartan",
        gpus="2",
        partition="gpu-h100",
        time="7-00:00:00",
        book=book,
    )
    config = calls[0][0]

    # Assert
    assert (
        config.project,
        config.host,
        config.partition,
        config.time,
        config.job_name,
        config.extra_sbatch_args,
    ) == (
        "serve",
        "spartan",
        "gpu-h100",
        "7-00:00:00",
        "h100-pair",
        ["--gpus=2"],
    )
