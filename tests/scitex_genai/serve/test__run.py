"""The runner drives the script's three loops -- proven with process doubles on real dirs.

No mocks (PA-306): ``popen``, ``sleep``, ``clock``, ``http_get`` and ``log``
are the runner's own parameters; the doubles below record what they were
asked and answer deterministically. Every path is under ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

from scitex_genai.serve._conf import EngineConf
from scitex_genai.serve._ready import FAILED, READY
from scitex_genai.serve._render import render
from scitex_genai.serve._run import EngineRunner
from scitex_genai.serve._settings import ServeSettings


class _Proc:
    """A child that is alive for ``alive_polls`` polls, then exits with ``rc``."""

    def __init__(self, alive_polls: int, rc: int = 0) -> None:
        self.alive_polls = alive_polls
        self.rc = rc
        self.polls = 0

    def poll(self):
        self.polls += 1
        return None if self.polls <= self.alive_polls else self.rc

    def wait(self) -> int:
        return self.rc


class _Spawner:
    """Records every spawn; hands out engine procs per the script, others exit at once."""

    def __init__(self, engine_alive_polls: int = 10**6) -> None:
        self.calls: list[tuple[list[str], dict[str, str]]] = []
        self.engine_alive_polls = engine_alive_polls

    def __call__(self, argv, stdout, stderr, env):
        self.calls.append((list(argv), dict(env)))
        if argv[1:2] == ["serve"]:
            return _Proc(self.engine_alive_polls)
        return _Proc(0)

    def argvs(self, word: str) -> list[list[str]]:
        return [argv for argv, _ in self.calls if word in argv[0]]


def _launch(tmp_path: Path):
    settings = ServeSettings(
        base=tmp_path / "base",
        logs=tmp_path / "logs",
        cache_root=tmp_path / "cache",
        vllm_bin=tmp_path / "venv" / "bin" / "vllm",
        litellm_bin=tmp_path / "venv" / "bin" / "litellm",
        bastion="bastion.example.org",
        bastion_user="me",
        litellm_master_key="sk-local",
    )
    conf = EngineConf(
        key="model-a",
        model_path=tmp_path / "weights",
        served_name="model-a",
        vllm_port=8768,
        litellm_port=4003,
        tunnel_port=18773,
        max_model_len=1024,
    )
    return render(
        settings, conf, {"PATH": "/usr/bin", "HOME": str(tmp_path / "realhome")}
    )


def _runner(
    tmp_path: Path,
    spawner: _Spawner,
    *,
    health_after: int = 1,
    stop_after_starts: int = 1,
):
    lines: list[str] = []
    probes = {"n": 0}

    def http_get(url: str) -> bool:
        probes["n"] += 1
        return probes["n"] > health_after

    runner = EngineRunner(
        _launch(tmp_path),
        popen=spawner,
        sleep=lambda s: (
            runner.stop() if runner.engine_starts >= stop_after_starts else None
        ),
        clock=lambda: float(probes["n"]),
        http_get=http_get,
        log=lines.append,
        idle_limit_s=3,
        poll_s=1,
    )
    runner.prepare()
    return runner, lines


def test_prepare_writes_the_sidecar_config(tmp_path: Path):
    # Arrange
    launch = _launch(tmp_path)
    runner = EngineRunner(launch, popen=_Spawner())

    # Act
    runner.prepare()

    # Assert
    assert Path(launch.litellm_config_path).read_text() == launch.litellm_config_text


def test_prepare_creates_every_cache_directory(tmp_path: Path):
    # Arrange
    launch = _launch(tmp_path)
    runner = EngineRunner(launch, popen=_Spawner())

    # Act
    runner.prepare()

    # Assert
    assert all(
        Path(launch.env[name]).is_dir()
        for name in ("HF_HOME", "VLLM_CACHE_ROOT", "FLASHINFER_WORKSPACE_BASE")
    )


def test_run_once_is_ready_when_health_answers(tmp_path: Path):
    # Arrange
    runner, _ = _runner(tmp_path, _Spawner())

    # Act
    readiness = runner.run_once()

    # Assert
    assert readiness.state == READY


def test_the_engine_runs_under_the_engine_env(tmp_path: Path):
    # Arrange
    spawner = _Spawner()
    runner, _ = _runner(tmp_path, spawner)

    # Act
    runner.run_once()
    engine_env = next(env for argv, env in spawner.calls if argv[1:2] == ["serve"])

    # Assert
    assert engine_env["HOME"] == str(tmp_path / "cache" / "model-a-cache" / "home")


def test_the_tunnel_opens_on_first_health_under_the_real_home(tmp_path: Path):
    # Arrange
    spawner = _Spawner()
    runner, _ = _runner(tmp_path, spawner, health_after=0)

    # Act
    runner.run_once()
    runner._tunnel_thread.join(timeout=5)
    tunnel_env = next(env for argv, env in spawner.calls if argv[0] == "ssh")

    # Assert
    assert tunnel_env["HOME"] == str(tmp_path / "realhome")


def test_a_ready_engine_is_announced(tmp_path: Path):
    # Arrange
    runner, lines = _runner(tmp_path, _Spawner())

    # Act
    runner.run_once()

    # Assert
    assert any(line.startswith("[serve] === READY: model-a") for line in lines)


def test_an_engine_that_exits_is_failed_and_opens_no_tunnel(tmp_path: Path):
    # Arrange
    runner, _ = _runner(tmp_path, _Spawner(engine_alive_polls=0), health_after=10**6)

    # Act
    readiness = runner.run_once()

    # Assert
    assert (readiness.state, runner.tunnel_started) == (FAILED, False)


def test_run_forever_restarts_the_engine_until_stopped(tmp_path: Path):
    # Arrange
    spawner = _Spawner()
    runner, _ = _runner(tmp_path, spawner, stop_after_starts=2)

    # Act
    runner.run_forever()

    # Assert
    assert runner.engine_starts == 2


def test_run_forever_starts_the_sidecar(tmp_path: Path):
    # Arrange
    spawner = _Spawner()
    runner, _ = _runner(tmp_path, spawner, stop_after_starts=1)

    # Act
    runner.run_forever()
    runner._sidecar_thread.join(timeout=5)

    # Assert
    assert spawner.argvs("litellm")[0][:2] == [
        str(tmp_path / "venv" / "bin" / "litellm"),
        "--config",
    ]
