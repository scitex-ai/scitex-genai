"""Run ONE rendered launch and hold it: the engine, its sidecar, its tunnel.

This is the supervising half of the captured script, with the same three
loops -- restart vLLM after it exits, restart LiteLLM every time it exits,
re-establish the reverse tunnel every time it drops -- and one difference
that matters: readiness is progress-bounded (``_ready``), so a long JIT no
longer strands a healthy engine with no tunnel.

The tunnel is started ONCE, the first time the engine answers ``/health``,
and then supervised independently of engine restarts: an engine restart
keeps the forward, so the fleet side sees a short refusal, not a dead port.

Process creation, sleeping and the clock are injected so the loops can be
driven in tests with hand-written doubles on real temporary directories.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ._ready import READY, Readiness, probe_health, wait_ready
from ._render import CACHE_SUBDIRS, Launch

Popen = Callable[..., Any]
Log = Callable[[str], None]


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class EngineRunner:
    """Hold one engine for the life of the allocation."""

    def __init__(
        self,
        launch: Launch,
        *,
        popen: Popen = subprocess.Popen,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        http_get: Callable[[str], bool] = probe_health,
        log: Log = print,
        engine_restart_s: float = 15.0,
        sidecar_restart_s: float = 10.0,
        tunnel_retry_s: float = 20.0,
        idle_limit_s: float = 1800.0,
        poll_s: float = 15.0,
    ) -> None:
        self.launch = launch
        self._popen = popen
        self._sleep = sleep
        self._clock = clock
        self._http_get = http_get
        self._log = log
        self.engine_restart_s = engine_restart_s
        self.sidecar_restart_s = sidecar_restart_s
        self.tunnel_retry_s = tunnel_retry_s
        self.idle_limit_s = idle_limit_s
        self.poll_s = poll_s
        self._stop = threading.Event()
        self._tunnel_thread: threading.Thread | None = None
        self._sidecar_thread: threading.Thread | None = None
        self.readiness: Readiness | None = None
        self.engine_starts = 0

    # -- preparation ---------------------------------------------------------
    def prepare(self) -> None:
        """Directories the launch expects, and the sidecar config, written fresh."""
        for name in CACHE_SUBDIRS:
            Path(self.launch.env[name]).mkdir(parents=True, exist_ok=True)
        Path(self.launch.env["HOME"], ".cache", "flashinfer").mkdir(
            parents=True, exist_ok=True
        )
        for log_path in (
            self.launch.vllm_log,
            self.launch.litellm_log,
            self.launch.tunnel_log,
        ):
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.launch.litellm_config_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.launch.litellm_config_path).write_text(
            self.launch.litellm_config_text
        )

    # -- the three loops -----------------------------------------------------
    def _spawn(self, argv: tuple[str, ...], log_path: Path, env: dict[str, str]) -> Any:
        handle = open(log_path, "ab")  # noqa: SIM115 -- the child owns it until it exits
        try:
            return self._popen(
                list(argv), stdout=handle, stderr=subprocess.STDOUT, env=env
            )
        finally:
            handle.close()

    def _sidecar_loop(self) -> None:
        while not self._stop.is_set():
            proc = self._spawn(
                self.launch.litellm_argv, self.launch.litellm_log, self.launch.env
            )
            proc.wait()
            self._append(
                self.launch.litellm_log,
                f"[{_stamp()}] litellm exited; restart {self.sidecar_restart_s:.0f}s",
            )
            self._sleep(self.sidecar_restart_s)

    def _tunnel_loop(self) -> None:
        while not self._stop.is_set():
            self._append(
                self.launch.tunnel_log,
                f"[{_stamp()}] establishing {self.launch.tunnel_argv[-2]}",
            )
            proc = self._spawn(
                self.launch.tunnel_argv, self.launch.tunnel_log, self.launch.tunnel_env
            )
            rc = proc.wait()
            if rc == 0:
                self._append(
                    self.launch.tunnel_log,
                    f"[{_stamp()}] WARNING: ssh -N returned 0 immediately; multiplexing may be active",
                )
            self._append(
                self.launch.tunnel_log,
                f"[{_stamp()}] tunnel exited rc={rc}; retry in {self.tunnel_retry_s:.0f}s",
            )
            self._sleep(self.tunnel_retry_s)

    def start_sidecar(self) -> threading.Thread:
        thread = threading.Thread(
            target=self._sidecar_loop, name=f"litellm-{self.launch.key}", daemon=True
        )
        thread.start()
        self._sidecar_thread = thread
        return thread

    def start_tunnel(self) -> threading.Thread:
        thread = threading.Thread(
            target=self._tunnel_loop, name=f"tunnel-{self.launch.key}", daemon=True
        )
        thread.start()
        self._tunnel_thread = thread
        return thread

    @property
    def tunnel_started(self) -> bool:
        return self._tunnel_thread is not None

    # -- the engine ------------------------------------------------------------
    def run_once(self) -> Readiness:
        """Start the engine, wait for it, open the tunnel on first health, hold until it exits."""
        self.engine_starts += 1
        self._append(
            self.launch.vllm_log, f"[{_stamp()}] starting engine {self.launch.key}"
        )
        proc = self._spawn(self.launch.vllm_argv, self.launch.vllm_log, self.launch.env)
        readiness = wait_ready(
            self.launch.health_url,
            process_alive=lambda: proc.poll() is None,
            cache_dir=self.launch.cache_dir,
            idle_limit_s=self.idle_limit_s,
            poll_s=self.poll_s,
            http_get=self._http_get,
            clock=self._clock,
            sleep=self._sleep,
        )
        self.readiness = readiness
        if readiness.state == READY:
            if not self.tunnel_started:
                self.start_tunnel()
            self._log(
                f"[serve] === READY: {self.launch.key} {self.launch.health_url} after "
                f"{readiness.waited_s:.0f}s; tunnel {self.launch.tunnel_argv[-2]} -- holding ==="
            )
        else:
            self._log(f"[serve] engine {self.launch.key} not ready: {readiness.reason}")
        proc.wait()
        self._append(
            self.launch.vllm_log,
            f"[{_stamp()}] engine exited; restart {self.engine_restart_s:.0f}s",
        )
        return readiness

    def run_forever(self) -> None:
        """The script's outer loop: prepare, start the sidecar, restart the engine until stopped."""
        self.prepare()
        self.start_sidecar()
        while not self._stop.is_set():
            self.run_once()
            if self._stop.is_set():
                break
            self._sleep(self.engine_restart_s)

    def stop(self) -> None:
        self._stop.set()

    # -- helpers -----------------------------------------------------------------
    @staticmethod
    def _append(path: Path, line: str) -> None:
        with open(path, "a") as handle:
            handle.write(line + "\n")
