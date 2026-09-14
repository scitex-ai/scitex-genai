"""Ship the gateway's systemd user unit from the package, not from a hand.

WHY. A relay of this kind ran for weeks from a script outside any package,
supervised by a unit somebody typed into ``~/.config/systemd/user``, with its
upstream list kept current by a hand-written watcher that expired one morning
and told nobody. None of it could be reproduced on the next host by anyone but
its author, and a step only one person can perform is a step that does not
exist. So the unit is rendered HERE, from the installed package, by one command
anyone can run on any host::

    scitex-genai-gateway install-unit

With no flags the unit runs the gateway exactly as ``scitex-genai-gateway``
would: settings come from ``~/.scitex/genai/config.yaml`` (see ``_settings``),
so the unit is byte-identical on every host and a settings change needs only a
restart. Flags given to ``install-unit`` are baked into the unit instead.

NO SHELL, AND WHY THAT CHANGED
------------------------------
This unit used to run ``ExecStart=/bin/bash -lc 'exec ...'`` -- a LOGIN shell --
for one reason, stated here at the time: the gateway refuses to start without
``SCITEX_GENAI_GATEWAY_API_KEY``, and the value lived in the user's profile as
an ``export NAME=value`` line that systemd's ``EnvironmentFile=`` cannot read.

That reason is gone. ``_secrets`` gives the key a home any process can read, and
``install-unit`` captures the current value into it before writing this unit, so
the key survives the change rather than being regenerated. The login shell is
therefore removed in the SAME change that removes its cause: a retired mechanism
left reachable is one that will keep being used.

What it bought us, besides one less moving part: a profile that fails, hangs, or
prompts no longer decides whether the gateway starts, and the unit no longer
depends on whatever the login shell puts on PATH.

``<interpreter> -m scitex_genai.gateway._cli`` rather than the console script:
the interpreter is the one running ``install-unit`` (``sys.executable``), an
absolute path known at install time.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from ._settings import (
    check_bool,
    check_count,
    check_host,
    check_port,
    check_timeout_s,
)

UNIT_NAME = "scitex-genai-gateway.service"
FRONTEND_SOCKET_UNIT = "scitex-genai-gateway-frontend.socket"
FRONTEND_SERVICE_UNIT = "scitex-genai-gateway-frontend.service"
BACKEND_UNIT_PREFIX = "scitex-genai-gateway-backend-"
DEFAULT_UNIT_DIR = Path.home() / ".config" / "systemd" / "user"
MODULE = "scitex_genai.gateway._cli"
SOCKET_PROXY = Path("/usr/lib/systemd/systemd-socket-proxyd")

Runner = Callable[[Sequence[str]], None]


def _systemctl(argv: Sequence[str]) -> None:
    """The real thing: a user-manager call that raises when systemd refuses."""
    subprocess.run(list(argv), check=True)


def gateway_command(
    *,
    host: str | None = None,
    port: int | None = None,
    inference_timeout_s: float | None = None,
    inference_capacity_per_upstream: int | None = None,
    inference_max_queue_size: int | None = None,
    inference_continuation_qos_enabled: bool | None = None,
    inference_continuation_qos_max_retries: int | None = None,
    inference_continuation_qos_min_preempt_tokens: int | None = None,
    inference_cache_report_enabled: bool | None = None,
    config: Path | str | None = None,
    interpreter: str | None = None,
    uds: Path | str | None = None,
    gateway_build: str | None = None,
    gateway_incarnation: str | None = None,
    frontend_generation: str | None = None,
    graceful_rollout_shutdown: bool = False,
) -> list[str]:
    """The argv the unit execs: this package's server under an absolute interpreter.

    Only what was given is baked in; everything else the server resolves from
    its settings file at each start.
    """
    argv = [str(interpreter or sys.executable), "-m", MODULE]
    if uds is not None:
        argv += ["--uds", str(uds)]
    if gateway_build is not None:
        argv += ["--gateway-build", gateway_build]
    if gateway_incarnation is not None:
        argv += ["--gateway-incarnation", gateway_incarnation]
    if frontend_generation is not None:
        argv += ["--frontend-generation", frontend_generation]
    if graceful_rollout_shutdown:
        argv.append("--graceful-rollout-shutdown")
    if config is not None:
        argv += ["--config", str(config)]
    if host is not None:
        argv += ["--host", check_host(host)]
    if port is not None:
        argv += ["--port", str(check_port(port))]
    if inference_timeout_s is not None:
        argv += ["--inference-timeout-s", str(check_timeout_s(inference_timeout_s))]
    if inference_capacity_per_upstream is not None:
        argv += [
            "--inference-capacity-per-upstream",
            str(
                check_count(
                    "inference_capacity_per_upstream",
                    inference_capacity_per_upstream,
                    minimum=1,
                )
            ),
        ]
    if inference_max_queue_size is not None:
        argv += [
            "--inference-max-queue-size",
            str(
                check_count(
                    "inference_max_queue_size", inference_max_queue_size, minimum=0
                )
            ),
        ]
    if inference_continuation_qos_enabled is not None:
        argv.append(
            "--inference-continuation-qos"
            if check_bool(
                "inference_continuation_qos_enabled",
                inference_continuation_qos_enabled,
            )
            else "--no-inference-continuation-qos"
        )
    if inference_continuation_qos_max_retries is not None:
        argv += [
            "--inference-continuation-qos-max-retries",
            str(
                check_count(
                    "inference_continuation_qos_max_retries",
                    inference_continuation_qos_max_retries,
                    minimum=0,
                )
            ),
        ]
    if inference_continuation_qos_min_preempt_tokens is not None:
        argv += [
            "--inference-continuation-qos-min-preempt-tokens",
            str(
                check_count(
                    "inference_continuation_qos_min_preempt_tokens",
                    inference_continuation_qos_min_preempt_tokens,
                    minimum=0,
                )
            ),
        ]
    if inference_cache_report_enabled is not None:
        argv.append(
            "--inference-cache-report"
            if check_bool(
                "inference_cache_report_enabled", inference_cache_report_enabled
            )
            else "--no-inference-cache-report"
        )
    return argv


def backend_unit_name(generation: str) -> str:
    """Systemd unit name for one already-validated generation label."""
    return f"{BACKEND_UNIT_PREFIX}{generation}.service"


def render_frontend_socket(*, host: str, port: int) -> str:
    """Stable fleet-facing listener retained across backend generations."""
    return (
        "[Unit]\n"
        "Description=scitex-genai stable gateway frontend socket\n\n"
        "[Socket]\n"
        f"ListenStream={check_host(host)}:{check_port(port)}\n"
        "NoDelay=true\n"
        f"Service={FRONTEND_SERVICE_UNIT}\n\n"
        "[Install]\n"
        "WantedBy=sockets.target\n"
    )


def render_frontend_service(*, current_socket: Path | str) -> str:
    """Existing systemd proxy primitive; it never retries an HTTP request."""
    return (
        "[Unit]\n"
        "Description=scitex-genai stable gateway frontend proxy\n"
        f"Requires={FRONTEND_SOCKET_UNIT}\n"
        f"After={FRONTEND_SOCKET_UNIT}\n\n"
        "[Service]\n"
        "Type=notify\n"
        f"ExecStart={SOCKET_PROXY} {current_socket}\n"
        "Restart=on-failure\n"
        "RestartSec=1\n"
    )


def render_backend_unit(
    *, command: Sequence[str], socket_path: Path | str, generation: str
) -> str:
    """A private generation backend whose SIGTERM drains existing ASGI tasks."""
    socket_path = Path(socket_path)
    if socket_path.name != f"{generation}.sock":
        raise ValueError("backend socket basename must match its generation")
    exec_start = " ".join(shlex.quote(arg) for arg in command)
    quoted_socket = shlex.quote(str(socket_path))
    return (
        "[Unit]\n"
        f"Description=scitex-genai gateway backend generation {generation}\n"
        "After=network-online.target\n"
        "Wants=network-online.target\n\n"
        "[Service]\n"
        "Type=simple\n"
        f"ExecStartPre=-/usr/bin/rm -f {quoted_socket}\n"
        f"ExecStart={exec_start}\n"
        f"ExecStopPost=-/usr/bin/rm -f {quoted_socket}\n"
        "TimeoutStopSec=infinity\n"
        "Restart=on-failure\n"
        "RestartSec=1\n"
        "Environment=PYTHONUNBUFFERED=1\n"
        "\n[Install]\n"
        "WantedBy=default.target\n"
    )


def install_rollout_frontend(
    *,
    host: str,
    port: int,
    current_socket: Path | str,
    unit_dir: Path | None = None,
    runner: Runner | None = None,
) -> tuple[Path, Path]:
    """Write but deliberately do not start the bootstrap-sensitive frontend."""
    target_dir = Path(unit_dir) if unit_dir is not None else DEFAULT_UNIT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    socket_unit = target_dir / FRONTEND_SOCKET_UNIT
    service_unit = target_dir / FRONTEND_SERVICE_UNIT
    socket_unit.write_text(render_frontend_socket(host=host, port=port))
    service_unit.write_text(render_frontend_service(current_socket=current_socket))
    (runner or _systemctl)(["systemctl", "--user", "daemon-reload"])
    return socket_unit, service_unit


def render_unit(
    *,
    host: str | None = None,
    port: int | None = None,
    inference_timeout_s: float | None = None,
    config: Path | str | None = None,
    interpreter: str | None = None,
    inference_capacity_per_upstream: int | None = None,
    inference_max_queue_size: int | None = None,
    inference_continuation_qos_enabled: bool | None = None,
    inference_continuation_qos_max_retries: int | None = None,
    inference_continuation_qos_min_preempt_tokens: int | None = None,
    inference_cache_report_enabled: bool | None = None,
) -> str:
    """The unit text, byte-for-byte what ``install_unit`` writes."""
    argv = gateway_command(
        host=host,
        port=port,
        inference_timeout_s=inference_timeout_s,
        inference_capacity_per_upstream=inference_capacity_per_upstream,
        inference_max_queue_size=inference_max_queue_size,
        inference_continuation_qos_enabled=inference_continuation_qos_enabled,
        inference_continuation_qos_max_retries=(inference_continuation_qos_max_retries),
        inference_continuation_qos_min_preempt_tokens=(
            inference_continuation_qos_min_preempt_tokens
        ),
        inference_cache_report_enabled=inference_cache_report_enabled,
        config=config,
        interpreter=interpreter,
    )
    exec_start = " ".join(shlex.quote(arg) for arg in argv)
    settings = str(config) if config is not None else "~/.scitex/genai/config.yaml"
    return (
        f"# {UNIT_NAME} -- written by `scitex-genai-gateway install-unit`.\n"
        "# Do not edit by hand: change the settings file and restart, or re-run\n"
        "# that command, so the next host gets the same unit from the same source.\n"
        "\n"
        "[Unit]\n"
        f"Description=scitex-genai gateway (Anthropic + OpenAI protocol relay; settings: {settings})\n"
        "After=network-online.target\n"
        "Wants=network-online.target\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        "# No shell: the auth key comes from ~/.scitex/genai/secrets, which a\n"
        "# plain process can read. A missing key still fails loud.\n"
        f"ExecStart={exec_start}\n"
        "# Let uvicorn drain admitted streams and wake queued requests on shutdown.\n"
        "TimeoutStopSec=infinity\n"
        "Restart=always\n"
        "RestartSec=3\n"
        "Environment=PYTHONUNBUFFERED=1\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def install_unit(
    *,
    host: str | None = None,
    port: int | None = None,
    inference_timeout_s: float | None = None,
    config: Path | str | None = None,
    unit_dir: Path | None = None,
    enable: bool = True,
    runner: Runner | None = None,
    inference_capacity_per_upstream: int | None = None,
    inference_max_queue_size: int | None = None,
    inference_continuation_qos_enabled: bool | None = None,
    inference_continuation_qos_max_retries: int | None = None,
    inference_continuation_qos_min_preempt_tokens: int | None = None,
    inference_cache_report_enabled: bool | None = None,
    interpreter: str | None = None,
) -> Path:
    """Write the unit, then reload the user manager and ``enable --now`` it.

    Idempotent: the file is overwritten in place, and systemd's reload and
    enable are safe to repeat. ``enable=False`` writes only -- for a host where
    the caller wants to inspect the unit before it runs. ``runner`` is the
    dependency that performs the ``systemctl`` calls; tests hand in a recorder.
    """
    target_dir = Path(unit_dir) if unit_dir is not None else DEFAULT_UNIT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / UNIT_NAME
    path.write_text(
        render_unit(
            host=host,
            port=port,
            inference_timeout_s=inference_timeout_s,
            inference_capacity_per_upstream=inference_capacity_per_upstream,
            inference_max_queue_size=inference_max_queue_size,
            inference_continuation_qos_enabled=(inference_continuation_qos_enabled),
            inference_continuation_qos_max_retries=(
                inference_continuation_qos_max_retries
            ),
            inference_continuation_qos_min_preempt_tokens=(
                inference_continuation_qos_min_preempt_tokens
            ),
            inference_cache_report_enabled=inference_cache_report_enabled,
            config=config,
            interpreter=interpreter,
        )
    )
    if enable:
        run = runner or _systemctl
        run(["systemctl", "--user", "daemon-reload"])
        run(["systemctl", "--user", "enable", "--now", UNIT_NAME])
    return path
