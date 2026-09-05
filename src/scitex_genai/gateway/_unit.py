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

TWO CHOICES THAT LOOK ODD AND ARE DELIBERATE
--------------------------------------------
``ExecStart=/bin/bash -lc 'exec ...'`` -- a LOGIN shell. The gateway refuses
to start without ``SCITEX_GENAI_GATEWAY_API_KEY``, and a user's profile is
where such a secret normally lives, as an ``export NAME=value`` line that
systemd's ``EnvironmentFile=`` cannot read. The profile is sourced the way an
interactive login sources it, and ``exec`` hands the PID to the gateway so
systemd supervises the server rather than the shell.

``<interpreter> -m scitex_genai.gateway._cli`` rather than the console script:
the interpreter is the one running ``install-unit`` (``sys.executable``), an
absolute path known at install time, so the unit does not depend on whatever
the login shell puts on PATH.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from ._settings import check_host, check_port, upstream_string

UNIT_NAME = "scitex-genai-gateway.service"
DEFAULT_UNIT_DIR = Path.home() / ".config" / "systemd" / "user"
MODULE = "scitex_genai.gateway._cli"

Runner = Callable[[Sequence[str]], None]


def _systemctl(argv: Sequence[str]) -> None:
    """The real thing: a user-manager call that raises when systemd refuses."""
    subprocess.run(list(argv), check=True)


def gateway_command(
    *,
    host: str | None = None,
    port: int | None = None,
    upstream: str | None = None,
    config: Path | str | None = None,
    interpreter: str | None = None,
) -> list[str]:
    """The argv the unit execs: this package's server under an absolute interpreter.

    Only what was given is baked in; everything else the server resolves from
    its settings file at each start.
    """
    argv = [str(interpreter or sys.executable), "-m", MODULE]
    if config is not None:
        argv += ["--config", str(config)]
    if host is not None:
        argv += ["--host", check_host(host)]
    if port is not None:
        argv += ["--port", str(check_port(port))]
    if upstream is not None and upstream_string(upstream):
        argv += ["--inference-upstream", upstream_string(upstream)]
    return argv


def render_unit(
    *,
    host: str | None = None,
    port: int | None = None,
    upstream: str | None = None,
    config: Path | str | None = None,
    interpreter: str | None = None,
) -> str:
    """The unit text, byte-for-byte what ``install_unit`` writes."""
    argv = gateway_command(
        host=host, port=port, upstream=upstream, config=config, interpreter=interpreter
    )
    inner = "exec " + " ".join(shlex.quote(arg) for arg in argv)
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
        "# A LOGIN shell, so the user's profile supplies SCITEX_GENAI_GATEWAY_API_KEY\n"
        "# (an `export NAME=value` line systemd's EnvironmentFile= cannot read).\n"
        "# The gateway refuses to start without it: a missing key fails loud.\n"
        f"ExecStart=/bin/bash -lc {shlex.quote(inner)}\n"
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
    upstream: str | None = None,
    config: Path | str | None = None,
    unit_dir: Path | None = None,
    enable: bool = True,
    runner: Runner | None = None,
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
            upstream=upstream,
            config=config,
            interpreter=interpreter,
        )
    )
    if enable:
        run = runner or _systemctl
        run(["systemctl", "--user", "daemon-reload"])
        run(["systemctl", "--user", "enable", "--now", UNIT_NAME])
    return path
