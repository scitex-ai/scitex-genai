"""Ship the gateway's systemd user unit from the package, not from a hand.

WHY (2026-09-05, scitex-compute-04). The relay that fronts the fleet's local
model ran for three weeks from a 369-line script outside any package,
supervised by a unit somebody typed into ``~/.config/systemd/user``, with its
upstream list kept current by a hand-written watcher that expired one morning
and told nobody. None of it could be reproduced on the next host by anyone but
its author, and the operator's ruling is that a step only one person can
perform is a step that does not exist. So the unit is rendered HERE, from the
installed package, by one command anyone can run on any host::

    scitex-genai-gateway install-unit --host 0.0.0.0 --port 18772 \\
        --inference-upstream http://127.0.0.1:18773,http://127.0.0.1:18774

TWO CHOICES THAT LOOK ODD AND ARE DELIBERATE
--------------------------------------------
``ExecStart=/bin/bash -lc 'exec ...'`` -- a LOGIN shell. The gateway refuses
to start without ``SCITEX_GENAI_GATEWAY_API_KEY``, and the fleet delivers that
secret through the dotfiles login profile (``~/.bash.d/secrets``, lines of the
form ``export NAME=value``). systemd's ``EnvironmentFile=`` cannot read the
``export`` form, so the profile is sourced the way an interactive login sources
it, and ``exec`` hands the PID to the gateway so systemd supervises the server
rather than the shell.

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

from ._inference import parse_upstreams

UNIT_NAME = "scitex-genai-gateway.service"
DEFAULT_UNIT_DIR = Path.home() / ".config" / "systemd" / "user"
MODULE = "scitex_genai.gateway._cli"

Runner = Callable[[Sequence[str]], None]


def _systemctl(argv: Sequence[str]) -> None:
    """The real thing: a user-manager call that raises when systemd refuses."""
    subprocess.run(list(argv), check=True)


def gateway_command(
    *,
    host: str,
    port: int,
    upstream: str = "",
    interpreter: str | None = None,
) -> list[str]:
    """The argv the unit execs: this package's server under an absolute interpreter.

    ``upstream`` is the same comma-separated list the server takes; empty means
    the Codex-account backend, exactly as it does on the serve command line.
    """
    if not host or any(ch.isspace() for ch in host):
        raise ValueError(f"host must be a bare address, got {host!r}")
    if not 0 < int(port) < 65536:
        raise ValueError(f"port must be within 1..65535, got {port!r}")
    argv = [
        str(interpreter or sys.executable),
        "-m",
        MODULE,
        "--host",
        host,
        "--port",
        str(int(port)),
    ]
    urls = parse_upstreams(upstream or "")
    if urls:
        argv += ["--inference-upstream", ",".join(urls)]
    return argv


def render_unit(
    *,
    host: str,
    port: int,
    upstream: str = "",
    interpreter: str | None = None,
) -> str:
    """The unit text, byte-for-byte what ``install_unit`` writes."""
    argv = gateway_command(
        host=host, port=port, upstream=upstream, interpreter=interpreter
    )
    inner = "exec " + " ".join(shlex.quote(arg) for arg in argv)
    return (
        f"# {UNIT_NAME} -- written by `scitex-genai-gateway install-unit`.\n"
        "# Do not edit by hand: re-run that command to change host, port or\n"
        "# upstreams, so the next host gets the same unit from the same source.\n"
        "\n"
        "[Unit]\n"
        f"Description=scitex-genai gateway on {host}:{int(port)} "
        "(Anthropic-format relay, sticky upstream pool)\n"
        "After=network-online.target\n"
        "Wants=network-online.target\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        "# A LOGIN shell: the fleet delivers SCITEX_GENAI_GATEWAY_API_KEY through\n"
        "# the dotfiles profile (.bash.d/secrets, `export NAME=value` lines), which\n"
        "# systemd's EnvironmentFile= cannot read. The gateway refuses to start\n"
        "# without that variable, so a missing profile fails loud, not silent.\n"
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
    host: str,
    port: int,
    upstream: str = "",
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
        render_unit(host=host, port=port, upstream=upstream, interpreter=interpreter)
    )
    if enable:
        run = runner or _systemctl
        run(["systemctl", "--user", "daemon-reload"])
        run(["systemctl", "--user", "enable", "--now", UNIT_NAME])
    return path
