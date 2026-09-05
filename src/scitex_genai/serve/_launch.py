"""Launch N engines inside ONE held allocation: the hold body the lease runs.

This is ``serve-pair.sh``'s job, put where it belongs. scitex-hpc owns the
allocation (``Reservation.book(..., persistent=True, hold_body=...)``) and its
walltime resubmit (the trap the lease wraps around the body); this module
renders the BODY -- one supervised step per engine, one engine per GPU
(operator ruling 2026-08-15: TP=1, one instance per card), ``wait`` at the
end so the batch shell stays alive for SLURM.

THREE THINGS THE BODY DELIBERATELY DOES NOT DO, each measured on Spartan:
- It sets no ``CUDA_VISIBLE_DEVICES``: each engine's conf exports its own and
  the serve command hands it to the engine; setting it here would put both
  replicas on one card.
- It installs no signal trap: ``--signal=B:USR1`` reaches the batch shell
  only, and the lease already owns that trap. Two resubmitters queue two
  successors for one pair.
- It runs the steps in the background and ``wait``s: a foreground child
  blocks the lease's trap delivery until it exits.

And one it does: it aborts on rc=127 (command not found), which never
self-heals, instead of retrying forever -- the failure that left an
allocation idling for hours with an unreadable log.
"""

from __future__ import annotations

import shlex
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ._conf import list_engines
from ._settings import ServeSettings

DEFAULT_RESTART_S = 30


def _check_keys(keys: Sequence[str], models_dir: Path | None) -> list[str]:
    cleaned = [str(k).strip() for k in keys if str(k).strip()]
    if not cleaned:
        raise ValueError("at least one engine key is required")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError(f"engine keys must be distinct, got {cleaned}")
    for key in cleaned:
        if "/" in key or any(ch.isspace() for ch in key):
            raise ValueError(f"engine key must be a bare name, got {key!r}")
    if models_dir is not None:
        known = set(list_engines(models_dir))
        missing = [k for k in cleaned if k not in known]
        if missing:
            available = ", ".join(sorted(known)) or "none"
            raise ValueError(
                f"no conf for {', '.join(missing)} under {models_dir}; available: {available}"
            )
    return cleaned


def render_hold_body(
    keys: Sequence[str],
    settings: ServeSettings,
    *,
    models_dir: Path | None = None,
    config_path: Path | None = None,
    restart_s: int = DEFAULT_RESTART_S,
) -> str:
    """The sbatch body: one restart-supervised ``srun --overlap`` step per engine."""
    engines = _check_keys(keys, models_dir)
    serve = [str(settings.serve_bin)]
    if config_path is not None:
        serve += ["--config", str(config_path)]
    if models_dir is not None:
        serve += ["--models-dir", str(models_dir)]
    serve_cmd = " ".join(shlex.quote(part) for part in serve)
    if settings.srun_bin is not None:
        srun_line = (
            f"SRUN={shlex.quote(str(settings.srun_bin))}\n"
            '[ -x "$SRUN" ] || { echo "FATAL: $SRUN is not executable" >&2; exit 2; }\n'
        )
    else:
        srun_line = 'SRUN=srun\ncommand -v "$SRUN" >/dev/null || { echo "FATAL: srun is not on PATH" >&2; exit 2; }\n'
    lines = [
        f"# === scitex-genai serve: {len(engines)} engine(s) inside this allocation ===",
        "set -uo pipefail",
        srun_line.rstrip("\n"),
        f"SERVE={shlex.quote(serve_cmd)}",
        "_step() {",
        "  local key=$1",
        "  while true; do",
        '    echo "[serve-launch] $(date -u +%FT%TZ) starting step for $key" >&2',
        '    "$SRUN" --overlap --ntasks=1 --exact bash -c "$SERVE $key"',
        "    rc=$?",
        f'    echo "[serve-launch] $(date -u +%FT%TZ) step $key exited rc=$rc; restart in {restart_s}s" >&2',
        '    if [ "$rc" -eq 127 ]; then',
        '      echo "[serve-launch] rc=127 is command-not-found and never self-heals; aborting" >&2',
        "      exit 127",
        "    fi",
        f"    sleep {restart_s}",
        "  done",
        "}",
        *[f"_step {shlex.quote(key)} &" for key in engines],
        "wait",
    ]
    return "\n".join(lines) + "\n"


def book_serve_lease(
    keys: Sequence[str],
    settings: ServeSettings,
    *,
    name: str,
    project: str,
    host: str,
    gpus: str,
    partition: str | None = None,
    time: str | None = None,
    models_dir: Path | None = None,
    config_path: Path | None = None,
    book: Callable[..., Any] | None = None,
) -> Any:
    """Book a PERSISTENT lease whose body serves ``keys`` -- through scitex-hpc.

    ``book`` is the dependency (``Reservation.book`` by default; tests hand
    in a recorder). The lease owns the walltime resubmit; this only shapes
    the request and the body.
    """
    body = render_hold_body(
        keys, settings, models_dir=models_dir, config_path=config_path
    )
    try:
        from scitex_hpc._config import JobConfig
    except (
        ImportError
    ) as exc:  # pragma: no cover - depends on the extra being installed
        raise RuntimeError(
            "serve launch needs scitex-hpc (install scitex-genai[serve])"
        ) from exc
    config = JobConfig(
        project=project,
        host=host,
        partition=partition,
        time=time,
        job_name=name,
        extra_sbatch_args=[f"--gpus={gpus}"],
    )
    if book is None:
        from scitex_hpc._reservation import Reservation

        book = Reservation.book
    return book(config, persistent=True, hold_body=body)
