"""``scitex-genai-serve <key>``: serve one engine from the user's settings, inside an allocation.

    scitex-genai-serve qwen38-27b                  # settings + conf from ~/.scitex/genai
    scitex-genai-serve qwen38-27b --dry-run        # print the rendered launch, start nothing
    scitex-genai-serve --list                      # keys with a conf

and, from the FLEET side, the allocation that runs N of those steps::

    scitex-genai-serve launch qwen38-27b qwen38-27b-b --lease h100-pair \\
        --host spartan --gpus 2 --partition gpu-h100 --time 7-00:00:00
    scitex-genai-serve launch ... --dry-run        # print the hold body only

``launch`` books a PERSISTENT scitex-hpc lease whose body is one supervised
step per key (see ``_launch``); the lease owns the walltime resubmit.

This runs ON THE NODE that holds the GPU (a SLURM step inside a held lease).
Where the allocation comes from is scitex-hpc's business; what runs inside it
is this.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

from ._conf import list_engines, load_engine
from ._launch import book_serve_lease, render_hold_body
from ._render import Launch, render
from ._run import EngineRunner
from ._settings import load_serve_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "key", nargs="?", help="engine key: ~/.scitex/genai/models.d/<key>.conf"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="settings file (default: ~/.scitex/genai/config.yaml)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="engine confs (default: ~/.scitex/genai/models.d)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the rendered launch and exit"
    )
    parser.add_argument(
        "--list", action="store_true", help="print the engine keys that have a conf"
    )
    return parser


def describe(launch: Launch) -> str:
    """The launch as the shell lines it amounts to -- what --dry-run prints."""
    return "\n".join(
        [
            f"# engine {launch.key}",
            f"cache:   {launch.cache_dir}",
            f"health:  {launch.health_url}",
            f"vllm:    {shlex.join(launch.vllm_argv)}",
            f"litellm: {shlex.join(launch.litellm_argv)}",
            f"tunnel:  {shlex.join(launch.tunnel_argv)}",
            f"logs:    {launch.vllm_log} {launch.litellm_log} {launch.tunnel_log}",
        ]
    )


LAUNCH = "launch"


def build_launch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scitex-genai-serve launch",
        description="Book a persistent scitex-hpc lease that serves the given engines, one per GPU.",
    )
    parser.add_argument("keys", nargs="+", help="engine keys (one step, one GPU each)")
    parser.add_argument(
        "--lease", required=True, help="lease name (the resource, not the model)"
    )
    parser.add_argument(
        "--host", required=True, help="scitex-hpc host alias of the cluster login"
    )
    parser.add_argument(
        "--gpus", required=True, help="SLURM --gpus spec, e.g. 2 or h100:2"
    )
    parser.add_argument(
        "--project", default="serve", help="scitex-hpc project name (default: serve)"
    )
    parser.add_argument("--partition", default=None)
    parser.add_argument("--time", default=None, help="walltime, e.g. 7-00:00:00")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="settings file (default: ~/.scitex/genai/config.yaml)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="engine confs (default: ~/.scitex/genai/models.d)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the hold body; book nothing"
    )
    return parser


def _launch_main(argv: list[str]) -> int:
    args = build_launch_parser().parse_args(argv)
    try:
        settings = load_serve_settings(args.config)
        models_dir = args.models_dir if args.models_dir is not None else None
        body = render_hold_body(
            args.keys, settings, models_dir=models_dir, config_path=args.config
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"scitex-genai-serve launch: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(body, end="")
        return 0
    lease = book_serve_lease(
        args.keys,
        settings,
        name=args.lease,
        project=args.project,
        host=args.host,
        gpus=args.gpus,
        partition=args.partition,
        time=args.time,
        models_dir=models_dir,
        config_path=args.config,
    )
    print(
        f"scitex-genai-serve: booked persistent lease {args.lease} on {args.host}: {lease}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == LAUNCH:
        return _launch_main(argv[1:])
    args = build_parser().parse_args(argv)
    if args.list:
        for key in list_engines(args.models_dir):
            print(key)
        return 0
    if not args.key:
        print(
            "scitex-genai-serve: an engine key is required (see --list)",
            file=sys.stderr,
        )
        return 2
    try:
        settings = load_serve_settings(args.config)
        conf = load_engine(args.key, args.models_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"scitex-genai-serve: {exc}", file=sys.stderr)
        return 2
    launch = render(settings, conf, dict(os.environ))
    if args.dry_run:
        print(describe(launch))
        return 0
    EngineRunner(launch).run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
