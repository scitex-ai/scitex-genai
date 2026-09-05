"""``scitex-genai-serve <key>``: serve one engine from the user's settings, inside an allocation.

    scitex-genai-serve qwen38-27b                  # settings + conf from ~/.scitex/genai
    scitex-genai-serve qwen38-27b --dry-run        # print the rendered launch, start nothing
    scitex-genai-serve --list                      # keys with a conf

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


def main(argv: list[str] | None = None) -> int:
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
