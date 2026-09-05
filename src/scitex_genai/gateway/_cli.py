"""CLI for the Anthropic-compatible gateway: Codex accounts or inference upstreams.

Two forms. With no subcommand the process IS the server::

    scitex-genai-gateway --host 127.0.0.1 --port 8765 \\
        [--inference-upstream URL,URL]

``install-unit`` writes the systemd user unit that runs that same command line
under supervision, reloads the user manager and enables it (see ``_unit``)::

    scitex-genai-gateway install-unit --host 0.0.0.0 --port 18772 \\
        --inference-upstream http://127.0.0.1:18773,http://127.0.0.1:18774
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ._accounts import CodexAccountPool
from ._codex import CodexBackend, CodexTransport
from ._inference import (
    DEFAULT_TIMEOUT_S,
    PREFIX_TELEMETRY_ENV,
    TIMEOUT_ENV,
    UPSTREAM_ENV,
    InferenceBackend,
    InferenceUpstreamPool,
    announce,
    telemetry_enabled,
)
from ._server import create_app
from ._unit import UNIT_NAME, install_unit

INSTALL_UNIT = "install-unit"


def _add_serve_args(parser: argparse.ArgumentParser) -> None:
    """The flags that describe ONE gateway; shared by serve and install-unit."""
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--inference-upstream",
        default=os.getenv(UPSTREAM_ENV, ""),
        help=(
            "Comma-separated base URLs of Anthropic-compatible inference "
            "upstreams (vLLM, LiteLLM). When set, /v1/messages is relayed to "
            f"that pool instead of the Codex accounts. Defaults to ${UPSTREAM_ENV}."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    _add_serve_args(parser)
    parser.add_argument(
        "--codex-base-url",
        default=os.getenv(
            "SCITEX_GENAI_CODEX_BASE_URL", "https://chatgpt.com/backend-api"
        ),
    )
    parser.add_argument("--log-level", default="info")
    commands = parser.add_subparsers(dest="command")
    unit = commands.add_parser(
        INSTALL_UNIT,
        help="write the systemd user unit for this gateway, reload, enable --now",
    )
    _add_serve_args(unit)
    unit.add_argument(
        "--unit-dir",
        type=Path,
        default=None,
        help="directory the unit is written to (default: ~/.config/systemd/user)",
    )
    unit.add_argument(
        "--no-enable",
        action="store_true",
        help="write the unit only; skip daemon-reload and enable --now",
    )
    return parser


def _telemetry_sink():
    """Stdout when ``HOIST_PREFIX_TELEMETRY`` asks for it, else off."""
    if not telemetry_enabled(os.getenv(PREFIX_TELEMETRY_ENV, "")):
        return None
    return lambda line: print(line, flush=True)


def _install_unit(args: argparse.Namespace) -> None:
    path = install_unit(
        host=args.host,
        port=args.port,
        upstream=args.inference_upstream,
        unit_dir=args.unit_dir,
        enable=not args.no_enable,
    )
    state = "written only" if args.no_enable else "reloaded and enabled --now"
    print(f"scitex-genai-gateway: {UNIT_NAME} -> {path} ({state})", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == INSTALL_UNIT:
        _install_unit(args)
        return
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install scitex-genai[gateway] to run the server") from exc
    if args.inference_upstream:
        pool = InferenceUpstreamPool.from_urls(args.inference_upstream)
        backend = InferenceBackend(
            pool,
            timeout_s=float(os.getenv(TIMEOUT_ENV, DEFAULT_TIMEOUT_S)),
            telemetry_sink=_telemetry_sink(),
        )
        print(announce(args.host, args.port, pool), flush=True)
    else:
        codex_pool = CodexAccountPool.discover()
        backend = CodexBackend(codex_pool, CodexTransport(base_url=args.codex_base_url))
    app = create_app(backend)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
