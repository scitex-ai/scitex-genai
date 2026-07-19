"""CLI for the Anthropic-compatible Codex subscription gateway."""

from __future__ import annotations

import argparse
import os

from ._accounts import CodexAccountPool
from ._codex import CodexBackend, CodexTransport
from ._server import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--codex-base-url",
        default=os.getenv("SCITEX_GENAI_CODEX_BASE_URL", "https://chatgpt.com/backend-api"),
    )
    parser.add_argument("--log-level", default="info")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install scitex-genai[gateway] to run the server") from exc
    pool = CodexAccountPool.discover()
    backend = CodexBackend(pool, CodexTransport(base_url=args.codex_base_url))
    app = create_app(backend)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
