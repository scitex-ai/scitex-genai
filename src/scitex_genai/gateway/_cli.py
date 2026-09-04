"""CLI for the Anthropic-compatible gateway: Codex accounts or inference upstreams."""

from __future__ import annotations

import argparse
import os

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--codex-base-url",
        default=os.getenv("SCITEX_GENAI_CODEX_BASE_URL", "https://chatgpt.com/backend-api"),
    )
    parser.add_argument(
        "--inference-upstream",
        default=os.getenv(UPSTREAM_ENV, ""),
        help=(
            "Comma-separated base URLs of Anthropic-compatible inference "
            "upstreams (vLLM, LiteLLM). When set, /v1/messages is relayed to "
            f"that pool instead of the Codex accounts. Defaults to ${UPSTREAM_ENV}."
        ),
    )
    parser.add_argument("--log-level", default="info")
    return parser


def _telemetry_sink():
    """Stdout when ``HOIST_PREFIX_TELEMETRY`` asks for it, else off."""
    if not telemetry_enabled(os.getenv(PREFIX_TELEMETRY_ENV, "")):
        return None
    return lambda line: print(line, flush=True)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
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
        backend = CodexBackend(
            codex_pool, CodexTransport(base_url=args.codex_base_url)
        )
    app = create_app(backend)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
