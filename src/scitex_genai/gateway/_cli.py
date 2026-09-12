"""CLI for the Anthropic-compatible gateway: Codex accounts or inference upstreams.

Settings resolve direct -> ``~/.scitex/genai/config.yaml`` -> environment ->
default (see ``_settings``), so the plain form needs no flags on a configured
host::

    scitex-genai-gateway                       # the process IS the server
    scitex-genai-gateway --host 127.0.0.1 --port 8765 --inference-upstream URL,URL

``install-unit`` writes the systemd user unit that runs that same command line
under supervision, reloads the user manager and enables it (see ``_unit``)::

    scitex-genai-gateway install-unit          # unit reads the settings file
    scitex-genai-gateway install-unit --host 0.0.0.0 --port 18772   # baked in
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from pathlib import Path

from ._accounts import CodexAccountPool
from ._codex import CodexBackend, CodexTransport
from ._errors import CredentialError
from ._external import ExternalProviderBackend, ExternalProviderPolicy
from ._inference import (
    PREFIX_TELEMETRY_ENV,
    InferenceBackend,
    InferenceUpstreamPool,
    announce,
    telemetry_enabled,
)
from ._secrets import (
    GATEWAY_KEY_ENV,
    default_secrets_path,
    resolve_gateway_key,
    write_key,
)
from ._server import create_app, run_uvicorn
from ._settings import load_settings
from ._unit import DEFAULT_UNIT_DIR, UNIT_NAME, install_unit

INSTALL_UNIT = "install-unit"


def _add_settings_args(
    parser: argparse.ArgumentParser, *, suppress_defaults: bool = False
) -> None:
    """The flags that describe ONE gateway; shared by serve and install-unit.

    Every default is ``None`` on purpose: an unset flag means "the settings
    file decides", so the command line never overrides silently.
    """
    default = argparse.SUPPRESS if suppress_defaults else None
    parser.add_argument(
        "--config",
        type=Path,
        default=default,
        help="settings file (default: ~/.scitex/genai/config.yaml)",
    )
    parser.add_argument(
        "--host",
        default=default,
        help="bind address (default: gateway.host, else 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=default,
        help="port (default: gateway.port, else 8765)",
    )
    parser.add_argument(
        "--inference-upstream",
        default=default,
        help=(
            "Comma-separated base URLs of Anthropic-compatible inference "
            "upstreams (vLLM, LiteLLM). When set, /v1/messages is relayed to "
            "that pool instead of the Codex accounts. Default: "
            "gateway.inference_upstreams in the settings file, else $HOIST_UPSTREAM."
        ),
    )
    parser.add_argument(
        "--inference-timeout-s",
        type=float,
        default=default,
        help=(
            "Upstream inference timeout in seconds (default: "
            "gateway.inference_timeout_s, else $HOIST_TIMEOUT_S, else "
            "$SCITEX_GATEWAY_INFERENCE_TIMEOUT_S, else 600)."
        ),
    )
    parser.add_argument(
        "--inference-capacity-per-upstream",
        type=int,
        default=default,
        help=(
            "Maximum concurrent admitted requests per inference upstream "
            "(default: gateway.inference_capacity_per_upstream, else 8)."
        ),
    )
    parser.add_argument(
        "--inference-max-queue-size",
        type=int,
        default=default,
        help=(
            "Maximum requests waiting for inference capacity across the pool "
            "(default: gateway.inference_max_queue_size, else 128)."
        ),
    )
    parser.add_argument(
        "--inference-token-capacity-per-upstream",
        type=int,
        default=default,
        help=(
            "Maximum estimated input tokens admitted concurrently per inference "
            "upstream (default: gateway.inference_token_capacity_per_upstream; "
            "unset disables the token budget)."
        ),
    )
    parser.add_argument(
        "--inference-continuation-qos",
        action=argparse.BooleanOptionalAction,
        default=default,
        help=(
            "Enable opt-in continuation handoff/preemption for explicit stable "
            "session IDs (default: gateway.inference_continuation_qos_enabled, "
            "else disabled)."
        ),
    )
    parser.add_argument(
        "--inference-continuation-qos-max-retries",
        type=int,
        default=default,
        help=(
            "Maximum transparent retries of a cooperatively preempted first turn "
            "(default: gateway.inference_continuation_qos_max_retries, else 1)."
        ),
    )
    parser.add_argument(
        "--inference-continuation-qos-min-preempt-tokens",
        type=int,
        default=default,
        help=(
            "Minimum estimated first-turn input tokens eligible for preemption "
            "(default: gateway.inference_continuation_qos_min_preempt_tokens, "
            "else 0)."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    _add_settings_args(parser)
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
    _add_settings_args(unit, suppress_defaults=True)
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


def _persist_key(*, replacing_a_unit: bool) -> None:
    """Give the key a home before anything depends on it having one.

    Run at install time, from whatever shell the operator used, so a key that
    exists ONLY as an ``export`` line in their profile is captured into the
    secrets file at its current value. That ordering is what lets the unit stop
    asking for a login shell without changing the key every client already
    presents.

    THE REFUSAL BELOW IS THE POINT OF THIS FUNCTION, not a detail of it. On a
    host that already has a gateway, clients are already holding a key. If this
    call had to INVENT one -- which happens when it is run from a shell where
    the old key does not resolve, exactly the non-login case this whole change
    is about -- then installing would replace a working key with one nobody
    has, and the symptom would be the outage we just fixed, reappearing at the
    moment we claimed to have fixed it. So it stops, and says which shell to
    re-run from.

    A first install has no such risk: there are no clients yet, and minting is
    what makes the host work with no manual step.
    """
    if replacing_a_unit:
        try:
            key = resolve_gateway_key()
        except CredentialError as exc:
            raise SystemExit(
                f"refusing to install: a gateway unit already exists here, so "
                f"clients already hold a key, but none resolved -- installing "
                f"would mint a NEW one and every client would start getting "
                f"401s. Re-run from a shell where {GATEWAY_KEY_ENV} resolves "
                f"(the login shell whose profile holds it), or write the "
                f"existing value into {default_secrets_path()} as "
                f"{GATEWAY_KEY_ENV}=<value> first."
            ) from exc
    else:
        key = resolve_gateway_key(create=True)

    # RESOLVING IS NOT PERSISTING, and this line is the whole function.
    # A key that resolved from the ENVIRONMENT lives only in this shell. The
    # unit written moments from now runs without one, so unless the value is
    # put in the file HERE, the next start finds nothing and mints a
    # replacement -- rotating the key every client holds. Measured on
    # scitex-compute-04 2026-09-07: install-unit printed "key environment
    # (kept; nothing written)" and wrote the shell-free unit anyway, leaving
    # exactly that gap open until the key was written by hand.
    stored = key.path if key.origin != "environment" else write_key(key.value)
    print(f"scitex-genai-gateway: key {key.origin} -> {stored}", flush=True)


def _install_unit(args: argparse.Namespace) -> None:
    unit_dir = args.unit_dir if args.unit_dir is not None else DEFAULT_UNIT_DIR
    _persist_key(replacing_a_unit=(Path(unit_dir) / UNIT_NAME).exists())
    path = install_unit(
        host=args.host,
        port=args.port,
        upstream=args.inference_upstream,
        inference_timeout_s=args.inference_timeout_s,
        inference_capacity_per_upstream=args.inference_capacity_per_upstream,
        inference_max_queue_size=args.inference_max_queue_size,
        inference_token_capacity_per_upstream=(
            args.inference_token_capacity_per_upstream
        ),
        inference_continuation_qos_enabled=args.inference_continuation_qos,
        inference_continuation_qos_max_retries=(
            args.inference_continuation_qos_max_retries
        ),
        inference_continuation_qos_min_preempt_tokens=(
            args.inference_continuation_qos_min_preempt_tokens
        ),
        config=args.config,
        unit_dir=args.unit_dir,
        enable=not args.no_enable,
    )
    state = "written only" if args.no_enable else "reloaded and enabled --now"
    print(f"scitex-genai-gateway: {UNIT_NAME} -> {path} ({state})", flush=True)


def main(
    argv: list[str] | None = None,
    *,
    server_runner: Callable[..., None] | None = None,
) -> None:
    args = build_parser().parse_args(argv)
    if args.command == INSTALL_UNIT:
        _install_unit(args)
        return
    try:
        __import__("uvicorn")
    except ImportError as exc:
        raise SystemExit("Install scitex-genai[gateway] to run the server") from exc
    settings = load_settings(
        args.config,
        host=args.host,
        port=args.port,
        inference_upstream=args.inference_upstream,
        inference_timeout_s=args.inference_timeout_s,
        inference_capacity_per_upstream=args.inference_capacity_per_upstream,
        inference_max_queue_size=args.inference_max_queue_size,
        inference_token_capacity_per_upstream=(
            args.inference_token_capacity_per_upstream
        ),
        inference_continuation_qos_enabled=args.inference_continuation_qos,
        inference_continuation_qos_max_retries=(
            args.inference_continuation_qos_max_retries
        ),
        inference_continuation_qos_min_preempt_tokens=(
            args.inference_continuation_qos_min_preempt_tokens
        ),
    )
    if settings.external_provider is not None:
        external = settings.external_provider
        upstream_key = os.getenv(external.upstream_auth_token_env, "").strip()
        if not upstream_key:
            raise SystemExit(
                "refusing to start external-provider relay: "
                f"{external.upstream_auth_token_env} is unset or empty"
            )
        pool = InferenceUpstreamPool.from_urls(
            [external.upstream],
            capacity_per_upstream=settings.inference_capacity_per_upstream,
            max_queue_size=settings.inference_max_queue_size,
            token_capacity_per_upstream=(
                settings.inference_token_capacity_per_upstream
            ),
        )
        backend = ExternalProviderBackend(
            pool,
            timeout_s=settings.inference_timeout_s,
            journal=lambda line: print(line, flush=True),
            policy=ExternalProviderPolicy(
                provider=external.provider,
                upstream_api_key=upstream_key,
                canonical_model=external.canonical_model,
                model_aliases=external.model_aliases,
                anthropic_path_prefix=external.anthropic_path_prefix,
                max_tokens_per_request=external.max_tokens_per_request,
                max_requests_per_run=external.max_requests_per_run,
                max_input_tokens_per_run=external.max_input_tokens_per_run,
                max_output_tokens_per_run=external.max_output_tokens_per_run,
                max_total_tokens_per_run=external.max_total_tokens_per_run,
                max_estimated_usd_per_run=external.max_estimated_usd_per_run,
                input_usd_per_million_tokens=external.input_usd_per_million_tokens,
                output_usd_per_million_tokens=external.output_usd_per_million_tokens,
            ),
        )
        print(
            "scitex-genai-gateway: external provider "
            f"{external.provider} -> {external.upstream}; outbound model policy: "
            f"{external.canonical_model} only",
            flush=True,
        )
    elif settings.inference_upstream:
        pool = InferenceUpstreamPool.from_urls(
            settings.inference_upstream,
            capacity_per_upstream=settings.inference_capacity_per_upstream,
            max_queue_size=settings.inference_max_queue_size,
            token_capacity_per_upstream=(
                settings.inference_token_capacity_per_upstream
            ),
        )
        backend = InferenceBackend(
            pool,
            timeout_s=settings.inference_timeout_s,
            telemetry_sink=_telemetry_sink(),
            journal=lambda line: print(line, flush=True),
            continuation_qos_enabled=(
                settings.inference_continuation_qos_enabled
            ),
            continuation_qos_max_retries=(
                settings.inference_continuation_qos_max_retries
            ),
            continuation_qos_min_preempt_tokens=(
                settings.inference_continuation_qos_min_preempt_tokens
            ),
        )
        print(announce(settings.host, settings.port, pool), flush=True)
    else:
        codex_pool = CodexAccountPool.discover()
        backend = CodexBackend(codex_pool, CodexTransport(base_url=args.codex_base_url))
    key = resolve_gateway_key(create=True)
    print(f"scitex-genai-gateway: key {key.origin}", flush=True)
    app = create_app(backend, api_key=key.value)
    app.state.scitex_backend = backend
    (server_runner or run_uvicorn)(
        app, host=settings.host, port=settings.port, log_level=args.log_level
    )


if __name__ == "__main__":
    main()
