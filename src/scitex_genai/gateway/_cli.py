"""CLI for the Anthropic-compatible gateway: Codex accounts or inference upstreams.

Settings resolve direct -> ``~/.scitex/genai/config.yaml`` -> environment ->
default (see ``_settings``), so the plain form needs no flags on a configured
host::

    scitex-genai-gateway                       # the process IS the server
    scitex-genai-gateway --host 127.0.0.1 --port 8765

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

import scitex_logging as slogging

from ._accounts import CodexAccountPool
from ._codex import CodexBackend, CodexTransport
from ._drain import (
    DEFAULT_DRAIN_TIMEOUT_S,
    DEFAULT_POLL_INTERVAL_S,
    DrainError,
    restart_when_drained,
)
from ._errors import CredentialError
from ._external import ExternalProviderBackend, ExternalProviderPolicy
from ._identity import gateway_identity
from ._inference import (
    PREFIX_TELEMETRY_ENV,
    InferenceBackend,
    InferenceUpstreamPool,
    announce,
    telemetry_enabled,
)
from ._rollout import (
    DEFAULT_HEALTH_TIMEOUT_S,
    RolloutError,
    default_current_socket_path,
    rollback,
    rollout,
)
from ._secrets import (
    GATEWAY_KEY_ENV,
    default_secrets_path,
    resolve_gateway_key,
    write_key,
)
from ._server import create_app, run_uvicorn
from ._session_state import GatewaySessionState
from ._settings import (
    default_admission_history_path,
    default_config_path,
    default_gateway_session_state_path,
    load_settings,
)
from ._unit import (
    DEFAULT_UNIT_DIR,
    UNIT_NAME,
    install_rollout_frontend,
    install_unit,
)

INSTALL_UNIT = "install-unit"
RESTART_UNIT = "restart-unit"
INSTALL_ROLLOUT = "install-rollout-units"
ROLLOUT = "rollout-generation"
ROLLBACK = "rollback-generation"

log = slogging.getLogger(__name__)


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
    parser.add_argument(
        "--inference-cache-report",
        action=argparse.BooleanOptionalAction,
        default=default,
        help=(
            "Request SGLang per-tier cache details on supported OpenAI routes "
            "(default: gateway.inference_cache_report_enabled, else disabled)."
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
    parser.add_argument("--uds", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gateway-build", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gateway-incarnation", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--frontend-generation", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--graceful-rollout-shutdown",
        action="store_true",
        help=argparse.SUPPRESS,
    )
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
    restart = commands.add_parser(
        RESTART_UNIT,
        help="close admission, wait until empty, then restart the systemd user unit",
    )
    restart.add_argument(
        "--config",
        type=Path,
        default=argparse.SUPPRESS,
        help="settings file used to derive the default loopback health URL",
    )
    restart.add_argument(
        "--health-url",
        default=None,
        help="gateway health URL (default: http://127.0.0.1:<configured port>/health)",
    )
    restart.add_argument(
        "--port",
        type=int,
        default=argparse.SUPPRESS,
        help="configured gateway port used to derive the default health URL",
    )
    restart.add_argument(
        "--drain-timeout-s",
        type=float,
        default=DEFAULT_DRAIN_TIMEOUT_S,
        help=f"maximum drain wait in seconds (default: {DEFAULT_DRAIN_TIMEOUT_S:g})",
    )
    restart.add_argument(
        "--poll-interval-s",
        type=float,
        default=DEFAULT_POLL_INTERVAL_S,
        help=(
            "deprecated compatibility option; the server-side barrier no longer "
            "polls health"
        ),
    )
    install_rollout = commands.add_parser(
        INSTALL_ROLLOUT,
        help="write the stable socket-proxy frontend units without starting them",
    )
    install_rollout.add_argument("--config", type=Path, default=None)
    install_rollout.add_argument("--unit-dir", type=Path, default=None)
    rollout_parser = commands.add_parser(
        ROLLOUT, help="verify and atomically promote one gateway generation"
    )
    rollout_parser.add_argument("--generation", required=True)
    rollout_parser.add_argument("--build", required=True)
    rollout_parser.add_argument("--config", type=Path, default=None)
    rollout_parser.add_argument("--unit-dir", type=Path, default=None)
    rollout_parser.add_argument("--runtime-dir", type=Path, default=None)
    rollout_parser.add_argument("--state-path", type=Path, default=None)
    rollout_parser.add_argument(
        "--health-timeout-s", type=float, default=DEFAULT_HEALTH_TIMEOUT_S
    )
    rollout_parser.add_argument(
        "--bootstrap-coordinated",
        action="store_true",
        help="acknowledge that all clients are paused for the one-time migration",
    )
    rollback_parser = commands.add_parser(
        ROLLBACK, help="atomically re-promote the retained previous generation"
    )
    rollback_parser.add_argument("--config", type=Path, default=None)
    rollback_parser.add_argument("--runtime-dir", type=Path, default=None)
    rollback_parser.add_argument("--state-path", type=Path, default=None)
    rollback_parser.add_argument(
        "--health-timeout-s", type=float, default=DEFAULT_HEALTH_TIMEOUT_S
    )
    return parser


def _telemetry_sink():
    """Log lines when ``HOIST_PREFIX_TELEMETRY`` asks for it, else off."""
    if not telemetry_enabled(os.getenv(PREFIX_TELEMETRY_ENV, "")):
        return None
    return lambda line: log.info(line)


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
    log.info(f"scitex-genai-gateway: key {key.origin} -> {stored}")


def _install_unit(args: argparse.Namespace) -> None:
    unit_dir = args.unit_dir if args.unit_dir is not None else DEFAULT_UNIT_DIR
    _persist_key(replacing_a_unit=(Path(unit_dir) / UNIT_NAME).exists())
    path = install_unit(
        host=args.host,
        port=args.port,
        inference_timeout_s=args.inference_timeout_s,
        inference_capacity_per_upstream=args.inference_capacity_per_upstream,
        inference_max_queue_size=args.inference_max_queue_size,
        inference_continuation_qos_enabled=args.inference_continuation_qos,
        inference_continuation_qos_max_retries=(
            args.inference_continuation_qos_max_retries
        ),
        inference_continuation_qos_min_preempt_tokens=(
            args.inference_continuation_qos_min_preempt_tokens
        ),
        inference_cache_report_enabled=args.inference_cache_report,
        config=args.config,
        unit_dir=args.unit_dir,
        enable=not args.no_enable,
    )
    state = "written only" if args.no_enable else "reloaded and enabled --now"
    log.info(f"scitex-genai-gateway: {UNIT_NAME} -> {path} ({state})")


def main(
    argv: list[str] | None = None,
    *,
    server_runner: Callable[..., None] | None = None,
    drain_runner: Callable[..., None] | None = None,
) -> None:
    args = build_parser().parse_args(argv)
    if args.command == INSTALL_UNIT:
        _install_unit(args)
        return
    if args.command == RESTART_UNIT:
        settings = load_settings(
            getattr(args, "config", None), port=getattr(args, "port", None)
        )
        health_url = args.health_url or f"http://127.0.0.1:{settings.port}/health"
        key = resolve_gateway_key()
        try:
            (drain_runner or restart_when_drained)(
                health_url=health_url,
                api_key=key.value,
                timeout_s=args.drain_timeout_s,
                poll_interval_s=args.poll_interval_s,
            )
        except (DrainError, ValueError) as exc:
            raise SystemExit(f"refusing to restart: {exc}") from exc
        return
    if args.command == INSTALL_ROLLOUT:
        settings = load_settings(args.config)
        paths = install_rollout_frontend(
            host=settings.host,
            port=settings.port,
            current_socket=default_current_socket_path(),
            unit_dir=args.unit_dir,
        )
        log.info(
            "scitex-genai-gateway: rollout frontend written but not started: "
            + ", ".join(str(path) for path in paths),
        )
        return
    if args.command in {ROLLOUT, ROLLBACK}:
        settings = load_settings(args.config)
        health_url = f"http://127.0.0.1:{settings.port}/health"
        try:
            if args.command == ROLLOUT:
                expected_members = {
                    upstream.label: {
                        "capacity": settings.inference_capacity_per_upstream,
                        "token_capacity": upstream.token_capacity,
                    }
                    for upstream in settings.inference_upstreams
                }
                rollout(
                    generation=args.generation,
                    build=args.build,
                    config=args.config or default_config_path(),
                    public_health_url=health_url,
                    runtime_dir=args.runtime_dir,
                    unit_dir=args.unit_dir,
                    state_path=args.state_path,
                    health_timeout_s=args.health_timeout_s,
                    bootstrap_coordinated=args.bootstrap_coordinated,
                    expected_members=(expected_members or None),
                )
            else:
                rollback(
                    public_health_url=health_url,
                    runtime_dir=args.runtime_dir,
                    state_path=args.state_path,
                    health_timeout_s=args.health_timeout_s,
                )
        except (RolloutError, ValueError) as exc:
            raise SystemExit(f"rollout refused: {exc}") from exc
        return
    try:
        __import__("uvicorn")
    except ImportError as exc:
        raise SystemExit("Install scitex-genai[gateway] to run the server") from exc
    settings = load_settings(
        args.config,
        host=args.host,
        port=args.port,
        inference_timeout_s=args.inference_timeout_s,
        inference_capacity_per_upstream=args.inference_capacity_per_upstream,
        inference_max_queue_size=args.inference_max_queue_size,
        inference_continuation_qos_enabled=args.inference_continuation_qos,
        inference_continuation_qos_max_retries=(
            args.inference_continuation_qos_max_retries
        ),
        inference_continuation_qos_min_preempt_tokens=(
            args.inference_continuation_qos_min_preempt_tokens
        ),
        inference_cache_report_enabled=args.inference_cache_report,
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
            token_capacity_per_upstream=None,
        )
        backend = ExternalProviderBackend(
            pool,
            timeout_s=settings.inference_timeout_s,
            journal=lambda line: log.info(line),
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
        log.info(
            "scitex-genai-gateway: external provider "
            f"{external.provider} -> {external.upstream}; outbound model policy: "
            f"{external.canonical_model} only",
        )
    elif settings.inference_upstreams:
        pool = InferenceUpstreamPool.from_specs(
            settings.inference_upstreams,
            capacity_per_upstream=settings.inference_capacity_per_upstream,
            max_queue_size=settings.inference_max_queue_size,
            max_admission_bypasses=settings.cache_admission.max_hot_bypasses,
            priority_aging_s=settings.cache_admission.starvation_age_s,
            cache_prediction_max_age_s=settings.cache_admission.evidence_max_age_s,
            cold_prefill_limit_per_upstream=(
                settings.cache_admission.cold_prefill_limit_per_upstream
                if settings.cache_admission.active
                else None
            ),
            cold_prefill_min_tokens=(
                settings.cache_admission.hot_max_uncached_tokens
                if settings.cache_admission.active
                else None
            ),
            session_state=GatewaySessionState(default_gateway_session_state_path()),
        )
        backend = InferenceBackend(
            pool,
            timeout_s=settings.inference_timeout_s,
            telemetry_sink=_telemetry_sink(),
            journal=lambda line: log.info(line),
            continuation_qos_enabled=(settings.inference_continuation_qos_enabled),
            continuation_qos_max_retries=(
                settings.inference_continuation_qos_max_retries
            ),
            continuation_qos_min_preempt_tokens=(
                settings.inference_continuation_qos_min_preempt_tokens
            ),
            cache_report_enabled=settings.inference_cache_report_enabled,
            cache_admission_settings=settings.cache_admission,
            admission_history_path=default_admission_history_path(),
        )
        log.info(announce(settings.host, settings.port, pool))
    else:
        codex_pool = CodexAccountPool.discover()
        backend = CodexBackend(codex_pool, CodexTransport(base_url=args.codex_base_url))
    key = resolve_gateway_key(create=True)
    log.info(f"scitex-genai-gateway: key {key.origin}")
    identity = gateway_identity(
        build=args.gateway_build,
        incarnation=args.gateway_incarnation,
        frontend_generation=args.frontend_generation,
    )
    app = create_app(backend, api_key=key.value, identity=identity)
    app.state.scitex_backend = backend
    listen = (
        {"uds": str(args.uds)}
        if args.uds is not None
        else {"host": settings.host, "port": settings.port}
    )
    server_kwargs: dict[str, object] = {**listen, "log_level": args.log_level}
    if args.graceful_rollout_shutdown:
        server_kwargs["close_admission_on_shutdown"] = False
    (server_runner or run_uvicorn)(app, **server_kwargs)


if __name__ == "__main__":
    main()
