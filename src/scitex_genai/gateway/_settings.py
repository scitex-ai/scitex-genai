"""The gateway's settings come from the user's config tree, never from the package.

OPERATOR RULING (Telegram, 2026-09-05): a package carries the reusable, public
mechanism; settings particular to one deployment -- which address, which port,
which upstreams -- live in that user's config tree under ``~/.scitex/genai/``;
anything that is STATE goes to the Postgres store; and where a scitex-dev
primitive exists it is used rather than re-invented.

So this module reads ONE file, ``~/.scitex/genai/config.yaml``, through
scitex-config's ``ScitexConfig`` (the ecosystem's YAML + environment cascade)
and answers with a fixed dataclass. Precedence is the primitive's own:
direct (command line) -> config file -> environment -> default.
``HOIST_UPSTREAM`` stays the environment name for the upstream list because
the relay's users already export it::

    # ~/.scitex/genai/config.yaml
    gateway:
      host: 0.0.0.0
      port: 18772
      inference_upstreams:
        - http://127.0.0.1:18773
        - http://127.0.0.1:18774
      inference_timeout_s: 1800

A missing file is not an error: the package must run for someone who has no
config yet, on the defaults the command line always had (127.0.0.1:8765, the
Codex backend). ``source`` says which case applied.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from scitex_config import ScitexConfig, get_scitex_dir, load_yaml

from ._inference import (
    DEFAULT_CAPACITY_PER_UPSTREAM,
    DEFAULT_CONTINUATION_QOS_ENABLED,
    DEFAULT_CONTINUATION_QOS_MAX_RETRIES,
    DEFAULT_CONTINUATION_QOS_MIN_PREEMPT_TOKENS,
    DEFAULT_MAX_QUEUE_SIZE,
    DEFAULT_TIMEOUT_S,
    DEFAULT_TOKEN_CAPACITY_PER_UPSTREAM,
    TIMEOUT_ENV,
    UPSTREAM_ENV,
    parse_upstreams,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
KEY_HOST = "gateway.host"
KEY_PORT = "gateway.port"
KEY_UPSTREAMS = "gateway.inference_upstreams"
KEY_TIMEOUT = "gateway.inference_timeout_s"
KEY_CAPACITY = "gateway.inference_capacity_per_upstream"
KEY_MAX_QUEUE = "gateway.inference_max_queue_size"
KEY_TOKEN_CAPACITY = "gateway.inference_token_capacity_per_upstream"
KEY_CONTINUATION_QOS = "gateway.inference_continuation_qos_enabled"
KEY_CONTINUATION_RETRIES = "gateway.inference_continuation_qos_max_retries"
KEY_CONTINUATION_MIN_TOKENS = "gateway.inference_continuation_qos_min_preempt_tokens"
KEY_CACHE_REPORT = "gateway.inference_cache_report_enabled"
KEY_EXTERNAL_PROVIDER = "gateway.external_provider"

SCITEX_TIMEOUT_ENV = "SCITEX_GATEWAY_INFERENCE_TIMEOUT_S"


@dataclass(frozen=True)
class ExternalGatewaySettings:
    """Non-secret deployment settings for one paid-provider relay."""

    provider: str
    upstream: str
    upstream_auth_token_env: str
    canonical_model: str
    model_aliases: tuple[str, ...] = ()
    anthropic_path_prefix: str = ""
    max_tokens_per_request: int = 16_384
    max_requests_per_run: int | None = 100
    max_input_tokens_per_run: int | None = 5_000_000
    max_output_tokens_per_run: int | None = 200_000
    max_total_tokens_per_run: int | None = 5_200_000
    max_estimated_usd_per_run: float | None = None
    input_usd_per_million_tokens: float = 0.0
    output_usd_per_million_tokens: float = 0.0

    @classmethod
    def from_mapping(cls, value: Any) -> "ExternalGatewaySettings | None":
        if value in (None, "", {}):
            return None
        if not isinstance(value, dict):
            raise ValueError("gateway.external_provider must be a mapping")
        required = (
            "provider",
            "upstream",
            "upstream_auth_token_env",
            "canonical_model",
        )
        missing = [name for name in required if not str(value.get(name) or "").strip()]
        if missing:
            raise ValueError(
                "gateway.external_provider is missing: " + ", ".join(missing)
            )
        aliases = value.get("model_aliases") or []
        if isinstance(aliases, str):
            aliases = [part.strip() for part in aliases.split(",") if part.strip()]
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(
                "gateway.external_provider has unknown keys: " + ", ".join(unknown)
            )
        return cls(**{**value, "model_aliases": tuple(aliases)})

    def __post_init__(self) -> None:
        upstream = self.upstream.rstrip("/")
        if not upstream.startswith(("http://", "https://")):
            raise ValueError("external_provider.upstream must be an http(s) URL")
        if any(ch.isspace() for ch in upstream):
            raise ValueError("external_provider.upstream cannot contain whitespace")
        if not self.upstream_auth_token_env.replace("_", "").isalnum():
            raise ValueError(
                "external_provider.upstream_auth_token_env is not an env name"
            )
        object.__setattr__(self, "upstream", upstream)


def default_config_path() -> Path:
    """``$SCITEX_DIR/genai/config.yaml`` -- ``~/.scitex/genai/config.yaml`` normally."""
    return Path(get_scitex_dir()) / "genai" / "config.yaml"


def check_host(host: Any) -> str:
    """A bare address or name: non-empty, no whitespace."""
    text = str(host)
    if not text or any(ch.isspace() for ch in text):
        raise ValueError(f"host must be a bare address, got {host!r}")
    return text


def check_port(port: Any) -> int:
    """A TCP port in 1..65535."""
    number = int(port)
    if not 0 < number < 65536:
        raise ValueError(f"port must be within 1..65535, got {port!r}")
    return number


def check_timeout_s(timeout_s: Any) -> float:
    """A finite positive request timeout, in seconds."""
    number = float(timeout_s)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(
            "inference_timeout_s must be a finite number greater than 0, "
            f"got {timeout_s!r}"
        )
    return number


def check_count(name: str, value: Any, *, minimum: int) -> int:
    """An integer admission bound at or above minimum."""
    number = int(value)
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return number


def check_bool(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"1", "true", "yes", "on"}:
        return True
    if isinstance(value, str) and value.lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def upstream_string(value: Any) -> str:
    """The comma-separated form the server takes; a list, a string or nothing."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = ",".join(str(item) for item in value)
    return ",".join(parse_upstreams(str(value)))


@dataclass(frozen=True)
class GatewaySettings:
    """One gateway, fully described. ``inference_upstream`` empty = Codex backend."""

    host: str
    port: int
    inference_upstream: str
    source: Path | None
    inference_timeout_s: float = DEFAULT_TIMEOUT_S
    inference_capacity_per_upstream: int = DEFAULT_CAPACITY_PER_UPSTREAM
    inference_max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE
    inference_token_capacity_per_upstream: int | None = (
        DEFAULT_TOKEN_CAPACITY_PER_UPSTREAM
    )
    inference_continuation_qos_enabled: bool = DEFAULT_CONTINUATION_QOS_ENABLED
    inference_continuation_qos_max_retries: int = DEFAULT_CONTINUATION_QOS_MAX_RETRIES
    inference_continuation_qos_min_preempt_tokens: int = (
        DEFAULT_CONTINUATION_QOS_MIN_PREEMPT_TOKENS
    )
    inference_cache_report_enabled: bool = False
    external_provider: ExternalGatewaySettings | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "host", check_host(self.host))
        object.__setattr__(self, "port", check_port(self.port))
        object.__setattr__(
            self, "inference_upstream", upstream_string(self.inference_upstream)
        )
        object.__setattr__(
            self, "inference_timeout_s", check_timeout_s(self.inference_timeout_s)
        )
        object.__setattr__(
            self,
            "inference_capacity_per_upstream",
            check_count(
                "inference_capacity_per_upstream",
                self.inference_capacity_per_upstream,
                minimum=1,
            ),
        )
        if self.inference_token_capacity_per_upstream is not None:
            object.__setattr__(
                self,
                "inference_token_capacity_per_upstream",
                check_count(
                    "inference_token_capacity_per_upstream",
                    self.inference_token_capacity_per_upstream,
                    minimum=1,
                ),
            )
        if self.external_provider is not None and self.inference_upstream:
            raise ValueError(
                "gateway.external_provider and gateway.inference_upstreams are mutually exclusive"
            )
        object.__setattr__(
            self,
            "inference_max_queue_size",
            check_count(
                "inference_max_queue_size", self.inference_max_queue_size, minimum=0
            ),
        )
        object.__setattr__(
            self,
            "inference_continuation_qos_enabled",
            check_bool(
                "inference_continuation_qos_enabled",
                self.inference_continuation_qos_enabled,
            ),
        )
        object.__setattr__(
            self,
            "inference_continuation_qos_max_retries",
            check_count(
                "inference_continuation_qos_max_retries",
                self.inference_continuation_qos_max_retries,
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "inference_continuation_qos_min_preempt_tokens",
            check_count(
                "inference_continuation_qos_min_preempt_tokens",
                self.inference_continuation_qos_min_preempt_tokens,
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "inference_cache_report_enabled",
            check_bool(
                "inference_cache_report_enabled",
                self.inference_cache_report_enabled,
            ),
        )


def load_settings(
    config_path: Path | str | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
    inference_upstream: str | None = None,
    inference_timeout_s: float | None = None,
    inference_capacity_per_upstream: int | None = None,
    inference_max_queue_size: int | None = None,
    inference_token_capacity_per_upstream: int | None = None,
    inference_continuation_qos_enabled: bool | None = None,
    inference_continuation_qos_max_retries: int | None = None,
    inference_continuation_qos_min_preempt_tokens: int | None = None,
    inference_cache_report_enabled: bool | None = None,
) -> GatewaySettings:
    """Resolve the gateway's settings: direct -> config file -> environment -> default."""
    path = Path(config_path) if config_path is not None else default_config_path()
    present = path.is_file()
    config = ScitexConfig(config_path=path if present else None)
    upstream = config.resolve(
        KEY_UPSTREAMS, direct_val=inference_upstream, default=None
    )
    if upstream is None:
        upstream = os.getenv(UPSTREAM_ENV, "")
    # Spell this cascade out: ScitexConfig.resolve() would otherwise put its
    # implicit SCITEX_GATEWAY_INFERENCE_TIMEOUT_S ahead of the older,
    # documented HOIST_TIMEOUT_S contract.
    timeout = inference_timeout_s
    if timeout is None:
        timeout = config.get(KEY_TIMEOUT)
    if timeout is None:
        timeout = (
            os.getenv(TIMEOUT_ENV) or os.getenv(SCITEX_TIMEOUT_ENV) or DEFAULT_TIMEOUT_S
        )
    raw_config = load_yaml(path) if present else {}
    raw_gateway = raw_config.get("gateway", {}) if isinstance(raw_config, dict) else {}
    external_mapping = (
        raw_gateway.get("external_provider") if isinstance(raw_gateway, dict) else None
    )
    return GatewaySettings(
        host=config.resolve(KEY_HOST, direct_val=host, default=DEFAULT_HOST),
        port=config.resolve(KEY_PORT, direct_val=port, default=DEFAULT_PORT, type=int),
        inference_upstream=upstream,
        source=path if present else None,
        inference_timeout_s=timeout,
        inference_capacity_per_upstream=config.resolve(
            KEY_CAPACITY,
            direct_val=inference_capacity_per_upstream,
            default=DEFAULT_CAPACITY_PER_UPSTREAM,
        ),
        inference_max_queue_size=config.resolve(
            KEY_MAX_QUEUE,
            direct_val=inference_max_queue_size,
            default=DEFAULT_MAX_QUEUE_SIZE,
        ),
        inference_token_capacity_per_upstream=config.resolve(
            KEY_TOKEN_CAPACITY,
            direct_val=inference_token_capacity_per_upstream,
            default=DEFAULT_TOKEN_CAPACITY_PER_UPSTREAM,
        ),
        inference_continuation_qos_enabled=config.resolve(
            KEY_CONTINUATION_QOS,
            direct_val=inference_continuation_qos_enabled,
            default=DEFAULT_CONTINUATION_QOS_ENABLED,
        ),
        inference_continuation_qos_max_retries=config.resolve(
            KEY_CONTINUATION_RETRIES,
            direct_val=inference_continuation_qos_max_retries,
            default=DEFAULT_CONTINUATION_QOS_MAX_RETRIES,
        ),
        inference_continuation_qos_min_preempt_tokens=config.resolve(
            KEY_CONTINUATION_MIN_TOKENS,
            direct_val=inference_continuation_qos_min_preempt_tokens,
            default=DEFAULT_CONTINUATION_QOS_MIN_PREEMPT_TOKENS,
        ),
        inference_cache_report_enabled=config.resolve(
            KEY_CACHE_REPORT,
            direct_val=inference_cache_report_enabled,
            default=False,
        ),
        external_provider=ExternalGatewaySettings.from_mapping(external_mapping),
    )
