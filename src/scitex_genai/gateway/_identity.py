"""Observable identity for one gateway process incarnation."""

from __future__ import annotations

import os
import secrets
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from scitex_genai import __version__

BUILD_ENV = "SCITEX_GENAI_GATEWAY_BUILD"
INCARNATION_ENV = "SCITEX_GENAI_GATEWAY_INCARNATION"
FRONTEND_GENERATION_ENV = "SCITEX_GENAI_GATEWAY_FRONTEND_GENERATION"


@dataclass(frozen=True)
class GatewayIdentity:
    """Build and routing identity exported by health and operator status."""

    build: str
    incarnation: str
    frontend_generation: str
    code_fingerprint: str = ""

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def gateway_identity(
    *,
    build: str | None = None,
    incarnation: str | None = None,
    frontend_generation: str | None = None,
) -> GatewayIdentity:
    """Resolve explicit values, deployment environment, then safe defaults."""
    return GatewayIdentity(
        build=(build or os.getenv(BUILD_ENV) or __version__).strip(),
        incarnation=(
            incarnation
            or os.getenv(INCARNATION_ENV)
            or os.getenv("INVOCATION_ID")
            or secrets.token_hex(12)
        ).strip(),
        frontend_generation=(
            frontend_generation or os.getenv(FRONTEND_GENERATION_ENV) or "direct"
        ).strip(),
        code_fingerprint=gateway_code_fingerprint(),
    )


def gateway_code_fingerprint(package_dir: Path | None = None) -> str:
    """Hash installed gateway sources so a build label cannot verify itself."""
    root = package_dir or Path(__file__).parent
    digest = sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
