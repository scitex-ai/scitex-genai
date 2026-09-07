"""The gateway's auth key, in a file a plain process can read.

WHY THIS EXISTS. The key had no home but a login shell. ``_unit`` renders
``ExecStart=/bin/bash -lc`` for exactly that reason -- its own docstring says
so -- because the value lived in the user's profile as an ``export NAME=value``
line that systemd's ``EnvironmentFile=`` cannot read. That worked for the one
process which got a login shell, and for nothing else. Measured 2026-09-05/07 on
scitex-compute-04: the gateway was healthy for 33 hours (``/health`` 200,
no restarts) and served ZERO completions, answering 444 consecutive 401s,
because every client was launched WITHOUT a login shell, resolved the variable
to the empty string, and presented an empty bearer token.

So the key gets a home next to the settings it belongs with::

    ~/.scitex/genai/config.yaml     what this gateway is      (_settings)
    ~/.scitex/genai/secrets         what opens it             (here)

Both under ``$SCITEX_DIR``, both outside git, neither needing a shell.

RESOLUTION ORDER is environment, then file. The environment wins so that a host
still exporting the key keeps working unchanged -- this adds a home, it does not
move one out from under anybody.

WHO MAY CREATE A KEY. Only the process that VALIDATES it: the server, and
``install-unit`` acting on the server's behalf. A client never creates. That
distinction is the whole safety property, and it is a role, not a hostname --
minting a key on a host with no gateway would produce a file that looks like a
configured system and opens nothing, which is worse than the missing file it
replaced, because a missing file says so.
"""

from __future__ import annotations

import os
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path

from scitex_config import get_scitex_dir

from ._errors import CredentialError

#: The variable a spec names. Specs stay readable: they point at this name, and
#: this name is the key under which the file stores the value.
GATEWAY_KEY_ENV = "SCITEX_GENAI_GATEWAY_API_KEY"

#: 32 bytes as hex -- the 64 characters the fleet's existing key already is.
KEY_BYTES = 32

FILE_MODE = 0o600
DIR_MODE = 0o700


def default_secrets_path() -> Path:
    """``$SCITEX_DIR/genai/secrets`` -- ``~/.scitex/genai/secrets`` normally."""
    return Path(get_scitex_dir()) / "genai" / "secrets"


@dataclass(frozen=True, repr=False)
class GatewayKey:
    """One resolved key, and where it came from.

    ``value`` is redacted from the representation: this object is logged, put in
    tracebacks, and returned to callers that print what they got.
    """

    value: str
    origin: str
    path: Path | None

    #: Every value ``origin`` may take. A caller branching on it gets a closed
    #: set, not a string to guess at.
    ORIGINS = ("environment", "file", "created")

    def __post_init__(self) -> None:
        if not self.value:
            raise CredentialError("gateway key resolved to an empty value")
        if self.origin not in self.ORIGINS:
            raise CredentialError(
                f"gateway key origin must be one of {self.ORIGINS}, got {self.origin!r}"
            )

    def __repr__(self) -> str:
        return f"GatewayKey(origin={self.origin!r}, path={self.path!r}, redacted=True)"


def parse_secrets(text: str) -> dict[str, str]:
    """``NAME=value`` lines to a mapping; blanks and ``#`` comments ignored.

    Deliberately not a shell. A line the file does not define -- ``export
    NAME=value`` being the one people will actually try, since that is the
    profile syntax this file replaces -- is refused by name and line number
    rather than parsed into a key called ``export NAME``. A secrets file that
    silently stores the wrong name is the failure this whole module exists to
    end.
    """
    values: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or not name:
            raise CredentialError(
                f"secrets line {number} is not NAME=value: {line!r}"
            )
        if any(character.isspace() for character in name):
            raise CredentialError(
                f"secrets line {number} has whitespace in the name {name!r}; "
                f"write NAME=value with no 'export' and no spaces"
            )
        values[name] = value.strip()
    return values


def read_secrets(path: Path | str | None = None) -> dict[str, str]:
    """The file's contents, or an empty mapping when there is no file yet."""
    target = Path(path) if path is not None else default_secrets_path()
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise CredentialError(f"cannot read gateway secrets file: {target}") from exc
    return parse_secrets(text)


def write_key(value: str, path: Path | str | None = None) -> Path:
    """Store ``value`` under :data:`GATEWAY_KEY_ENV`, owner-readable only.

    Other names already in the file are preserved, and the file is opened at
    ``0600`` before anything is written into it, so the value is never briefly
    world-readable.

    A directory we CREATE is made ``0700``. One that already exists is left
    exactly as the user has it. That asymmetry matters: this path is shared with
    ``config.yaml``, which on a real host is a symlink into the user's dotfiles,
    so silently tightening an existing ``~/.scitex/genai`` would change access to
    a directory this function was only asked to add a file to. Protecting the
    secret is the file's mode; re-permissioning someone's config directory is a
    side effect nobody asked for.
    """
    if not value:
        raise CredentialError("refusing to write an empty gateway key")
    target = Path(path) if path is not None else default_secrets_path()
    existing = read_secrets(target)
    existing[GATEWAY_KEY_ENV] = value

    if not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(target.parent, DIR_MODE)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, FILE_MODE)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(
            "# scitex-genai gateway secrets. Written by the gateway; read by\n"
            "# clients WITHOUT a login shell. NAME=value, one per line.\n"
        )
        for name in sorted(existing):
            handle.write(f"{name}={existing[name]}\n")
    os.chmod(target, FILE_MODE)
    return target


def new_key() -> str:
    """A fresh key: 64 hex characters, from the system CSPRNG."""
    return secrets.token_hex(KEY_BYTES)


def resolve_gateway_key(
    path: Path | str | None = None, *, create: bool = False
) -> GatewayKey:
    """Resolve the gateway key: environment, then file, then optionally mint one.

    ``create`` is for the gateway and for ``install-unit`` only -- see this
    module's docstring on why a client must never mint. When nothing resolves
    and ``create`` is false, this raises and names both places it looked and the
    command that fixes it.
    """
    target = Path(path) if path is not None else default_secrets_path()

    from_environment = os.getenv(GATEWAY_KEY_ENV, "").strip()
    if from_environment:
        return GatewayKey(value=from_environment, origin="environment", path=None)

    from_file = read_secrets(target).get(GATEWAY_KEY_ENV, "").strip()
    if from_file:
        return GatewayKey(value=from_file, origin="file", path=target)

    if not create:
        raise CredentialError(
            f"no gateway key: {GATEWAY_KEY_ENV} is unset and {target} does not "
            f"supply it. This host is a CLIENT, so it must be given the key the "
            f"gateway already validates -- generating one here would open "
            f"nothing. Run `scitex-genai-gateway install-unit` on the gateway's "
            f"own host, then distribute {target}."
        )

    value = new_key()
    written = write_key(value, target)
    return GatewayKey(value=value, origin="created", path=written)


def secrets_file_mode(path: Path | str | None = None) -> int:
    """The file's permission bits, for a caller that wants to check them."""
    target = Path(path) if path is not None else default_secrets_path()
    return stat.S_IMODE(target.stat().st_mode)
