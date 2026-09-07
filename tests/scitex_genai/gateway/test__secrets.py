"""The gateway key's home: resolution order, who may mint, and file safety.

Every test here isolates SCITEX_GENAI_GATEWAY_API_KEY through a fixture that
touches the REAL ``os.environ`` and restores it on teardown. That is deliberate
twice over. These tests assert on a value read from the process environment, and
this fleet injects that class of variable into agent containers -- a test that
inherits it silently stops testing the logic and starts testing the ambient
environment, green in CI where the variable is unset and disarmed inside any
injected container. And the isolation uses the real environment rather than a
patching fixture, because the production code reads the real one.
"""

from __future__ import annotations

import contextlib
import os
import stat

import pytest

from scitex_genai.gateway._errors import CredentialError
from scitex_genai.gateway._secrets import (
    GATEWAY_KEY_ENV,
    GatewayKey,
    default_secrets_path,
    new_key,
    parse_secrets,
    read_secrets,
    resolve_gateway_key,
    write_key,
)
from scitex_genai.gateway._settings import default_config_path


@pytest.fixture(autouse=True)
def gateway_key_env():
    """Clear the real variable, yield a setter, restore what was there before."""
    original = os.environ.pop(GATEWAY_KEY_ENV, None)
    yield lambda value: os.environ.__setitem__(GATEWAY_KEY_ENV, value)
    os.environ.pop(GATEWAY_KEY_ENV, None)
    if original is not None:
        os.environ[GATEWAY_KEY_ENV] = original


@pytest.fixture
def secrets_path(tmp_path):
    """A real path under tmp_path; nothing is created until a test writes."""
    return tmp_path / "genai" / "secrets"


# --- parsing ---------------------------------------------------------------


def test_parse_reads_a_name_value_line():
    # Arrange
    text = "NAME=value"
    # Act
    values = parse_secrets(text)
    # Assert
    assert values == {"NAME": "value"}


def test_parse_ignores_comments_and_blank_lines():
    # Arrange
    text = "# note\n\nNAME=value\n"
    # Act
    values = parse_secrets(text)
    # Assert
    assert values == {"NAME": "value"}


def test_parse_keeps_a_value_containing_equals():
    # Arrange
    text = "NAME=a=b"
    # Act
    values = parse_secrets(text)
    # Assert
    assert values["NAME"] == "a=b"


def test_parse_strips_surrounding_whitespace_from_a_value():
    # Arrange
    text = "NAME=  value  "
    # Act
    values = parse_secrets(text)
    # Assert
    assert values["NAME"] == "value"


def test_parse_refuses_a_line_with_no_equals():
    # Arrange
    text = "NAME"
    # Act / Assert is one raising call
    # Assert
    with pytest.raises(CredentialError, match="not NAME=value"):
        parse_secrets(text)


def test_parse_refuses_an_export_line_rather_than_storing_a_wrong_name():
    # Arrange: the profile syntax this file replaces must fail loud, not parse
    text = "export NAME=value"
    # Act
    # Assert
    with pytest.raises(CredentialError, match="whitespace in the name"):
        parse_secrets(text)


def test_parse_names_the_offending_line_number():
    # Arrange
    text = "GOOD=1\nbad line\n"
    # Act
    # Assert
    with pytest.raises(CredentialError, match="line 2"):
        parse_secrets(text)


# --- reading ---------------------------------------------------------------


def test_read_returns_empty_when_there_is_no_file(secrets_path):
    # Arrange: secrets_path deliberately does not exist
    target = secrets_path
    # Act
    values = read_secrets(target)
    # Assert
    assert values == {}


def test_read_returns_the_stored_key(secrets_path):
    # Arrange
    write_key("abc", secrets_path)
    # Act
    values = read_secrets(secrets_path)
    # Assert
    assert values[GATEWAY_KEY_ENV] == "abc"


# --- writing ---------------------------------------------------------------


def test_write_creates_the_file(secrets_path):
    # Arrange
    target = secrets_path
    # Act
    written = write_key("abc", target)
    # Assert
    assert written.is_file()


def test_write_makes_the_file_owner_only(secrets_path):
    # Arrange
    target = secrets_path
    # Act
    write_key("abc", target)
    # Assert
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_write_makes_a_directory_it_creates_owner_only(secrets_path):
    # Arrange: the parent does not exist yet
    target = secrets_path
    # Act
    write_key("abc", target)
    # Assert
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700


def test_write_leaves_an_existing_directory_permissions_alone(secrets_path):
    """~/.scitex/genai is shared with config.yaml; do not re-permission it."""
    # Arrange
    secrets_path.parent.mkdir(parents=True)
    secrets_path.parent.chmod(0o755)
    # Act
    write_key("abc", secrets_path)
    # Assert
    assert stat.S_IMODE(secrets_path.parent.stat().st_mode) == 0o755


def test_write_still_protects_the_file_in_an_existing_directory(secrets_path):
    # Arrange
    secrets_path.parent.mkdir(parents=True)
    secrets_path.parent.chmod(0o755)
    # Act
    write_key("abc", secrets_path)
    # Assert
    assert stat.S_IMODE(secrets_path.stat().st_mode) == 0o600


def test_write_preserves_other_names_already_in_the_file(secrets_path):
    # Arrange
    secrets_path.parent.mkdir(parents=True)
    secrets_path.write_text("OTHER=keep\n", encoding="utf-8")
    # Act
    write_key("abc", secrets_path)
    # Assert
    assert read_secrets(secrets_path)["OTHER"] == "keep"


def test_write_replaces_an_existing_key(secrets_path):
    # Arrange
    write_key("first", secrets_path)
    # Act
    write_key("second", secrets_path)
    # Assert
    assert read_secrets(secrets_path)[GATEWAY_KEY_ENV] == "second"


def test_write_refuses_an_empty_value(secrets_path):
    # Arrange
    target = secrets_path
    # Act
    # Assert
    with pytest.raises(CredentialError, match="empty gateway key"):
        write_key("", target)


# --- resolution order ------------------------------------------------------


def test_environment_wins_over_the_file(gateway_key_env, secrets_path):
    # Arrange: adding a home must not move one out from under a host exporting it
    write_key("from-file", secrets_path)
    gateway_key_env("from-env")
    # Act
    key = resolve_gateway_key(secrets_path)
    # Assert
    assert key.value == "from-env"


def test_environment_origin_is_reported(gateway_key_env, secrets_path):
    # Arrange
    gateway_key_env("from-env")
    # Act
    key = resolve_gateway_key(secrets_path)
    # Assert
    assert key.origin == "environment"


def test_file_is_used_when_the_environment_is_unset(secrets_path):
    # Arrange
    write_key("from-file", secrets_path)
    # Act
    key = resolve_gateway_key(secrets_path)
    # Assert
    assert key.value == "from-file"


def test_file_origin_is_reported(secrets_path):
    # Arrange
    write_key("from-file", secrets_path)
    # Act
    key = resolve_gateway_key(secrets_path)
    # Assert
    assert key.origin == "file"


def test_an_empty_environment_value_does_not_win(gateway_key_env, secrets_path):
    # Arrange: the outage's exact shape, the variable set to the empty string
    write_key("from-file", secrets_path)
    gateway_key_env("")
    # Act
    key = resolve_gateway_key(secrets_path)
    # Assert
    assert key.value == "from-file"


def test_a_whitespace_environment_value_does_not_win(gateway_key_env, secrets_path):
    # Arrange
    write_key("from-file", secrets_path)
    gateway_key_env("   ")
    # Act
    key = resolve_gateway_key(secrets_path)
    # Assert
    assert key.value == "from-file"


# --- who may mint ----------------------------------------------------------


def test_a_client_refuses_to_mint(secrets_path):
    # Arrange: a key minted where no gateway runs opens nothing
    target = secrets_path
    # Act
    # Assert
    with pytest.raises(CredentialError, match="no gateway key"):
        resolve_gateway_key(target)


def test_the_refusal_names_the_file_to_distribute(secrets_path):
    # Arrange
    target = secrets_path
    # Act
    # Assert
    with pytest.raises(CredentialError, match=str(secrets_path)):
        resolve_gateway_key(target)


def test_the_refusal_names_the_variable(secrets_path):
    # Arrange
    target = secrets_path
    # Act
    # Assert
    with pytest.raises(CredentialError, match=GATEWAY_KEY_ENV):
        resolve_gateway_key(target)


def test_a_client_does_not_create_the_file_when_it_refuses(secrets_path):
    # Arrange
    target = secrets_path
    # Act
    with contextlib.suppress(CredentialError):
        resolve_gateway_key(target)
    # Assert
    assert not target.exists()


def test_create_mints_when_nothing_resolves(secrets_path):
    # Arrange
    target = secrets_path
    # Act
    key = resolve_gateway_key(target, create=True)
    # Assert
    assert key.origin == "created"


def test_a_minted_key_is_persisted(secrets_path):
    # Arrange
    minted = resolve_gateway_key(secrets_path, create=True)
    # Act
    stored = read_secrets(secrets_path)[GATEWAY_KEY_ENV]
    # Assert
    assert stored == minted.value


def test_a_minted_key_is_written_owner_only(secrets_path):
    # Arrange
    target = secrets_path
    # Act
    resolve_gateway_key(target, create=True)
    # Assert
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_create_does_not_replace_a_key_already_in_the_file(secrets_path):
    # Arrange: install-unit runs repeatedly and must not rotate the fleet's key
    write_key("existing", secrets_path)
    # Act
    key = resolve_gateway_key(secrets_path, create=True)
    # Assert
    assert key.value == "existing"


def test_create_captures_the_environment_key_rather_than_minting(
    gateway_key_env, secrets_path
):
    # Arrange
    gateway_key_env("from-profile")
    # Act
    key = resolve_gateway_key(secrets_path, create=True)
    # Assert
    assert key.origin == "environment"


# --- the key itself --------------------------------------------------------


def test_a_new_key_is_sixty_four_characters():
    # Arrange
    # Act
    value = new_key()
    # Assert
    assert len(value) == 64


def test_a_new_key_is_hexadecimal():
    # Arrange
    # Act
    value = new_key()
    # Assert
    assert all(character in "0123456789abcdef" for character in value)


def test_two_new_keys_differ():
    # Arrange
    first = new_key()
    # Act
    second = new_key()
    # Assert
    assert first != second


# --- the dataclass ---------------------------------------------------------


def test_the_key_is_redacted_in_its_representation():
    # Arrange
    key = GatewayKey(value="s3cret", origin="file", path=None)
    # Act
    shown = repr(key)
    # Assert
    assert "s3cret" not in shown


def test_the_representation_still_names_the_origin():
    # Arrange
    key = GatewayKey(value="s3cret", origin="file", path=None)
    # Act
    shown = repr(key)
    # Assert
    assert "file" in shown


def test_an_empty_value_is_refused_at_construction():
    # Arrange
    value = ""
    # Act
    # Assert
    with pytest.raises(CredentialError, match="empty value"):
        GatewayKey(value=value, origin="file", path=None)


def test_an_unknown_origin_is_refused_at_construction():
    # Arrange
    origin = "guessed"
    # Act
    # Assert
    with pytest.raises(CredentialError, match="origin must be one of"):
        GatewayKey(value="abc", origin=origin, path=None)


# --- the default location --------------------------------------------------


def test_the_default_path_sits_beside_the_settings_file():
    # Arrange
    settings_directory = default_config_path().parent
    # Act
    secrets_directory = default_secrets_path().parent
    # Assert
    assert secrets_directory == settings_directory


def test_the_default_path_is_named_secrets():
    # Arrange
    # Act
    path = default_secrets_path()
    # Assert
    assert path.name == "secrets"


def test_the_default_path_is_absolute():
    # Arrange
    # Act
    path = default_secrets_path()
    # Assert
    assert path.is_absolute()
