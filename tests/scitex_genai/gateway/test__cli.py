"""The gateway CLI keeps its serve form and gains ``install-unit``.

The serve path is not started here (it binds a port and runs forever); what is
proved is the parse, and that ``install-unit`` writes a real file and returns
without touching a server.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import pytest

from scitex_genai.gateway._cli import INSTALL_UNIT, build_parser, main
from scitex_genai.gateway._secrets import (
    GATEWAY_KEY_ENV,
    default_secrets_path,
    read_secrets,
)
from scitex_genai.gateway._unit import UNIT_NAME, render_unit

UPSTREAM = "http://127.0.0.1:18773,http://127.0.0.1:18774"
INSTALL_ARGS = [
    INSTALL_UNIT,
    "--host",
    "0.0.0.0",
    "--port",
    "18772",
    "--inference-upstream",
    UPSTREAM,
]


def test_no_subcommand_is_the_serve_form():
    # Arrange
    parser = build_parser()

    # Act
    args = parser.parse_args(["--host", "0.0.0.0", "--port", "18772"])

    # Assert
    assert args.command is None


def test_serve_flags_default_to_unset_so_the_settings_file_decides():
    # Arrange
    parser = build_parser()

    # Act
    args = parser.parse_args([])

    # Assert
    assert (
        args.config,
        args.host,
        args.port,
        args.inference_upstream,
        args.inference_timeout_s,
        args.inference_capacity_per_upstream,
        args.inference_max_queue_size,
    ) == (None, None, None, None, None, None, None)


def test_install_unit_is_recognised():
    # Arrange
    parser = build_parser()

    # Act
    args = parser.parse_args([INSTALL_UNIT])

    # Assert
    assert args.command == INSTALL_UNIT


def test_install_unit_takes_the_settings_flags():
    # Arrange
    parser = build_parser()

    # Act
    args = parser.parse_args(INSTALL_ARGS)

    # Assert
    assert (args.host, args.port, args.inference_upstream) == (
        "0.0.0.0",
        18772,
        UPSTREAM,
    )


def test_install_unit_takes_an_inference_timeout():
    # Arrange
    parser = build_parser()

    # Act
    args = parser.parse_args([INSTALL_UNIT, "--inference-timeout-s", "1800"])

    # Assert
    assert args.inference_timeout_s == 1800.0


def test_timeout_before_install_unit_is_not_erased_by_subparser_defaults():
    # Arrange
    parser = build_parser()

    # Act
    args = parser.parse_args(["--inference-timeout-s", "123", INSTALL_UNIT])

    # Assert
    assert args.inference_timeout_s == 123.0


def test_all_shared_settings_accept_the_same_parent_or_subcommand_placement():
    # Arrange
    parser = build_parser()
    settings = [
        "--config",
        "/srv/genai/config.yaml",
        "--host",
        "0.0.0.0",
        "--port",
        "18772",
        "--inference-upstream",
        "http://one",
        "--inference-timeout-s",
        "123",
        "--inference-capacity-per-upstream",
        "3",
        "--inference-max-queue-size",
        "9",
    ]

    # Act
    before = parser.parse_args([*settings, INSTALL_UNIT])
    after = parser.parse_args([INSTALL_UNIT, *settings])

    # Assert
    names = (
        "config",
        "host",
        "port",
        "inference_upstream",
        "inference_timeout_s",
        "inference_capacity_per_upstream",
        "inference_max_queue_size",
    )
    assert tuple(getattr(before, name) for name in names) == tuple(
        getattr(after, name) for name in names
    )


def test_install_unit_takes_a_unit_dir():
    # Arrange
    parser = build_parser()

    # Act
    args = parser.parse_args([INSTALL_UNIT, "--unit-dir", "/somewhere/units"])

    # Assert
    assert args.unit_dir == Path("/somewhere/units")


def test_install_unit_takes_no_enable():
    # Arrange
    parser = build_parser()

    # Act
    args = parser.parse_args([INSTALL_UNIT, "--no-enable"])

    # Assert
    assert args.no_enable is True


def test_main_install_unit_writes_the_unit_without_starting_a_server(tmp_path: Path):
    # Arrange
    argv = [*INSTALL_ARGS, "--unit-dir", str(tmp_path), "--no-enable"]

    # Act
    main(argv)

    # Assert
    assert (tmp_path / UNIT_NAME).read_text() == render_unit(
        host="0.0.0.0", port=18772, upstream=UPSTREAM
    )


def test_main_install_unit_reports_the_path_and_the_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    # Arrange
    argv = [INSTALL_UNIT, "--unit-dir", str(tmp_path), "--no-enable"]

    # Act
    main(argv)

    # Assert
    assert (str(tmp_path / UNIT_NAME) in capsys.readouterr().out, "written only") == (
        True,
        "written only",
    )


def test_main_forwards_timeout_to_foreground_inference_backend(gateway_key_env):
    # Arrange
    gateway_key_env("test-key")
    calls = []

    def record(app, **kwargs):
        calls.append((app, kwargs))

    argv = [
        "--inference-upstream",
        "http://127.0.0.1:18773",
        "--inference-timeout-s",
        "123",
        "--inference-capacity-per-upstream",
        "3",
        "--inference-max-queue-size",
        "9",
    ]

    # Act
    main(argv, server_runner=record)

    # Assert
    app, kwargs = calls[0]
    pool = app.state.scitex_backend.pool
    assert (
        app.state.scitex_backend.timeout_s,
        kwargs,
        pool.upstreams[0].capacity,
        pool.max_queue_size,
    ) == (
        123.0,
        {"host": "127.0.0.1", "port": 8765, "log_level": "info"},
        3,
        9,
    )


def test_main_builds_external_backend_without_exposing_vendor_key(
    tmp_path: Path, gateway_key_env
):
    # Arrange
    gateway_key_env("local-key")
    os.environ["TEST_VENDOR_KEY"] = "vendor-key"
    path = tmp_path / "config.yaml"
    path.write_text(
        "gateway:\n"
        "  external_provider:\n"
        "    provider: deepseek\n"
        "    upstream: https://api.deepseek.com\n"
        "    upstream_auth_token_env: TEST_VENDOR_KEY\n"
        "    canonical_model: deepseek-flash\n"
        "    model_aliases: [deepseek-v4-flash]\n"
        "    anthropic_path_prefix: /anthropic\n"
    )
    calls = []
    # Act
    try:
        main(["--config", str(path)], server_runner=lambda app, **kw: calls.append(app))
    finally:
        os.environ.pop("TEST_VENDOR_KEY", None)
    backend = calls[0].state.scitex_backend
    # Assert
    assert (
        backend.provider,
        backend.policy.canonical_model,
        backend.policy.upstream_api_key,
    ) == ("external:deepseek", "deepseek-flash", "vendor-key")


def test_main_refuses_external_backend_when_vendor_key_is_absent(
    tmp_path: Path, gateway_key_env
):
    # Arrange
    gateway_key_env("local-key")
    os.environ.pop("ABSENT_VENDOR_KEY", None)
    path = tmp_path / "config.yaml"
    path.write_text(
        "gateway:\n"
        "  external_provider:\n"
        "    provider: deepseek\n"
        "    upstream: https://api.deepseek.com\n"
        "    upstream_auth_token_env: ABSENT_VENDOR_KEY\n"
        "    canonical_model: deepseek-flash\n"
    )
    # Act
    # Assert
    with pytest.raises(SystemExit, match="ABSENT_VENDOR_KEY"):
        main(["--config", str(path)], server_runner=lambda app, **kw: None)


def test_main_forwards_admission_bounds_to_generated_unit(
    tmp_path: Path, gateway_key_env
):
    # Arrange
    gateway_key_env("test-key")

    # Act
    main(
        [
            "--inference-capacity-per-upstream",
            "3",
            INSTALL_UNIT,
            "--inference-max-queue-size",
            "9",
            "--unit-dir",
            str(tmp_path),
            "--no-enable",
        ]
    )

    # Assert
    assert (tmp_path / UNIT_NAME).read_text() == render_unit(
        inference_capacity_per_upstream=3, inference_max_queue_size=9
    )


def test_main_forwards_parent_form_timeout_to_generated_unit(
    tmp_path: Path, gateway_key_env
):
    # Arrange
    gateway_key_env("test-key")
    argv = [
        "--inference-timeout-s",
        "123",
        INSTALL_UNIT,
        "--unit-dir",
        str(tmp_path),
        "--no-enable",
    ]

    # Act
    main(argv)

    # Assert
    assert (tmp_path / UNIT_NAME).read_text() == render_unit(inference_timeout_s=123)


def test_main_forwards_subcommand_form_timeout_to_generated_unit(
    tmp_path: Path, gateway_key_env
):
    # Arrange
    gateway_key_env("test-key")
    argv = [
        INSTALL_UNIT,
        "--inference-timeout-s",
        "321",
        "--unit-dir",
        str(tmp_path),
        "--no-enable",
    ]

    # Act
    main(argv)

    # Assert
    assert (tmp_path / UNIT_NAME).read_text() == render_unit(inference_timeout_s=321)


def test_a_first_install_mints_a_key_so_a_new_host_needs_no_manual_step(
    tmp_path: Path,
):
    # Arrange
    argv = [INSTALL_UNIT, "--unit-dir", str(tmp_path), "--no-enable"]

    # Act
    main(argv)

    # Assert
    assert len(read_secrets(default_secrets_path())[GATEWAY_KEY_ENV]) == 64


def test_installing_over_an_existing_gateway_refuses_to_mint_a_new_key(
    tmp_path: Path,
):
    """The regression this guard exists for.

    A host that already has a unit has clients holding a key. Re-running the
    install from a shell where the old key does not resolve -- the non-login
    case this whole change is about -- would mint a replacement nobody has, and
    every client would start getting 401s: the outage, recreated by the fix.
    """
    # Arrange
    (tmp_path / UNIT_NAME).write_text("[Service]\n", encoding="utf-8")
    argv = [INSTALL_UNIT, "--unit-dir", str(tmp_path), "--no-enable"]

    # Act
    # Assert
    with pytest.raises(SystemExit, match="refusing to install"):
        main(argv)


def test_the_refusal_says_which_shell_to_re_run_from(tmp_path: Path):
    # Arrange
    (tmp_path / UNIT_NAME).write_text("[Service]\n", encoding="utf-8")
    argv = [INSTALL_UNIT, "--unit-dir", str(tmp_path), "--no-enable"]

    # Act
    # Assert
    with pytest.raises(SystemExit, match=GATEWAY_KEY_ENV):
        main(argv)


def test_the_refusal_leaves_the_existing_unit_untouched(tmp_path: Path):
    # Arrange
    (tmp_path / UNIT_NAME).write_text("[Service]\n", encoding="utf-8")
    argv = [INSTALL_UNIT, "--unit-dir", str(tmp_path), "--no-enable"]

    # Act
    with contextlib.suppress(SystemExit):
        main(argv)

    # Assert
    assert (tmp_path / UNIT_NAME).read_text(encoding="utf-8") == "[Service]\n"


def test_install_unit_persists_a_key_that_only_the_environment_had(
    tmp_path: Path, gateway_key_env
):
    """Resolving is not persisting -- the bug this test exists for.

    The unit written by this command runs with NO login shell. A key that lives
    only in the installing shell's environment is therefore invisible to it, and
    the next start would mint a replacement and rotate every client's key.
    """
    # Arrange
    gateway_key_env("a-key-only-this-shell-has")
    argv = [INSTALL_UNIT, "--unit-dir", str(tmp_path), "--no-enable"]

    # Act
    main(argv)

    # Assert
    assert (
        read_secrets(default_secrets_path())[GATEWAY_KEY_ENV]
        == "a-key-only-this-shell-has"
    )


def test_install_unit_reports_where_the_key_was_stored(
    tmp_path: Path, gateway_key_env, capsys: pytest.CaptureFixture[str]
):
    # Arrange
    gateway_key_env("a-key-only-this-shell-has")
    argv = [INSTALL_UNIT, "--unit-dir", str(tmp_path), "--no-enable"]

    # Act
    main(argv)

    # Assert
    assert str(default_secrets_path()) in capsys.readouterr().out
