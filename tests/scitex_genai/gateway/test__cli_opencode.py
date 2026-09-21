"""The gateway CLI selects the OpenCode serve-harness backend when asked.

Every case isolates ``SCITEX_GENAI_OPENCODE_SERVE_URL`` by hand: the ambient
environment may or may not define it, and the cascade under test reads it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from scitex_genai.gateway._cli import build_parser, main
from scitex_genai.gateway._opencode import OpenCodeBackend
from scitex_genai.gateway._server import create_app
from scitex_genai.gateway._settings import OPENCODE_SERVE_ENV

SERVE_ENV = OPENCODE_SERVE_ENV


@pytest.fixture
def isolated_serve_env():
    """The cascade reads one env name; hold its ambient value aside."""
    saved = os.environ.pop(SERVE_ENV, None)
    try:
        yield
    finally:
        os.environ.pop(SERVE_ENV, None)
        if saved is not None:
            os.environ[SERVE_ENV] = saved


def _collect(argv: list[str], gateway_key_env) -> tuple[Any, dict]:
    gateway_key_env("test-key")
    calls: list[tuple[Any, dict]] = []
    main(argv, server_runner=lambda app, **kwargs: calls.append((app, kwargs)))
    return calls[0]


def test_flag_defaults_to_unset_so_the_settings_file_decides(
    isolated_serve_env,
) -> None:
    # Arrange
    parser = build_parser()
    # Act
    args = parser.parse_args([])
    # Assert
    assert args.opencode_serve_url is None


def test_flag_parses_the_serve_url(isolated_serve_env) -> None:
    # Arrange
    parser = build_parser()
    # Act
    args = parser.parse_args(["--opencode-serve-url", "http://127.0.0.1:4096"])
    # Assert
    assert args.opencode_serve_url == "http://127.0.0.1:4096"


def _write_gateway_config(tmp_path: Path, body: str) -> str:
    config = tmp_path / "config.yaml"
    config.write_text(body)
    return str(config)


def test_main_selects_opencode_backend_type_from_flag(
    tmp_path: Path, gateway_key_env, isolated_serve_env
) -> None:
    # Arrange
    config = _write_gateway_config(tmp_path, "gateway:\n  port: 8765\n")
    argv = ["--config", config, "--opencode-serve-url", "http://127.0.0.1:4096"]
    # Act
    backend = _collect(argv, gateway_key_env)[0].state.scitex_backend
    # Assert
    assert isinstance(backend, OpenCodeBackend)


def test_main_selects_opencode_serve_url_from_flag(
    tmp_path: Path, gateway_key_env, isolated_serve_env
) -> None:
    # Arrange
    config = _write_gateway_config(tmp_path, "gateway:\n  port: 8765\n")
    argv = ["--config", config, "--opencode-serve-url", "http://127.0.0.1:4096"]
    # Act
    backend = _collect(argv, gateway_key_env)[0].state.scitex_backend
    # Assert
    assert backend.serve_url == "http://127.0.0.1:4096"


def test_main_selects_opencode_backend_type_from_settings_file(
    tmp_path: Path, gateway_key_env, isolated_serve_env
) -> None:
    # Arrange
    config = _write_gateway_config(
        tmp_path,
        "gateway:\n  port: 8765\n  opencode_serve_url: http://127.0.0.1:4097\n",
    )
    # Act
    backend = _collect(["--config", config], gateway_key_env)[0].state.scitex_backend
    # Assert
    assert isinstance(backend, OpenCodeBackend)


def test_main_selects_opencode_serve_url_from_settings_file(
    tmp_path: Path, gateway_key_env, isolated_serve_env
) -> None:
    # Arrange
    config = _write_gateway_config(
        tmp_path,
        "gateway:\n  port: 8765\n  opencode_serve_url: http://127.0.0.1:4097\n",
    )
    # Act
    backend = _collect(["--config", config], gateway_key_env)[0].state.scitex_backend
    # Assert
    assert backend.serve_url == "http://127.0.0.1:4097"


def test_main_selects_opencode_backend_type_from_environment(
    tmp_path: Path, gateway_key_env, isolated_serve_env
) -> None:
    # Arrange
    config = _write_gateway_config(tmp_path, "gateway:\n  port: 8765\n")
    os.environ[SERVE_ENV] = "http://127.0.0.1:4098"
    # Act
    backend = _collect(["--config", config], gateway_key_env)[0].state.scitex_backend
    # Assert
    assert isinstance(backend, OpenCodeBackend)


def test_main_selects_opencode_serve_url_from_environment(
    tmp_path: Path, gateway_key_env, isolated_serve_env
) -> None:
    # Arrange
    config = _write_gateway_config(tmp_path, "gateway:\n  port: 8765\n")
    os.environ[SERVE_ENV] = "http://127.0.0.1:4098"
    # Act
    backend = _collect(["--config", config], gateway_key_env)[0].state.scitex_backend
    # Assert
    assert backend.serve_url == "http://127.0.0.1:4098"


def test_flag_beats_settings_file_and_environment(
    tmp_path: Path, gateway_key_env, isolated_serve_env
) -> None:
    # Arrange
    config = tmp_path / "config.yaml"
    config.write_text(
        "gateway:\n  port: 8765\n  opencode_serve_url: http://127.0.0.1:4097\n"
    )
    os.environ[SERVE_ENV] = "http://127.0.0.1:4098"
    argv = [
        "--config",
        str(config),
        "--opencode-serve-url",
        "http://127.0.0.1:4099",
    ]
    # Act
    app, _kwargs = _collect(argv, gateway_key_env)
    # Assert
    assert app.state.scitex_backend.serve_url == "http://127.0.0.1:4099"


def test_settings_file_beats_environment(
    tmp_path: Path, gateway_key_env, isolated_serve_env
) -> None:
    # Arrange
    config = tmp_path / "config.yaml"
    config.write_text(
        "gateway:\n  port: 8765\n  opencode_serve_url: http://127.0.0.1:4097\n"
    )
    os.environ[SERVE_ENV] = "http://127.0.0.1:4098"
    argv = ["--config", str(config)]
    # Act
    app, _kwargs = _collect(argv, gateway_key_env)
    # Assert
    assert app.state.scitex_backend.serve_url == "http://127.0.0.1:4097"


def test_opencode_and_inference_upstreams_are_mutually_exclusive(
    tmp_path: Path, gateway_key_env, isolated_serve_env
) -> None:
    # Arrange
    config = tmp_path / "config.yaml"
    config.write_text(
        "gateway:\n"
        "  opencode_serve_url: http://127.0.0.1:4096\n"
        "  inference_upstreams:\n"
        "    - label: qwen-tp1\n"
        "      url: http://127.0.0.1:18773\n"
        "      token_capacity: 100\n"
    )
    # Act
    # Assert
    with pytest.raises(ValueError, match="mutually exclusive"):
        _collect(["--config", str(config)], gateway_key_env)


def _models_response(gateway_key_env):
    gateway_key_env("test-key")
    app = create_app(OpenCodeBackend(), api_key="test-key")
    client = TestClient(app)
    return client.get("/v1/models", headers={"Authorization": "Bearer test-key"})


def test_gateway_models_reports_success_status(
    gateway_key_env, isolated_serve_env
) -> None:
    # Arrange
    # Act
    response = _models_response(gateway_key_env)
    # Assert
    assert response.status_code == 200


def test_gateway_models_lists_opencode_model_entry(
    gateway_key_env, isolated_serve_env
) -> None:
    # Arrange
    # Act
    payload = _models_response(gateway_key_env).json()
    # Assert
    assert payload == {
        "object": "list",
        "data": [
            {
                "id": "muse-spark-1.3-contributor-free",
                "object": "model",
                "owned_by": "opencode",
            }
        ],
    }


def _run_mock_serve_completion(gateway_key_env) -> tuple[int, str, list[str]]:
    import http.server
    import threading

    seen: list[dict] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, payload: bytes) -> None:
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("content-length") or 0)
            raw = self.rfile.read(length) if length else b""
            seen.append({"path": self.path, "body": json.loads(raw or b"{}")})
            if self.path == "/session":
                self._send(b'{"id": "ses_smoke"}')
            else:
                self._send(
                    b'{"info": {"modelID": "m1"},'
                    b' "parts": [{"type": "text", "text": "SERVE_SMOKE"}]}'
                )

        def log_message(self, *args: Any) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        serve_url = f"http://127.0.0.1:{server.server_address[1]}"
        gateway_key_env("test-key")
        app = create_app(
            OpenCodeBackend(serve_url=serve_url), api_key="test-key"
        )
        client = TestClient(app)
        # Act
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-key"},
            json={"model": "m1", "messages": [{"role": "user", "content": "Hi"}]},
        )
    finally:
        server.shutdown()
        server.server_close()
    return (
        response.status_code,
        response.json()["choices"][0]["message"]["content"],
        [entry["path"] for entry in seen],
    )


def test_gateway_completions_reports_success_status(gateway_key_env) -> None:
    # Arrange
    # Act
    status, _text, _paths = _run_mock_serve_completion(gateway_key_env)
    # Assert
    assert status == 200


def test_gateway_completions_returns_serve_text(gateway_key_env) -> None:
    # Arrange
    # Act
    _status, text, _paths = _run_mock_serve_completion(gateway_key_env)
    # Assert
    assert text == "SERVE_SMOKE"


def test_gateway_completions_creates_session_before_messaging(
    gateway_key_env,
) -> None:
    # Arrange
    # Act
    _status, _text, paths = _run_mock_serve_completion(gateway_key_env)
    # Assert
    assert paths == ["/session", "/session/ses_smoke/message"]
