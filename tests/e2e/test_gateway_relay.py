"""E2E: gateway relays one request to a real loopback upstream (PS-212).

The full story — ``install-unit`` writes a real unit, ``create_app`` serves
a real ``InferenceBackend`` pool pointed at a real ``http.server`` in a
thread, and a request posted through the ASGI app comes back with the
upstream's bytes. Auth is enforced end to end (401 without the key).
No network beyond loopback, no credentials beyond tmp files.
"""

from __future__ import annotations

import asyncio
import http.server
import threading
from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("fastapi")

import httpx

from scitex_genai.gateway._inference import (
    InferenceBackend,
    InferenceUpstreamPool,
)
from scitex_genai.gateway._server import create_app
from scitex_genai.gateway._unit import UNIT_NAME, render_unit

pytestmark = pytest.mark.e2e

API_KEY = "e2e-relay-secret"


class _Upstream(http.server.BaseHTTPRequestHandler):
    """A vLLM-shaped upstream: reads the body, replies chunked JSON."""

    protocol_version = "HTTP/1.1"

    def _serve(self) -> None:
        length = int(self.headers.get("content-length") or 0)
        if length:
            self.rfile.read(length)
        chunk = b'{"id":"resp-e2e","object":"response","output":"hello-e2e"}'
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    do_GET = _serve
    do_POST = _serve

    def log_message(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        pass


@pytest.fixture
def upstream_url() -> Iterator[str]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_unit_text_names_the_configured_port(tmp_path) -> None:
    # Arrange — the unit is what the gateway actually execs, so the port
    # baked in here is the port the deployment serves.
    config = tmp_path / "config.yaml"
    config.write_text("gateway:\n  port: 18772\n")

    # Act
    text = render_unit(host="127.0.0.1", port=18772, config=config)

    # Assert
    assert ("ExecStart=" in text, "--port" in text, "18772" in text) == (
        True,
        True,
        True,
    )
    assert UNIT_NAME == "scitex-genai-gateway.service"


async def _post(client: httpx.AsyncClient, headers: dict) -> httpx.Response:
    return await client.post(
        "/v1/responses",
        json={"model": "m", "input": "hello"},
        headers=headers,
    )


def test_relay_returns_the_upstream_bytes(upstream_url: str) -> None:
    # Arrange
    backend = InferenceBackend(InferenceUpstreamPool.from_urls(upstream_url))
    app = create_app(backend, api_key=API_KEY)

    async def run() -> tuple[httpx.Response, httpx.Response]:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://gateway.test",
            ) as client:
                ok = await _post(client, {"Authorization": f"Bearer {API_KEY}"})
                denied = await _post(client, {})
                return ok, denied

    # Act
    ok, denied = asyncio.run(run())

    # Assert
    assert (ok.status_code, "hello-e2e" in ok.text) == (200, True)
    assert denied.status_code == 401


def test_health_reports_the_real_upstream(upstream_url: str) -> None:
    # Arrange
    backend = InferenceBackend(InferenceUpstreamPool.from_urls(upstream_url))
    app = create_app(backend, api_key=API_KEY)

    async def run() -> httpx.Response:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://gateway.test",
            ) as client:
                return await client.get("/health")

    # Act
    response = asyncio.run(run())

    # Assert
    assert (response.status_code, response.json()["status"]) == (200, "ok")
