"""Gateway fixtures: a REAL HTTP upstream on an ephemeral loopback port.

Not a mock (PA-306). It is Python's ``http.server`` in a thread answering over
a socket, so the relay under test opens a real connection, sends real bytes,
and reads a real chunked reply — the same path a vLLM upstream exercises.
"""

from __future__ import annotations

import http.server
import os
import socket
import threading
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from scitex_genai.gateway._secrets import GATEWAY_KEY_ENV


@pytest.fixture(autouse=True)
def isolate_the_scitex_store(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """No gateway test may read or write the developer's real ``~/.scitex``.

    ``install-unit`` LEGITIMATELY writes the gateway key to
    ``$SCITEX_DIR/genai/secrets``, so any test that drives the CLI writes a real
    key into a real home unless the store is redirected. Measured 2026-09-07:
    it did exactly that on the machine this was written on, before this fixture
    existed -- a 64-character key appeared in ``~/.scitex/genai/secrets`` as a
    side effect of running the suite.

    The key variable is cleared for the same reason it is cleared in
    ``test__secrets``: this fleet injects ``SCITEX_*`` names into agent
    containers, and a test that inherits one stops testing the logic and starts
    testing the ambient environment -- passing in CI where it is unset and
    behaving differently everywhere it is set.

    Autouse and in ``conftest`` on purpose: a future test that calls the CLI
    cannot forget to do this, which is the difference between a rule and a
    barrier.
    """
    previous = {
        name: os.environ.get(name) for name in ("SCITEX_DIR", GATEWAY_KEY_ENV)
    }
    os.environ["SCITEX_DIR"] = str(tmp_path_factory.mktemp("scitex"))
    os.environ.pop(GATEWAY_KEY_ENV, None)
    yield
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@pytest.fixture
def gateway_key_env():
    """Set the real key variable for one test.

    Restoring is ``isolate_the_scitex_store``'s job -- it snapshots and puts
    back whatever was there, so a test that sets a key cannot leak it into the
    next one.
    """
    return lambda value: os.environ.__setitem__(GATEWAY_KEY_ENV, value)


class RecordingUpstream:
    """Replays one scripted reply per request and records what it was asked."""

    def __init__(
        self,
        *,
        status: int = 200,
        content_type: str = "application/json",
        chunks: tuple[bytes, ...] = (b'{"ok": true}',),
    ) -> None:
        self.status = status
        self.content_type = content_type
        self.chunks = chunks
        self.requests: list[dict[str, Any]] = []
        upstream = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _serve(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                upstream.requests.append(
                    {
                        "method": self.command,
                        "path": self.path,
                        "headers": {
                            name.lower(): value for name, value in self.headers.items()
                        },
                        "body": self.rfile.read(length) if length else b"",
                    }
                )
                self.send_response(upstream.status)
                self.send_header("content-type", upstream.content_type)
                self.send_header("transfer-encoding", "chunked")
                self.end_headers()
                for chunk in upstream.chunks:
                    self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

            do_GET = _serve
            do_POST = _serve

            def log_message(self, *args: Any) -> None:
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        # A short poll so ``shutdown()`` returns promptly at fixture teardown;
        # the default 0.5 s would add half a second per upstream to the suite.
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def upstream_factory() -> Iterator[Callable[..., RecordingUpstream]]:
    created: list[RecordingUpstream] = []

    def factory(**kwargs: Any) -> RecordingUpstream:
        upstream = RecordingUpstream(**kwargs)
        created.append(upstream)
        return upstream

    yield factory
    for upstream in created:
        upstream.close()


@pytest.fixture
def dead_url_factory() -> Iterator[Callable[[], str]]:
    """Loopback URLs nothing listens on: a port bound, released, and left closed."""

    def factory() -> str:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return f"http://127.0.0.1:{sock.getsockname()[1]}"

    yield factory
