"""Gateway fixtures: a REAL HTTP upstream on an ephemeral loopback port.

Not a mock (PA-306). It is Python's ``http.server`` in a thread answering over
a socket, so the relay under test opens a real connection, sends real bytes,
and reads a real chunked reply — the same path a vLLM upstream exercises.
"""

from __future__ import annotations

import http.server
import socket
import threading
from collections.abc import Callable, Iterator
from typing import Any

import pytest


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
