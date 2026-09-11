#!/usr/bin/env python3
"""Black-box acceptance test for disconnect cancellation during cold prefill."""

from __future__ import annotations

import argparse
import http.client
import json
import socket
import threading
import time
import uuid
from urllib.parse import urlsplit


def _json_get(base_url: str, path: str, timeout: float = 5.0) -> dict:
    parsed = urlsplit(base_url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"GET {path} returned {response.status}: {body[:200]!r}")
        return json.loads(body)
    finally:
        connection.close()


def _load(base_url: str) -> tuple[int, int]:
    payload = _json_get(base_url, "/v1/loads?include=core")
    loads = payload.get("loads", [])
    return (
        sum(int(item.get("num_running_reqs", 0)) for item in loads),
        sum(int(item.get("num_waiting_reqs", 0)) for item in loads),
    )


class DisconnectingRequest:
    def __init__(self, base_url: str, payload: dict):
        parsed = urlsplit(base_url)
        if parsed.scheme != "http" or not parsed.hostname or not parsed.port:
            raise ValueError("acceptance target must be an explicit http://HOST:PORT URL")
        self.host = parsed.hostname
        self.port = parsed.port
        self.body = json.dumps(payload, separators=(",", ":")).encode()
        self.sent = threading.Event()
        self.disconnect = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()
        if not self.sent.wait(30):
            raise TimeoutError("request body was not sent within 30 seconds")
        if self.error is not None:
            raise RuntimeError("request sender failed") from self.error

    def close(self) -> None:
        self.disconnect.set()
        self.thread.join(10)
        if self.thread.is_alive():
            raise TimeoutError("request socket did not close")
        if self.error is not None:
            raise RuntimeError("request sender failed") from self.error

    def _run(self) -> None:
        sock: socket.socket | None = None
        try:
            sock = socket.create_connection((self.host, self.port), timeout=30)
            headers = (
                "POST /v1/chat/completions HTTP/1.1\r\n"
                f"Host: {self.host}:{self.port}\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(self.body)}\r\n"
                "Connection: close\r\n\r\n"
            ).encode()
            sock.sendall(headers + self.body)
            self.sent.set()
            self.disconnect.wait(120)
        except BaseException as exc:  # propagated to the controller thread
            self.error = exc
            self.sent.set()
        finally:
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                sock.close()


def _wait_for(base_url: str, predicate, timeout: float, description: str) -> None:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = _load(base_url)
        if predicate(last):
            return
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for {description}; last load={last}")


def run(args: argparse.Namespace) -> None:
    _wait_for(args.base_url, lambda value: value == (0, 0), 60, "an idle server")
    run_tag = f"scitex-cancel-{uuid.uuid4()}"
    # UUID makes each prompt a cold cache miss; the repeated payload forces more
    # than one configured 32K-token prefill chunk.
    prefix = f"{run_tag}\n"
    content = prefix + ("cold-prefill-token " * ((args.prompt_bytes - len(prefix)) // 19))
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 32,
        "stream": True,
        "rid": run_tag,
    }

    request = DisconnectingRequest(args.base_url, payload)
    request.start()
    _wait_for(
        args.base_url,
        lambda value: sum(value) > 0,
        args.busy_timeout,
        "the cold request to enter SGLang",
    )
    request.close()
    _wait_for(
        args.base_url,
        lambda value: value == (0, 0),
        args.drain_timeout,
        "the disconnected request to drain",
    )

    if args.log:
        log_text = open(args.log, encoding="utf-8", errors="replace").read()
        if run_tag not in log_text:
            raise AssertionError(f"request id {run_tag} was not found in the server log")
        bad_lines = [
            line
            for line in log_text.splitlines()
            if run_tag in line and "state was deleted" in line.lower()
        ]
        if bad_lines:
            raise AssertionError("TokenizerManager deleted request state: " + bad_lines[-1])

    print(f"PASS: {run_tag} disconnected during cold prefill and server returned idle")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--log")
    parser.add_argument("--prompt-bytes", type=int, default=2_400_000)
    parser.add_argument("--busy-timeout", type=float, default=120)
    parser.add_argument("--drain-timeout", type=float, default=120)
    args = parser.parse_args()
    if args.prompt_bytes < 256_000:
        parser.error("--prompt-bytes must be at least 256000 to exercise chunked prefill")
    run(args)


if __name__ == "__main__":
    main()

