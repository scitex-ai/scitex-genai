"""E2E: benchmark replay against a real loopback canary (PS-212).

The full story — scenario file on disk, ``run_scenario`` over a real
socket to a real ``http.server`` in a thread, usage + scheduler/cache
telemetry compared — with zero network beyond loopback and zero API
keys. The canary speaks the SSE shape a vLLM ``/v1/chat/completions``
stream returns, so this exercises the same parse path production does.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from scitex_genai.benchmark._sglang_ab import load_scenario, run_scenario

pytestmark = pytest.mark.e2e

FRAMES = [
    'data: {"choices":[{"delta":{"content":"a"}}]}\n\n',
    'data: {"choices":[{"delta":{"content":"b"}}]}\n\n',
    'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,'
    '"total_tokens":12,"prompt_tokens_details":{"cached_tokens":8}},'
    '"meta_info":{"device_hit_tokens":8,"host_hit_tokens":1,'
    '"newly_computed_tokens":1,"queue_time_s":0.25}}\n\n',
    "data: [DONE]\n\n",
]

SCENARIO = {
    "schema_version": 1,
    "name": "e2e-loopback",
    "seed": 7,
    "configuration": {"scheduler": "fcfs"},
    "metadata": {"warmup": "done"},
    "requests": [
        {
            "id": "cold",
            "arrival_ms": 0,
            "cache_state": "cold",
            "expected_prompt_tokens": 10,
            "body": {"model": "fake", "stream": True, "prompt": "fixed", "seed": 7},
        },
        {
            "id": "warm",
            "arrival_ms": 0,
            "cache_state": "warm",
            "expected_prompt_tokens": 10,
            "body": {"model": "fake", "stream": True, "prompt": "same", "seed": 7},
        },
    ],
}


class _Canary(http.server.BaseHTTPRequestHandler):
    """A minimal SSE canary: reads the body, streams fixed frames."""

    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length") or 0)
        if length:
            self.rfile.read(length)
        body = "".join(FRAMES).encode()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        pass


@pytest.fixture
def canary_url() -> Iterator[str]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Canary)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"
    server.shutdown()
    server.server_close()


def test_replay_compares_usage_and_cache_telemetry(
    tmp_path, canary_url: str
) -> None:
    # Arrange
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(SCENARIO))
    scenario = load_scenario(path)

    # Act
    rows = asyncio.run(
        run_scenario(
            scenario,
            endpoint=canary_url,
            acknowledge_isolated_canary=True,
            run_id="e2e-r1",
        )
    )

    # Assert
    assert [row["request_label"] for row in rows] == ["cold", "warm"]
    assert [row["prompt_tokens_match"] for row in rows] == [True, True]
    assert [row["error"] for row in rows] == [None, None]
    assert rows[0]["cache"] == {
        "cached_tokens": 8,
        "device_hit_tokens": 8,
        "host_hit_tokens": 1,
        "newly_computed_tokens": 1,
    }
    assert rows[0]["scheduler"] == {"queue_time_s": 0.25}
    assert rows[0]["request_id"] == "sglang-ab-e2e-r1-00-cold"


def test_replay_without_acknowledgement_sends_nothing(
    tmp_path, canary_url: str
) -> None:
    # Arrange
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(SCENARIO))

    # Act
    with pytest.raises(PermissionError, match="isolated-canary"):
        asyncio.run(
            run_scenario(
                load_scenario(path),
                endpoint=canary_url,
                acknowledge_isolated_canary=False,
                run_id="e2e-r1",
            )
        )

    # Assert — nothing to assert on the wire (the refusal happens before
    # any request is built); the raises above IS the assertion.
