from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parents[1]
IMAGE_DIR = ROOT / "containers" / "sglang-cancel-safe"
PATCH = IMAGE_DIR / "patches" / (
    "0001-preserve-dispatched-state-and-retry-chunked-abort.patch"
)
BASE_DIGEST = "45e39d4c5bcfd89d171b3358ba78899354ab26a85bc746a17621ad818f8394aa"
BASE_COMMIT = "4ccff141dbe992794f9da6c3aa23535b4f72000d"
PATCH_DIGEST = "36d9ea8c3b386608e00967b881e96115c3ee5d0ec423857315e55ef1b4c7bfeb"


def _load_acceptance_module():
    spec = importlib.util.spec_from_file_location(
        "sglang_cancel_acceptance", IMAGE_DIR / "acceptance_disconnect.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_definition_pins_base_source_and_patch_digest():
    definition = (IMAGE_DIR / "Apptainer.def").read_text()
    assert f"sha256:{BASE_DIGEST}" in definition
    assert BASE_COMMIT in definition
    assert hashlib.sha256(PATCH.read_bytes()).hexdigest() == PATCH_DIGEST


def test_patch_contains_both_cancellation_guards():
    patch = PATCH.read_text()
    assert "dispatched: bool = False" in patch
    assert "abort_sent: bool = False" in patch
    assert "_release_req_states_on_failure(request_rids)" in patch
    assert "self.abort_request(AbortReq(rid=req.rid))" in patch


def test_tp1_launcher_has_required_isolation_and_yarn_guards():
    launcher = (IMAGE_DIR / "run_tp1_pair_experiment.sh").read_text()
    assert '--bind "${run_dir}"' in launcher
    assert "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1" in launcher
    assert '--context-length 400000' in launcher
    assert '--mem-fraction-static 0.75' in launcher


def test_builder_rejects_non_scratch_output_before_invoking_runtime():
    result = subprocess.run(
        [str(IMAGE_DIR / "build.sh"), "/var/tmp/scitex-test-unsafe.sif"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "refusing output outside scratch storage" in result.stderr


def test_disconnect_acceptance_closes_socket_and_observes_drain():
    acceptance = _load_acceptance_module()
    state = SimpleNamespace(active=False, disconnected=threading.Event())

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format, *_args):
            return

        def do_GET(self):
            assert self.path == "/v1/loads?include=core"
            active = state.active and not state.disconnected.is_set()
            body = json.dumps(
                {"loads": [{"num_running_reqs": int(active), "num_waiting_reqs": 0}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            size = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(size))
            assert payload["stream"] is True
            assert payload["rid"].startswith("scitex-cancel-")
            state.active = True
            self.rfile.read(1)
            state.disconnected.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        args = SimpleNamespace(
            base_url=f"http://127.0.0.1:{server.server_port}",
            model="test-model",
            log=None,
            prompt_bytes=256_000,
            busy_timeout=5,
            drain_timeout=5,
        )
        acceptance.run(args)
        assert state.disconnected.wait(1)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
