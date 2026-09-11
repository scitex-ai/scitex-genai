#!/usr/bin/env python3
"""Run serialized, cold-cache exact-token SGLang capacity probes."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import uuid


def request_json(base_url: str, path: str, payload=None, timeout: float = 900):
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    request = urllib.request.Request(
        base_url + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def load(base_url: str) -> dict:
    return request_json(base_url, "/v1/loads?include=core")["loads"][0]


def wait_idle(base_url: str, timeout: float = 120) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = load(base_url)
        if not snapshot["num_running_reqs"] and not snapshot["num_waiting_reqs"]:
            return snapshot
        time.sleep(0.1)
    raise TimeoutError(f"server did not return idle: {snapshot}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sizes", default="65536,131072,262144")
    args = parser.parse_args()
    sizes = [int(value) for value in args.sizes.split(",")]

    report = {
        "base_url": args.base_url,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "initial_load": wait_idle(args.base_url),
        "probes": [],
    }
    for size in sizes:
        rid = f"capacity-{size}-{uuid.uuid4()}"
        started = time.monotonic()
        response = request_json(
            args.base_url,
            "/generate",
            {
                "input_ids": [42] * size,
                "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                "rid": rid,
                "cache_salt": rid,
            },
        )
        elapsed = time.monotonic() - started
        after = wait_idle(args.base_url)
        meta = response.get("meta_info", {})
        if meta.get("prompt_tokens") != size or meta.get("cached_tokens") != 0:
            raise AssertionError(f"probe was not an exact cold request: {meta}")
        report["probes"].append(
            {
                "tokens": size,
                "wall_seconds": elapsed,
                "input_tokens_per_second": size / elapsed,
                "meta_info": meta,
                "after": after,
            }
        )
        print(f"{size} tokens: {elapsed:.3f}s", flush=True)

    with open(args.output, "x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
