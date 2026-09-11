"""Deterministic replay harness for an isolated SGLang scheduler/cache canary.

There is deliberately no default endpoint.  Both an explicit URL and an
explicit acknowledgement are required before this module sends any request.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

ACK_FLAG = "--i-understand-this-sends-load-to-an-isolated-canary"
_CACHE_KEYS = {
    "cached_tokens",
    "cache_hit_tokens",
    "cache_miss_tokens",
    "device_cached_tokens",
    "device_hit_tokens",
    "gpu_cache_usage",
    "gpu_cache_utilization",
    "host_cached_tokens",
    "host_hit_tokens",
    "host_cache_usage",
    "host_cache_utilization",
    "cpu_cache_usage",
    "storage_cached_tokens",
    "storage_hit_tokens",
    "storage_cache_usage",
    "storage_cache_utilization",
    "newly_computed_tokens",
    "evicted_tokens",
    "eviction_count",
    "cache_utilization",
    "cache_usage",
}
_SCHEDULER_KEYS = {
    "queue_time",
    "queue_time_s",
    "queue_duration",
    "running_requests",
    "waiting_requests",
    "preemptions",
    "preemption_count",
    "starvation_events",
    "starvation_count",
}


def load_scenario(path: Path) -> dict[str, Any]:
    """Load and strictly validate a version-one replay manifest."""
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("scenario schema_version must be 1")
    if not isinstance(data.get("name"), str) or not data["name"].strip():
        raise ValueError("scenario name must be a non-empty string")
    if not isinstance(data.get("seed"), int):
        raise ValueError("scenario seed must be an integer")
    requests = data.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("scenario requests must be a non-empty list")
    seen: set[str] = set()
    previous = -1
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            raise ValueError(f"request {index} must be an object")
        label = request.get("id")
        if not isinstance(label, str) or not label or label in seen:
            raise ValueError(f"request {index} id must be a unique non-empty string")
        seen.add(label)
        arrival = request.get("arrival_ms")
        if not isinstance(arrival, int) or arrival < 0 or arrival < previous:
            raise ValueError("arrival_ms must be non-negative and nondecreasing")
        previous = arrival
        expected = request.get("expected_prompt_tokens")
        if not isinstance(expected, int) or expected < 1:
            raise ValueError(f"request {label} expected_prompt_tokens must be positive")
        body = request.get("body")
        if not isinstance(body, dict):
            raise ValueError(f"request {label} body must be an object")
        if body.get("stream") is not True:
            raise ValueError(f"request {label} body.stream must be true")
        sampling = body.get("sampling_params", {})
        body_seed = body.get("seed")
        if body_seed is None and isinstance(sampling, dict):
            body_seed = sampling.get("seed")
        if body_seed != data["seed"]:
            raise ValueError(
                f"request {label} seed must equal scenario seed {data['seed']}"
            )
    return data


def _request_id(run_id: str, index: int, label: str) -> str:
    safe_run = re.sub(r"[^A-Za-z0-9_.-]", "-", run_id).strip("-")
    safe_label = re.sub(r"[^A-Za-z0-9_.-]", "-", label).strip("-")
    if not safe_run:
        raise ValueError("run_id must contain at least one safe character")
    return f"sglang-ab-{safe_run}-{index:02d}-{safe_label}"


def _walk_metrics(value: Any, wanted: set[str], output: dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in wanted and isinstance(child, (int, float)):
                output[key] = child
            _walk_metrics(child, wanted, output)
    elif isinstance(value, list):
        for child in value:
            _walk_metrics(child, wanted, output)


def _content_present(event: Mapping[str, Any]) -> bool:
    text = event.get("text")
    if isinstance(text, str) and text:
        return True
    choices = event.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        delta = choice.get("delta", {})
        if isinstance(delta, Mapping) and delta.get("content"):
            return True
        if choice.get("text"):
            return True
    return False


def _token_count(event: Mapping[str, Any]) -> int:
    """Count streamed token-bearing chunks when usage is not yet available."""
    return int(_content_present(event))


async def _run_request(
    client: Any,
    *,
    endpoint: str,
    request: Mapping[str, Any],
    request_id: str,
    scenario: Mapping[str, Any],
    run_id: str,
    timeout_s: float,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    started = clock()
    first_token_at: float | None = None
    last_token_at: float | None = None
    streamed_token_chunks = 0
    usage: dict[str, Any] = {}
    raw_metrics: dict[str, Any] = {}
    cache: dict[str, Any] = {}
    scheduler: dict[str, Any] = {}
    error: str | None = None
    status_code: int | None = None
    try:
        headers = {"x-request-id": request_id, "accept": "text/event-stream"}
        async with client.stream(
            "POST", endpoint, json=request["body"], headers=headers, timeout=timeout_s
        ) as response:
            status_code = response.status_code
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                event = json.loads(payload)
                if not isinstance(event, dict):
                    continue
                raw_metrics.update(event)
                event_usage = event.get("usage")
                if isinstance(event_usage, dict):
                    usage.update(event_usage)
                meta = event.get("meta_info")
                if isinstance(meta, dict):
                    raw_metrics.update(meta)
                    for key in (
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                        "input_tokens",
                        "output_tokens",
                    ):
                        if key in meta:
                            usage[key] = meta[key]
                _walk_metrics(event, _CACHE_KEYS, cache)
                _walk_metrics(event, _SCHEDULER_KEYS, scheduler)
                if _content_present(event):
                    observed = clock()
                    first_token_at = first_token_at or observed
                    last_token_at = observed
                    streamed_token_chunks += _token_count(event)
    except Exception as exc:  # a failed request is a benchmark result, not a lost row
        error = f"{exc.__class__.__name__}: {exc}"
    ended = clock()
    output_tokens = usage.get(
        "completion_tokens", usage.get("output_tokens", streamed_token_chunks)
    )
    prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    ttft = first_token_at - started if first_token_at is not None else None
    generation_s = (
        last_token_at - first_token_at
        if first_token_at is not None and last_token_at is not None
        else None
    )
    tpot = (
        generation_s / (output_tokens - 1)
        if generation_s is not None and isinstance(output_tokens, int) and output_tokens > 1
        else None
    )
    result = {
        "schema_version": 1,
        "run_id": run_id,
        "scenario": scenario["name"],
        "scenario_seed": scenario["seed"],
        "scenario_metadata": scenario.get("metadata", {}),
        "configuration": scenario.get("configuration", {}),
        "request_id": request_id,
        "request_label": request["id"],
        "arrival_ms": request["arrival_ms"],
        "expected_prompt_tokens": request["expected_prompt_tokens"],
        "prompt_tokens": prompt_tokens,
        "prompt_tokens_match": (
            prompt_tokens == request["expected_prompt_tokens"]
            if isinstance(prompt_tokens, int)
            else None
        ),
        "output_tokens": output_tokens,
        "status_code": status_code,
        "error": error,
        "ttft_s": ttft,
        "tpot_s": tpot,
        "e2e_s": ended - started,
        "output_tokens_per_s": (
            output_tokens / (ended - first_token_at)
            if first_token_at is not None
            and isinstance(output_tokens, int)
            and ended > first_token_at
            else None
        ),
        "usage": usage,
        "cache": cache,
        "scheduler": scheduler,
    }
    # Keep vendor-specific telemetry without echoing generated text.
    result["raw_metric_fields"] = {
        key: value
        for key, value in raw_metrics.items()
        if key not in {"choices", "text", "output", "usage", "meta_info"}
    }
    return result


async def run_scenario(
    scenario: Mapping[str, Any],
    *,
    endpoint: str,
    acknowledge_isolated_canary: bool,
    run_id: str,
    client: Any | None = None,
    timeout_s: float = 1800.0,
) -> list[dict[str, Any]]:
    """Run one fixed replay, returning results in declared arrival order."""
    if not endpoint or not endpoint.startswith(("http://", "https://")):
        raise ValueError("an explicit http(s) endpoint is required")
    if not acknowledge_isolated_canary:
        raise PermissionError(f"refusing to send load without {ACK_FLAG}")
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("benchmark requires scitex-genai[benchmark]") from exc

    owned_client = client is None
    if owned_client:
        headers = {}
        token = os.environ.get("SCITEX_SGLANG_BENCHMARK_API_KEY")
        if token:
            headers["authorization"] = f"Bearer {token}"
        client = httpx.AsyncClient(headers=headers)
    start = asyncio.get_running_loop().time()

    async def scheduled(index: int, request: Mapping[str, Any]) -> dict[str, Any]:
        due = start + request["arrival_ms"] / 1000
        await asyncio.sleep(max(0.0, due - asyncio.get_running_loop().time()))
        return await _run_request(
            client,
            endpoint=endpoint,
            request=request,
            request_id=_request_id(run_id, index, request["id"]),
            scenario=scenario,
            run_id=run_id,
            timeout_s=timeout_s,
        )

    try:
        tasks = [
            asyncio.create_task(scheduled(index, request))
            for index, request in enumerate(scenario["requests"])
        ]
        return await asyncio.gather(*tasks)
    finally:
        if owned_client:
            await client.aclose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument("--endpoint", required=True, help="full isolated-canary URL")
    parser.add_argument("--run-id", required=True, help="stable A/B run identifier")
    parser.add_argument("--output", type=Path, help="JSONL path (default: stdout)")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument(ACK_FLAG, action="store_true", dest="acknowledge")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        scenario = load_scenario(args.scenario)
        results = asyncio.run(
            run_scenario(
                scenario,
                endpoint=args.endpoint,
                acknowledge_isolated_canary=args.acknowledge,
                run_id=args.run_id,
                timeout_s=args.timeout,
            )
        )
    except (OSError, ValueError, PermissionError, RuntimeError) as exc:
        print(f"scitex-genai-sglang-ab: {exc}", file=sys.stderr)
        return 2
    lines = _jsonl(results)
    if args.output:
        args.output.write_text(lines)
    else:
        sys.stdout.write(lines)
    return int(any(row["error"] or row["prompt_tokens_match"] is False for row in results))


def _jsonl(results: list[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in results)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
