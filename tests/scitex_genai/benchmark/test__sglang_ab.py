from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from scitex_genai.benchmark._sglang_ab import (
    _jsonl,
    _run_request,
    load_scenario,
    run_scenario,
)


def _scenario() -> dict:
    return {
        "schema_version": 1,
        "name": "tiny-fixed-order",
        "seed": 7,
        "configuration": {"scheduler": "fcfs"},
        "metadata": {"warmup": "done"},
        "requests": [
            {
                "id": "cold",
                "arrival_ms": 0,
                "expected_prompt_tokens": 10,
                "body": {
                    "model": "fake",
                    "stream": True,
                    "prompt": "fixed",
                    "seed": 7,
                },
            },
            {
                "id": "warm",
                "arrival_ms": 0,
                "expected_prompt_tokens": 10,
                "body": {
                    "model": "fake",
                    "stream": True,
                    "prompt": "same",
                    "seed": 7,
                },
            },
        ],
    }


@pytest.mark.asyncio
async def test_refuses_without_explicit_acknowledgement():
    # Arrange
    ctx = pytest.raises(PermissionError, match="isolated-canary")
    # Act
    # Assert
    with ctx:
        await run_scenario(
            _scenario(),
            endpoint="http://canary.invalid/v1/chat/completions",
            acknowledge_isolated_canary=False,
            run_id="a",
        )


@pytest.mark.asyncio
async def test_requires_an_explicit_http_endpoint():
    # Arrange
    ctx = pytest.raises(ValueError, match="explicit http")
    # Act
    # Assert
    with ctx:
        await run_scenario(
            _scenario(),
            endpoint="",
            acknowledge_isolated_canary=True,
            run_id="a",
        )


@pytest.mark.asyncio
async def test_stream_results_keep_order_ids_usage_and_cache_fields():
    # Arrange
    seen = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.headers["x-request-id"], json.loads(request.content)))
        frames = [
            'data: {"choices":[{"delta":{"content":"a"}}]}\n\n',
            'data: {"choices":[{"delta":{"content":"b"}}]}\n\n',
            (
                'data: {"choices":[],"usage":{"prompt_tokens":10,'
                '"completion_tokens":2,"total_tokens":12,'
                '"prompt_tokens_details":{"cached_tokens":8}},'
                '"meta_info":{"device_hit_tokens":8,"host_hit_tokens":1,'
                '"storage_hit_tokens":0,"newly_computed_tokens":1,'
                '"queue_time_s":0.25,"running_requests":2,'
                '"waiting_requests":1}}\n\n'
            ),
            "data: [DONE]\n\n",
        ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join(frames),
        )

    expected = (
        ["cold", "warm"],
        ["sglang-ab-fcfs-r1-00-cold", "sglang-ab-fcfs-r1-01-warm"],
        [True, True],
        {
            "cached_tokens": 8,
            "device_hit_tokens": 8,
            "host_hit_tokens": 1,
            "storage_hit_tokens": 0,
            "newly_computed_tokens": 1,
        },
        0.25,
        ["fixed", "same"],
    )

    # Act
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rows = await run_scenario(
            _scenario(),
            endpoint="http://canary.invalid/v1/chat/completions",
            acknowledge_isolated_canary=True,
            run_id="fcfs-r1",
            client=client,
        )
    actual = (
        [row["request_label"] for row in rows],
        [row["request_id"] for row in rows],
        [row["prompt_tokens_match"] for row in rows],
        rows[0]["cache"],
        rows[0]["scheduler"]["queue_time_s"],
        [item[1]["prompt"] for item in seen],
    )

    # Assert
    assert actual == expected


def test_manifest_rejects_unfixed_or_reordered_arrivals(tmp_path: Path):
    # Arrange
    scenario = _scenario()
    scenario["requests"][1]["arrival_ms"] = -1
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(scenario))

    # Act
    ctx = pytest.raises(ValueError, match="non-negative and nondecreasing")

    # Assert
    with ctx:
        load_scenario(path)


def test_jsonl_is_one_parseable_object_per_result():
    # Arrange
    rows = [{"request_id": "one"}, {"request_id": "two"}]

    # Act
    rendered = _jsonl(rows)
    request_ids = [json.loads(line)["request_id"] for line in rendered.splitlines()]

    # Assert
    assert request_ids == ["one", "two"]


def test_manifest_requires_the_fixed_seed_in_each_body(tmp_path: Path):
    # Arrange
    scenario = _scenario()
    scenario["requests"][0]["body"].pop("seed")
    path = tmp_path / "missing-seed.json"
    path.write_text(json.dumps(scenario))

    # Act
    ctx = pytest.raises(ValueError, match="seed must equal scenario seed")

    # Assert
    with ctx:
        load_scenario(path)


def test_manifest_marks_two_cold_requests_as_a_crash_probe(tmp_path: Path):
    # Arrange
    scenario = _scenario()
    for request in scenario["requests"]:
        request["cache_state"] = "cold"
    path = tmp_path / "unmarked-cold-cold.json"
    path.write_text(json.dumps(scenario))

    # Act
    ctx = pytest.raises(ValueError, match=r"cold\+cold.*crash-probe")

    # Assert
    with ctx:
        load_scenario(path)


def test_crash_probe_requires_a_dedicated_canary_target(tmp_path: Path):
    # Arrange
    scenario = _scenario()
    scenario["risk_class"] = "crash-probe"
    path = tmp_path / "unsafe-target.json"
    path.write_text(json.dumps(scenario))

    # Act
    ctx = pytest.raises(ValueError, match="target_scope dedicated-canary")

    # Assert
    with ctx:
        load_scenario(path)


@pytest.mark.asyncio
async def test_crash_probe_requires_its_second_acknowledgement():
    # Arrange
    scenario = _scenario()
    scenario.update(risk_class="crash-probe", target_scope="dedicated-canary")
    ctx = pytest.raises(PermissionError, match="may-crash-the-isolated-canary")

    # Act
    # Assert
    with ctx:
        await run_scenario(
            scenario,
            endpoint="http://canary.invalid/v1/chat/completions",
            acknowledge_isolated_canary=True,
            acknowledge_crash_probe=False,
            run_id="crash-probe",
        )


@pytest.mark.asyncio
async def test_stream_timing_calculates_ttft_tpot_and_e2e():
    # Arrange
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(
                'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"b"}}],'
                '"usage":{"prompt_tokens":10,"completion_tokens":2}}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    ticks = iter([10.0, 10.5, 11.0, 12.0])

    # Act
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        row = await _run_request(
            client,
            endpoint="http://canary.invalid/v1/chat/completions",
            request=_scenario()["requests"][0],
            request_id="request-1",
            scenario=_scenario(),
            run_id="run-1",
            timeout_s=1,
            clock=lambda: next(ticks),
        )

    # Assert
    assert (row["ttft_s"], row["tpot_s"], row["e2e_s"]) == (0.5, 0.5, 2.0)
