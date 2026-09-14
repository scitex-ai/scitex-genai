"""Bounded parsing of the SGLang scheduler metrics contract."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

SGLANG_METRICS_PATH = "/metrics"
ENGINE_GENERATION_LABEL = "scitex_engine_generation"
MAX_METRICS_BYTES = 1_048_576

_SAMPLE = re.compile(
    r"^sglang:(num_running_reqs|num_queue_reqs|token_usage)"
    r"\{([^}]*)\}\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)
_GENERATION = re.compile(r'(?:^|,)scitex_engine_generation="([0-9a-f]{32})"(?:,|$)')


@dataclass(frozen=True)
class SGLangSchedulerObservation:
    """One internally consistent scheduler sample from one engine process."""

    engine_generation: str
    running: int
    queued: int
    token_usage: float


def parse_sglang_metrics(payload: str) -> SGLangSchedulerObservation:
    """Extract the generation-bearing gauges; reject partial/mixed snapshots."""
    samples: dict[str, list[float]] = {
        "num_running_reqs": [],
        "num_queue_reqs": [],
        "token_usage": [],
    }
    generations: set[str] = set()
    for line in payload.splitlines():
        match = _SAMPLE.fullmatch(line.strip())
        if match is None:
            continue
        generation = _GENERATION.search(match.group(2))
        if generation is None:
            continue
        generations.add(generation.group(1))
        samples[match.group(1)].append(float(match.group(3)))
    if len(generations) != 1 or any(not values for values in samples.values()):
        raise ValueError("SGLang metrics lack one complete engine generation")
    running_values = samples["num_running_reqs"]
    queued_values = samples["num_queue_reqs"]
    token_values = samples["token_usage"]
    if any(value < 0 or not value.is_integer() for value in running_values):
        raise ValueError("SGLang running-request metrics are invalid")
    if any(value < 0 or not value.is_integer() for value in queued_values):
        raise ValueError("SGLang queued-request metrics are invalid")
    if any(not 0 <= value <= 1 for value in token_values):
        raise ValueError("SGLang token-usage metrics are invalid")
    return SGLangSchedulerObservation(
        engine_generation=next(iter(generations)),
        running=sum(map(int, running_values)),
        queued=sum(map(int, queued_values)),
        token_usage=max(token_values),
    )


async def probe_sglang_metrics(
    base_url: str,
    timeout_s: float,
    *,
    transport: Any = None,
) -> SGLangSchedulerObservation:
    """Read at most one MiB from the non-payload SGLang metrics endpoint."""
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError(
            "Gateway metrics probes require scitex-genai[gateway]"
        ) from exc

    async def request() -> SGLangSchedulerObservation:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s),
            transport=transport,
            follow_redirects=False,
        ) as client:
            async with client.stream(
                "GET", f"{base_url.rstrip('/')}{SGLANG_METRICS_PATH}"
            ) as response:
                response.raise_for_status()
                chunks: list[bytes] = []
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > MAX_METRICS_BYTES:
                        raise ValueError("SGLang metrics response exceeds one MiB")
                    chunks.append(chunk)
        return parse_sglang_metrics(b"".join(chunks).decode("utf-8"))

    return await asyncio.wait_for(request(), timeout=timeout_s)
