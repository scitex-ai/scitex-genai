from __future__ import annotations

import httpx
import pytest

from scitex_genai.gateway._sglang_metrics import (
    MAX_METRICS_BYTES,
    parse_sglang_metrics,
    probe_sglang_metrics,
)

GENERATION = "0123456789abcdef0123456789abcdef"


def _metrics(generation: str = GENERATION) -> str:
    labels = f'engine_type="unified",scitex_engine_generation="{generation}"'
    return "\n".join(
        (
            f"sglang:num_running_reqs{{{labels}}} 2",
            f"sglang:num_queue_reqs{{{labels}}} 3",
            f"sglang:token_usage{{{labels}}} 2.5e-1",
        )
    )


def test_generation_bearing_scheduler_metrics_are_one_atomic_observation() -> None:
    # Arrange
    payload = _metrics()

    # Act
    observed = parse_sglang_metrics(payload)

    # Assert
    assert (
        observed.engine_generation,
        observed.running,
        observed.queued,
        observed.token_usage,
    ) == (GENERATION, 2, 3, 0.25)


@pytest.mark.parametrize(
    "payload",
    [
        _metrics().replace(f',scitex_engine_generation="{GENERATION}"', ""),
        _metrics().replace(GENERATION, "f" * 32, 1),
        _metrics().replace("2.5e-1", "1.1"),
    ],
)
def test_partial_mixed_or_invalid_metrics_fail_closed(payload: str) -> None:
    # Arrange
    expected = "SGLang.*metrics"

    # Act
    def parse() -> None:
        parse_sglang_metrics(payload)

    # Assert
    with pytest.raises(ValueError, match=expected):
        parse()


@pytest.mark.asyncio
async def test_probe_reads_the_real_metrics_route_with_a_body_bound() -> None:
    # Arrange
    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, text=_metrics())

    # Act
    observed = await probe_sglang_metrics(
        "http://engine.test",
        1,
        transport=httpx.MockTransport(handler),
    )

    # Assert
    assert (seen, observed.engine_generation) == (["/metrics"], GENERATION)


@pytest.mark.asyncio
async def test_probe_refuses_an_oversized_metrics_document() -> None:
    # Arrange

    async def oversized(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * (MAX_METRICS_BYTES + 1))

    # Act
    async def probe() -> None:
        await probe_sglang_metrics(
            "http://engine.test", 1, transport=httpx.MockTransport(oversized)
        )

    # Assert
    with pytest.raises(ValueError, match="exceeds one MiB"):
        await probe()
