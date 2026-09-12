from __future__ import annotations

import asyncio
import contextlib
import gc
import hashlib
import json
import re
from collections.abc import AsyncIterator

import pytest

from scitex_genai.gateway._errors import (
    InferenceAdmissionError,
    NoAccountAvailable,
    UpstreamReloading,
    UpstreamUnreachable,
)
from scitex_genai.gateway._inference import (
    InferenceBackend,
    InferenceUpstreamPool,
    ResponseTokenReport,
    accepts_session_id,
    adapt_openai_roles,
    announce,
    conversation_key,
    estimate_input_tokens,
    hoist_system,
    hoists_on,
    inject_cache_report_request,
    parse_upstreams,
    prefix_report,
    request_prefix_fingerprint,
    request_session_key,
    telemetry_enabled,
)

LISTING = "Available agent types for the Agent tool: general-purpose, Explore"
SECRET = "Only the byte count of this sentence may ever leave the process. "


def _request(*, system: str = "You are terse.", later: tuple[dict, ...] = ()) -> dict:
    """What Claude Code >= v2.1.x sends: a role=system entry INSIDE messages."""
    return {
        "model": "local-model",
        "max_tokens": 32,
        "system": system,
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": LISTING},
            *later,
        ],
    }


async def _collect(body: AsyncIterator[bytes]) -> bytes:
    return b"".join([chunk async for chunk in body])


async def _raised_async(awaitable) -> BaseException | None:
    try:
        await awaitable
    except BaseException as exc:
        return exc
    return None


def test_parse_upstreams_strips_whitespace_and_drops_empties() -> None:
    # Arrange
    value = " http://a:1 ,http://b:2,, "
    # Act
    parsed = parse_upstreams(value)
    # Assert
    assert parsed == ["http://a:1", "http://b:2"]


@pytest.mark.parametrize("size, expected", [(0, 0), (1, 1), (4, 1), (5, 2)])
def test_input_token_estimate_uses_the_documented_four_byte_approximation(
    size: int, expected: int
) -> None:
    # Arrange
    body = b"x" * size
    # Act
    estimated = estimate_input_tokens(body)
    # Assert
    assert estimated == expected


def test_request_prefix_fingerprint_is_stable_bounded_and_payload_free() -> None:
    # Arrange
    shared = SECRET.encode() * 400
    same_prefix = shared + b"different tail"
    changed_prefix = b"x" + shared
    # Act
    fingerprints = [
        request_prefix_fingerprint(value)
        for value in (shared, same_prefix, changed_prefix)
    ]
    # Assert -- the tail beyond 16 KiB does not matter and no source text leaks.
    assert (
        fingerprints[0] == fingerprints[1],
        fingerprints[0] != fingerprints[2],
        all(SECRET not in fingerprint for fingerprint in fingerprints),
    ) == (True, True, True)


def test_cache_report_extension_is_opt_in_and_openai_only() -> None:
    # Arrange
    source = json.dumps({"stream": True, "messages": []}).encode()
    # Act
    openai, changed = inject_cache_report_request(
        source, "/v1/chat/completions", enabled=True
    )
    anthropic, anthropic_changed = inject_cache_report_request(
        source, "/v1/messages", enabled=True
    )
    disabled, disabled_changed = inject_cache_report_request(
        source, "/v1/chat/completions", enabled=False
    )
    payload = json.loads(openai)
    # Assert
    assert (
        changed,
        payload["return_cached_tokens_details"],
        payload["stream_options"]["include_usage"],
        anthropic is source,
        anthropic_changed,
        disabled is source,
        disabled_changed,
    ) == (True, True, True, True, False, True, False)


def test_response_token_report_reads_sglang_and_anthropic_streams_incrementally() -> (
    None
):
    # Arrange -- exact field shapes observed from the live SGLang 2026-09-12.
    report = ResponseTokenReport()
    stream = (
        b'data: {"type":"message_start","message":{"usage":{"input_tokens":65,'
        b'"output_tokens":0,"cache_read_input_tokens":64}}}\n\n'
        b'data: {"sglext":{"cached_tokens_details":{"device":32,"host":24,'
        b'"storage":8,"storage_backend":"HiCacheFile"}}}\n\n'
        b'data: {"usage":{"prompt_tokens":65,"completion_tokens":2,'
        b'"prompt_tokens_details":{"cached_tokens":64}}}\n\n'
    )
    # Act -- split inside JSON to prove framing is not chunk-boundary dependent.
    for chunk in (stream[:31], stream[31:117], stream[117:]):
        report.feed(chunk)
    report.finish()
    # Assert
    assert report.fields() == (
        "reported_input_tokens=65 reported_output_tokens=2 cached_tokens=64 "
        "cache_device_tokens=32 cache_host_tokens=24 cache_storage_tokens=8 "
        "cache_storage_backend=HiCacheFile"
    )


def test_telemetry_enabled_matches_the_script_truthiness() -> None:
    # Arrange
    values = ("", "0", "false", "FALSE", "1", "yes")
    # Act
    flags = [telemetry_enabled(value) for value in values]
    # Assert
    assert flags == [False, False, False, False, True, True]


def test_hoist_moves_system_messages_behind_the_top_level_system() -> None:
    # Arrange
    payload = {
        "system": "Top",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "First hoisted"},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
            {"role": "system", "content": [{"type": "text", "text": "Second hoisted"}]},
        ],
    }
    # Act
    hoisted, count = hoist_system(payload)
    # Assert
    assert (count, hoisted["system"], [m["role"] for m in hoisted["messages"]]) == (
        2,
        [
            {"type": "text", "text": "Top"},
            {"type": "text", "text": "First hoisted"},
            {"type": "text", "text": "Second hoisted"},
        ],
        ["user", "assistant"],
    )


def test_hoist_leaves_a_payload_without_system_messages_untouched() -> None:
    # Arrange
    payload = {"system": "Top", "messages": [{"role": "user", "content": "Hello"}]}
    before = json.dumps(payload, sort_keys=True)
    # Act
    hoisted, count = hoist_system(payload)
    # Assert
    assert (count, json.dumps(hoisted, sort_keys=True)) == (0, before)


def test_conversation_key_is_stable_while_messages_grow() -> None:
    # Arrange
    first_turn = _request()
    later_turn = _request(
        later=(
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Again"},
        )
    )
    # Act
    keys = (conversation_key(first_turn), conversation_key(later_turn))
    # Assert
    assert keys[0] == keys[1]


def test_conversation_key_is_none_without_messages() -> None:
    # Arrange
    payload = {"system": "Top", "messages": []}
    # Act
    key = conversation_key(payload)
    # Assert
    assert key is None


def test_prefix_report_carries_sizes_and_never_content() -> None:
    # Arrange
    payload = {"system": SECRET * 100}
    # Act
    report = prefix_report(payload, "abcdef0123456789")
    # Assert
    assert (
        SECRET in report,
        report.startswith("conv=abcdef012345 system_bytes="),
        "p4k=" in report,
    ) == (False, True, True)


@pytest.mark.asyncio
async def test_pool_places_new_conversations_round_robin() -> None:
    # Arrange
    pool = InferenceUpstreamPool.from_urls("http://a:1,http://b:2,http://c:3")
    placed: list[str] = []
    # Act
    for session in ("s1", "s2", "s3", "s4"):
        upstream = await pool.acquire(session)
        placed.append(upstream.alias)
        await pool.release(upstream, session_id=session)
    # Assert
    assert placed == ["http://a:1", "http://b:2", "http://c:3", "http://a:1"]


@pytest.mark.asyncio
async def test_pool_serializes_overlapping_turns_for_one_conversation() -> None:
    # Arrange
    pool = InferenceUpstreamPool.from_urls("http://a:1,http://b:2")
    # Act
    first = await pool.acquire("s1")
    concurrent = asyncio.create_task(pool.acquire("s1"))
    await _wait_for_queue(pool, 1)
    while_first_active = (concurrent.done(), first.in_flight)
    await pool.release(first, session_id="s1")
    readmitted = await concurrent
    await pool.release(readmitted, session_id="s1")
    # Assert
    assert (first.alias, readmitted.alias, while_first_active) == (
        "http://a:1",
        "http://a:1",
        (False, 1),
    )


@pytest.mark.asyncio
async def test_pool_admits_different_conversations_concurrently() -> None:
    # Arrange
    pool = InferenceUpstreamPool.from_urls("http://only:1", capacity_per_upstream=2)
    # Act
    first = await pool.acquire("s1")
    second = await pool.acquire("s2")
    observed = (first.in_flight, sum(member.queued for member in pool.upstreams))
    await pool.release(first, session_id="s1")
    await pool.release(second, session_id="s2")
    # Assert
    assert observed == (2, 0)


async def _wait_for_queue(pool: InferenceUpstreamPool, size: int) -> None:
    for _ in range(100):
        if sum(member.queued for member in pool.upstreams) == size:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"queue did not reach {size}: {pool.status()}")


async def _wait_for_requests(upstream, size: int) -> None:
    for _ in range(200):
        if len(upstream.requests) >= size:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(
        f"upstream saw {len(upstream.requests)} requests, wanted {size}"
    )


async def _wait_for_in_flight(pool: InferenceUpstreamPool, value: int) -> None:
    for _ in range(200):
        if pool.status()[0]["in_flight"] == value:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"in_flight did not reach {value}: {pool.status()}")


@pytest.mark.asyncio
async def test_capacity_is_enforced_and_waiters_are_admitted_after_release() -> None:
    # Arrange
    pool = InferenceUpstreamPool.from_urls(
        "http://only:1", capacity_per_upstream=8, max_queue_size=4
    )
    tasks = [
        asyncio.create_task(pool.acquire(f"session-{index}")) for index in range(12)
    ]

    # Act
    await _wait_for_queue(pool, 4)
    saturated = pool.status()
    expected = [
        {
            "url": "http://only:1",
            "active": True,
            "in_flight": 8,
            "queued": 4,
            "capacity": 8,
        }
    ]

    admitted = [task.result() for task in tasks if task.done()]
    for member in admitted:
        await pool.release(member)
    remaining = await asyncio.gather(*(task for task in tasks if not task.done()))
    for member in remaining:
        await pool.release(member)
    # Assert
    assert (saturated, pool.status()[0]["in_flight"]) == (expected, 0)


@pytest.mark.asyncio
async def test_priority_waiters_are_fifo_ahead_of_ordinary_work() -> None:
    # Arrange
    pool = InferenceUpstreamPool.from_urls(
        "http://only:1", capacity_per_upstream=1, max_queue_size=3
    )
    active = await pool.acquire("active")
    ordinary = asyncio.create_task(pool.acquire("ordinary"))
    await _wait_for_queue(pool, 1)
    priority_one = asyncio.create_task(pool.acquire("priority-1", priority=True))
    await _wait_for_queue(pool, 2)
    priority_two = asyncio.create_task(pool.acquire("priority-2", priority=True))
    await _wait_for_queue(pool, 3)

    # Act
    await pool.release(active, session_id="active")
    first = await priority_one
    order = ["priority-1"]
    await pool.release(first, session_id="priority-1")
    second = await priority_two
    order.append("priority-2")
    await pool.release(second, session_id="priority-2")
    third = await ordinary
    order.append("ordinary")
    await pool.release(third, session_id="ordinary")

    # Assert
    assert (order, pool.status()[0]["in_flight"]) == (
        ["priority-1", "priority-2", "ordinary"],
        0,
    )


@pytest.mark.asyncio
async def test_token_capacity_queues_a_large_request_while_short_work_fits() -> None:
    # Arrange
    pool = InferenceUpstreamPool.from_urls(
        "http://only:1",
        capacity_per_upstream=8,
        token_capacity_per_upstream=1_000,
        max_queue_size=2,
    )
    first = await pool.acquire("long-a", input_tokens=700)
    second = await pool.acquire("short", input_tokens=200)
    waiting = asyncio.create_task(pool.acquire("long-b", input_tokens=400))
    await _wait_for_queue(pool, 1)

    # Act
    saturated = pool.status()[0]
    await pool.release(second, input_tokens=200, session_id="short")
    still_waiting = not waiting.done()
    await pool.release(first, input_tokens=700, session_id="long-a")
    admitted = await waiting
    await pool.release(admitted, input_tokens=400, session_id="long-b")

    # Assert
    assert (
        saturated["input_tokens_in_flight"],
        saturated["input_tokens_queued"],
        still_waiting,
        pool.status()[0]["input_tokens_in_flight"],
    ) == (900, 400, True, 0)


@pytest.mark.asyncio
async def test_token_capacity_rejects_a_request_that_can_never_fit() -> None:
    # Arrange
    pool = InferenceUpstreamPool.from_urls(
        "http://only:1", token_capacity_per_upstream=1_000
    )

    # Act
    refused = await _raised_async(pool.acquire("too-large", input_tokens=1_001))

    # Assert
    assert (
        isinstance(refused, InferenceAdmissionError),
        "1001/1000" in str(refused),
        pool.status()[0]["in_flight"],
    ) == (True, True, 0)


@pytest.mark.asyncio
async def test_token_waiter_keeps_its_cached_home_when_another_member_is_idle() -> None:
    # Arrange
    pool = InferenceUpstreamPool.from_urls(
        "http://a:1,http://b:2", token_capacity_per_upstream=1_000
    )
    home = await pool.acquire("sticky", input_tokens=700)
    waiting = asyncio.create_task(pool.acquire("sticky", input_tokens=400))
    await _wait_for_queue(pool, 1)

    # Act
    aliases = [member.alias for member in pool.upstreams if member.in_flight]
    await pool.release(home, input_tokens=700, session_id="sticky")
    readmitted = await waiting
    await pool.release(readmitted, input_tokens=400, session_id="sticky")

    # Assert
    assert (aliases, readmitted.alias) == (["http://a:1"], "http://a:1")


@pytest.mark.asyncio
async def test_queue_bound_refuses_overload_with_503_error() -> None:
    # Arrange
    pool = InferenceUpstreamPool.from_urls(
        "http://only:1", capacity_per_upstream=1, max_queue_size=1
    )
    admitted = await pool.acquire("first")
    waiting = asyncio.create_task(pool.acquire("second"))
    await _wait_for_queue(pool, 1)

    # Act
    refused = await _raised_async(pool.acquire("third"))
    waiting.cancel()
    cancelled = await _raised_async(waiting)
    await pool.release(admitted, session_id="first")

    # Assert
    assert (
        isinstance(refused, InferenceAdmissionError),
        getattr(refused, "status_code", None),
        "queue is full" in str(refused),
        isinstance(cancelled, asyncio.CancelledError),
    ) == (True, 503, True, True)


@pytest.mark.asyncio
async def test_waiting_preserves_sticky_home_even_when_another_member_is_idle() -> None:
    # Arrange
    pool = InferenceUpstreamPool.from_urls(
        "http://a:1,http://b:2", capacity_per_upstream=1
    )
    home = await pool.acquire("sticky")
    waiting = asyncio.create_task(pool.acquire("sticky"))
    await _wait_for_queue(pool, 1)

    # Act
    waiting_state = [
        (member.alias, member.in_flight, member.queued) for member in pool.upstreams
    ]
    expected = [
        ("http://a:1", 1, 1),
        ("http://b:2", 0, 0),
    ]
    await pool.release(home, session_id="sticky")
    readmitted = await waiting
    await pool.release(readmitted, session_id="sticky")

    # Assert
    assert (waiting_state, readmitted.alias) == (expected, "http://a:1")


@pytest.mark.asyncio
async def test_cancelled_waiter_releases_its_queue_slot() -> None:
    # Arrange
    pool = InferenceUpstreamPool.from_urls(
        "http://only:1", capacity_per_upstream=1, max_queue_size=1
    )
    admitted = await pool.acquire("first")
    waiting = asyncio.create_task(pool.acquire("second"))
    await _wait_for_queue(pool, 1)

    # Act
    waiting.cancel()
    cancelled = await _raised_async(waiting)
    state = pool.status()[0]
    await pool.release(admitted, session_id="first")

    # Assert
    assert (
        isinstance(cancelled, asyncio.CancelledError),
        state["in_flight"],
        state["queued"],
    ) == (True, 1, 0)


@pytest.mark.asyncio
async def test_shutdown_wakes_waiters_and_rejects_new_admission() -> None:
    # Arrange
    pool = InferenceUpstreamPool.from_urls(
        "http://only:1", capacity_per_upstream=1, max_queue_size=1
    )
    admitted = await pool.acquire("first")
    waiting = asyncio.create_task(pool.acquire("second"))
    await _wait_for_queue(pool, 1)

    # Act
    await pool.close()
    waiting_error = await _raised_async(waiting)
    new_error = await _raised_async(pool.acquire("third"))
    state = pool.status()[0]
    expected = {
        "url": "http://only:1",
        "active": False,
        "in_flight": 1,
        "queued": 0,
        "capacity": 1,
    }
    await pool.release(admitted, session_id="first")

    # Assert
    assert (
        isinstance(waiting_error, InferenceAdmissionError),
        isinstance(new_error, InferenceAdmissionError),
        state,
    ) == (True, True, expected)


@pytest.mark.asyncio
async def test_explicit_session_ids_separate_identical_prompts_and_stay_sticky(
    upstream_factory,
) -> None:
    # Arrange -- two Hermes agents can have byte-identical startup prompts.
    first = upstream_factory()
    second = upstream_factory()
    backend = InferenceBackend(InferenceUpstreamPool.from_urls([first.url, second.url]))
    startup_body = json.dumps(_request()).encode()
    continued_body = json.dumps(
        _request(
            later=(
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "Again"},
            )
        )
    ).encode()

    # Act -- identical starts separate, then a changed a-turn must return home.
    for session_id, body in (
        ("hermes-session-a", startup_body),
        ("hermes-session-b", startup_body),
        ("hermes-session-a", continued_body),
    ):
        relayed = await backend.relay(
            "POST",
            "/v1/messages",
            body=body,
            headers={"X-SciTeX-Session-ID": session_id},
        )
        await _collect(relayed.body)

    message_counts = (
        [len(json.loads(request["body"])["messages"]) for request in first.requests],
        [len(json.loads(request["body"])["messages"]) for request in second.requests],
    )
    protocol_safe = all(
        "session_id" not in json.loads(request["body"])
        and "x-scitex-session-id" not in request["headers"]
        for request in (*first.requests, *second.requests)
    )

    # Assert
    assert (message_counts, protocol_safe) == (([1, 3], [1]), True)


def test_pool_refuses_with_inference_wording_when_empty() -> None:
    # Arrange
    urls = ""

    # Act
    def build() -> InferenceUpstreamPool:
        return InferenceUpstreamPool.from_urls(urls)

    # Assert
    with pytest.raises(NoAccountAvailable, match="No inference upstreams"):
        build()


def test_pool_refuses_duplicate_urls_with_inference_wording() -> None:
    # Arrange
    urls = "http://a:1,http://a:1"

    # Act
    def build() -> InferenceUpstreamPool:
        return InferenceUpstreamPool.from_urls(urls)

    # Assert
    with pytest.raises(ValueError, match="Inference upstream URLs"):
        build()


def test_announce_names_every_upstream() -> None:
    # Arrange
    pool = InferenceUpstreamPool.from_urls("http://a:1,http://b:2")
    # Act
    line = announce("0.0.0.0", 18772, pool)
    # Assert
    assert line == (
        "scitex-genai-gateway: listening 0.0.0.0:18772 -> 2 inference upstream(s): "
        "http://a:1, http://b:2  [sticky per conversation]"
    )


@pytest.mark.asyncio
async def test_relay_hoists_the_body_and_keeps_the_query_string(
    upstream_factory,
) -> None:
    # Arrange
    upstream = upstream_factory()
    backend = InferenceBackend(InferenceUpstreamPool.from_urls(upstream.url))
    # Act
    relayed = await backend.relay(
        "POST",
        "/v1/messages?beta=true",
        body=json.dumps(_request()).encode(),
        headers={"content-type": "application/json"},
    )
    await _collect(relayed.body)
    seen = upstream.requests[0]
    sent = json.loads(seen["body"])
    # Assert
    assert (seen["path"], [m["role"] for m in sent["messages"]], sent["system"]) == (
        "/v1/messages?beta=true",
        ["user"],
        [
            {"type": "text", "text": "You are terse."},
            {"type": "text", "text": LISTING},
        ],
    )


@pytest.mark.asyncio
async def test_stream_holds_capacity_until_its_body_is_drained(
    upstream_factory,
) -> None:
    # Arrange
    upstream = upstream_factory(chunks=(b"first",))
    pool = InferenceUpstreamPool.from_urls(
        upstream.url, capacity_per_upstream=1, max_queue_size=1
    )
    backend = InferenceBackend(pool)
    first = await backend.relay(
        "POST", "/v1/messages", body=b"{}", headers={"x-session-id": "first"}
    )

    # Act
    second_task = asyncio.create_task(
        backend.relay(
            "POST", "/v1/messages", body=b"{}", headers={"x-session-id": "second"}
        )
    )
    await _wait_for_queue(pool, 1)
    was_waiting = not second_task.done()
    first_body = await _collect(first.body)
    second = await second_task
    second_body = await _collect(second.body)

    # Assert
    assert (was_waiting, first_body, second_body, pool.status()[0]["in_flight"]) == (
        True,
        b"first",
        b"first",
        0,
    )


@pytest.mark.asyncio
async def test_relay_streams_the_upstream_sse_verbatim(upstream_factory) -> None:
    # Arrange
    frames = (
        b'event: message_start\ndata: {"type":"message_start"}\n\n',
        b'event: content_block_delta\ndata: {"delta":{"text":"Hi"}}\n\n',
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    )
    upstream = upstream_factory(content_type="text/event-stream", chunks=frames)
    backend = InferenceBackend(InferenceUpstreamPool.from_urls(upstream.url))
    # Act
    relayed = await backend.relay(
        "POST", "/v1/messages", body=json.dumps(_request()).encode(), headers={}
    )
    received = await _collect(relayed.body)
    # Assert
    assert (relayed.content_type, received) == ("text/event-stream", b"".join(frames))


@pytest.mark.asyncio
async def test_continuation_qos_aborts_then_retries_first_turn(
    upstream_factory,
) -> None:
    # Arrange: gateway capacity is deliberately 2. The handoff must happen
    # before dispatch even though a second gateway slot is available.
    release = __import__("threading").Event()
    release.set()
    upstream = upstream_factory(block_until=release, abort_releases=True)
    pool = InferenceUpstreamPool.from_urls(upstream.url, capacity_per_upstream=2)
    backend = InferenceBackend(pool, continuation_qos_enabled=True)
    body = json.dumps({"model": "m", "input": "hello", "stream": True}).encode()

    known = await backend.relay(
        "POST", "/v1/responses", body=body, headers={"x-scitex-session-id": "known"}
    )
    await _collect(known.body)
    release.clear()

    # Act
    first_task = asyncio.create_task(
        backend.relay(
            "POST", "/v1/responses", body=body, headers={"x-scitex-session-id": "new"}
        )
    )
    await _wait_for_requests(upstream, 2)
    continuation_task = asyncio.create_task(
        backend.relay(
            "POST", "/v1/responses", body=body, headers={"x-scitex-session-id": "known"}
        )
    )
    continuation = await continuation_task
    await _collect(continuation.body)
    retried = await first_task
    await _collect(retried.body)

    # Assert: initial success, cold attempt, explicit abort, continuation,
    # transparent cold retry. The gateway owns/overrides every SGLang rid.
    paths = [request["path"] for request in upstream.requests]
    abort = json.loads(upstream.requests[2]["body"])["rid"]
    cold_rid = json.loads(upstream.requests[1]["body"])["rid"]
    retry_rid = json.loads(upstream.requests[4]["body"])["rid"]
    snapshot = backend.continuation_qos.snapshot()
    assert (
        paths,
        abort == cold_rid,
        retry_rid != cold_rid,
        pool.status()[0]["in_flight"],
        snapshot["first_turns_preempted"],
        snapshot["first_turn_retries"],
    ) == (
        [
            "/v1/responses",
            "/v1/responses",
            "/abort_request",
            "/v1/responses",
            "/v1/responses",
        ],
        True,
        True,
        0,
        1,
        1,
    )


@pytest.mark.asyncio
async def test_client_cancellation_during_handoff_leaks_no_capacity(
    upstream_factory,
) -> None:
    # Arrange
    release = __import__("threading").Event()
    release.set()
    upstream = upstream_factory(block_until=release, abort_releases=True)
    pool = InferenceUpstreamPool.from_urls(upstream.url, capacity_per_upstream=2)
    backend = InferenceBackend(pool, continuation_qos_enabled=True)
    body = json.dumps({"model": "m", "input": "hello", "stream": True}).encode()
    known = await backend.relay(
        "POST", "/v1/responses", body=body, headers={"x-scitex-session-id": "known"}
    )
    await _collect(known.body)
    release.clear()
    first_task = asyncio.create_task(
        backend.relay(
            "POST", "/v1/responses", body=body, headers={"x-scitex-session-id": "new"}
        )
    )
    await _wait_for_requests(upstream, 2)
    continuation = await backend.relay(
        "POST", "/v1/responses", body=body, headers={"x-scitex-session-id": "known"}
    )

    # Act
    first_task.cancel()
    cancelled = await _raised_async(first_task)
    await anext(continuation.body)
    await continuation.body.aclose()

    # Assert
    assert (
        isinstance(cancelled, asyncio.CancelledError),
        pool.status()[0]["in_flight"],
        backend.continuation_qos.snapshot()["replay_safe_first_turns"],
    ) == (True, 0, 0)


@pytest.mark.asyncio
async def test_disconnect_before_retry_headers_aborts_replayed_first_turn(
    upstream_factory,
) -> None:
    # Arrange: reproduce a cold first turn preempted for a known continuation,
    # then make its fresh-rid retry block before response headers.
    release = __import__("threading").Event()
    release.set()
    upstream = upstream_factory(block_until=release, abort_releases=True)
    pool = InferenceUpstreamPool.from_urls(upstream.url, capacity_per_upstream=2)
    backend = InferenceBackend(pool, continuation_qos_enabled=True)
    body = json.dumps({"model": "m", "input": "hello", "stream": True}).encode()
    known = await backend.relay(
        "POST", "/v1/responses", body=body, headers={"x-scitex-session-id": "known"}
    )
    await _collect(known.body)
    release.clear()
    disconnected = asyncio.Event()

    async def client_disconnected() -> bool:
        return disconnected.is_set()

    cold = asyncio.create_task(
        backend.relay(
            "POST",
            "/v1/responses",
            body=body,
            headers={"x-scitex-session-id": "cold"},
            client_disconnected=client_disconnected,
        )
    )
    await _wait_for_requests(upstream, 2)
    continuation = await backend.relay(
        "POST", "/v1/responses", body=body, headers={"x-scitex-session-id": "known"}
    )
    release.clear()
    await _collect(continuation.body)
    await _wait_for_requests(upstream, 5)

    # Act
    disconnected.set()
    disconnected_response = await cold
    disconnected_body = await _collect(disconnected_response.body)
    request_paths = [request["path"] for request in upstream.requests]
    request_rids = [
        json.loads(request["body"])["rid"]
        for request in upstream.requests
        if request["path"] == "/v1/responses"
    ]
    abort_rids = [
        json.loads(request["body"])["rid"]
        for request in upstream.requests
        if request["path"] == "/abort_request"
    ]

    # Assert: both cold attempts were explicitly aborted, the retry received a
    # fresh rid, and the downstream disconnect cannot strand admission.
    assert (
        (disconnected_response.status_code, disconnected_body),
        request_paths,
        len(set(request_rids)),
        abort_rids == [request_rids[1], request_rids[3]],
        pool.status()[0]["in_flight"],
        backend.continuation_qos.snapshot()["replay_safe_first_turns"],
    ) == (
        (499, b""),
        [
            "/v1/responses",
            "/v1/responses",
            "/abort_request",
            "/v1/responses",
            "/v1/responses",
            "/abort_request",
        ],
        4,
        True,
        0,
        0,
    )


@pytest.mark.asyncio
async def test_continuation_qos_is_disabled_and_body_compatible_by_default(
    upstream_factory,
) -> None:
    # Arrange
    upstream = upstream_factory()
    backend = InferenceBackend(InferenceUpstreamPool.from_urls(upstream.url))
    original = {"model": "m", "input": "hello", "rid": "caller-rid"}
    # Act
    relayed = await backend.relay(
        "POST",
        "/v1/responses",
        body=json.dumps(original).encode(),
        headers={"x-scitex-session-id": "stable"},
    )
    await _collect(relayed.body)

    # Assert
    assert (
        json.loads(upstream.requests[0]["body"])["rid"],
        backend.continuation_qos.snapshot()["mode"],
    ) == ("caller-rid", "disabled")


@pytest.mark.asyncio
async def test_partial_2xx_stream_does_not_establish_a_continuation(
    upstream_factory,
) -> None:
    # Arrange
    upstream = upstream_factory(chunks=(b"first", b"second"))
    backend = InferenceBackend(
        InferenceUpstreamPool.from_urls(upstream.url),
        continuation_qos_enabled=True,
    )
    body = json.dumps({"model": "m", "input": "hello", "stream": True}).encode()
    first = await backend.relay(
        "POST", "/v1/responses", body=body, headers={"x-scitex-session-id": "stable"}
    )
    await anext(first.body)
    await first.body.aclose()

    # Act
    again = await backend.relay(
        "POST", "/v1/responses", body=body, headers={"x-scitex-session-id": "stable"}
    )
    classified = backend.continuation_qos.snapshot()
    await _collect(again.body)

    # Assert
    assert (
        classified["first_turn"],
        classified["continuation"],
    ) == (2, 0)


@pytest.mark.asyncio
async def test_only_canonical_session_header_classifies_qos(upstream_factory) -> None:
    # Arrange
    upstream = upstream_factory()
    backend = InferenceBackend(
        InferenceUpstreamPool.from_urls(upstream.url),
        continuation_qos_enabled=True,
    )
    body = json.dumps({"model": "m", "input": "hello"}).encode()

    # Act
    legacy = await backend.relay(
        "POST", "/v1/responses", body=body, headers={"x-session-id": "legacy"}
    )
    await _collect(legacy.body)
    canonical = await backend.relay(
        "POST",
        "/v1/responses",
        body=body,
        headers={"x-scitex-session-id": "canonical"},
    )
    await _collect(canonical.body)
    snapshot = backend.continuation_qos.snapshot()

    # Assert
    assert (
        snapshot["unclassified"],
        snapshot["first_turn"],
        snapshot["known_successful_sessions"],
    ) == (1, 1, 1)


@pytest.mark.asyncio
async def test_client_cancel_before_headers_aborts_engine_and_releases_capacity(
    upstream_factory,
) -> None:
    # Arrange
    release = __import__("threading").Event()
    upstream = upstream_factory(block_until=release, abort_releases=True)
    pool = InferenceUpstreamPool.from_urls(upstream.url)
    backend = InferenceBackend(pool, continuation_qos_enabled=True)
    body = json.dumps({"model": "m", "input": "hello", "stream": True}).encode()
    task = asyncio.create_task(
        backend.relay(
            "POST",
            "/v1/responses",
            body=body,
            headers={"x-scitex-session-id": "new"},
        )
    )
    await _wait_for_requests(upstream, 1)

    # Act
    task.cancel()
    cancelled = await _raised_async(task)

    # Assert
    assert (
        isinstance(cancelled, asyncio.CancelledError),
        [request["path"] for request in upstream.requests],
        pool.status()[0]["in_flight"],
    ) == (True, ["/v1/responses", "/abort_request"], 0)


@pytest.mark.asyncio
async def test_invalid_body_is_never_registered_for_preemption(
    upstream_factory,
) -> None:
    # Arrange
    release = __import__("threading").Event()
    upstream = upstream_factory(block_until=release, abort_releases=True)
    backend = InferenceBackend(
        InferenceUpstreamPool.from_urls(upstream.url),
        continuation_qos_enabled=True,
    )
    task = asyncio.create_task(
        backend.relay(
            "POST",
            "/v1/responses",
            body=b"not-json",
            headers={"x-scitex-session-id": "new"},
        )
    )
    await _wait_for_requests(upstream, 1)

    # Act
    while_waiting = backend.continuation_qos.snapshot()["replay_safe_first_turns"]
    release.set()
    relayed = await task
    await _collect(relayed.body)

    # Assert
    assert (
        while_waiting,
        [request["path"] for request in upstream.requests],
    ) == (0, ["/v1/responses"])


@pytest.mark.asyncio
async def test_small_first_turn_is_below_configured_preemption_threshold(
    upstream_factory,
) -> None:
    # Arrange
    release = __import__("threading").Event()
    upstream = upstream_factory(block_until=release)
    backend = InferenceBackend(
        InferenceUpstreamPool.from_urls(upstream.url),
        continuation_qos_enabled=True,
        continuation_qos_min_preempt_tokens=1000,
    )
    task = asyncio.create_task(
        backend.relay(
            "POST",
            "/v1/responses",
            body=b'{"model":"m","input":"tiny"}',
            headers={"x-scitex-session-id": "new"},
        )
    )
    await _wait_for_requests(upstream, 1)

    # Act
    snapshot = backend.continuation_qos.snapshot()
    release.set()
    relayed = await task
    await _collect(relayed.body)

    # Assert
    assert (
        snapshot["min_preempt_tokens"],
        snapshot["replay_safe_first_turns"],
    ) == (1000, 0)


@pytest.mark.asyncio
async def test_failed_midstream_abort_retains_capacity_until_upstream_eof(
    upstream_factory,
) -> None:
    # Arrange
    finish = __import__("threading").Event()
    upstream = upstream_factory(
        chunks=(b"first", b"second"),
        abort_status=500,
        block_after_first_chunk=finish,
    )
    pool = InferenceUpstreamPool.from_urls(upstream.url)
    backend = InferenceBackend(pool, continuation_qos_enabled=True)
    body = json.dumps({"model": "m", "input": "hello", "stream": True}).encode()
    relayed = await backend.relay(
        "POST",
        "/v1/responses",
        body=body,
        headers={"x-scitex-session-id": "new"},
    )
    await anext(relayed.body)

    # Act
    close_task = asyncio.create_task(relayed.body.aclose())
    await _wait_for_requests(upstream, 2)
    before_eof = (close_task.done(), pool.status()[0]["in_flight"])
    finish.set()
    await close_task
    request_rid = json.loads(upstream.requests[0]["body"])["rid"]
    abort_rid = json.loads(upstream.requests[1]["body"])["rid"]

    # Assert
    assert (
        before_eof,
        pool.status()[0]["in_flight"],
        request_rid == abort_rid,
    ) == ((False, 1), 0, True)


@pytest.mark.asyncio
async def test_cancelled_chunk_wait_retains_capacity_when_abort_fails(
    upstream_factory,
) -> None:
    # Arrange
    finish = __import__("threading").Event()
    upstream = upstream_factory(
        chunks=(b"first", b"second"),
        abort_status=500,
        block_after_first_chunk=finish,
    )
    pool = InferenceUpstreamPool.from_urls(upstream.url)
    backend = InferenceBackend(pool, continuation_qos_enabled=True)
    relayed = await backend.relay(
        "POST",
        "/v1/responses",
        body=b'{"model":"m","input":"hello","stream":true}',
        headers={"x-scitex-session-id": "new"},
    )
    await anext(relayed.body)
    waiting = asyncio.create_task(anext(relayed.body))
    await asyncio.sleep(0.02)

    # Act
    waiting.cancel()
    await _wait_for_requests(upstream, 2)
    before_eof = (waiting.done(), pool.status()[0]["in_flight"])
    finish.set()
    cancelled = await _raised_async(waiting)

    # Assert
    assert (
        before_eof,
        isinstance(cancelled, asyncio.CancelledError),
        pool.status()[0]["in_flight"],
    ) == ((False, 1), True, 0)


@pytest.mark.asyncio
async def test_cancelled_chunk_wait_retrieves_concurrent_end_of_stream(
    upstream_factory,
) -> None:
    # Arrange: this scheduler-controlled response ends its read task and then
    # cancels the relay reader before the shielded parent can retrieve the
    # resulting StopAsyncIteration.  This is the ordering observed when a
    # client closed a replay just as the real SGLang stream ended.
    upstream = upstream_factory()
    pool = InferenceUpstreamPool.from_urls(upstream.url)
    backend = InferenceBackend(pool, continuation_qos_enabled=True)
    member = await pool.acquire("new", input_tokens=10)
    reader: dict[str, asyncio.Task[bytes]] = {}
    loop = asyncio.get_running_loop()
    unhandled: list[dict] = []
    previous_handler = loop.get_exception_handler()

    class EndingResponse:
        def aiter_bytes(self):
            async def ending_stream():
                loop.call_soon(reader["task"].cancel)
                if False:
                    yield b""

            return ending_stream()

        async def aclose(self) -> None:
            return None

    class ClosingClient:
        async def aclose(self) -> None:
            return None

    relayed = backend._drain(
        ClosingClient(),
        EndingResponse(),
        member,
        input_tokens=10,
        session_id="new",
        request_id="rid-ending",
    )
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))

    # Act
    try:
        reader["task"] = asyncio.create_task(anext(relayed))
        cancelled = await _raised_async(reader["task"])
        await asyncio.sleep(0)
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)
        await backend.close()

    # Assert
    unretrieved = [
        context
        for context in unhandled
        if context.get("message") == "Task exception was never retrieved"
    ]
    assert (
        isinstance(cancelled, asyncio.CancelledError),
        pool.status()[0]["in_flight"],
        unretrieved,
    ) == (True, 0, [])


@pytest.mark.asyncio
async def test_cleanup_reaper_retries_abort_then_releases_exactly_once(
    upstream_factory,
) -> None:
    # Arrange
    upstream = upstream_factory(abort_status=500)
    pool = InferenceUpstreamPool.from_urls(upstream.url)
    backend = InferenceBackend(pool, continuation_qos_enabled=True)
    member = await pool.acquire("held", input_tokens=10)
    backend._schedule_cleanup_reaper(
        member,
        "rid-held",
        headers={},
        input_tokens=10,
        session_id="held",
    )
    await _wait_for_requests(upstream, 1)

    # Act
    upstream.abort_status = 200
    reapers = tuple(backend._cleanup_reapers)
    await _wait_for_in_flight(pool, 0)
    await asyncio.wait_for(asyncio.gather(*reapers), timeout=2)
    snapshot = backend.continuation_qos.snapshot()
    await backend.close()

    # Assert
    assert (
        snapshot["cleanup_reapers_active"],
        snapshot["cleanup_reaper_attempts"] >= 2,
        snapshot["cleanup_reaper_recoveries"],
        pool.status()[0]["in_flight"],
    ) == (0, True, 1, 0)


@pytest.mark.asyncio
async def test_non_addressable_cleanup_releases_instead_of_spawning_infinite_reaper(
    upstream_factory,
) -> None:
    # Arrange: the Anthropic adapter does not propagate a gateway-owned rid.
    # Reproduce the post-header disconnect path after draining the transport
    # failed, which left one 685,947-token admission permanently reserved in
    # production on 2026-09-12.
    upstream = upstream_factory()
    pool = InferenceUpstreamPool.from_urls(
        upstream.url, token_capacity_per_upstream=1_100_000
    )
    backend = InferenceBackend(pool, continuation_qos_enabled=True)
    member = await pool.acquire("anthropic", input_tokens=685_947)

    class ClosedResponse:
        async def aclose(self) -> None:
            return None

    class ClosedClient:
        async def aclose(self) -> None:
            return None

    # Act: false means the engine state was not confirmed through an explicit
    # abort. An empty request id makes such confirmation impossible.
    await backend._finish(
        ClosedClient(),
        ClosedResponse(),
        member,
        input_tokens=685_947,
        session_id="anthropic",
        release_capacity=False,
        request_id="",
    )
    snapshot = backend.continuation_qos.snapshot()

    # Assert: no keyless reaper can spin forever and poison both slot and token
    # accounting. Closing the non-addressable transport is the terminal cleanup
    # event; SGLang's native Anthropic StreamingResponse owns engine abort on
    # downstream disconnect.
    assert (
        pool.status()[0]["in_flight"],
        pool.status()[0]["input_tokens_in_flight"],
        snapshot["cleanup_reapers_active"],
        snapshot["non_addressable_cleanup_releases"],
    ) == (0, 0, 0, 1)


@pytest.mark.asyncio
async def test_cleanup_reaper_rejects_missing_engine_request_id(
    upstream_factory,
) -> None:
    # Arrange
    upstream = upstream_factory()
    pool = InferenceUpstreamPool.from_urls(upstream.url)
    backend = InferenceBackend(pool, continuation_qos_enabled=True)
    member = await pool.acquire("held")

    # Act
    with pytest.raises(
        ValueError, match="cleanup reaper requires a non-empty request id"
    ):
        backend._schedule_cleanup_reaper(
            member,
            "",
            headers={},
            input_tokens=0,
            session_id="held",
        )

    # Assert
    await pool.release(member, session_id="held")


@pytest.mark.asyncio
async def test_close_cancels_visible_reaper_without_double_release(
    upstream_factory,
) -> None:
    # Arrange
    upstream = upstream_factory(abort_status=500)
    pool = InferenceUpstreamPool.from_urls(upstream.url)
    backend = InferenceBackend(pool, continuation_qos_enabled=True)
    member = await pool.acquire("held")
    backend._schedule_cleanup_reaper(
        member,
        "rid-held",
        headers={},
        input_tokens=0,
        session_id="held",
    )
    await _wait_for_requests(upstream, 1)
    before = backend.continuation_qos.snapshot()["cleanup_reapers_active"]

    # Act
    await backend.close()
    snapshot = backend.continuation_qos.snapshot()

    # Assert
    assert (
        before,
        snapshot["cleanup_reapers_active"],
        snapshot["cleanup_reaper_cancellations"],
        pool.status()[0]["in_flight"],
    ) == (1, 0, 1, 1)


@pytest.mark.asyncio
async def test_relay_returns_the_upstream_status_and_body_verbatim(
    upstream_factory,
) -> None:
    # Arrange
    rejection = b'{"error":{"type":"invalid_request_error","message":"nope"}}'
    upstream = upstream_factory(status=400, chunks=(rejection,))
    backend = InferenceBackend(InferenceUpstreamPool.from_urls(upstream.url))
    # Act
    relayed = await backend.relay(
        "POST", "/v1/messages", body=json.dumps(_request()).encode(), headers={}
    )
    received = await _collect(relayed.body)
    # Assert
    assert (relayed.status_code, received) == (400, rejection)


@pytest.mark.asyncio
async def test_relay_forwards_a_non_json_body_untouched(upstream_factory) -> None:
    # Arrange
    upstream = upstream_factory()
    backend = InferenceBackend(InferenceUpstreamPool.from_urls(upstream.url))
    # Act
    relayed = await backend.relay(
        "POST", "/v1/messages", body=b"not json at all", headers={}
    )
    await _collect(relayed.body)
    # Assert
    assert upstream.requests[0]["body"] == b"not json at all"


@pytest.mark.asyncio
async def test_relay_drops_hop_by_hop_headers_and_forwards_the_rest(
    upstream_factory,
) -> None:
    # Arrange
    upstream = upstream_factory()
    backend = InferenceBackend(InferenceUpstreamPool.from_urls(upstream.url))
    headers = {
        "x-api-key": "agent-key",
        "anthropic-version": "2023-06-01",
        "host": "gateway.test",
        "content-length": "1",
    }
    # Act
    relayed = await backend.relay(
        "POST", "/v1/messages", body=json.dumps(_request()).encode(), headers=headers
    )
    await _collect(relayed.body)
    seen = upstream.requests[0]["headers"]
    # Assert
    assert (seen["x-api-key"], seen["anthropic-version"], seen["host"]) == (
        "agent-key",
        "2023-06-01",
        upstream.url.removeprefix("http://"),
    )


@pytest.mark.asyncio
async def test_relay_holds_the_upstream_in_flight_while_streaming(
    upstream_factory,
) -> None:
    # Arrange
    upstream = upstream_factory(chunks=(b"first", b"second"))
    pool = InferenceUpstreamPool.from_urls(upstream.url)
    lines: list[str] = []
    backend = InferenceBackend(pool, journal=lines.append)
    # Act
    relayed = await backend.relay(
        "POST", "/v1/messages", body=json.dumps(_request()).encode(), headers={}
    )
    await anext(relayed.body)
    while_streaming = pool.upstreams[0].in_flight
    await relayed.body.aclose()
    # Assert
    assert (
        while_streaming,
        pool.upstreams[0].in_flight,
        any("outcome=client_disconnected" in line for line in lines),
    ) == (1, 0, True)


@pytest.mark.asyncio
async def test_relay_rotates_past_an_unreachable_upstream(
    upstream_factory, dead_url_factory
) -> None:
    # Arrange
    live = upstream_factory()
    pool = InferenceUpstreamPool.from_urls([dead_url_factory(), live.url])
    backend = InferenceBackend(pool)
    # Act
    relayed = await backend.relay(
        "POST", "/v1/messages", body=json.dumps(_request()).encode(), headers={}
    )
    await _collect(relayed.body)
    # Assert
    assert (
        relayed.status_code,
        len(live.requests),
        pool.upstreams[0].cooldown_until > 0,
    ) == (
        200,
        1,
        True,
    )


@pytest.mark.asyncio
async def test_relay_refuses_naming_inference_upstreams_when_none_answers(
    dead_url_factory,
) -> None:
    # Arrange
    dead = [dead_url_factory(), dead_url_factory()]
    backend = InferenceBackend(InferenceUpstreamPool.from_urls(dead))
    # Round robin tries them in order, so the message names them in order.
    expected = (
        r"^No inference upstream answered: "
        + re.escape(dead[0])
        + r" \(.*\); "
        + re.escape(dead[1])
        + r" \(.*\)\. Configured inference upstreams \(2\): "
    )

    # Act
    async def relay() -> None:
        await backend.relay(
            "POST", "/v1/messages", body=json.dumps(_request()).encode(), headers={}
        )

    # Assert
    with pytest.raises(UpstreamUnreachable, match=expected):
        await relay()


@pytest.mark.asyncio
async def test_relay_holds_a_conversation_whose_home_just_died(
    dead_url_factory,
) -> None:
    # Arrange -- the conversation was placed on the one upstream, which then
    # produced no response. Measured 2026-09-05: re-placing it elsewhere is how
    # the request that killed one replica killed the other. The home stays
    # pinned and the caller is told to retry.
    backend = InferenceBackend(
        InferenceUpstreamPool.from_urls(dead_url_factory()), wait_for_home_s=0.0
    )
    body = json.dumps(_request()).encode()
    with contextlib.suppress(UpstreamUnreachable):
        await backend.relay("POST", "/v1/messages", body=body, headers={})

    # Act
    async def relay() -> None:
        await backend.relay("POST", "/v1/messages", body=body, headers={})

    # Assert
    with pytest.raises(UpstreamReloading, match="stays pinned to its home upstream"):
        await relay()


@pytest.mark.asyncio
async def test_relay_waits_out_a_reloading_home_and_then_serves_from_it(
    upstream_factory,
) -> None:
    # Arrange -- the conversation's home is live but was marked cooling a
    # moment ago with a short cooldown: the shape of a replica mid-reload.
    # Measured 2026-09-05: Codex ends its turn on a 503, so the relay must
    # keep the request open and try the home again once its cooldown lapses.
    upstream = upstream_factory()
    lines: list[str] = []
    backend = InferenceBackend(
        InferenceUpstreamPool.from_urls(upstream.url), telemetry_sink=lines.append
    )
    body = json.dumps(_request()).encode()
    first = await backend.relay("POST", "/v1/messages", body=body, headers={})
    await _collect(first.body)
    home = backend.pool.upstreams[0]
    await backend.pool.cool_down(home, 0.3)

    # Act
    relayed = await backend.relay("POST", "/v1/messages", body=body, headers={})
    await _collect(relayed.body)

    # Assert -- served by the home after a wait, never refused.
    waited = [line for line in lines if "waiting" in line and "for its home" in line]
    assert (relayed.status_code, len(upstream.requests), len(waited) >= 1) == (
        200,
        2,
        True,
    )


@pytest.mark.asyncio
async def test_relay_refuses_with_the_cooling_message_while_all_are_cooling(
    dead_url_factory,
) -> None:
    # Arrange -- a body with no conversation key has no home to wait for; when
    # every upstream is cooling it gets the honest "all cooling" refusal.
    backend = InferenceBackend(InferenceUpstreamPool.from_urls(dead_url_factory()))
    keyless = json.dumps({"model": "m", "max_tokens": 8}).encode()
    with contextlib.suppress(UpstreamUnreachable):
        await backend.relay("POST", "/v1/messages", body=keyless, headers={})

    # Act
    async def relay() -> None:
        await backend.relay("POST", "/v1/messages", body=keyless, headers={})

    # Assert
    with pytest.raises(
        UpstreamUnreachable, match="All inference upstreams are cooling down"
    ):
        await relay()


@pytest.mark.asyncio
async def test_telemetry_sink_receives_a_size_only_report(upstream_factory) -> None:
    # Arrange
    upstream = upstream_factory()
    lines: list[str] = []
    backend = InferenceBackend(
        InferenceUpstreamPool.from_urls(upstream.url), telemetry_sink=lines.append
    )
    # Act
    relayed = await backend.relay(
        "POST",
        "/v1/messages",
        body=json.dumps(_request(system=SECRET * 40)).encode(),
        headers={},
    )
    await _collect(relayed.body)
    # Assert
    prefix = [line for line in lines if line.startswith("[prefix]")]
    assert (
        len(prefix),
        prefix[0].startswith("[prefix] conv="),
        SECRET in prefix[0],
    ) == (
        1,
        True,
        False,
    )


@pytest.mark.asyncio
async def test_the_journal_says_which_request_went_where_and_how_it_ended(
    upstream_factory,
) -> None:
    # Arrange -- the one place that knows both the request and the upstream.
    upstream = upstream_factory()
    lines: list[str] = []
    backend = InferenceBackend(
        InferenceUpstreamPool.from_urls(upstream.url), telemetry_sink=lines.append
    )
    body = json.dumps(_request(system=SECRET * 40)).encode()

    # Act
    relayed = await backend.relay("POST", "/v1/messages", body=body, headers={})
    await _collect(relayed.body)

    # Assert -- a send line naming the upstream and size, a finish line with the
    # status, and never the payload itself.
    relay = [line for line in lines if line.startswith("[relay]")]
    assert (
        len(relay),
        f"-> {upstream.url} POST /v1/messages bytes=" in relay[0],
        "estimated_input_tokens=" in relay[0],
        "admitted_input_tokens=" in relay[0],
        f"<- {upstream.url} status=200 outcome=complete bytes=" in relay[1],
        any(SECRET in line for line in relay),
    ) == (2, True, True, True, True, False)


@pytest.mark.asyncio
async def test_relay_journal_reports_ttft_prefix_and_upstream_cache_tiers(
    upstream_factory,
) -> None:
    # Arrange
    upstream = upstream_factory(
        chunks=(
            b'data: {"sglext":{"cached_tokens_details":{"device":400,'
            b'"host":20,"storage":5,"storage_backend":"HiCacheFile"}}}\n\n',
            b'data: {"usage":{"prompt_tokens":500,"completion_tokens":7,'
            b'"prompt_tokens_details":{"cached_tokens":425}}}\n\n',
            b"data: [DONE]\n\n",
        )
    )
    lines: list[str] = []
    backend = InferenceBackend(
        InferenceUpstreamPool.from_urls(upstream.url),
        journal=lines.append,
        cache_report_enabled=True,
    )
    body = json.dumps(
        {
            "model": "local",
            "stream": True,
            "messages": [{"role": "user", "content": SECRET}],
        }
    ).encode()

    # Act
    relayed = await backend.relay("POST", "/v1/chat/completions", body=body, headers={})
    await _collect(relayed.body)

    # Assert
    sent = json.loads(upstream.requests[0]["body"])
    assert (
        sent["return_cached_tokens_details"],
        sent["stream_options"]["include_usage"],
        "prefix_fingerprint=" in lines[0],
        "cache_report_requested=true" in lines[0],
        "queue_s=" in lines[0],
        "ttft_s=" in lines[1],
        "total_s=" in lines[1],
        "reported_input_tokens=500" in lines[1],
        "reported_output_tokens=7" in lines[1],
        "cached_tokens=425" in lines[1],
        "cache_device_tokens=400" in lines[1],
        "cache_host_tokens=20" in lines[1],
        "cache_storage_tokens=5" in lines[1],
        SECRET in "\n".join(lines),
    ) == (
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    )


@pytest.mark.asyncio
async def test_the_journal_is_written_without_the_prefix_telemetry_opt_in(
    upstream_factory,
) -> None:
    # Arrange -- no telemetry sink (the production default until the env
    # flag is set); only the journal is wired, as the CLI now does.
    # Measured 2026-09-05: with the journal behind the opt-in flag, a night
    # of relayed requests left zero [relay] lines to join a crash against.
    upstream = upstream_factory()
    lines: list[str] = []
    backend = InferenceBackend(
        InferenceUpstreamPool.from_urls(upstream.url), journal=lines.append
    )

    # Act
    relayed = await backend.relay(
        "POST", "/v1/messages", body=json.dumps(_request()).encode(), headers={}
    )
    await _collect(relayed.body)

    # Assert
    assert [line.startswith("[relay]") for line in lines] == [True, True]


@pytest.mark.asyncio
async def test_the_journal_names_the_upstream_that_gave_no_response(
    dead_url_factory,
) -> None:
    # Arrange
    dead = dead_url_factory()
    lines: list[str] = []
    backend = InferenceBackend(
        InferenceUpstreamPool.from_urls(dead), telemetry_sink=lines.append
    )

    # Act
    async def relay() -> None:
        await backend.relay(
            "POST", "/v1/messages", body=json.dumps(_request()).encode(), headers={}
        )

    with contextlib.suppress(UpstreamUnreachable):
        await relay()

    # Assert
    relay_lines = [line for line in lines if line.startswith("[relay]")]
    assert [f"<- {dead} no response (" in line for line in relay_lines] == [False, True]


# ---------------------------------------------------------------------------
# The OpenAI protocol (2026-09-05, Codex): one sticky key for three shapes,
# and the hoist confined to the route whose shape asks for it.
# ---------------------------------------------------------------------------


def test_conversation_key_is_unchanged_for_a_hoisted_anthropic_body() -> None:
    # Arrange -- the fleet's existing bodies must hash exactly as before, or
    # every sticky assignment flushes on deploy.
    payload, _ = hoist_system(_request())
    legacy_seed = json.dumps(
        [payload.get("system"), payload["messages"][0]], sort_keys=True, default=str
    )
    # Act
    key = conversation_key(payload)
    # Assert
    assert key == hashlib.sha256(legacy_seed.encode()).hexdigest()


def test_conversation_key_reads_a_responses_body_with_a_string_input() -> None:
    # Arrange -- Codex's shape: instructions + input, input allowed to be bare.
    payload = {"model": "m", "instructions": "You are terse.", "input": "Hello"}
    # Act
    key = conversation_key(payload)
    # Assert
    assert key is not None


def test_conversation_key_skips_the_shared_system_message_of_a_chat_body() -> None:
    # Arrange -- two chat sessions of the same agent in the same cwd share
    # messages[0]; keying on it would pin both to one replica.
    shared = {"role": "system", "content": "Same instructions"}
    one = {"messages": [shared, {"role": "user", "content": "first task"}]}
    two = {"messages": [shared, {"role": "user", "content": "second task"}]}
    # Act
    keys = (conversation_key(one), conversation_key(two))
    # Assert
    assert keys[0] != keys[1]


def test_session_header_is_case_insensitive_bounded_and_opaque() -> None:
    # Arrange
    upper_case = {"X-SCITEX-SESSION-ID": "  session-a  "}
    lower_case = {"x-scitex-session-id": "session-a"}

    # Act
    normalized = request_session_key(upper_case)
    same = request_session_key(lower_case)

    # Assert
    assert (normalized == same, len(normalized), "session-a" in normalized) == (
        True,
        64,
        False,
    )


def test_scitex_session_header_has_precedence_over_legacy_spellings() -> None:
    # Arrange
    headers = {
        "x-session-id": "legacy-last",
        "session_id": "legacy-first",
        "X-SciTeX-Session-ID": "canonical",
    }

    # Act
    actual = request_session_key(headers)
    canonical = request_session_key({"x-scitex-session-id": "canonical"})

    # Assert
    assert actual == canonical


@pytest.mark.parametrize(
    "path,payload,expected_injection",
    [
        (
            "/v1/messages",
            {
                "model": "m",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "hi"}],
            },
            False,
        ),
        (
            "/v1/chat/completions",
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            True,
        ),
        ("/v1/responses", {"model": "m", "input": "hi"}, True),
    ],
)
def test_explicit_session_is_injected_without_changing_protocol_shape(
    path: str,
    payload: dict,
    expected_injection: bool,
) -> None:
    # Arrange
    raw_identity = "customer/alice/session-123"
    key = request_session_key({"X-SciTeX-Session-ID": raw_identity})
    backend = InferenceBackend(InferenceUpstreamPool.from_urls("http://127.0.0.1:9"))

    # Act
    forwarded, session = backend.prepare(
        json.dumps(payload).encode(),
        hoist=hoists_on(path),
        affinity_key=key,
        inject_session_id=accepts_session_id(path),
    )
    sent = json.loads(forwarded)

    injected = sent.pop("session_id", None)

    # Assert -- only protocol models that propagate the extension receive it.
    assert (session, injected, sent, raw_identity not in forwarded.decode()) == (
        key,
        key if expected_injection else None,
        payload,
        True,
    )


def test_explicit_header_replaces_a_raw_body_session_id() -> None:
    # Arrange
    payload = {"model": "m", "input": "hi", "session_id": "raw-body-identity"}
    key = request_session_key({"session_id": "header-wins"})
    backend = InferenceBackend(InferenceUpstreamPool.from_urls("http://127.0.0.1:9"))

    # Act
    forwarded, _ = backend.prepare(
        json.dumps(payload).encode(), hoist=False, affinity_key=key
    )

    # Assert
    assert json.loads(forwarded)["session_id"] == key


@pytest.mark.parametrize("body", [b"not-json", b"[1,2,3]"])
def test_uninjectable_body_is_forwarded_unchanged_with_session_affinity(
    body: bytes,
) -> None:
    # Arrange
    key = request_session_key({"x-session-id": "stable"})
    backend = InferenceBackend(InferenceUpstreamPool.from_urls("http://127.0.0.1:9"))

    # Act
    prepared = backend.prepare(body, hoist=False, affinity_key=key)

    # Assert
    assert prepared == (body, key)


def test_no_caller_session_does_not_add_a_body_session_id() -> None:
    # Arrange
    body = json.dumps({"model": "m", "input": "hi"}).encode()
    backend = InferenceBackend(InferenceUpstreamPool.from_urls("http://127.0.0.1:9"))

    # Act
    forwarded, _ = backend.prepare(body, hoist=False)

    # Assert
    assert forwarded == body


@pytest.mark.asyncio
async def test_relay_strips_raw_session_headers_and_sends_only_opaque_body_id(
    upstream_factory,
) -> None:
    # Arrange
    upstream = upstream_factory()
    backend = InferenceBackend(InferenceUpstreamPool.from_urls(upstream.url))
    headers = {
        "content-type": "application/json",
        "X-SciTeX-Session-ID": "raw-canonical",
        "session_id": "raw-legacy",
        "x-session-id": "raw-other",
    }

    # Act
    relayed = await backend.relay(
        "POST",
        "/v1/responses",
        body=json.dumps({"model": "m", "input": "hi"}).encode(),
        headers=headers,
    )
    await _collect(relayed.body)
    request = upstream.requests[0]

    headers_are_private = not {
        "x-scitex-session-id",
        "session_id",
        "x-session-id",
    } & set(request["headers"])
    opaque_session = json.loads(request["body"])["session_id"]

    # Assert
    assert (headers_are_private, opaque_session) == (
        True,
        request_session_key(headers),
    )


def test_blank_session_header_is_rejected() -> None:
    # Arrange
    headers = {"X-SciTeX-Session-ID": " \t "}

    # Act
    blank = request_session_key(headers)

    # Assert
    assert blank == ""


def test_blank_session_header_keeps_body_derived_affinity() -> None:
    # Arrange
    payload, _ = hoist_system(_request())
    body = json.dumps(_request()).encode()
    backend = InferenceBackend(InferenceUpstreamPool.from_urls("http://127.0.0.1:9"))

    # Act
    _, fallback_key = backend.prepare(
        body,
        affinity_key=request_session_key({"X-SciTeX-Session-ID": " \t "}),
    )

    # Assert
    assert fallback_key == conversation_key(payload)


def test_hoists_on_is_true_only_for_the_messages_route() -> None:
    # Arrange
    paths = ("/v1/messages?beta=true", "/v1/chat/completions", "/v1/responses")
    # Act
    verdicts = tuple(hoists_on(path) for path in paths)
    # Assert
    assert verdicts == (True, False, False)


def test_prepare_without_hoist_forwards_the_bytes_untouched() -> None:
    # Arrange -- a chat/completions body whose system message must survive.
    body = json.dumps(
        {
            "messages": [
                {"role": "system", "content": "Keep me"},
                {"role": "user", "content": "hi"},
            ]
        }
    ).encode()
    backend = InferenceBackend(InferenceUpstreamPool.from_urls("http://127.0.0.1:9"))
    # Act
    forwarded, _ = backend.prepare(body, hoist=False)
    # Assert
    assert forwarded == body


@pytest.mark.asyncio
async def test_relay_keeps_a_chat_completions_system_message_in_place(
    upstream_factory,
) -> None:
    # Arrange -- measured 2026-09-05: hoisting this body made vLLM discard
    # the instructions silently. The route must not hoist.
    upstream = upstream_factory()
    backend = InferenceBackend(InferenceUpstreamPool.from_urls(upstream.url))
    body = {
        "messages": [
            {"role": "system", "content": "Keep me"},
            {"role": "user", "content": "hi"},
        ]
    }
    # Act
    relayed = await backend.relay(
        "POST",
        "/v1/chat/completions",
        body=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    await _collect(relayed.body)
    sent = json.loads(upstream.requests[0]["body"])
    # Assert
    assert ("system" in sent, [m["role"] for m in sent["messages"]]) == (
        False,
        ["system", "user"],
    )


# ---------------------------------------------------------------------------
# One system preamble, first (2026-09-05, the first live codex turns): vLLM
# refuses the developer role, then refuses a second system message.
# ---------------------------------------------------------------------------


def test_responses_developer_items_fold_into_the_instructions() -> None:
    # Arrange -- Codex's shape: top-level instructions AND a developer item.
    payload = {
        "instructions": "Base rules.",
        "input": [
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": "Be terse."}],
            },
            {"role": "user", "content": "hi"},
        ],
    }
    # Act
    adapted, _ = adapt_openai_roles(payload)
    # Assert -- one preamble, in `instructions`; the item is gone from input.
    assert (adapted["instructions"], [i["role"] for i in adapted["input"]]) == (
        "Base rules.\n\nBe terse.",
        ["user"],
    )


def test_responses_without_instructions_still_get_one_preamble() -> None:
    # Arrange
    payload = {
        "input": [
            {"role": "developer", "content": "Be terse."},
            {"role": "user", "content": "hi"},
        ]
    }
    # Act
    adapted, changed = adapt_openai_roles(payload)
    # Assert
    assert (adapted["instructions"], changed) == ("Be terse.", True)


def test_chat_preamble_messages_merge_into_one_system_message_first() -> None:
    # Arrange -- a developer message after the user turn, plus a system one.
    payload = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "developer", "content": "Be terse."},
            {"role": "system", "content": "Base rules."},
        ]
    }
    # Act
    adapted, _ = adapt_openai_roles(payload)
    # Assert
    assert [m["role"] for m in adapted["messages"]] == ["system", "user"]


def test_a_chat_body_already_in_shape_is_untouched() -> None:
    # Arrange -- one system message, already first: nothing to do.
    payload = {
        "messages": [
            {"role": "system", "content": "x"},
            {"role": "user", "content": "y"},
        ]
    }
    # Act
    _, changed = adapt_openai_roles(payload)
    # Assert
    assert changed is False


def test_a_string_input_is_left_alone() -> None:
    # Arrange -- the Responses schema allows a bare string.
    payload = {"input": "hello"}
    # Act
    adapted, changed = adapt_openai_roles(payload)
    # Assert
    assert (adapted["input"], changed) == ("hello", False)


@pytest.mark.asyncio
async def test_relay_sends_one_preamble_on_the_responses_route(
    upstream_factory,
) -> None:
    # Arrange
    upstream = upstream_factory()
    backend = InferenceBackend(InferenceUpstreamPool.from_urls(upstream.url))
    body = {
        "model": "m",
        "instructions": "Base rules.",
        "input": [
            {"role": "developer", "content": "Be terse."},
            {"role": "user", "content": "hi"},
        ],
    }
    # Act
    relayed = await backend.relay(
        "POST",
        "/v1/responses",
        body=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    await _collect(relayed.body)
    sent = json.loads(upstream.requests[0]["body"])
    # Assert
    assert (sent["instructions"], [i["role"] for i in sent["input"]]) == (
        "Base rules.\n\nBe terse.",
        ["user"],
    )
