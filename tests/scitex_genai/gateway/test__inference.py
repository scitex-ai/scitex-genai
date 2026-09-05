from __future__ import annotations

import contextlib
import hashlib
import json
import re
from collections.abc import AsyncIterator

import pytest

from scitex_genai.gateway._errors import NoAccountAvailable, UpstreamUnreachable
from scitex_genai.gateway._inference import (
    InferenceBackend,
    InferenceUpstreamPool,
    adapt_openai_roles,
    announce,
    conversation_key,
    hoist_system,
    hoists_on,
    parse_upstreams,
    prefix_report,
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


def test_parse_upstreams_strips_whitespace_and_drops_empties() -> None:
    # Arrange
    value = " http://a:1 ,http://b:2,, "
    # Act
    parsed = parse_upstreams(value)
    # Assert
    assert parsed == ["http://a:1", "http://b:2"]


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
        await pool.release(upstream)
    # Assert
    assert placed == ["http://a:1", "http://b:2", "http://c:3", "http://a:1"]


@pytest.mark.asyncio
async def test_pool_keeps_a_conversation_sticky_even_while_it_is_busy() -> None:
    # Arrange
    pool = InferenceUpstreamPool.from_urls("http://a:1,http://b:2")
    # Act
    first = await pool.acquire("s1")
    concurrent = await pool.acquire("s1")
    # Assert
    assert (first.alias, concurrent.alias, first.in_flight) == (
        "http://a:1",
        "http://a:1",
        2,
    )


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
async def test_relay_hoists_the_body_and_keeps_the_query_string(upstream_factory) -> None:
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
async def test_relay_returns_the_upstream_status_and_body_verbatim(upstream_factory) -> None:
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
async def test_relay_drops_hop_by_hop_headers_and_forwards_the_rest(upstream_factory) -> None:
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
async def test_relay_holds_the_upstream_in_flight_while_streaming(upstream_factory) -> None:
    # Arrange
    upstream = upstream_factory(chunks=(b"first", b"second"))
    pool = InferenceUpstreamPool.from_urls(upstream.url)
    backend = InferenceBackend(pool)
    # Act
    relayed = await backend.relay(
        "POST", "/v1/messages", body=json.dumps(_request()).encode(), headers={}
    )
    await anext(relayed.body)
    while_streaming = pool.upstreams[0].in_flight
    await relayed.body.aclose()
    # Assert
    assert (while_streaming, pool.upstreams[0].in_flight) == (1, 0)


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
    assert (relayed.status_code, len(live.requests), pool.upstreams[0].cooldown_until > 0) == (
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
async def test_relay_refuses_with_the_cooling_message_while_all_are_cooling(
    dead_url_factory,
) -> None:
    # Arrange
    backend = InferenceBackend(InferenceUpstreamPool.from_urls(dead_url_factory()))
    body = json.dumps(_request()).encode()
    with contextlib.suppress(UpstreamUnreachable):
        await backend.relay("POST", "/v1/messages", body=body, headers={})

    # Act
    async def relay() -> None:
        await backend.relay("POST", "/v1/messages", body=body, headers={})

    # Assert
    with pytest.raises(UpstreamUnreachable, match="All inference upstreams are cooling down"):
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
    assert (len(lines), lines[0].startswith("[prefix] conv="), SECRET in lines[0]) == (
        1,
        True,
        False,
    )


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
# The developer role (2026-09-05, first live codex turn): vLLM answers 400
# "Unexpected message role." to it, so the gateway rewrites it to system.
# ---------------------------------------------------------------------------


def test_developer_items_become_system_in_a_responses_body() -> None:
    # Arrange -- Codex's shape: instructions as a developer item, then the turn.
    payload = {
        "input": [
            {"role": "user", "content": "hi"},
            {"role": "developer", "content": "You are terse."},
        ]
    }
    # Act
    adapted, _ = adapt_openai_roles(payload)
    # Assert -- rewritten AND moved first.
    assert [m["role"] for m in adapted["input"]] == ["system", "user"]


def test_developer_messages_become_system_in_a_chat_body() -> None:
    # Arrange
    payload = {
        "messages": [
            {"role": "developer", "content": "x"},
            {"role": "user", "content": "y"},
        ]
    }
    # Act
    adapted, changed = adapt_openai_roles(payload)
    # Assert
    assert (adapted["messages"][0]["role"], changed) == ("system", True)


def test_a_string_input_is_left_alone() -> None:
    # Arrange -- the Responses schema allows a bare string.
    payload = {"input": "hello"}
    # Act
    adapted, changed = adapt_openai_roles(payload)
    # Assert
    assert (adapted["input"], changed) == ("hello", False)


@pytest.mark.asyncio
async def test_relay_rewrites_the_developer_role_on_the_responses_route(
    upstream_factory,
) -> None:
    # Arrange
    upstream = upstream_factory()
    backend = InferenceBackend(InferenceUpstreamPool.from_urls(upstream.url))
    body = {
        "model": "m",
        "input": [
            {"role": "developer", "content": "You are terse."},
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
    assert [m["role"] for m in sent["input"]] == ["system", "user"]

