from __future__ import annotations

import contextlib
import hashlib
import json
import re
from collections.abc import AsyncIterator

import pytest

from scitex_genai.gateway._errors import (
    NoAccountAvailable,
    UpstreamReloading,
    UpstreamUnreachable,
)
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
        f"<- {upstream.url} status=200 bytes=" in relay[1],
        any(SECRET in line for line in relay),
    ) == (2, True, True, False)


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
