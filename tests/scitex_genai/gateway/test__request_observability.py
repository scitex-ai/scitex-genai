from __future__ import annotations

import json

import pytest

from scitex_genai.gateway._request_observability import (
    RequestLifecycleRegistry,
    request_agent_label,
    request_session_label,
)


def test_lifecycle_reports_queue_admission_and_returned_cache_without_identity() -> (
    None
):
    # Arrange
    now = 10.0
    registry = RequestLifecycleRegistry(
        clock=lambda: now,
        request_label_factory=lambda: "request-1",
    )
    raw_agent = "scitex-01/private-agent"
    raw_session = "native-session-secret"

    # Act
    record = registry.start(
        agent_label=request_agent_label({"X-SciTeX-Agent-ID": raw_agent}),
        session_label=request_session_label(raw_session),
        estimated_input_tokens=700_000,
    )
    now = 12.5
    registry.upstream_inflight(
        record,
        upstream="http://compute:18773",
        admitted_input_tokens=700_000,
        gateway_input_tokens_admitted=1_570_000,
        predicted_uncached_tokens=640_000,
        cache_classification="cold",
    )
    owned_while_inflight = registry.snapshot()["requests"][0]["gateway_capacity_owned"]
    registry.capacity_released(record)
    now = 15.0
    registry.completed(
        record,
        cache={"cached_tokens": 60_000, "device_cached_tokens": 60_000},
    )
    now = 99.0

    snapshot = registry.snapshot()
    row = snapshot["requests"][0]
    serialized = json.dumps(snapshot)

    # Assert
    assert (
        owned_while_inflight,
        snapshot["active"],
        snapshot["terminal_total"],
        row["phase"],
        row["queue_elapsed_s"],
        row["elapsed_s"],
        row["phase_elapsed_s"],
        row["estimated_input_tokens"],
        row["admitted_input_tokens"],
        row["gateway_input_tokens_admitted"],
        row["gateway_capacity_owned"],
        row["cache"]["device_cached_tokens"],
        raw_agent not in serialized,
        raw_session not in serialized,
    ) == (
        True,
        0,
        {"completed": 1},
        "completed",
        2.5,
        5.0,
        0.0,
        700_000,
        700_000,
        1_570_000,
        False,
        60_000,
        True,
        True,
    )


def test_active_requests_are_never_evicted_by_bounded_terminal_history() -> None:
    # Arrange
    counter = 0

    def next_label() -> str:
        nonlocal counter
        counter += 1
        return f"request-{counter}"

    registry = RequestLifecycleRegistry(max_recent=1, request_label_factory=next_label)

    # Act
    active = registry.start(
        agent_label="agent", session_label="session", estimated_input_tokens=1
    )
    first = registry.start(
        agent_label="agent", session_label="first", estimated_input_tokens=2
    )
    registry.completed(first)
    second = registry.start(
        agent_label="agent", session_label="second", estimated_input_tokens=3
    )
    registry.disconnected(second)

    snapshot = registry.snapshot()
    health = registry.health_snapshot()

    # Assert
    assert (
        snapshot["active"],
        snapshot["terminal_total"],
        [row["request_label"] for row in snapshot["requests"]],
        snapshot["requests"][0]["phase"],
        snapshot["requests"][1]["phase"],
        "requests" not in health,
    ) == (
        1,
        {"completed": 1, "disconnected": 1},
        [active.request_label, second.request_label],
        "admission_queued",
        "disconnected",
        True,
    )


def test_terminal_disconnect_remains_active_until_capacity_is_released() -> None:
    # Arrange
    registry = RequestLifecycleRegistry(request_label_factory=lambda: "request")
    record = registry.start(
        agent_label="agent", session_label="session", estimated_input_tokens=10
    )
    # Act
    registry.upstream_inflight(
        record,
        upstream="http://compute:1",
        admitted_input_tokens=10,
        gateway_input_tokens_admitted=10,
        predicted_uncached_tokens=10,
        cache_classification="unknown",
    )
    registry.admission_queued(record)
    requeued = registry.snapshot()["requests"][0]
    registry.upstream_inflight(
        record,
        upstream="http://compute:1",
        admitted_input_tokens=10,
        gateway_input_tokens_admitted=10,
        predicted_uncached_tokens=10,
        cache_classification="unknown",
    )

    registry.disconnected(record)
    held = registry.snapshot()
    registry.capacity_released(record)
    released = registry.snapshot()

    # Assert
    assert (
        requeued["upstream"],
        requeued["admitted_input_tokens"],
        requeued["gateway_capacity_owned"],
        held["active"],
        held["active_by_phase"],
        held["requests"][0]["gateway_capacity_owned"],
        released["active"],
        released["requests"][0]["gateway_capacity_owned"],
    ) == (None, None, False, 1, {"disconnected": 1}, True, 0, False)


def test_lifecycle_rejects_negative_recent_bound() -> None:
    # Arrange
    invalid_bound = -1

    # Act
    def construct() -> RequestLifecycleRegistry:
        return RequestLifecycleRegistry(max_recent=invalid_bound)

    # Assert
    with pytest.raises(ValueError, match="max_recent"):
        construct()
