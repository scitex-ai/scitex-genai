from __future__ import annotations

import json
from pathlib import Path

import pytest

from scitex_genai.gateway._inference import (
    ContinuationQoS,
    InferenceUpstream,
    InferenceUpstreamPool,
)
from scitex_genai.gateway._session_state import GatewaySessionState


def test_state_persists_hashed_route_and_success_without_session_text(tmp_path: Path):
    # Arrange
    path = tmp_path / "sessions.json"
    first = GatewaySessionState(path)
    first.remember_route("private-agent-session", "qwen-a")
    first.remember_success("private-agent-session")

    # Act
    restored = GatewaySessionState(path)

    # Assert
    assert (
        restored.route("private-agent-session", {"qwen-a", "qwen-b"}),
        restored.successful("private-agent-session"),
        "private-agent-session" in path.read_text(),
        path.stat().st_mode & 0o777,
    ) == ("qwen-a", True, False, 0o600)


def test_route_restore_rejects_a_member_absent_from_the_new_config(tmp_path: Path):
    # Arrange
    state = GatewaySessionState(tmp_path / "sessions.json")
    state.remember_route("session", "removed-member")

    # Act
    route = state.route("session", {"current-member"})

    # Assert
    assert route is None


def test_namespaces_are_bounded_and_updates_merge(tmp_path: Path):
    # Arrange
    path = tmp_path / "sessions.json"
    state = GatewaySessionState(path, max_sessions=2)
    for name in ("one", "two", "three"):
        state.remember_route(name, "qwen")
        state.remember_success(name)

    # Act
    payload = json.loads(path.read_text())

    # Assert
    assert (len(payload["routes"]), len(payload["successful"])) == (2, 2)


@pytest.mark.asyncio
async def test_new_pool_generation_restores_the_prior_sticky_member(tmp_path: Path):
    # Arrange
    state = GatewaySessionState(tmp_path / "sessions.json")
    first = InferenceUpstreamPool(
        [InferenceUpstream("qwen-a"), InferenceUpstream("qwen-b")],
        choose=lambda candidates: candidates[-1],
        session_state=state,
    )
    replacement = InferenceUpstreamPool(
        [InferenceUpstream("qwen-a"), InferenceUpstream("qwen-b")],
        choose=lambda candidates: candidates[0],
        session_state=GatewaySessionState(state.path),
    )

    # Act
    first_route = await first.route_alias("agent-session")
    restored_route = await replacement.route_alias("agent-session")

    # Assert
    assert (first_route, restored_route) == ("qwen-b", "qwen-b")


def test_new_qos_generation_restores_successful_continuation_class(tmp_path: Path):
    # Arrange
    state = GatewaySessionState(tmp_path / "sessions.json")
    first = ContinuationQoS(enabled=True, session_state=state)
    first.mark_successful("agent-session")

    # Act
    replacement = ContinuationQoS(
        enabled=True, session_state=GatewaySessionState(state.path)
    )

    # Assert
    assert replacement.kind("agent-session") == "continuation"
