"""Tests for persistent openhands-rest sessions.

A persistent session owns one agent-server conversation: every follow-up is
appended to the same conversation, so the server-side event log carries the
agent's memory. The agent-server is mocked with respx plus one scripted event
stream per turn.
"""

from __future__ import annotations

from typing import Any

import pytest
import respx
from httpx import Response

from agenter import AutonomousCodingAgent, CodingStatus
from agenter.coding_backends.openhands_rest.client import OpenHandsRestClient

SERVER = "http://oh.test"
CONVERSATION = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def conversation_info(
    *,
    status: str = "idle",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost: float = 0.0,
) -> dict[str, Any]:
    return {
        "id": CONVERSATION,
        "execution_status": status,
        "metrics": {
            "model_name": "test-model",
            "accumulated_cost": cost,
            "accumulated_token_usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        },
    }


class ScriptedStreams:
    """Serves one scripted frame list per ``__aiter__`` call, in order."""

    def __init__(self, per_turn: list[list[dict[str, Any]]]) -> None:
        self._turns = list(per_turn)
        self.turn = 0
        self.requested_after_seqs: list[int | None] = []
        self.closed = False

    async def _iterate(self, frames: list[dict[str, Any]]):
        for frame in frames:
            yield frame

    def __aiter__(self):
        frames = self._turns[self.turn] if self.turn < len(self._turns) else []
        self.turn += 1
        return self._iterate(frames)

    async def aclose(self) -> None:
        self.closed = True


def state_event(status: str, event_id: str) -> dict[str, Any]:
    return {
        "id": event_id,
        "timestamp": "2026-01-01T00:00:04",
        "source": "environment",
        "kind": "ConversationStateUpdateEvent",
        "key": "execution_status",
        "value": status,
    }


def message_event(text: str, event_id: str) -> dict[str, Any]:
    return {
        "id": event_id,
        "timestamp": "2026-01-01T00:00:00",
        "source": "agent",
        "kind": "MessageEvent",
        "llm_message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def sync(through_seq: int | None = None) -> dict[str, Any]:
    return {"type": "sync", "from_seq": None, "through_seq": through_seq}


def durable(seq: int, event: dict[str, Any]) -> dict[str, Any]:
    return {"type": "durable", "seq": seq, "event": event}


def patch_streams(monkeypatch, streams: ScriptedStreams) -> None:
    """Serve scripted streams and record the cursor each turn asked for."""

    def factory(self, conversation_id, *, after_seq=None):
        streams.requested_after_seqs.append(after_seq)
        return streams

    monkeypatch.setattr(OpenHandsRestClient, "open_event_stream", factory)


def make_agent(**kwargs: Any) -> AutonomousCodingAgent:
    params: dict[str, Any] = {
        "backend": "openhands-rest",
        "openhands_rest_base_url": SERVER,
        "openhands_rest_api_key": "server-key",
        "openhands_rest_llm_api_key": "llm-key",
        "validators": [],
    }
    params.update(kwargs)
    return AutonomousCodingAgent(**params)


def pin_backend(agent: AutonomousCodingAgent):
    """Create the backend once so tests can tune it and keep one instance."""
    backend = agent._create_backend()
    backend._status_poll_delay = 0.0
    agent._create_backend = lambda: backend
    return backend


@respx.mock
@pytest.mark.asyncio
async def test_open_session_reuses_one_conversation_across_followups(tmp_path, monkeypatch) -> None:
    create_route = respx.post(f"{SERVER}/api/conversations").mock(return_value=Response(201, json=conversation_info()))
    send_route = respx.post(f"{SERVER}/api/conversations/{CONVERSATION}/events").mock(
        return_value=Response(200, json={"success": True})
    )
    respx.get(f"{SERVER}/api/conversations/{CONVERSATION}/git/changes").mock(return_value=Response(200, json=[]))
    turn = {"n": 0}

    def info_response(request):
        turn["n"] += 1
        return Response(
            200,
            json=conversation_info(status="finished", prompt_tokens=100 * turn["n"], completion_tokens=10),
        )

    respx.get(f"{SERVER}/api/conversations/{CONVERSATION}").mock(side_effect=info_response)
    streams = ScriptedStreams(
        [
            [sync(), durable(1, message_event("first done", "e1")), durable(2, state_event("finished", "s1"))],
            [sync(2), durable(3, message_event("second done", "e2")), durable(4, state_event("finished", "s2"))],
        ]
    )
    patch_streams(monkeypatch, streams)

    agent = make_agent()
    pin_backend(agent)

    session = await agent.open_session(str(tmp_path))
    assert session.session_id == CONVERSATION

    first = await session.execute("create the first file")
    second = await session.execute("now create the second file")

    assert create_route.call_count == 1, "a persistent session must reuse one conversation"
    assert send_route.call_count == 2, "each follow-up appends one message"
    assert streams.requested_after_seqs == [None, None], "follow-ups subscribe live, never replayed"
    assert first.status == CodingStatus.COMPLETED
    assert second.status == CodingStatus.COMPLETED
    # Usage deltas come from the server's cumulative counters: 110, then +100.
    assert first.total_tokens == 110
    assert second.total_tokens == 100
    assert second.session_total_tokens == 210
    await session.close()


@respx.mock
@pytest.mark.asyncio
async def test_open_session_cancel_interrupts_the_conversation(tmp_path, monkeypatch) -> None:
    respx.post(f"{SERVER}/api/conversations").mock(return_value=Response(201, json=conversation_info()))
    interrupt = respx.post(f"{SERVER}/api/conversations/{CONVERSATION}/interrupt").mock(
        return_value=Response(200, json={"success": True})
    )
    patch_streams(monkeypatch, ScriptedStreams([[]]))

    agent = make_agent()
    pin_backend(agent)

    session = await agent.open_session(str(tmp_path))
    await session.cancel()
    assert interrupt.called
    await session.close()


@respx.mock
@pytest.mark.asyncio
async def test_open_session_resume_continues_an_existing_conversation(tmp_path, monkeypatch) -> None:
    create_route = respx.post(f"{SERVER}/api/conversations")
    respx.get(f"{SERVER}/api/conversations/{CONVERSATION}/git/changes").mock(return_value=Response(200, json=[]))
    respx.post(f"{SERVER}/api/conversations/{CONVERSATION}/events").mock(
        return_value=Response(200, json={"success": True})
    )
    calls = {"n": 0}

    def info_response(request):
        # First read is the resume baseline; later reads include this turn.
        calls["n"] += 1
        if calls["n"] == 1:
            return Response(
                200,
                json=conversation_info(status="finished", prompt_tokens=500, completion_tokens=50),
            )
        return Response(
            200,
            json=conversation_info(status="finished", prompt_tokens=560, completion_tokens=60),
        )

    respx.get(f"{SERVER}/api/conversations/{CONVERSATION}").mock(side_effect=info_response)
    patch_streams(
        monkeypatch,
        ScriptedStreams([[sync(9), durable(10, state_event("finished", "s1"))]]),
    )

    agent = make_agent()
    backend = pin_backend(agent)

    session = await agent.open_session(str(tmp_path), resume_session_id=CONVERSATION)
    assert session.session_id == CONVERSATION
    assert not create_route.called, "resuming must not create a new conversation"
    assert backend.usage().total_tokens == 550, "pre-existing usage is the baseline"

    result = await session.execute("keep going")

    assert result.status == CodingStatus.COMPLETED
    # 620 after the turn minus the 550 baseline: only this turn is billed.
    assert result.total_tokens == 70
    await session.close()


@respx.mock
@pytest.mark.asyncio
async def test_open_session_delete_on_disconnect_removes_the_conversation(tmp_path, monkeypatch) -> None:
    respx.post(f"{SERVER}/api/conversations").mock(return_value=Response(201, json=conversation_info()))
    delete = respx.delete(f"{SERVER}/api/conversations/{CONVERSATION}").mock(
        return_value=Response(200, json={"success": True})
    )
    patch_streams(monkeypatch, ScriptedStreams([[]]))

    agent = make_agent(openhands_rest_delete_on_disconnect=True)
    pin_backend(agent)

    session = await agent.open_session(str(tmp_path))
    await session.close()
    assert delete.called


@pytest.mark.asyncio
async def test_persistent_session_still_rejects_other_backends() -> None:
    """The gate widened for openhands-rest only, not for every backend."""
    from agenter.data_models import ConfigurationError

    agent = AutonomousCodingAgent(backend="codex", model="o4-mini")
    with pytest.raises(ConfigurationError, match=r"open_session\(\) currently supports only"):
        await agent.open_session("/tmp")
