"""Tests for the OpenHands REST backend.

The agent-server is mocked at two seams:

- REST calls go through ``respx``, which intercepts httpx at the transport
  layer, so the payloads the server would receive are asserted for real.
- The event socket is replaced by ``FakeEventStream``, a scripted frame
  sequence. The real ``EventStream`` (cursor handling, reconnect, close codes)
  has its own tests in ``TestEventStreamReconnect``, which fake websockets.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import respx
from httpx import Response
from pydantic import BaseModel

from agenter import AutonomousCodingAgent
from agenter.coding_backends.openhands_rest import OpenHandsRestBackend
from agenter.coding_backends.openhands_rest.client import (
    BackendTransportError,
    OpenHandsRestClient,
)
from agenter.data_models import (
    BackendError,
    ConfigurationError,
    ContentModifiedFiles,
    PathsModifiedFiles,
    PromptMessage,
    RefusalMessage,
    TextMessage,
    ToolCallMessage,
    ToolResult,
)
from agenter.data_models.types import CodingRequest, CodingStatus

SERVER = "http://oh.test"
CONVERSATION = "11111111-2222-3333-4444-555555555555"


# --------------------------------------------------------------------------- #
# Event / payload builders
# --------------------------------------------------------------------------- #


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def message_event(text: str, *, source: str = "agent", event_id: str = "ev-msg") -> dict[str, Any]:
    return {
        "id": event_id,
        "timestamp": "2026-01-01T00:00:00",
        "source": source,
        "kind": "MessageEvent",
        "llm_message": {"role": "assistant", "content": [text_block(text)]},
    }


def action_event(
    tool_name: str,
    action: dict[str, Any],
    *,
    event_id: str = "ev-act",
) -> dict[str, Any]:
    return {
        "id": event_id,
        "timestamp": "2026-01-01T00:00:01",
        "source": "agent",
        "kind": "ActionEvent",
        "tool_name": tool_name,
        "action": action,
        "tool_call_id": "call-1",
    }


def observation_event(
    tool_name: str,
    text: str,
    *,
    is_error: bool = False,
    event_id: str = "ev-obs",
) -> dict[str, Any]:
    return {
        "id": event_id,
        "timestamp": "2026-01-01T00:00:02",
        "source": "environment",
        "kind": "ObservationEvent",
        "tool_name": tool_name,
        "tool_call_id": "call-1",
        "action_id": "ev-act",
        "observation": {"content": [text_block(text)], "is_error": is_error, "kind": "FileEditorObservation"},
    }


def agent_error_event(
    error: str,
    *,
    kind: str = "transient",
    retryable: bool = True,
    event_id: str = "ev-err",
) -> dict[str, Any]:
    return {
        "id": event_id,
        "timestamp": "2026-01-01T00:00:03",
        "source": "agent",
        "kind": "AgentErrorEvent",
        "tool_name": None,
        "error": error,
        "classification": {"kind": kind, "retryable": retryable},
    }


def state_event(status: str, *, key: str = "execution_status", event_id: str = "ev-state") -> dict[str, Any]:
    value: Any = status if key == "execution_status" else {"execution_status": status}
    return {
        "id": event_id,
        "timestamp": "2026-01-01T00:00:04",
        "source": "environment",
        "kind": "ConversationStateUpdateEvent",
        "key": key,
        "value": value,
    }


def sync_frame(through_seq: int | None = None) -> dict[str, Any]:
    return {"type": "sync", "from_seq": None, "through_seq": through_seq}


def durable(seq: int, event: dict[str, Any]) -> dict[str, Any]:
    return {"type": "durable", "seq": seq, "event": event}


def conversation_info(
    *,
    status: str = "idle",
    conversation_id: str = CONVERSATION,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost: float = 0.0,
) -> dict[str, Any]:
    return {
        "id": conversation_id,
        "execution_status": status,
        "metrics": {
            "model_name": "test-model",
            "accumulated_cost": cost,
            "accumulated_token_usage": {
                "model": "test-model",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        },
    }


class FakeEventStream:
    """Scripted replacement for ``EventStream``.

    Records the ``after_seq`` the backend asked for and replays frames. With
    ``fail_after`` set, the stream raises ``BackendTransportError`` after that
    many frames, mirroring an exhausted reconnect budget.
    """

    def __init__(self, frames: list[dict[str, Any]], *, fail_after: int | None = None) -> None:
        self.frames = frames
        self.fail_after = fail_after
        self.requested_after_seq: int | None = None
        self.closed = False
        self.consumed = 0

    async def _iterate(self):
        for index, frame in enumerate(self.frames):
            if self.fail_after is not None and index >= self.fail_after:
                raise BackendTransportError("stream dropped")
            self.consumed += 1
            yield frame
        if self.fail_after is not None:
            raise BackendTransportError("stream dropped")

    def __aiter__(self):
        return self._iterate()

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def fake_stream(monkeypatch):
    """Patch the client's stream factory and expose the created fake stream.

    The stream is pre-created so a test can set its ``frames`` before executing.
    """
    created: dict[str, Any] = {"stream": FakeEventStream([])}

    def factory(self, conversation_id, *, after_seq=None):
        stream: FakeEventStream = created["stream"]
        stream.requested_after_seq = after_seq
        created["conversation_id"] = conversation_id
        return stream

    monkeypatch.setattr(OpenHandsRestClient, "open_event_stream", factory)
    return created


def make_backend(**kwargs: Any) -> OpenHandsRestBackend:
    params: dict[str, Any] = {
        "base_url": SERVER,
        "api_key": "server-key",
        "llm_api_key": "llm-key",
        "model": "openai/gpt-4o",
    }
    params.update(kwargs)
    backend = OpenHandsRestBackend(**params)
    # Keep the terminal-verification poll instant in tests.
    backend._status_poll_delay = 0.0
    return backend


def mock_create_conversation(**info_kwargs: Any) -> None:
    respx.post(f"{SERVER}/api/conversations").mock(return_value=Response(201, json=conversation_info(**info_kwargs)))


def mock_conversation_info(status: str, **info_kwargs: Any) -> None:
    respx.get(f"{SERVER}/api/conversations/{CONVERSATION}").mock(
        return_value=Response(200, json=conversation_info(status=status, **info_kwargs))
    )


def mock_turn(git: Any = None) -> tuple[respx.Route, respx.Route]:
    """Mock the REST surface an ``execute()`` turn always touches.

    Returns the (message-send, git-changes) routes. ``git`` may be a list of
    changes (the default is "nothing changed") or a ``Response`` when the test
    needs the git call itself to fail.
    """
    send = respx.post(f"{SERVER}/api/conversations/{CONVERSATION}/events").mock(
        return_value=Response(200, json={"success": True})
    )
    body = git if isinstance(git, Response) else (git or [])
    changes = respx.get(f"{SERVER}/api/conversations/{CONVERSATION}/git/changes").mock(
        return_value=body if isinstance(body, Response) else Response(200, json=body)
    )
    return send, changes


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


class TestConfiguration:
    def test_extra_tools_are_rejected(self) -> None:
        class Tool:
            name = "custom"

        with pytest.raises(ConfigurationError, match="does not support custom tools"):
            make_backend(extra_tools=[Tool()])

    def test_default_model(self) -> None:
        assert OpenHandsRestBackend(base_url=SERVER).model == "openai/gpt-4o"

    def test_base_url_required(self, monkeypatch) -> None:
        monkeypatch.delenv("ACA_OPENHANDS_REST_BASE_URL", raising=False)
        backend = OpenHandsRestBackend()
        with pytest.raises(ConfigurationError, match="URL is required"):
            backend._make_client()

    def test_base_url_must_be_http(self) -> None:
        with pytest.raises(ConfigurationError, match="http://"):
            OpenHandsRestClient(base_url="oh.test")

    def test_env_fallback_and_precedence(self, monkeypatch) -> None:
        monkeypatch.setenv("ACA_OPENHANDS_REST_BASE_URL", "http://env.test")
        monkeypatch.setenv("ACA_OPENHANDS_REST_API_KEY", "env-server-key")
        monkeypatch.setenv("ACA_OPENHANDS_REST_LLM_API_KEY", "env-llm-key")
        monkeypatch.setenv("ACA_OPENHANDS_REST_LLM_BASE_URL", "http://llm.test")

        from_env = OpenHandsRestBackend()
        assert from_env._base_url == "http://env.test"
        assert from_env._api_key == "env-server-key"
        assert from_env._resolve_llm_api_key() == "env-llm-key"
        assert from_env._llm_base_url == "http://llm.test"

        explicit = OpenHandsRestBackend(
            base_url="http://explicit.test",
            api_key="explicit-server-key",
            llm_api_key="explicit-llm-key",
            llm_base_url="http://explicit-llm.test",
        )
        assert explicit._base_url == "http://explicit.test"
        assert explicit._api_key == "explicit-server-key"
        assert explicit._resolve_llm_api_key() == "explicit-llm-key"
        assert explicit._llm_base_url == "http://explicit-llm.test"

    def test_llm_key_falls_back_to_provider_key(self, monkeypatch) -> None:
        monkeypatch.delenv("ACA_OPENHANDS_REST_LLM_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")

        assert OpenHandsRestBackend(model="openai/gpt-4o")._resolve_llm_api_key() == "openai-key"
        assert (
            OpenHandsRestBackend(model="anthropic/claude-sonnet-4-5-20250929")._resolve_llm_api_key() == "anthropic-key"
        )

    @pytest.mark.asyncio
    async def test_missing_llm_credentials_raise(self, monkeypatch, tmp_path) -> None:
        for var in (
            "ACA_OPENHANDS_REST_LLM_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "AWS_BEARER_TOKEN_BEDROCK",
        ):
            monkeypatch.delenv(var, raising=False)
        backend = OpenHandsRestBackend(base_url=SERVER)
        with pytest.raises(ConfigurationError, match="No LLM API key available"):
            await backend.connect(str(tmp_path))

    @respx.mock
    @pytest.mark.asyncio
    async def test_api_key_rejected_is_configuration_error(self, tmp_path) -> None:
        respx.post(f"{SERVER}/api/conversations").mock(
            return_value=Response(401, json={"detail": "Invalid session API key"})
        )
        backend = make_backend()
        with pytest.raises(ConfigurationError, match="rejected the API key"):
            await backend.connect(str(tmp_path))

    @pytest.mark.asyncio
    async def test_execute_before_connect_raises(self) -> None:
        backend = make_backend()
        with pytest.raises(BackendError, match="not connected"):
            [msg async for msg in backend.execute("go")]


# --------------------------------------------------------------------------- #
# connect / conversation creation
# --------------------------------------------------------------------------- #


class TestConnect:
    @respx.mock
    @pytest.mark.asyncio
    async def test_create_conversation_payload(self, tmp_path) -> None:
        route = respx.post(f"{SERVER}/api/conversations").mock(return_value=Response(201, json=conversation_info()))
        backend = make_backend()
        await backend.connect(str(tmp_path), system_prompt="Follow the house style.")

        payload = json.loads(route.calls[0].request.content)
        assert payload["agent"]["kind"] == "Agent"
        assert payload["agent"]["llm"] == {"model": "openai/gpt-4o", "usage_id": "agenter", "api_key": "llm-key"}
        assert payload["agent"]["tools"] == [
            {"name": "terminal", "params": {}},
            {"name": "file_editor", "params": {}},
            {"name": "task_tracker", "params": {}},
        ]
        assert str(tmp_path) in payload["agent"]["system_prompt"]
        assert "Follow the house style." in payload["agent"]["system_prompt"]
        assert payload["workspace"] == {"kind": "LocalWorkspace", "working_dir": str(tmp_path)}
        assert payload["confirmation_policy"] == {"kind": "NeverConfirm"}
        assert payload["max_iterations"] == 500

        assert backend.session_id == CONVERSATION
        assert backend.conversation_id == CONVERSATION
        await backend.disconnect()

    @respx.mock
    @pytest.mark.asyncio
    async def test_server_api_key_is_sent_on_rest_calls(self, tmp_path) -> None:
        route = respx.post(f"{SERVER}/api/conversations").mock(return_value=Response(201, json=conversation_info()))
        backend = make_backend()
        await backend.connect(str(tmp_path))
        assert route.calls[0].request.headers["X-Session-API-Key"] == "server-key"
        await backend.disconnect()

    @respx.mock
    @pytest.mark.asyncio
    async def test_custom_iterations_and_llm_base_url(self, tmp_path) -> None:
        route = respx.post(f"{SERVER}/api/conversations").mock(return_value=Response(201, json=conversation_info()))
        backend = make_backend(max_iterations=7, llm_base_url="http://llm.test/v1")
        await backend.connect(str(tmp_path))

        payload = json.loads(route.calls[0].request.content)
        assert payload["max_iterations"] == 7
        assert payload["agent"]["llm"]["base_url"] == "http://llm.test/v1"
        await backend.disconnect()

    @respx.mock
    @pytest.mark.asyncio
    async def test_resume_reuses_conversation_and_baselines_usage(self, tmp_path) -> None:
        create_route = respx.post(f"{SERVER}/api/conversations")
        respx.get(f"{SERVER}/api/conversations/{CONVERSATION}").mock(
            return_value=Response(200, json=conversation_info(prompt_tokens=100, completion_tokens=20, cost=0.5))
        )
        backend = make_backend()
        await backend.connect(str(tmp_path), resume_session_id=CONVERSATION)

        assert not create_route.called, "resuming must not create a new conversation"
        assert backend.session_id == CONVERSATION
        # Usage accumulated before the resume is the baseline, so the session's
        # delta math cannot attribute it to the new request.
        usage = backend.usage()
        assert (usage.input_tokens, usage.output_tokens) == (100, 20)
        assert usage.cost_usd == pytest.approx(0.5)
        await backend.disconnect()

    @respx.mock
    @pytest.mark.asyncio
    async def test_resume_missing_conversation_raises(self, tmp_path) -> None:
        respx.get(f"{SERVER}/api/conversations/{CONVERSATION}").mock(
            return_value=Response(404, json={"detail": "Conversation not found"})
        )
        backend = make_backend()
        with pytest.raises(BackendError, match="404"):
            await backend.connect(str(tmp_path), resume_session_id=CONVERSATION)


# --------------------------------------------------------------------------- #
# execute: message mapping and turn end
# --------------------------------------------------------------------------- #


class TestExecuteMapping:
    @respx.mock
    @pytest.mark.asyncio
    async def test_streams_text_and_tool_calls(self, tmp_path, fake_stream) -> None:
        mock_create_conversation()
        mock_conversation_info("finished", prompt_tokens=10, completion_tokens=5, cost=0.01)
        mock_turn()
        fake_stream["stream"].frames = [
            sync_frame(),
            durable(1, message_event("Working on it")),
            durable(2, action_event("file_editor", {"command": "create", "path": f"{tmp_path}/app.py"})),
            durable(3, observation_event("file_editor", "File created")),
            durable(4, message_event("Done.", event_id="ev-msg-2")),
            durable(5, state_event("finished")),
        ]
        backend = make_backend()
        await backend.connect(str(tmp_path))

        messages = [msg async for msg in backend.execute("Create app.py")]

        assert isinstance(messages[0], PromptMessage)
        assert messages[0].user_prompt == "Create app.py"
        assert [type(m) for m in messages[1:]] == [TextMessage, ToolCallMessage, ToolResult, TextMessage]
        assert messages[1].content == "Working on it"
        assert messages[2].tool_name == "file_editor"
        assert messages[3].success is True
        assert messages[3].output == "File created"
        assert messages[4].content == "Done."
        assert backend.turn_error() is None
        await backend.disconnect()

    @respx.mock
    @pytest.mark.asyncio
    async def test_user_messages_are_not_echoed(self, tmp_path, fake_stream) -> None:
        mock_create_conversation()
        mock_conversation_info("finished")
        mock_turn()
        fake_stream["stream"].frames = [
            sync_frame(),
            durable(1, message_event("Create app.py", source="user", event_id="ev-user")),
            durable(2, state_event("finished")),
        ]
        backend = make_backend()
        await backend.connect(str(tmp_path))
        messages = [msg async for msg in backend.execute("Create app.py")]
        assert not [m for m in messages if isinstance(m, TextMessage)]
        await backend.disconnect()

    @respx.mock
    @pytest.mark.asyncio
    async def test_subscription_happens_before_the_message_is_sent(self, tmp_path, fake_stream) -> None:
        mock_create_conversation()
        mock_conversation_info("finished")
        sent, _ = mock_turn()
        fake_stream["stream"].frames = [sync_frame(), durable(1, state_event("finished"))]
        backend = make_backend()
        await backend.connect(str(tmp_path))
        [msg async for msg in backend.execute("go")]

        assert fake_stream["stream"].requested_after_seq is None, "live-only subscription expected"
        assert sent.called
        await backend.disconnect()

    @respx.mock
    @pytest.mark.asyncio
    async def test_drains_tail_after_terminal_status(self, tmp_path, fake_stream) -> None:
        """Frames after the status flip belong to the turn and must be mapped."""
        mock_create_conversation()
        mock_conversation_info("finished")
        mock_turn()
        fake_stream["stream"].frames = [
            sync_frame(),
            durable(1, state_event("finished")),
            durable(2, message_event("final answer", event_id="ev-late")),
        ]
        backend = make_backend()
        await backend.connect(str(tmp_path))
        messages = [msg async for msg in backend.execute("go")]
        assert [m.content for m in messages if isinstance(m, TextMessage)] == ["final answer"]
        assert fake_stream["stream"].closed
        await backend.disconnect()

    @respx.mock
    @pytest.mark.asyncio
    async def test_grace_drain_is_bounded(self, tmp_path, fake_stream) -> None:
        """With a zero grace window, only the terminal frame itself is read."""
        mock_create_conversation()
        mock_conversation_info("finished")
        mock_turn()
        fake_stream["stream"].frames = [
            sync_frame(),
            durable(1, state_event("finished")),
            durable(2, message_event("too late", event_id="ev-late")),
        ]
        backend = make_backend()
        backend._grace_drain_seconds = 0.0
        await backend.connect(str(tmp_path))
        messages = [msg async for msg in backend.execute("go")]
        assert not [m for m in messages if isinstance(m, TextMessage)]
        assert fake_stream["stream"].consumed == 2  # sync + terminal frame only
        await backend.disconnect()

    @respx.mock
    @pytest.mark.asyncio
    async def test_full_state_snapshot_carries_the_status(self, tmp_path, fake_stream) -> None:
        """``full_state`` events report the status too, not just the keyed form."""
        mock_create_conversation()
        mock_turn()
        mock_conversation_info("finished")
        fake_stream["stream"].frames = [
            sync_frame(),
            durable(1, state_event("running", key="full_state", event_id="ev-running")),
            durable(2, state_event("finished", key="full_state", event_id="ev-finished")),
        ]
        backend = make_backend()
        await backend.connect(str(tmp_path))
        [msg async for msg in backend.execute("go")]
        assert backend._execution_status == "finished"
        await backend.disconnect()

    @respx.mock
    @pytest.mark.asyncio
    async def test_replayed_history_is_ignored_after_reconnect(self, tmp_path, fake_stream) -> None:
        """A full-history replay must not re-emit, nor end the turn immediately."""
        mock_create_conversation()
        mock_conversation_info("finished")
        mock_turn()
        # through_seq=3: events 1..3 were already on disk before this turn.
        fake_stream["stream"].frames = [
            sync_frame(through_seq=3),
            durable(1, message_event("old text", event_id="ev-old")),
            durable(2, state_event("finished", event_id="ev-old-state")),
            durable(4, message_event("this turn", event_id="ev-new")),
            durable(5, state_event("finished", event_id="ev-new-state")),
        ]
        backend = make_backend()
        await backend.connect(str(tmp_path))
        messages = [msg async for msg in backend.execute("go")]
        assert [m.content for m in messages if isinstance(m, TextMessage)] == ["this turn"]
        await backend.disconnect()

    @respx.mock
    @pytest.mark.asyncio
    async def test_refusal_tool_call_is_captured(self, tmp_path, fake_stream) -> None:
        mock_create_conversation()
        mock_conversation_info("finished")
        mock_turn()
        fake_stream["stream"].frames = [
            sync_frame(),
            durable(1, action_event("Refusal", {"reason": "Not allowed", "category": "policy"})),
            durable(2, state_event("finished")),
        ]
        backend = make_backend()
        await backend.connect(str(tmp_path))
        messages = [msg async for msg in backend.execute("do something bad")]

        refusals = [m for m in messages if isinstance(m, RefusalMessage)]
        assert len(refusals) == 1
        assert refusals[0].reason == "Not allowed"
        assert refusals[0].category == "policy"
        assert backend.refusal() is not None
        assert backend.refusal().reason == "Not allowed"
        await backend.disconnect()

    @respx.mock
    @pytest.mark.asyncio
    async def test_replayed_events_are_deduplicated_by_id(self, tmp_path, fake_stream) -> None:
        mock_create_conversation()
        mock_conversation_info("finished")
        mock_turn()
        duplicate = message_event("once", event_id="ev-dup")
        fake_stream["stream"].frames = [
            sync_frame(),
            durable(1, duplicate),
            durable(2, duplicate),
            durable(3, state_event("finished")),
        ]
        backend = make_backend()
        await backend.connect(str(tmp_path))
        messages = [msg async for msg in backend.execute("go")]
        assert [m.content for m in messages if isinstance(m, TextMessage)] == ["once"]
        await backend.disconnect()


# --------------------------------------------------------------------------- #
# execute: failures
# --------------------------------------------------------------------------- #


class TestExecuteFailures:
    @respx.mock
    @pytest.mark.asyncio
    async def test_agent_error_ends_turn_when_status_is_error(self, tmp_path, fake_stream) -> None:
        mock_create_conversation()
        mock_conversation_info("error")
        mock_turn()
        fake_stream["stream"].frames = [
            sync_frame(),
            durable(1, agent_error_event("model stream disconnected", kind="transient", retryable=True)),
            durable(2, state_event("error")),
        ]
        backend = make_backend()
        await backend.connect(str(tmp_path))
        messages = [msg async for msg in backend.execute("go")]

        failures = [m for m in messages if isinstance(m, ToolResult) and not m.success]
        assert len(failures) == 1
        assert "model stream disconnected" in failures[0].output

        turn_error = backend.turn_error()
        assert turn_error is not None
        assert turn_error.kind == "provider_disconnect"
        assert turn_error.retryable is True
        await backend.disconnect()

    @respx.mock
    @pytest.mark.asyncio
    async def test_agent_error_does_not_fail_a_finished_turn(self, tmp_path, fake_stream) -> None:
        """A correctable agent error mid-turn is not a turn failure."""
        mock_create_conversation()
        mock_conversation_info("finished")
        mock_turn()
        fake_stream["stream"].frames = [
            sync_frame(),
            durable(1, agent_error_event("bad tool call", kind="agent_action", retryable=True)),
            durable(2, message_event("recovered", event_id="ev-recovered")),
            durable(3, state_event("finished")),
        ]
        backend = make_backend()
        await backend.connect(str(tmp_path))
        messages = [msg async for msg in backend.execute("go")]

        assert backend.turn_error() is None
        assert any(isinstance(m, TextMessage) and m.content == "recovered" for m in messages)
        await backend.disconnect()

    @respx.mock
    @pytest.mark.asyncio
    async def test_stuck_status_is_not_retryable(self, tmp_path, fake_stream) -> None:
        mock_create_conversation()
        mock_conversation_info("stuck")
        mock_turn()
        fake_stream["stream"].frames = [sync_frame(), durable(1, state_event("stuck"))]
        backend = make_backend()
        await backend.connect(str(tmp_path))
        [msg async for msg in backend.execute("go")]

        turn_error = backend.turn_error()
        assert turn_error is not None
        assert turn_error.retryable is False
        assert turn_error.stop_reason == "stuck"
        await backend.disconnect()

    @respx.mock
    @pytest.mark.asyncio
    async def test_run_still_active_after_drop_is_an_unknown_outcome(self, tmp_path, fake_stream) -> None:
        mock_create_conversation()
        mock_conversation_info("running")
        mock_turn()
        fake_stream["stream"].fail_after = 1  # sync frame, then the socket dies

        backend = make_backend()
        await backend.connect(str(tmp_path))
        messages = [msg async for msg in backend.execute("go")]
        assert len(messages) == 1  # only the PromptMessage made it out

        turn_error = backend.turn_error()
        assert turn_error is not None
        assert turn_error.kind == "unknown"
        assert turn_error.retryable is False
        assert "did not reach a terminal status" in turn_error.reason
        await backend.disconnect()

    @respx.mock
    @pytest.mark.asyncio
    async def test_drop_but_run_finished_is_not_a_failure(self, tmp_path, fake_stream) -> None:
        """A dropped socket must not turn a run that did finish into a failure."""
        mock_create_conversation()
        mock_conversation_info("finished")
        mock_turn()
        fake_stream["stream"].fail_after = 1

        backend = make_backend()
        await backend.connect(str(tmp_path))
        [msg async for msg in backend.execute("go")]
        assert backend.turn_error() is None
        await backend.disconnect()


# --------------------------------------------------------------------------- #
# usage / structured output / modified files
# --------------------------------------------------------------------------- #


class TestUsageAndOutput:
    @respx.mock
    @pytest.mark.asyncio
    async def test_usage_is_cumulative_from_server_metrics(self, tmp_path, fake_stream) -> None:
        mock_create_conversation()
        mock_turn()
        state = {"info": conversation_info(status="finished", prompt_tokens=120, completion_tokens=30, cost=0.25)}
        respx.get(f"{SERVER}/api/conversations/{CONVERSATION}").mock(
            side_effect=lambda request: Response(200, json=state["info"])
        )
        fake_stream["stream"].frames = [sync_frame(), durable(1, state_event("finished"))]

        backend = make_backend()
        await backend.connect(str(tmp_path))
        assert backend.usage().total_tokens == 0

        [msg async for msg in backend.execute("go")]
        usage = backend.usage()
        assert (usage.input_tokens, usage.output_tokens) == (120, 30)
        assert usage.cost_usd == pytest.approx(0.25)
        assert usage.provider == "openhands-rest"

        # The next turn reports the server's new cumulative totals.
        state["info"] = conversation_info(status="finished", prompt_tokens=200, completion_tokens=60, cost=0.4)
        [msg async for msg in backend.execute("again")]
        usage = backend.usage()
        assert (usage.input_tokens, usage.output_tokens) == (200, 60)
        assert usage.cost_usd == pytest.approx(0.4)
        await backend.disconnect()

    @respx.mock
    @pytest.mark.asyncio
    async def test_structured_output_from_final_response(self, tmp_path, fake_stream) -> None:
        class Report(BaseModel):
            summary: str

        mock_create_conversation()
        mock_conversation_info("finished")
        mock_turn()
        respx.get(f"{SERVER}/api/conversations/{CONVERSATION}/agent_final_response").mock(
            return_value=Response(200, json={"response": '{"summary": "all good"}'})
        )
        fake_stream["stream"].frames = [sync_frame(), durable(1, state_event("finished"))]

        backend = make_backend()
        await backend.connect(str(tmp_path), output_type=Report)
        [msg async for msg in backend.execute("report please")]

        output = backend.structured_output()
        assert isinstance(output, Report)
        assert output.summary == "all good"
        await backend.disconnect()

    @respx.mock
    @pytest.mark.asyncio
    async def test_modified_files_fetches_contents(self, tmp_path, fake_stream) -> None:
        mock_create_conversation()
        mock_conversation_info("finished")
        mock_turn(
            git=[
                {"status": "ADDED", "path": "app.py"},
                {"status": "UPDATED", "path": "lib/util.py"},
                {"status": "DELETED", "path": "old.py"},
            ]
        )
        contents = {"app.py": "print(1)", "lib/util.py": "x = 2"}

        def download(request):
            path = request.url.params["path"]
            match = next(suffix for suffix in contents if path.endswith(suffix))
            return Response(200, text=contents[match])

        respx.get(f"{SERVER}/api/conversations/{CONVERSATION}/file/download").mock(side_effect=download)
        fake_stream["stream"].frames = [sync_frame(), durable(1, state_event("finished"))]

        backend = make_backend()
        await backend.connect(str(tmp_path))
        [msg async for msg in backend.execute("go")]

        files = backend.modified_files()
        assert isinstance(files, ContentModifiedFiles)
        assert files.files == {"app.py": "print(1)", "lib/util.py": "x = 2"}
        await backend.disconnect()

    @respx.mock
    @pytest.mark.asyncio
    async def test_modified_files_falls_back_to_tracked_paths(self, tmp_path, fake_stream) -> None:
        mock_create_conversation()
        mock_conversation_info("finished")
        mock_turn(git=Response(400, json={"detail": "not a git repository"}))
        fake_stream["stream"].frames = [
            sync_frame(),
            durable(1, action_event("file_editor", {"command": "create", "path": f"{tmp_path}/app.py"})),
            durable(2, state_event("finished")),
        ]

        backend = make_backend()
        await backend.connect(str(tmp_path))
        [msg async for msg in backend.execute("go")]

        files = backend.modified_files()
        assert isinstance(files, PathsModifiedFiles)
        assert files.file_paths == ["app.py"]
        await backend.disconnect()

    @respx.mock
    @pytest.mark.asyncio
    async def test_modified_files_content_cap(self, tmp_path, fake_stream) -> None:
        mock_create_conversation()
        mock_conversation_info("finished")
        mock_turn(git=[{"status": "ADDED", "path": f"file{i}.py"} for i in range(5)])
        fake_stream["stream"].frames = [sync_frame(), durable(1, state_event("finished"))]

        backend = make_backend(max_content_files=2)
        await backend.connect(str(tmp_path))
        [msg async for msg in backend.execute("go")]

        files = backend.modified_files()
        assert isinstance(files, PathsModifiedFiles)
        assert len(files.file_paths) == 5
        await backend.disconnect()

    @respx.mock
    @pytest.mark.asyncio
    async def test_modified_files_untracked_git_paths_use_cwd_for_download(self, tmp_path, fake_stream) -> None:
        """Relative git paths are resolved against the workspace before download."""
        mock_create_conversation()
        mock_conversation_info("finished")
        mock_turn(git=[{"status": "ADDED", "path": "src/new.py"}])
        download = respx.get(f"{SERVER}/api/conversations/{CONVERSATION}/file/download").mock(
            return_value=Response(200, text="x = 1\n")
        )
        fake_stream["stream"].frames = [sync_frame(), durable(1, state_event("finished"))]

        backend = make_backend()
        await backend.connect(str(tmp_path))
        [msg async for msg in backend.execute("go")]

        assert download.calls[0].request.url.params["path"] == f"{tmp_path}/src/new.py"
        await backend.disconnect()


# --------------------------------------------------------------------------- #
# cancel / disconnect / facade
# --------------------------------------------------------------------------- #


class TestLifecycleAndFacade:
    @respx.mock
    @pytest.mark.asyncio
    async def test_cancel_interrupts(self, tmp_path) -> None:
        mock_create_conversation()
        interrupt = respx.post(f"{SERVER}/api/conversations/{CONVERSATION}/interrupt").mock(
            return_value=Response(200, json={"success": True})
        )
        backend = make_backend()
        await backend.connect(str(tmp_path))
        await backend.cancel()
        assert interrupt.called
        await backend.disconnect()

    @pytest.mark.asyncio
    async def test_cancel_before_connect_raises(self) -> None:
        backend = make_backend()
        with pytest.raises(BackendError, match="not connected"):
            await backend.cancel()

    @respx.mock
    @pytest.mark.asyncio
    async def test_disconnect_keeps_conversation_by_default(self, tmp_path) -> None:
        mock_create_conversation()
        delete = respx.delete(f"{SERVER}/api/conversations/{CONVERSATION}")
        backend = make_backend()
        await backend.connect(str(tmp_path))
        await backend.disconnect()

        assert not delete.called
        assert backend.session_id is None
        assert backend.cwd is None

    @respx.mock
    @pytest.mark.asyncio
    async def test_disconnect_can_delete_conversation(self, tmp_path) -> None:
        mock_create_conversation()
        delete = respx.delete(f"{SERVER}/api/conversations/{CONVERSATION}").mock(
            return_value=Response(200, json={"success": True})
        )
        backend = make_backend(delete_on_disconnect=True)
        await backend.connect(str(tmp_path))
        await backend.disconnect()
        assert delete.called

    def test_backend_is_exported(self) -> None:
        from agenter.coding_backends import OpenHandsRestBackend as Exported

        assert Exported is OpenHandsRestBackend

    def test_facade_passes_model_and_options(self) -> None:
        agent = AutonomousCodingAgent(
            backend="openhands-rest",
            model="openai/gpt-4o-mini",
            openhands_rest_base_url=SERVER,
            openhands_rest_api_key="server-key",
            openhands_rest_llm_api_key="llm-key",
            openhands_rest_max_iterations=9,
            openhands_rest_delete_on_disconnect=True,
        )
        backend = agent._create_backend()

        assert isinstance(backend, OpenHandsRestBackend)
        assert backend.model == "openai/gpt-4o-mini"
        assert backend._base_url == SERVER
        assert backend._api_key == "server-key"
        assert backend._max_iterations == 9
        assert backend.delete_on_disconnect is True

    def test_facade_rejects_unknown_backend(self) -> None:
        with pytest.raises(ConfigurationError, match="Unknown backend"):
            AutonomousCodingAgent(backend="openhands-rest-typo")


# --------------------------------------------------------------------------- #
# end-to-end through CodingSession (status mapping)
# --------------------------------------------------------------------------- #


class TestThroughSession:
    @respx.mock
    @pytest.mark.asyncio
    async def test_completed_run_yields_completed_status(self, tmp_path, fake_stream) -> None:
        mock_create_conversation()
        mock_conversation_info("finished")
        mock_turn(git=[{"status": "ADDED", "path": "app.py"}])
        respx.get(f"{SERVER}/api/conversations/{CONVERSATION}/file/download").mock(
            return_value=Response(200, text="print('hi')\n")
        )
        fake_stream["stream"].frames = [
            sync_frame(),
            durable(1, action_event("file_editor", {"command": "create", "path": f"{tmp_path}/app.py"})),
            durable(2, observation_event("file_editor", "created")),
            durable(3, message_event("done", event_id="ev-done")),
            durable(4, state_event("finished")),
        ]

        agent = AutonomousCodingAgent(
            backend="openhands-rest",
            openhands_rest_base_url=SERVER,
            openhands_rest_api_key="server-key",
            openhands_rest_llm_api_key="llm-key",
        )
        backend = agent._create_backend()
        backend._status_poll_delay = 0.0
        agent._create_backend = lambda: backend  # reuse the configured instance

        result = await agent.execute(CodingRequest(prompt="create app.py", cwd=str(tmp_path)))

        assert result.status == CodingStatus.COMPLETED
        assert result.files == {"app.py": "print('hi')\n"}

    @respx.mock
    @pytest.mark.asyncio
    async def test_transport_failure_maps_to_error_status(self, tmp_path, fake_stream) -> None:
        mock_create_conversation()
        mock_turn()
        respx.get(f"{SERVER}/api/conversations/{CONVERSATION}").mock(
            return_value=Response(500, json={"detail": "boom"})
        )
        fake_stream["stream"].frames = [sync_frame()]

        agent = AutonomousCodingAgent(
            backend="openhands-rest",
            openhands_rest_base_url=SERVER,
            openhands_rest_api_key="server-key",
            openhands_rest_llm_api_key="llm-key",
        )
        backend = agent._create_backend()
        backend._status_poll_delay = 0.0
        agent._create_backend = lambda: backend

        result = await agent.execute(CodingRequest(prompt="go", cwd=str(tmp_path)))

        assert result.status == CodingStatus.ERROR
        assert result.error_kind == "provider_disconnect"
        assert result.retryable is True


# --------------------------------------------------------------------------- #
# The real EventStream: cursor handling, reconnect, close codes
# --------------------------------------------------------------------------- #


class FakeClose(Exception):
    """Stand-in for a websockets ``ConnectionClosed`` with a close code."""

    def __init__(self, code: int) -> None:
        super().__init__(f"closed with code {code}")
        self.code = code


class FakeSocket:
    """Minimal stand-in for a websockets client connection."""

    def __init__(
        self,
        frames: list[dict[str, Any]],
        *,
        close_code: int | None = None,
        fail_after: int | None = None,
    ) -> None:
        self._raws = [json.dumps(frame) for frame in frames]
        self.sent: list[str] = []
        self.closed = False
        self.close_code = close_code
        self.fail_after = fail_after

    async def send(self, raw: str) -> None:
        self.sent.append(raw)

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for index, raw in enumerate(self._raws):
            if self.fail_after is not None and index >= self.fail_after:
                raise FakeClose(self.close_code) if self.close_code is not None else ConnectionError("dropped")
            yield raw
        if self.fail_after is not None:
            if self.close_code is not None:
                raise FakeClose(self.close_code)
            raise ConnectionError("dropped")


async def collect_frames(stream, limit: int) -> list[dict[str, Any]]:
    """Collect up to ``limit`` frames, then close the stream deterministically."""
    frames: list[dict[str, Any]] = []
    async for frame in stream:
        frames.append(frame)
        if len(frames) >= limit:
            await stream.aclose()
            break
    return frames


class TestEventStreamReconnect:
    @pytest.mark.asyncio
    async def test_auth_frame_is_sent_first(self, monkeypatch) -> None:
        import websockets.asyncio.client as ws_client

        socket = FakeSocket([sync_frame()])
        seen_urls: list[str] = []

        async def fake_connect(url, **kwargs):
            seen_urls.append(url)
            return socket

        monkeypatch.setattr(ws_client, "connect", fake_connect)
        client = OpenHandsRestClient(base_url=SERVER, api_key="server-key")
        stream = client.open_event_stream(CONVERSATION)
        frames = await collect_frames(stream, limit=1)

        assert seen_urls == [f"ws://oh.test/sockets/session/{CONVERSATION}"]
        assert json.loads(socket.sent[0]) == {"type": "auth", "session_api_key": "server-key"}
        assert frames == [sync_frame()]
        await client.aclose()

    @pytest.mark.asyncio
    async def test_no_auth_frame_without_api_key(self, monkeypatch) -> None:
        import websockets.asyncio.client as ws_client

        socket = FakeSocket([sync_frame()])

        async def fake_connect(url, **kwargs):
            return socket

        monkeypatch.setattr(ws_client, "connect", fake_connect)
        client = OpenHandsRestClient(base_url=SERVER)
        stream = client.open_event_stream(CONVERSATION)
        await collect_frames(stream, limit=1)
        assert socket.sent == []
        await client.aclose()

    @pytest.mark.asyncio
    async def test_reconnect_resumes_from_last_durable_seq(self, monkeypatch) -> None:
        import websockets.asyncio.client as ws_client

        first = FakeSocket([sync_frame(), durable(1, message_event("one", event_id="e1"))], fail_after=2)
        second = FakeSocket([sync_frame(), durable(2, message_event("two", event_id="e2"))])
        sockets = [first, second]
        urls: list[str] = []

        async def fake_connect(url, **kwargs):
            urls.append(url)
            return sockets.pop(0)

        monkeypatch.setattr(ws_client, "connect", fake_connect)
        client = OpenHandsRestClient(base_url=SERVER, stream_retry_backoff_seconds=0.0)
        stream = client.open_event_stream(CONVERSATION)
        frames = await collect_frames(stream, limit=4)  # both sync frames, durable 1, durable 2

        assert urls[0] == f"ws://oh.test/sockets/session/{CONVERSATION}"
        assert urls[1] == f"ws://oh.test/sockets/session/{CONVERSATION}?after_seq=1"
        assert [f["seq"] for f in frames if f["type"] == "durable"] == [1, 2]
        assert stream.last_seq == 2
        await client.aclose()

    @pytest.mark.asyncio
    async def test_reconnect_without_cursor_requests_full_history(self, monkeypatch) -> None:
        """With no durable frame seen, a resume must not risk a gap."""
        import websockets.asyncio.client as ws_client

        first = FakeSocket([], fail_after=0)  # drops before sending anything
        second = FakeSocket([sync_frame(through_seq=0), durable(0, message_event("replayed", event_id="e0"))])
        sockets = [first, second]
        urls: list[str] = []

        async def fake_connect(url, **kwargs):
            urls.append(url)
            return sockets.pop(0)

        monkeypatch.setattr(ws_client, "connect", fake_connect)
        client = OpenHandsRestClient(base_url=SERVER, stream_retry_backoff_seconds=0.0)
        stream = client.open_event_stream(CONVERSATION)
        await collect_frames(stream, limit=1)

        assert urls[1] == f"ws://oh.test/sockets/session/{CONVERSATION}?after_seq=-1"
        await client.aclose()

    @pytest.mark.asyncio
    async def test_full_history_request_is_honoured_on_first_connect(self, monkeypatch) -> None:
        import websockets.asyncio.client as ws_client

        socket = FakeSocket([sync_frame()])
        urls: list[str] = []

        async def fake_connect(url, **kwargs):
            urls.append(url)
            return socket

        monkeypatch.setattr(ws_client, "connect", fake_connect)
        client = OpenHandsRestClient(base_url=SERVER)
        stream = client.open_event_stream(CONVERSATION, after_seq=-1)
        await collect_frames(stream, limit=1)
        assert urls == [f"ws://oh.test/sockets/session/{CONVERSATION}?after_seq=-1"]
        await client.aclose()

    @pytest.mark.asyncio
    async def test_auth_close_is_not_retried(self, monkeypatch) -> None:
        import websockets.asyncio.client as ws_client

        attempts = 0

        async def fake_connect(url, **kwargs):
            nonlocal attempts
            attempts += 1
            return FakeSocket([], close_code=4001, fail_after=0)

        monkeypatch.setattr(ws_client, "connect", fake_connect)
        client = OpenHandsRestClient(base_url=SERVER, stream_retry_backoff_seconds=0.0)
        stream = client.open_event_stream(CONVERSATION)
        with pytest.raises(FakeClose):
            await collect_frames(stream, limit=1)
        assert attempts == 1, "an auth failure must not be retried"
        await client.aclose()

    @pytest.mark.asyncio
    async def test_unknown_conversation_close_is_not_retried(self, monkeypatch) -> None:
        import websockets.asyncio.client as ws_client

        attempts = 0

        async def fake_connect(url, **kwargs):
            nonlocal attempts
            attempts += 1
            return FakeSocket([], close_code=4004, fail_after=0)

        monkeypatch.setattr(ws_client, "connect", fake_connect)
        client = OpenHandsRestClient(base_url=SERVER, stream_retry_backoff_seconds=0.0)
        stream = client.open_event_stream(CONVERSATION)
        with pytest.raises(FakeClose):
            await collect_frames(stream, limit=1)
        assert attempts == 1
        await client.aclose()

    @pytest.mark.asyncio
    async def test_exhausted_retries_raise_transport_error(self, monkeypatch) -> None:
        import websockets.asyncio.client as ws_client

        attempts = 0

        async def fake_connect(url, **kwargs):
            nonlocal attempts
            attempts += 1
            return FakeSocket([], fail_after=0)

        monkeypatch.setattr(ws_client, "connect", fake_connect)
        client = OpenHandsRestClient(
            base_url=SERVER,
            max_stream_disconnect_retries=2,
            stream_retry_backoff_seconds=0.0,
        )
        stream = client.open_event_stream(CONVERSATION)
        with pytest.raises(BackendTransportError, match="dropped"):
            await collect_frames(stream, limit=1)
        assert attempts == 3  # initial attempt + 2 retries
        await client.aclose()


class TestMissingOptionalDependency:
    """A missing extra must fail fast and clearly, not look like a flaky socket."""

    @pytest.mark.asyncio
    async def test_missing_websockets_raises_configuration_error(self, monkeypatch) -> None:
        import sys

        import websockets  # noqa: F401 - ensures the module is importable in this env

        monkeypatch.setitem(sys.modules, "websockets.asyncio.client", None)
        client = OpenHandsRestClient(base_url=SERVER, stream_retry_backoff_seconds=0.0)
        stream = client.open_event_stream(CONVERSATION)

        with pytest.raises(ConfigurationError, match="websockets is required"):
            await collect_frames(stream, limit=1)
        assert stream.closed
        await client.aclose()


def conversation_error_event(
    code: str = "LLMBadRequestError",
    detail: str = "litellm.BadRequestError: the model rejected the request",
    *,
    kind: str = "config",
    retryable: bool = False,
    event_id: str = "ev-conv-err",
) -> dict[str, Any]:
    return {
        "id": event_id,
        "timestamp": "2026-01-01T00:00:05",
        "source": "environment",
        "kind": "ConversationErrorEvent",
        "code": code,
        "detail": detail,
        "classification": {"kind": kind, "retryable": retryable, "user_action": "settings"},
    }


class TestConversationErrorEvent:
    """A conversation-level error is the only place some failures are explained.

    Shape taken from a live agent-server: a model-compatibility failure produced
    ``ConversationErrorEvent{code=LLMBadRequestError, detail=..., classification}``
    and no AgentErrorEvent at all.
    """

    @respx.mock
    @pytest.mark.asyncio
    async def test_conversation_error_explains_the_failed_turn(self, tmp_path, fake_stream) -> None:
        mock_create_conversation()
        mock_conversation_info("error")
        mock_turn()
        fake_stream["stream"].frames = [
            sync_frame(),
            durable(1, conversation_error_event()),
            durable(2, state_event("error")),
        ]
        backend = make_backend()
        await backend.connect(str(tmp_path))
        [msg async for msg in backend.execute("go")]

        turn_error = backend.turn_error()
        assert turn_error is not None
        assert "LLMBadRequestError" in turn_error.reason
        assert "the model rejected the request" in turn_error.reason
        assert turn_error.retryable is False
        await backend.disconnect()

    @respx.mock
    @pytest.mark.asyncio
    async def test_conversation_error_classification_drives_retryability(self, tmp_path, fake_stream) -> None:
        mock_create_conversation()
        mock_conversation_info("error")
        mock_turn()
        fake_stream["stream"].frames = [
            sync_frame(),
            durable(1, conversation_error_event(kind="transient", retryable=True)),
            durable(2, state_event("error")),
        ]
        backend = make_backend()
        await backend.connect(str(tmp_path))
        [msg async for msg in backend.execute("go")]

        turn_error = backend.turn_error()
        assert turn_error is not None
        assert turn_error.kind == "provider_disconnect"
        assert turn_error.retryable is True
        await backend.disconnect()


class TestUsageFromStats:
    """Usage lives at ``stats.usage_to_metrics[usage_id]``; ``metrics`` is often null."""

    @staticmethod
    def info_with_stats(
        *,
        entries: dict[str, Any],
        metrics: Any = None,
        status: str = "finished",
    ) -> dict[str, Any]:
        payload = conversation_info(status=status)
        payload["metrics"] = metrics
        payload["stats"] = {"usage_to_metrics": entries}
        return payload

    @staticmethod
    def usage_entry(prompt: int, completion: int, cost: float) -> dict[str, Any]:
        return {
            "model_name": "test-model",
            "accumulated_cost": cost,
            "accumulated_token_usage": {"prompt_tokens": prompt, "completion_tokens": completion},
        }

    @respx.mock
    @pytest.mark.asyncio
    async def test_reads_our_usage_id(self, tmp_path, fake_stream) -> None:
        mock_create_conversation()
        mock_turn()
        respx.get(f"{SERVER}/api/conversations/{CONVERSATION}").mock(
            return_value=Response(
                200,
                json=self.info_with_stats(
                    entries={
                        "agenter": self.usage_entry(120, 30, 0.25),
                        "condenser": self.usage_entry(1000, 500, 5.0),
                    }
                ),
            )
        )
        fake_stream["stream"].frames = [sync_frame(), durable(1, state_event("finished"))]

        backend = make_backend()
        await backend.connect(str(tmp_path))
        [msg async for msg in backend.execute("go")]

        usage = backend.usage()
        assert (usage.input_tokens, usage.output_tokens) == (120, 30)
        assert usage.cost_usd == pytest.approx(0.25)
        assert usage.reported is True
        await backend.disconnect()

    @respx.mock
    @pytest.mark.asyncio
    async def test_sums_ids_when_ours_is_absent(self, tmp_path, fake_stream) -> None:
        """A conversation created elsewhere has no 'agenter' entry to read."""
        mock_create_conversation()
        mock_turn()
        respx.get(f"{SERVER}/api/conversations/{CONVERSATION}").mock(
            return_value=Response(
                200,
                json=self.info_with_stats(
                    entries={
                        "ui-llm": self.usage_entry(10, 5, 0.1),
                        "ui-condenser": self.usage_entry(20, 5, 0.2),
                    }
                ),
            )
        )
        fake_stream["stream"].frames = [sync_frame(), durable(1, state_event("finished"))]

        backend = make_backend()
        await backend.connect(str(tmp_path))
        [msg async for msg in backend.execute("go")]

        usage = backend.usage()
        assert (usage.input_tokens, usage.output_tokens) == (30, 10)
        assert usage.cost_usd == pytest.approx(0.3, abs=1e-9)
        assert usage.reported is True
        await backend.disconnect()

    @respx.mock
    @pytest.mark.asyncio
    async def test_falls_back_to_metrics_field(self, tmp_path, fake_stream) -> None:
        mock_create_conversation()
        mock_turn()
        info = conversation_info(status="finished")
        info["metrics"] = self.usage_entry(7, 3, 0.05)
        info["stats"] = {}
        respx.get(f"{SERVER}/api/conversations/{CONVERSATION}").mock(return_value=Response(200, json=info))
        fake_stream["stream"].frames = [sync_frame(), durable(1, state_event("finished"))]

        backend = make_backend()
        await backend.connect(str(tmp_path))
        [msg async for msg in backend.execute("go")]

        usage = backend.usage()
        assert (usage.input_tokens, usage.output_tokens) == (7, 3)
        assert usage.reported is True
        await backend.disconnect()

    @respx.mock
    @pytest.mark.asyncio
    async def test_no_usage_reported_is_flagged(self, tmp_path, fake_stream) -> None:
        """Unavailable usage must not read as a measured zero."""
        mock_create_conversation()
        mock_turn()
        info = conversation_info(status="finished")
        info["metrics"] = None
        info["stats"] = {}
        respx.get(f"{SERVER}/api/conversations/{CONVERSATION}").mock(return_value=Response(200, json=info))
        fake_stream["stream"].frames = [sync_frame(), durable(1, state_event("finished"))]

        backend = make_backend()
        await backend.connect(str(tmp_path))
        [msg async for msg in backend.execute("go")]

        usage = backend.usage()
        assert usage.total_tokens == 0
        assert usage.reported is False
        await backend.disconnect()


class TestModelEnvFallback:
    def test_model_falls_back_to_env(self, monkeypatch) -> None:
        monkeypatch.setenv("ACA_OPENHANDS_REST_MODEL", "openai/deepseek-chat")
        assert OpenHandsRestBackend(base_url=SERVER).model == "openai/deepseek-chat"

    def test_explicit_model_beats_env(self, monkeypatch) -> None:
        monkeypatch.setenv("ACA_OPENHANDS_REST_MODEL", "openai/deepseek-chat")
        assert OpenHandsRestBackend(base_url=SERVER, model="openai/gpt-4o").model == "openai/gpt-4o"


class TestBrowserObservation:
    """Browser screenshots ride along in ToolResult.metadata for the UI."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_browser_observation_screenshot_in_metadata(self, tmp_path, fake_stream) -> None:
        mock_create_conversation()
        mock_conversation_info("finished")
        mock_turn()
        shot = {"id": "ev-shot", "timestamp": "2026-01-01T00:00:02", "source": "environment",
                "kind": "ObservationEvent", "tool_name": "browser", "tool_call_id": "c9",
                "action_id": "ev-act", "observation": {
                    "content": [text_block("page loaded")], "is_error": False,
                    "kind": "BrowserObservation", "screenshot_data": "aGVsbG8="}}
        fake_stream["stream"].frames = [
            sync_frame(),
            durable(1, action_event("browser", {"url": "https://x.test"})),
            durable(2, shot),
            durable(3, state_event("finished")),
        ]
        backend = make_backend()
        await backend.connect(str(tmp_path))
        messages = [msg async for msg in backend.execute("open the page")]

        results = [m for m in messages if isinstance(m, ToolResult)]
        assert len(results) == 1
        assert results[0].metadata.get("browser_screenshot") == "aGVsbG8="
        await backend.disconnect()
