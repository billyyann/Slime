"""OpenHands REST backend: drives a remote agent-server over HTTP + WebSocket.

Unlike ``OpenHandsBackend`` (which links openhands-sdk *into this process* and
therefore reaches straight into the local filesystem), this backend talks to an
already-running OpenHands agent-server. The server owns the agent loop, the
workspace and the model credentials:

    Agenter process                     agent-server process
    ────────────────                    ─────────────────────
    CodingBackend ──HTTP/WS──►  conversation + agent + workspace

That split is the point of this backend:

- **Workspace isolation is the server's job.** The agent runs where the server
  runs, and the server rejects file/git requests whose ``path`` escapes the
  conversation workspace (HTTP 422). ``sandbox`` therefore always holds; setting
  it to False does not widen anything.
- **Validation works without a shared filesystem.** Modified file contents are
  read back over HTTP, so Agenter can validate a workspace it cannot see. Only
  when that fails does the backend fall back to path-only reporting, which
  requires the paths to be readable locally.
- **The model key belongs to the server.** ``llm_api_key``/``llm_base_url`` are
  sent once, when the conversation is created, for the server's LLM calls.

Requires the optional dependency:

    pip install agenter[openhands-rest]

Known limitations (deliberate, not silent):

- Agenter's custom tools (``extra_tools``) are rejected: injecting tools into a
  remote conversation needs the server's ``client_tools`` protocol, which
  executes them back on this side. Not implemented yet.
- The refusal tool is not advertised in the system prompt, because it would have
  to exist in the server's tool registry to be callable. A Refusal-shaped tool
  call is still detected if the server happens to expose one.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import structlog

from ...data_models import (
    BackendError,
    BackendMessage,
    ConfigurationError,
    ContentModifiedFiles,
    ModifiedFiles,
    PathsModifiedFiles,
    PromptMessage,
    TextMessage,
    ToolCallMessage,
    ToolError,
    ToolErrorCode,
    ToolResult,
    TurnError,
    Usage,
)
from ...pricing import calculate_cost_usd
from ..base import BaseBackend
from ..output_parser import parse_structured_output
from ..refusal import parse_refusal_from_tool_call
from .client import BackendTransportError, OpenHandsRestClient, is_terminal_status
from .constants import (
    DEFAULT_CONVERSATION_MAX_ITERATIONS,
    DEFAULT_GRACE_DRAIN_MAX_FRAMES,
    DEFAULT_GRACE_DRAIN_SECONDS,
    DEFAULT_MAX_STREAM_DISCONNECT_RETRIES,
    DEFAULT_MODEL_OPENHANDS_REST,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_STATUS_POLL_ATTEMPTS,
    DEFAULT_STATUS_POLL_DELAY_SECONDS,
    DEFAULT_STREAM_RETRY_BACKOFF_SECONDS,
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_LLM_API_KEY,
    ENV_LLM_BASE_URL,
    ENV_MODEL,
    FILE_MODIFICATION_TOOLS,
    FRAME_DURABLE,
    FRAME_ERROR,
    FRAME_ITEM_ABORTED,
    FRAME_SYNC,
    FRAME_TRANSIENT,
    KIND_ACTION_EVENT,
    KIND_AGENT_ERROR_EVENT,
    KIND_CONVERSATION_ERROR_EVENT,
    KIND_MESSAGE_EVENT,
    KIND_OBSERVATION_EVENT,
    KIND_SERVER_ERROR_EVENT,
    KIND_STATE_UPDATE_EVENT,
    KIND_USER_REJECT_OBSERVATION,
    LLM_USAGE_ID,
    OPENHANDS_REST_PROMPT,
    PATH_INPUT_KEYS,
    PATH_STATS_USAGE_TO_METRICS,
    SOURCE_AGENT,
    STATE_KEY_EXECUTION_STATUS,
    STATE_KEY_FULL_STATE,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic import BaseModel

    from ...tools import Tool

logger = structlog.get_logger(__name__)

TurnErrorKind = Literal[
    "provider_disconnect",
    "provider_capacity",
    "rpc_error",
    "cancelled",
    "max_tokens",
    "unknown",
]

#: Cap on how many modified files have their contents fetched per validation
#: round. Beyond it the backend reports paths only, so a run that rewrites
#: thousands of files cannot turn validation into thousands of HTTP calls.
DEFAULT_MAX_CONTENT_FILES = 50

# The server's failure vocabulary and the closest Agenter turn-error kind.
_FAILURE_KIND_TO_TURN_ERROR: dict[str, TurnErrorKind] = {
    "transient": "provider_disconnect",
    "rate_limit": "provider_capacity",
    "quota": "provider_capacity",
    "internal": "rpc_error",
    "auth": "unknown",
    "config": "unknown",
    "agent_action": "unknown",
    "unknown": "unknown",
}


def _text_from_content(content: Any) -> str:
    """Join the text blocks of a serialized content list."""
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


class OpenHandsRestBackend(BaseBackend):
    """Backend driving a remote OpenHands agent-server.

    Example:
        backend = OpenHandsRestBackend(
            base_url="http://127.0.0.1:18000",
            api_key=os.environ["OH_SESSION_KEY"],   # server access
            llm_api_key=os.environ["OPENAI_API_KEY"],  # server's model access
        )
        await backend.connect("/path/to/project")
        async for message in backend.execute("Fix the bug"):
            print(message)
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        max_iterations: int = DEFAULT_CONVERSATION_MAX_ITERATIONS,
        delete_on_disconnect: bool = False,
        timeout_seconds: float | None = None,
        max_stream_disconnect_retries: int = DEFAULT_MAX_STREAM_DISCONNECT_RETRIES,
        stream_retry_backoff_seconds: float = DEFAULT_STREAM_RETRY_BACKOFF_SECONDS,
        max_content_files: int = DEFAULT_MAX_CONTENT_FILES,
        extra_tools: list[Tool] | None = None,
    ) -> None:
        """Initialize the backend.

        Args:
            model: Model identifier in litellm format (e.g. "anthropic/claude-sonnet-4-5-20250929",
                "openai/gpt-4o"). The server resolves it. Falls back to the
                ``ACA_OPENHANDS_REST_MODEL`` environment variable.
            base_url: Root URL of the agent-server. Falls back to the
                ``ACA_OPENHANDS_REST_BASE_URL`` environment variable.
            api_key: Server-level session API key, sent as ``X-Session-API-Key``
                and as the socket's first-message auth frame. Falls back to
                ``ACA_OPENHANDS_REST_API_KEY``. Omit when the server runs without
                authentication.
            llm_api_key: Model API key for the *server's* LLM calls, embedded in
                the conversation it creates. Falls back to
                ``ACA_OPENHANDS_REST_LLM_API_KEY``, then to ``ANTHROPIC_API_KEY``
                or ``OPENAI_API_KEY`` based on the model prefix.
            llm_base_url: Optional custom LLM base URL for the server. Falls back
                to ``ACA_OPENHANDS_REST_LLM_BASE_URL``.
            max_iterations: Cap on agent steps within a single run, enforced by
                the server. This is not Agenter's retry budget.
            delete_on_disconnect: Delete the conversation (and its event log) on
                the server when this backend disconnects. Off by default so a
                finished conversation stays inspectable.
            timeout_seconds: Per-request HTTP timeout.
            max_stream_disconnect_retries: How many times to transparently
                reconnect the event stream after a transport drop.
            stream_retry_backoff_seconds: Linear backoff base between stream
                reconnects; attempt N waits N * this many seconds.
            max_content_files: Cap on files whose contents are fetched for
                validation (see ``DEFAULT_MAX_CONTENT_FILES``).
            extra_tools: Not supported by this backend; passing any raises
                ``ConfigurationError``.
        """
        if extra_tools:
            raise ConfigurationError(
                "OpenHandsRestBackend does not support custom tools yet: the tool set of a "
                "remote conversation lives on the agent-server, and injecting client-side "
                "tools requires its client_tools protocol.",
                parameter="tools",
                value=f"{len(extra_tools)} tool(s)",
            )

        self.model = model or os.environ.get(ENV_MODEL) or DEFAULT_MODEL_OPENHANDS_REST
        self._base_url = base_url if base_url is not None else os.environ.get(ENV_BASE_URL)
        self._api_key = api_key if api_key is not None else os.environ.get(ENV_API_KEY)
        self._llm_api_key = llm_api_key if llm_api_key is not None else os.environ.get(ENV_LLM_API_KEY)
        self._llm_base_url = llm_base_url if llm_base_url is not None else os.environ.get(ENV_LLM_BASE_URL)
        self._max_iterations = max_iterations
        self.delete_on_disconnect = delete_on_disconnect
        self._timeout_seconds = timeout_seconds
        self._max_stream_disconnect_retries = max_stream_disconnect_retries
        self._stream_retry_backoff_seconds = stream_retry_backoff_seconds
        self._max_content_files = max_content_files
        # Terminal-status verification budget (instance-level so tests and
        # callers can tighten it).
        self._status_poll_attempts = DEFAULT_STATUS_POLL_ATTEMPTS
        self._status_poll_delay = DEFAULT_STATUS_POLL_DELAY_SECONDS
        self._grace_drain_seconds = DEFAULT_GRACE_DRAIN_SECONDS

        self._init_state()

        # Connection state.
        self._cwd: Path | None = None
        self._custom_system_prompt: str | None = None
        self._client: OpenHandsRestClient | None = None
        self._conversation_id: str | None = None
        self._stream: Any = None

        # Per-session tracking.
        self._files_modified: list[str] = []
        self._last_text_content: str = ""
        self._modified_files_cache: ModifiedFiles | None = None

        # Per-turn signals.
        self._turn_error: TurnError | None = None
        self._execution_status: str | None = None
        self._seen_event_ids: set[str] = set()
        self._message_sent = False
        # Whether the server ever reported usage for this session.
        self._usage_seen = False
        self._last_agent_error: dict[str, Any] | None = None
        self._stream_error: BackendTransportError | None = None

    # -- properties --------------------------------------------------------

    @property
    def cwd(self) -> Path | None:
        """The workspace path as the *server* sees it, if connected."""
        return self._cwd

    @property
    def conversation_id(self) -> str | None:
        """The live conversation identifier, if connected."""
        return self._conversation_id

    @property
    def _effective_timeout(self) -> float:
        """The request timeout actually in force."""
        if self._timeout_seconds is not None:
            return self._timeout_seconds
        return DEFAULT_REQUEST_TIMEOUT_SECONDS

    @property
    def session_id(self) -> str | None:
        """Alias of ``conversation_id`` for the persistent-session contract."""
        return self._conversation_id

    # -- configuration -----------------------------------------------------

    def _resolve_llm_api_key(self) -> str | None:
        """Resolve the key the *server* uses to call the model.

        Explicit configuration wins; otherwise fall back to the key the model's
        provider would conventionally use, so a single-provider setup needs no
        extra variables.
        """
        if self._llm_api_key:
            return self._llm_api_key
        model = self.model.lower()
        if model.startswith("anthropic/") or model.startswith("bedrock/"):
            return os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        if model.startswith("openai/"):
            return os.environ.get("OPENAI_API_KEY")
        return os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")

    def _make_client(self) -> OpenHandsRestClient:
        """Create the HTTP/WS client, validating the connection settings."""
        if not self._base_url:
            raise ConfigurationError(
                "The OpenHands agent-server URL is required for backend='openhands-rest'. "
                f"Pass base_url or set {ENV_BASE_URL}.",
                parameter="base_url",
                value=None,
            )
        if self._client is None:
            kwargs: dict[str, Any] = {}
            if self._timeout_seconds is not None:
                kwargs["timeout_seconds"] = self._timeout_seconds
            self._client = OpenHandsRestClient(
                base_url=self._base_url,
                api_key=self._api_key,
                max_stream_disconnect_retries=self._max_stream_disconnect_retries,
                stream_retry_backoff_seconds=self._stream_retry_backoff_seconds,
                **kwargs,
            )
        return self._client

    def _build_system_prompt(self) -> str:
        """Build the system prompt sent to the server's agent."""
        assert self._cwd is not None
        base = OPENHANDS_REST_PROMPT.format(cwd=self._cwd)
        if self._custom_system_prompt:
            return f"{base}\n\n{self._custom_system_prompt}"
        return base

    # -- lifecycle ---------------------------------------------------------

    async def connect(
        self,
        cwd: str,
        allowed_write_paths: list[str] | None = None,
        resume_session_id: str | None = None,
        output_type: type[BaseModel] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """Create (or resume) the server-side conversation.

        The conversation is created here rather than on the first ``execute()``
        so that ``session_id`` is available as soon as the backend is connected,
        which the persistent-session contract relies on.

        Args:
            cwd: Workspace directory, resolved on the *server*'s filesystem. It
                must be an absolute path the server can read; with a remote
                server that means a shared or server-local path, not a client
                path.
            allowed_write_paths: Not supported (the server confines all
                operations to the conversation workspace). Logged and ignored.
            resume_session_id: An existing conversation id to continue. Its
                server-side event log is the agent's memory, so follow-up
                requests keep full context. Usage already accumulated by that
                conversation is treated as the starting baseline.
            output_type: Optional Pydantic model for structured output, parsed
                from the agent's final response.
            system_prompt: Extra instructions appended to the built-in prompt.
        """
        self._custom_system_prompt = system_prompt
        self._cwd = Path(cwd).resolve()
        self._files_modified = []
        self._last_text_content = ""
        self._modified_files_cache = None
        self._turn_error = None
        self._execution_status = None
        self._stream_error = None
        self._reset_state()
        self._output_type = output_type

        if output_type:
            logger.debug("structured_output_enabled", output_type=output_type.__name__)

        if allowed_write_paths:
            logger.warning(
                "allowed_write_paths is not enforced by OpenHandsRestBackend. "
                "The conversation workspace on the agent-server is the boundary."
            )

        client = self._make_client()
        if resume_session_id:
            info = await client.get_conversation(resume_session_id)
            self._conversation_id = resume_session_id
            self._record_metrics(info)
            logger.debug(
                "backend_resumed",
                conversation_id=resume_session_id,
                cwd=str(self._cwd),
                model=self.model,
            )
            return

        llm_api_key = self._resolve_llm_api_key()
        if not llm_api_key and not self._llm_base_url:
            raise ConfigurationError(
                "No LLM API key available for the OpenHands agent-server. Pass llm_api_key "
                f"or set {ENV_LLM_API_KEY} (or the provider's own key such as OPENAI_API_KEY).",
                parameter="llm_api_key",
                value=None,
            )
        info = await client.create_conversation(
            cwd=str(self._cwd),
            model=self.model,
            llm_api_key=llm_api_key,
            llm_base_url=self._llm_base_url,
            system_prompt=self._build_system_prompt(),
            max_iterations=self._max_iterations,
        )
        self._conversation_id = str(info["id"])
        self._record_metrics(info)
        logger.debug(
            "backend_connected",
            conversation_id=self._conversation_id,
            cwd=str(self._cwd),
            model=self.model,
        )

    async def disconnect(self) -> None:
        """Close the stream/client, optionally deleting the conversation.

        By default the conversation stays on the server (its event log is the
        record of the run). ``delete_on_disconnect=True`` removes it, which also
        removes that record.
        """
        stream, self._stream = self._stream, None
        if stream is not None:
            await stream.aclose()

        client, self._client = self._client, None
        if client is not None:
            if self.delete_on_disconnect and self._conversation_id:
                try:
                    await client.delete_conversation(self._conversation_id)
                except BackendError as e:
                    logger.warning(
                        "conversation_delete_failed",
                        conversation_id=self._conversation_id,
                        error=str(e)[:200],
                    )
            await client.aclose()

        self._conversation_id = None
        self._cwd = None
        self._files_modified = []
        self._last_text_content = ""
        self._modified_files_cache = None
        self._turn_error = None
        self._execution_status = None
        self._stream_error = None
        self._reset_state()

    async def cancel(self) -> None:
        """Interrupt the in-flight turn on the server."""
        if self._client is None or self._conversation_id is None:
            raise BackendError(
                "OpenHandsRestBackend is not connected. Call connect() before cancel().",
                backend="openhands-rest",
            )
        await self._client.interrupt(self._conversation_id)

    # -- execution ---------------------------------------------------------

    async def execute(self, prompt: str) -> AsyncIterator[BackendMessage]:
        """Send one prompt and stream the resulting turn.

        The event socket is opened and subscribed *before* the message is sent,
        so no event of this turn can be missed. Frames are mapped to
        ``BackendMessage``s as they arrive; the turn ends when the conversation
        reaches a terminal status, which is then confirmed against the REST API
        because the socket alone is not a reliable end-of-turn signal.

        Args:
            prompt: The user prompt to execute.

        Yields:
            BackendMessage objects for each significant step.

        Raises:
            BackendError: If not connected, or the conversation is gone.
            ConfigurationError: If the server rejects this client's credentials.
        """
        if self._cwd is None or self._conversation_id is None or self._client is None:
            raise BackendError(
                "OpenHandsRestBackend is not connected. Call connect() before execute().",
                backend="openhands-rest",
            )

        self._turn_error = None
        self._stream_error = None
        self._last_agent_error = None
        self._seen_event_ids = set()
        self._message_sent = False
        # Durability boundary for this turn: the highest seq on disk when the
        # stream first connected. Events at or below it belong to earlier turns
        # and are ignored if a reconnect replays them.
        self._turn_seq_boundary: int | None = None

        client = self._client
        conversation_id = self._conversation_id
        yield PromptMessage(user_prompt=prompt, system_prompt=self._build_system_prompt())

        stream = client.open_event_stream(conversation_id)
        self._stream = stream
        try:
            iterator = stream.__aiter__()
            # The server sends a sync frame only after registering this
            # subscriber, so receiving any frame proves the subscription exists.
            try:
                first = await asyncio.wait_for(iterator.__anext__(), timeout=self._effective_timeout)
            except TimeoutError as e:
                raise BackendTransportError(
                    f"Agent-server event stream for conversation {conversation_id} did not "
                    f"subscribe within {self._effective_timeout:.0f}s."
                ) from e
            self._note_boundary(first)
            if first.get("type") != FRAME_SYNC:
                for message in self._map_frame(first):
                    yield message

            await client.send_message(conversation_id, prompt, run=True)
            self._message_sent = True

            async for frame in iterator:
                if not self._accept_frame(frame):
                    continue
                for message in self._map_frame(frame):
                    yield message
                if self._message_sent and is_terminal_status(self._execution_status):
                    # The status flip is published before the facts that explain
                    # it (a ConversationErrorEvent for a failed run, the last
                    # messages of a finished one). Keep reading briefly instead
                    # of breaking on the status frame itself.
                    for message in await self._grace_drain(iterator):
                        yield message
                    break
        except BackendTransportError as e:
            # The stream died and reconnects were exhausted. The run itself may
            # still be alive, so this is resolved by the status check below.
            self._stream_error = e
            logger.warning(
                "event_stream_exhausted",
                conversation_id=conversation_id,
                error=str(e)[:200],
            )
        finally:
            self._stream = None
            await stream.aclose()

        info = await self._run_post_turn(conversation_id)
        self._record_metrics(info)
        await self._refresh_modified_files(conversation_id)
        await self._finalize_turn(conversation_id)

    async def _grace_drain(self, iterator: AsyncIterator[dict[str, Any]]) -> list[BackendMessage]:
        """Collect the frames that follow a terminal status frame.

        The server flips ``execution_status`` and then persists/publishes the
        events that explain the outcome, so the tail of the turn arrives just
        after the frame that ended the loop. Read until the socket is quiet for
        ``DEFAULT_GRACE_DRAIN_SECONDS`` (or the frame budget runs out), mapping
        whatever arrives; a quiet socket means everything durable has landed.
        """
        messages: list[BackendMessage] = []
        quiet_seconds = self._grace_drain_seconds
        for _ in range(DEFAULT_GRACE_DRAIN_MAX_FRAMES):
            try:
                frame = await asyncio.wait_for(iterator.__anext__(), timeout=quiet_seconds)
            except StopAsyncIteration:
                break  # the stream ended; everything it had is already mapped
            except TimeoutError:
                break  # quiet socket: the tail has landed
            except BackendTransportError:
                break  # reconnects exhausted; the outcome check decides
            for message in self._map_frame(frame):
                messages.append(message)
        return messages

    async def _run_post_turn(self, conversation_id: str) -> dict[str, Any]:
        """Confirm the turn outcome, degrading transport failures to a TurnError.

        The messages of the turn have already been yielded by this point, so a
        failure to *read* the outcome must not turn into an exception that loses
        them: it becomes a retryable turn error, and the session reports it as
        CodingStatus.ERROR.
        """
        try:
            return await self._confirm_terminal(conversation_id)
        except BackendTransportError as e:
            if self._turn_error is None:
                self._turn_error = TurnError(
                    reason=f"The turn outcome could not be confirmed: {e}",
                    kind="provider_disconnect",
                    retryable=True,
                )
            logger.warning(
                "turn_outcome_unconfirmed",
                conversation_id=conversation_id,
                error=str(e)[:200],
            )
            return {}

    async def _confirm_terminal(self, conversation_id: str) -> dict[str, Any]:
        """Return the conversation info once the run reached a terminal status.

        The socket's status frames are the primary signal; this polls the REST
        API as the backstop for the cases where they were missed or the stream
        dropped. If the run is still active after the poll budget, the turn is
        reported as an unknown outcome rather than as success.
        """
        assert self._client is not None
        info = await self._client.get_conversation(conversation_id)
        status = self._status_of(info)
        attempts = 0
        while not is_terminal_status(status) and attempts < self._status_poll_attempts:
            attempts += 1
            await asyncio.sleep(self._status_poll_delay)
            info = await self._client.get_conversation(conversation_id)
            status = self._status_of(info)
        self._execution_status = status

        if not is_terminal_status(status):
            stream_note = (
                f" The event stream also failed: {self._stream_error}" if self._stream_error is not None else ""
            )
            self._turn_error = TurnError(
                reason=(
                    f"Conversation {conversation_id} did not reach a terminal status after "
                    f"{self._status_poll_attempts} checks (last status: {status!r}); the run may "
                    f"still be in flight on the agent-server, so this turn's outcome is unknown."
                    f"{stream_note}"
                ),
                kind="unknown",
                retryable=False,
            )
        elif status == "error":
            self._turn_error = self._turn_error_from_agent_error()
        elif status == "stuck":
            self._turn_error = TurnError(
                reason=(
                    "The agent-server's stuck detection stopped the run, so no result was "
                    "produced. Retrying the same prompt is unlikely to help."
                ),
                kind="unknown",
                retryable=False,
                stop_reason="stuck",
            )
        return info

    def _turn_error_from_agent_error(self) -> TurnError:
        """Map the last agent error's classification onto a TurnError."""
        error = self._last_agent_error or {}
        message = str(error.get("error") or "The conversation ended in an error state.")
        classification = error.get("classification")
        kind = "unknown"
        retryable = False
        if isinstance(classification, dict):
            kind = _FAILURE_KIND_TO_TURN_ERROR.get(str(classification.get("kind") or "unknown"), "unknown")
            retryable = bool(classification.get("retryable"))
        return TurnError(reason=message, kind=kind, retryable=retryable)  # type: ignore[arg-type]

    async def _finalize_turn(self, conversation_id: str) -> None:
        """Resolve structured output for the turn that just ended."""
        if self._output_type is None or self._turn_error is not None:
            return
        assert self._client is not None
        final_response = ""
        try:
            final_response = await self._client.agent_final_response(conversation_id)
        except BackendError as e:
            logger.warning(
                "final_response_fetch_failed",
                conversation_id=conversation_id,
                error=str(e)[:200],
            )
        # The server's final response covers the Finish action too, which never
        # appears as a MessageEvent; fall back to the last text we streamed.
        text = final_response or self._last_text_content
        if final_response and final_response != self._last_text_content:
            self._last_text_content = final_response
        self._structured_output = parse_structured_output(text, self._output_type)

    # -- frame / event mapping --------------------------------------------

    def _note_boundary(self, frame: dict[str, Any]) -> None:
        """Record the pre-turn durability boundary from the first sync frame.

        Only the first sync frame counts: sync frames of later reconnects
        describe a log that already contains this turn, so using them would let
        replayed history through.
        """
        if frame.get("type") != FRAME_SYNC or self._turn_seq_boundary is not None:
            return
        through_seq = frame.get("through_seq")
        if isinstance(through_seq, int):
            self._turn_seq_boundary = through_seq

    def _accept_frame(self, frame: dict[str, Any]) -> bool:
        """Whether a frame belongs to the current turn.

        Live frames always do. A replayed durable frame only does when its seq
        is past the turn boundary — this is what stops a reconnect's full-history
        replay from re-emitting earlier turns, whose terminal status would
        otherwise end this turn immediately.
        """
        if self._turn_seq_boundary is None or frame.get("type") != FRAME_DURABLE:
            return True
        seq = frame.get("seq")
        if isinstance(seq, int) and seq <= self._turn_seq_boundary:
            logger.debug("dropping_pre_turn_frame", seq=seq, boundary=self._turn_seq_boundary)
            return False
        return True

    def _map_frame(self, frame: dict[str, Any]) -> list[BackendMessage]:
        """Map one session-socket frame to backend messages."""
        frame_type = frame.get("type")
        if frame_type == FRAME_ERROR:
            logger.warning(
                "agent_server_socket_error",
                code=frame.get("code"),
                detail=frame.get("detail"),
            )
            return []
        if frame_type == FRAME_ITEM_ABORTED:
            logger.warning("agent_stream_aborted", reason=frame.get("reason"))
            return []
        if frame_type in (FRAME_DURABLE, FRAME_TRANSIENT):
            event = frame.get("event")
            if isinstance(event, dict):
                return self._map_event(event)
        return []

    def _map_event(self, event: dict[str, Any]) -> list[BackendMessage]:
        """Map one conversation event to backend messages (deduplicated by id)."""
        event_id = event.get("id")
        if isinstance(event_id, str):
            if event_id in self._seen_event_ids:
                return []
            self._seen_event_ids.add(event_id)

        kind = event.get("kind")
        if kind == KIND_MESSAGE_EVENT:
            return self._map_message_event(event)
        if kind == KIND_ACTION_EVENT:
            return self._map_action_event(event)
        if kind == KIND_OBSERVATION_EVENT:
            return self._map_observation_event(event)
        if kind == KIND_USER_REJECT_OBSERVATION:
            reason = str(event.get("rejection_reason") or "The action was rejected.")
            return [
                ToolResult(
                    tool_name=event.get("tool_name"),
                    output=reason,
                    success=False,
                    error=ToolError(code=ToolErrorCode.EXECUTION_ERROR, message=reason),
                )
            ]
        if kind == KIND_AGENT_ERROR_EVENT:
            return self._map_agent_error_event(event)
        if kind == KIND_STATE_UPDATE_EVENT:
            self._record_state_event(event)
            return []
        if kind == KIND_CONVERSATION_ERROR_EVENT:
            return self._map_conversation_error_event(event)
        if kind == KIND_SERVER_ERROR_EVENT:
            logger.warning(
                "agent_server_conversation_error",
                code=event.get("code"),
                detail=event.get("detail"),
            )
            return []
        return []

    def _map_message_event(self, event: dict[str, Any]) -> list[BackendMessage]:
        """Map a MessageEvent; only the agent's own text becomes a TextMessage."""
        if event.get("source") != SOURCE_AGENT:
            return []
        text = _text_from_content((event.get("llm_message") or {}).get("content"))
        if not text:
            return []
        self._last_text_content = text
        return [TextMessage(content=text)]

    def _map_action_event(self, event: dict[str, Any]) -> list[BackendMessage]:
        """Map an ActionEvent, detecting refusals and tracking touched paths."""
        tool_name = str(event.get("tool_name") or "unknown")
        action = event.get("action")
        args: dict[str, Any] = action if isinstance(action, dict) else {}

        refusal = parse_refusal_from_tool_call(tool_name, args)
        if refusal is not None:
            self._capture_refusal(refusal.reason, refusal.category)
            logger.info("refusal_captured", reason=refusal.reason, category=refusal.category)
            return [refusal]

        self._track_file_paths(tool_name, args)
        return [ToolCallMessage(tool_name=tool_name, args=args)]

    def _map_observation_event(self, event: dict[str, Any]) -> list[BackendMessage]:
        """Map an ObservationEvent to a tool result."""
        tool_name = event.get("tool_name")
        observation = event.get("observation")
        output = ""
        is_error = False
        if isinstance(observation, dict):
            output = _text_from_content(observation.get("content"))
            is_error = bool(observation.get("is_error"))
        elif observation is not None:
            output = str(observation)
        result = ToolResult(tool_name=tool_name, output=output, success=not is_error)
        if is_error:
            result.error = ToolError(code=ToolErrorCode.EXECUTION_ERROR, message=output or "Tool failed")
        if isinstance(observation, dict) and observation.get("kind") == "BrowserObservation":
            screenshot = observation.get("screenshot_data")
            if screenshot:
                result.metadata["browser_screenshot"] = screenshot
        return [result]

    def _map_agent_error_event(self, event: dict[str, Any]) -> list[BackendMessage]:
        """Map an AgentErrorEvent to a failed tool result, remembering it.

        The error is remembered rather than raised: the agent often recovers, and
        only a conversation that *ends* in the error state makes it a turn error.
        """
        message = str(event.get("error") or "The agent reported an error.")
        self._last_agent_error = {"error": message, "classification": event.get("classification")}
        return [
            ToolResult(
                tool_name=event.get("tool_name"),
                output=message,
                success=False,
                error=ToolError(code=ToolErrorCode.EXECUTION_ERROR, message=message),
            )
        ]

    def _map_conversation_error_event(self, event: dict[str, Any]) -> list[BackendMessage]:
        """Remember a conversation-level error so the turn can explain itself.

        This is the durable counterpart of AgentErrorEvent: the run failed
        before (or instead of) the agent scaffolding producing its own error, so
        it is the only place the reason exists. It is remembered rather than
        raised because the conversation may still recover; a run that ends in
        the error state reports it through ``turn_error()``.
        """
        code = str(event.get("code") or "")
        detail = str(event.get("detail") or "The conversation reported an error.")
        self._last_agent_error = {
            "error": f"{code}: {detail}" if code else detail,
            "classification": event.get("classification"),
        }
        logger.warning("conversation_error", code=code, detail=detail[:300])
        return []

    def _record_state_event(self, event: dict[str, Any]) -> None:
        """Track the execution status carried by a state-update event."""
        key = event.get("key")
        value = event.get("value")
        if key == STATE_KEY_FULL_STATE and isinstance(value, dict):
            status = value.get("execution_status")
            if isinstance(status, str):
                self._execution_status = status
        elif key == STATE_KEY_EXECUTION_STATUS and isinstance(value, str):
            self._execution_status = value

    def _track_file_paths(self, tool_name: str, args: dict[str, Any]) -> None:
        """Remember paths written by file-editing tools.

        Only a fallback for ``modified_files()``: shell commands can write files
        without naming them, which is why git changes are the primary source.
        """
        if tool_name not in FILE_MODIFICATION_TOOLS:
            return
        for key in PATH_INPUT_KEYS:
            raw = args.get(key)
            if not raw or not isinstance(raw, str):
                continue
            path = Path(raw)
            if path.is_absolute() and self._cwd is not None:
                try:
                    path = path.relative_to(self._cwd)
                except ValueError:
                    logger.debug("file_outside_workspace", tool_name=tool_name, path=raw)
                    break
            relative = str(path)
            if relative not in self._files_modified:
                self._files_modified.append(relative)
            break

    # -- usage / files -----------------------------------------------------

    def _status_of(self, info: dict[str, Any]) -> str | None:
        status = info.get("execution_status")
        return status if isinstance(status, str) else None

    def _record_metrics(self, info: dict[str, Any]) -> None:
        """Store the server's cumulative usage for this conversation.

        The values are cumulative server-side, which is exactly the contract
        ``usage()`` must satisfy: the session subtracts the previous reading to
        get per-iteration deltas.

        Usage normally lives at ``info["stats"]["usage_to_metrics"][usage_id]``
        (the ``info["metrics"]`` field is often null). Our own ``usage_id`` is
        preferred so other LLMs of the conversation — a condenser, for example —
        do not land in the total; a conversation created elsewhere falls back to
        summing every entry, which is the honest reading of "what this
        conversation spent".
        """
        metrics = self._metrics_from_stats(info)
        if metrics is None:
            candidate = info.get("metrics")
            metrics = candidate if isinstance(candidate, dict) else None
        if metrics is None:
            self._usage_seen = False
            return
        tokens = metrics.get("accumulated_token_usage")
        if isinstance(tokens, dict):
            self._input_tokens = int(tokens.get("prompt_tokens") or 0)
            self._output_tokens = int(tokens.get("completion_tokens") or 0)
            self._usage_seen = True
        cost = metrics.get("accumulated_cost")
        if isinstance(cost, (int, float)):
            self._cost_usd = float(cost)
            self._usage_seen = True

    def _metrics_from_stats(self, info: dict[str, Any]) -> dict[str, Any] | None:
        """Return the per-usage metrics block from ``stats``, if present."""
        node: Any = info
        for key in PATH_STATS_USAGE_TO_METRICS:
            if not isinstance(node, dict):
                return None
            node = node.get(key)
        if not isinstance(node, dict):
            return None
        preferred = node.get(LLM_USAGE_ID)
        if isinstance(preferred, dict):
            return preferred
        totals: dict[str, Any] = {}
        prompt = completion = 0
        cost = 0.0
        seen = False
        for entry in node.values():
            if not isinstance(entry, dict):
                continue
            tokens = entry.get("accumulated_token_usage")
            if isinstance(tokens, dict):
                prompt += int(tokens.get("prompt_tokens") or 0)
                completion += int(tokens.get("completion_tokens") or 0)
                seen = True
            entry_cost = entry.get("accumulated_cost")
            if isinstance(entry_cost, (int, float)):
                cost += float(entry_cost)
                seen = True
        if not seen:
            return None
        totals["accumulated_token_usage"] = {"prompt_tokens": prompt, "completion_tokens": completion}
        totals["accumulated_cost"] = cost
        return totals

    def usage(self) -> Usage:
        """Return cumulative token usage and cost for this conversation."""
        cost = self._cost_usd or calculate_cost_usd(self.model, self._input_tokens, self._output_tokens)
        return Usage(
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            cost_usd=cost,
            model=self.model,
            provider="openhands-rest",
            reported=self._usage_seen,
        )

    async def _refresh_modified_files(self, conversation_id: str) -> None:
        """Refresh the modified-file cache from the workspace's git state.

        Contents are fetched over HTTP so validation does not depend on this
        process sharing a filesystem with the server. Any failure degrades to
        path-only reporting instead of failing the turn.
        """
        assert self._client is not None
        if self._cwd is None:
            return
        try:
            changes = await self._client.git_changes(conversation_id, str(self._cwd))
        except BackendError as e:
            logger.warning("git_changes_failed", error=str(e)[:200])
            self._modified_files_cache = self._fallback_paths()
            return

        relative_paths = [
            str(change["path"]) for change in changes if isinstance(change.get("path"), str) and change["path"]
        ]
        if not relative_paths:
            # Either nothing changed, or the workspace is not a repository: fall
            # back to what the action stream told us.
            self._modified_files_cache = self._fallback_paths() if self._files_modified else ContentModifiedFiles()
            return

        writable = [
            str(change["path"])
            for change in changes
            if change.get("status") != "DELETED" and isinstance(change.get("path"), str) and change["path"]
        ]
        if len(writable) > self._max_content_files:
            logger.warning(
                "modified_files_content_cap_reached",
                changed=len(writable),
                cap=self._max_content_files,
            )
            self._modified_files_cache = PathsModifiedFiles(file_paths=writable)
            return

        files: dict[str, str] = {}
        for relative in writable:
            content = await self._fetch_file_content(conversation_id, relative)
            if content is not None:
                files[relative] = content
        if not files and writable:
            # Nothing could be read back; report paths so validation can still
            # resolve them locally when the filesystem is shared.
            self._modified_files_cache = PathsModifiedFiles(file_paths=writable)
            return
        self._modified_files_cache = ContentModifiedFiles(files=files)

    async def _fetch_file_content(self, conversation_id: str, relative_path: str) -> str | None:
        """Fetch one changed file's contents, or None when it cannot be read."""
        assert self._client is not None
        assert self._cwd is not None
        absolute = Path(relative_path)
        if not absolute.is_absolute():
            absolute = self._cwd / relative_path
        try:
            raw = await self._client.download_file(conversation_id, str(absolute))
        except BackendError as e:
            logger.debug("file_content_fetch_failed", path=relative_path, error=str(e)[:200])
            return None
        return raw.decode("utf-8", errors="replace")

    def _fallback_paths(self) -> ModifiedFiles:
        """Path-only report built from action-event tracking."""
        if self._files_modified:
            return PathsModifiedFiles(file_paths=list(self._files_modified))
        return ContentModifiedFiles()

    def modified_files(self) -> ModifiedFiles:
        """Return files modified in the conversation workspace.

        With content when the server could provide it (the normal case), so
        validation works without a shared filesystem; path-only otherwise, which
        requires this process to be able to read those paths.
        """
        if self._modified_files_cache is not None:
            return self._modified_files_cache
        return self._fallback_paths()

    # -- turn signals ------------------------------------------------------

    def turn_error(self) -> TurnError | None:
        """Return the turn's transport/provider failure, if the turn had one."""
        return self._turn_error
