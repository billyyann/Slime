"""HTTP/WebSocket client for the OpenHands agent-server.

Talks to a *running* agent-server over its REST API and its session event
socket. Nothing here imports openhands-sdk: the server is a separate process
(possibly on another host), and this client only needs ``httpx`` and
``websockets``.

Endpoints used (verified against openhands-agent-server 1.49.x):

    POST   /api/conversations                      create a conversation
    GET    /api/conversations/{id}                 status + cumulative metrics
    DELETE /api/conversations/{id}                 delete a conversation
    POST   /api/conversations/{id}/events          append a user message (run=true)
    POST   /api/conversations/{id}/interrupt       stop the in-flight turn
    GET    /api/conversations/{id}/agent_final_response
    GET    /api/conversations/{id}/git/changes     workspace git status
    GET    /api/conversations/{id}/file/download   file contents
    WS     /sockets/session/{id}?after_seq=N       durable event stream

Error mapping is deliberate: authentication problems become
``ConfigurationError`` (retrying cannot fix them), 409s become
``BackendConflictError`` (a run is already in flight), and transport/5xx
failures become ``BackendTransportError`` so callers can tell a retryable fault
from a task failure.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlencode

import structlog

from ...data_models import BackendError, ConfigurationError
from .constants import (
    DEFAULT_CONVERSATION_MAX_ITERATIONS,
    DEFAULT_MAX_STREAM_DISCONNECT_RETRIES,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_SERVER_TOOLS,
    DEFAULT_STREAM_RETRY_BACKOFF_SECONDS,
    ENV_BASE_URL,
    FRAME_DURABLE,
    PATH_CONVERSATIONS,
    PATH_SOCKETS_SESSION,
    SESSION_API_KEY_HEADER,
    TERMINAL_EXECUTION_STATUSES,
    WS_AUTH_FRAME_TYPE,
    WS_CLOSE_AUTH_FAILED,
    WS_CLOSE_CONVERSATION_NOT_FOUND,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import TracebackType

    import httpx

logger = structlog.get_logger(__name__)

#: Largest frame the server emits; a bigger one drops the connection.
DEFAULT_MAX_FRAME_BYTES = 4 * 1024 * 1024

# httpx is imported on first use rather than at module scope so that importing
# Agenter never depends on the REST extra being installed.
_httpx_module: Any = None


def _require_httpx() -> Any:
    """Return the httpx module, or raise if the optional dependency is absent."""
    global _httpx_module
    if _httpx_module is None:
        try:
            import httpx
        except ImportError as e:
            raise ConfigurationError(
                "httpx is required for backend='openhands-rest'. Install with: pip install agenter[openhands-rest]",
                parameter="httpx",
                value=None,
            ) from e
        _httpx_module = httpx
    return _httpx_module


class BackendTransportError(BackendError):
    """A retryable transport failure talking to the agent-server."""

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(message, backend="openhands-rest", cause=cause)


class BackendConflictError(BackendError):
    """The server rejected the request because a run is already in flight."""

    def __init__(self, message: str) -> None:
        super().__init__(message, backend="openhands-rest")


class EventStream:
    """Async iterator over session-socket frames, resuming across disconnects.

    Yields raw frame dictionaries (``{"type": "durable", "seq": 3, "event": ...}``).
    The cursor advances on durable frames only: the server persists those under
    an index, so they are the only frames a reconnect can resume from.

    A connection that fails with an auth close (4001) or an unknown-conversation
    close (4004) raises immediately — retrying cannot help. Transport drops are
    retried up to ``max_stream_disconnect_retries`` with linear backoff; when the
    budget is exhausted the caller sees ``BackendTransportError`` and decides
    whether the turn actually failed.
    """

    def __init__(
        self,
        client: OpenHandsRestClient,
        conversation_id: str,
        *,
        after_seq: int | None = None,
    ) -> None:
        self._client = client
        self._conversation_id = conversation_id
        # Cursor for the *first* connection, exactly as the caller asked for.
        self._initial_cursor = after_seq
        # Cursor of the last durable frame seen; None until one arrives.
        self._cursor: int | None = None
        self._socket: Any = None
        self._closed = False

    @property
    def last_seq(self) -> int | None:
        """Highest durable seq seen, or None if no durable frame arrived yet."""
        return self._cursor

    @property
    def closed(self) -> bool:
        """Whether the stream was closed deliberately."""
        return self._closed

    def _connect_cursor(self, *, first: bool) -> int | None:
        """The ``after_seq`` to use for a connection attempt.

        The first attempt honours the caller's request (None = live-only). A
        reconnect resumes from the last durable seq; without one it asks for the
        full log and lets the backend drop duplicates by event id, because a
        gap would silently lose part of the turn.
        """
        if not first:
            return self._cursor if self._cursor is not None else -1
        return self._initial_cursor

    async def _connect(self, *, first: bool) -> None:
        try:
            from websockets.asyncio.client import connect
        except ImportError as e:
            raise ConfigurationError(
                "websockets is required for backend='openhands-rest'. "
                "Install with: pip install agenter[openhands-rest]",
                parameter="websockets",
                value=None,
            ) from e

        cursor = self._connect_cursor(first=first)
        query = "" if cursor is None else f"?{urlencode({'after_seq': cursor})}"
        url = f"{self._client.ws_base_url}{PATH_SOCKETS_SESSION}/{self._conversation_id}{query}"
        self._socket = await connect(
            url,
            additional_headers=self._client.auth_headers or None,
            open_timeout=self._client.timeout_seconds,
            max_size=self._client.max_frame_bytes,
        )
        if self._client.api_key:
            await self._socket.send(json.dumps({"type": WS_AUTH_FRAME_TYPE, "session_api_key": self._client.api_key}))
        logger.debug(
            "event_stream_connected",
            conversation_id=self._conversation_id,
            after_seq=cursor,
        )

    async def _close_socket(self) -> None:
        socket, self._socket = self._socket, None
        if socket is None:
            return
        try:
            await socket.close()
        except Exception:
            logger.debug("event_stream_close_failed", conversation_id=self._conversation_id)

    async def aclose(self) -> None:
        """Stop iterating and close the socket."""
        self._closed = True
        await self._close_socket()

    async def __aenter__(self) -> EventStream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        attempt = 0
        first = True
        while not self._closed:
            try:
                if self._socket is None:
                    await self._connect(first=first)
                async for raw in self._socket:
                    frame = json.loads(raw)
                    if not isinstance(frame, dict):
                        continue
                    seq = frame.get("seq")
                    if frame.get("type") == FRAME_DURABLE and isinstance(seq, int):
                        self._cursor = seq
                    yield frame
                if self._closed:
                    return
                # The server closed the socket without an error frame; treat it
                # like a drop so the reconnect path resumes from the cursor.
                raise ConnectionError("event stream closed by server")
            except asyncio.CancelledError:
                raise
            except ConfigurationError:
                # A missing dependency or bad URL cannot be fixed by reconnecting.
                await self.aclose()
                raise
            except Exception as e:
                close_code = getattr(e, "code", None) or getattr(getattr(e, "rcvd", None), "code", None)
                if close_code in (WS_CLOSE_AUTH_FAILED, WS_CLOSE_CONVERSATION_NOT_FOUND):
                    await self.aclose()
                    raise
                if self._closed:
                    return
                attempt += 1
                if attempt > self._client.max_stream_disconnect_retries:
                    await self.aclose()
                    raise BackendTransportError(
                        f"Event stream for conversation {self._conversation_id} dropped {attempt} times; giving up.",
                        cause=e if isinstance(e, Exception) else None,
                    ) from e
                backoff = self._client.stream_retry_backoff_seconds * attempt
                logger.warning(
                    "event_stream_reconnect",
                    conversation_id=self._conversation_id,
                    attempt=attempt,
                    max_attempts=self._client.max_stream_disconnect_retries,
                    cursor=self.last_seq,
                    error=str(e)[:200],
                )
                first = False
                await self._close_socket()
                if backoff > 0:
                    await asyncio.sleep(backoff)


class OpenHandsRestClient:
    """Thin async client over the agent-server REST API and event socket."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        max_stream_disconnect_retries: int = DEFAULT_MAX_STREAM_DISCONNECT_RETRIES,
        stream_retry_backoff_seconds: float = DEFAULT_STREAM_RETRY_BACKOFF_SECONDS,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
    ) -> None:
        if not base_url:
            raise ConfigurationError(
                "OpenHands agent-server base URL is required. Pass base_url or set "
                f"the {ENV_BASE_URL} environment variable.",
                parameter="base_url",
                value=base_url,
            )
        self.base_url = base_url.rstrip("/")
        if self.base_url.startswith("https://"):
            self.ws_base_url = "wss://" + self.base_url[len("https://") :]
        elif self.base_url.startswith("http://"):
            self.ws_base_url = "ws://" + self.base_url[len("http://") :]
        else:
            raise ConfigurationError(
                "OpenHands agent-server base URL must start with http:// or https://.",
                parameter="base_url",
                value=base_url,
            )
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_stream_disconnect_retries = max_stream_disconnect_retries
        self.stream_retry_backoff_seconds = stream_retry_backoff_seconds
        self.max_frame_bytes = max_frame_bytes
        self._http: httpx.AsyncClient | None = None

    # -- lifecycle ---------------------------------------------------------

    @property
    def auth_headers(self) -> dict[str, str]:
        """Headers carrying the server-level session API key."""
        return {SESSION_API_KEY_HEADER: self.api_key} if self.api_key else {}

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            httpx = _require_httpx()
            self._http = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.auth_headers,
                timeout=self.timeout_seconds,
            )
        return self._http

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        http, self._http = self._http, None
        if http is not None:
            await http.aclose()

    # -- REST --------------------------------------------------------------

    def _raise_for_status(self, exc: httpx.HTTPStatusError) -> None:
        status = exc.response.status_code
        try:
            payload = exc.response.json()
            detail = str(payload.get("detail", payload))[:300] if isinstance(payload, dict) else str(payload)[:300]
        except Exception:
            detail = exc.response.text[:300]
        if status in (401, 403):
            raise ConfigurationError(
                f"Agent-server rejected the API key (HTTP {status}): {detail}",
                parameter=SESSION_API_KEY_HEADER,
                value=None,
            ) from exc
        if status == 409:
            raise BackendConflictError(f"Agent-server conflict (HTTP 409): {detail}") from exc
        if status >= 500:
            raise BackendTransportError(f"Agent-server error (HTTP {status}): {detail}") from exc
        raise BackendError(
            f"Agent-server request failed (HTTP {status}): {detail}",
            backend="openhands-rest",
        ) from exc

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        httpx = _require_httpx()
        client = await self._client()
        try:
            response = await client.request(method, path, **kwargs)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            self._raise_for_status(e)
            raise AssertionError("unreachable") from e  # pragma: no cover
        except httpx.HTTPError as e:
            raise BackendTransportError(f"Agent-server request failed: {e}", cause=e) from e
        return response

    async def create_conversation(
        self,
        *,
        cwd: str,
        model: str,
        llm_api_key: str | None,
        llm_base_url: str | None,
        system_prompt: str,
        max_iterations: int = DEFAULT_CONVERSATION_MAX_ITERATIONS,
    ) -> dict[str, Any]:
        """Create a conversation on the server and return its info payload.

        The LLM credentials belong to the *server*: it runs the agent loop, so
        it needs the model key, while this process only needs the server key.
        """
        llm: dict[str, Any] = {"model": model, "usage_id": "agenter"}
        if llm_api_key:
            llm["api_key"] = llm_api_key
        if llm_base_url:
            llm["base_url"] = llm_base_url
        payload = {
            "agent": {
                "kind": "Agent",
                "llm": llm,
                "system_prompt": system_prompt,
                "tools": list(DEFAULT_SERVER_TOOLS),
            },
            "workspace": {"kind": "LocalWorkspace", "working_dir": cwd},
            "confirmation_policy": {"kind": "NeverConfirm"},
            "max_iterations": max_iterations,
        }
        response = await self._request("POST", PATH_CONVERSATIONS, json=payload)
        data = response.json()
        if not isinstance(data, dict) or not data.get("id"):
            raise BackendTransportError(f"Agent-server returned an unexpected conversation payload: {str(data)[:200]}")
        return data

    async def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        """Fetch conversation info (execution status and cumulative metrics)."""
        response = await self._request("GET", f"{PATH_CONVERSATIONS}/{quote(conversation_id)}")
        data = response.json()
        if not isinstance(data, dict):
            raise BackendTransportError("Agent-server returned an unexpected conversation payload")
        return data

    async def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation (and its persisted event log) on the server."""
        await self._request("DELETE", f"{PATH_CONVERSATIONS}/{quote(conversation_id)}")

    async def send_message(self, conversation_id: str, text: str, *, run: bool = True) -> None:
        """Append a user message and, by default, start a run."""
        payload: dict[str, Any] = {
            "role": "user",
            "content": [{"type": "text", "text": text}],
            "run": run,
        }
        await self._request("POST", f"{PATH_CONVERSATIONS}/{quote(conversation_id)}/events", json=payload)

    async def interrupt(self, conversation_id: str) -> None:
        """Ask the server to stop the in-flight turn."""
        await self._request("POST", f"{PATH_CONVERSATIONS}/{quote(conversation_id)}/interrupt")

    async def agent_final_response(self, conversation_id: str) -> str:
        """Return the agent's final response text (empty before it produces one)."""
        response = await self._request("GET", f"{PATH_CONVERSATIONS}/{quote(conversation_id)}/agent_final_response")
        data = response.json()
        if isinstance(data, dict) and isinstance(data.get("response"), str):
            return data["response"]
        return ""

    async def git_changes(self, conversation_id: str, path: str) -> list[dict[str, Any]]:
        """Return the workspace's git change list (empty for non-repo paths)."""
        response = await self._request(
            "GET",
            f"{PATH_CONVERSATIONS}/{quote(conversation_id)}/git/changes",
            params={"path": path},
        )
        data = response.json()
        if not isinstance(data, list):
            raise BackendTransportError("Agent-server returned an unexpected git change payload")
        return [item for item in data if isinstance(item, dict)]

    async def download_file(self, conversation_id: str, path: str) -> bytes:
        """Download one file from the conversation workspace."""
        response = await self._request(
            "GET",
            f"{PATH_CONVERSATIONS}/{quote(conversation_id)}/file/download",
            params={"path": path},
        )
        return response.content

    # -- WebSocket ---------------------------------------------------------

    def open_event_stream(self, conversation_id: str, *, after_seq: int | None = None) -> EventStream:
        """Open the durable event stream for a conversation.

        ``after_seq`` follows the server protocol: omitted means live-only, ``-1``
        replays the whole log, and ``N`` replays everything after seq N. The
        returned stream reconnects transparently from its own cursor.
        """
        return EventStream(self, conversation_id, after_seq=after_seq)


def is_terminal_status(status: str | None) -> bool:
    """Whether an ``execution_status`` value ends a run."""
    return status in TERMINAL_EXECUTION_STATUSES
