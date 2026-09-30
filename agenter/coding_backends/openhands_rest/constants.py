"""Constants for the OpenHands REST backend.

These map to the agent-server's REST/WebSocket surface (``openhands.agent_server``)
and to the tool names it registers at startup. When the agent-server updates,
these may need to be reviewed.

The server side of this contract was verified against openhands-agent-server
1.49.x; see ``client.py`` for the endpoint list.
"""

from typing import Final

# Default model in litellm format (the server routes models through litellm,
# so the same "provider/model" notation as the SDK backend applies).
DEFAULT_MODEL_OPENHANDS_REST: Final[str] = "openai/gpt-4o"

# Environment variables. Constructor arguments take precedence over these.
ENV_BASE_URL: Final[str] = "ACA_OPENHANDS_REST_BASE_URL"
ENV_API_KEY: Final[str] = "ACA_OPENHANDS_REST_API_KEY"
ENV_LLM_API_KEY: Final[str] = "ACA_OPENHANDS_REST_LLM_API_KEY"
ENV_LLM_BASE_URL: Final[str] = "ACA_OPENHANDS_REST_LLM_BASE_URL"
ENV_MODEL: Final[str] = "ACA_OPENHANDS_REST_MODEL"

# Header carrying the server-level session API key on REST calls.
SESSION_API_KEY_HEADER: Final[str] = "X-Session-API-Key"

# First frame sent on an event socket when authenticating after connect.
# The server closes with 4001 if it is missing or wrong.
WS_AUTH_FRAME_TYPE: Final[str] = "auth"

# Default server-side cap on agent steps within one run. This is the
# agent-server's own default; it bounds a single execute() call, not the
# Agenter retry loop (which is governed by Budget.max_iterations).
DEFAULT_CONVERSATION_MAX_ITERATIONS: Final[int] = 500

# Connection / stream policy.
DEFAULT_REQUEST_TIMEOUT_SECONDS: Final[float] = 60.0
DEFAULT_MAX_STREAM_DISCONNECT_RETRIES: Final[int] = 3
DEFAULT_STREAM_RETRY_BACKOFF_SECONDS: Final[float] = 1.0
# Bounded verification polls used to confirm the terminal status after the
# event stream ends (the stream is not the only source of truth).
DEFAULT_STATUS_POLL_ATTEMPTS: Final[int] = 3
DEFAULT_STATUS_POLL_DELAY_SECONDS: Final[float] = 1.0
# After a terminal status frame, the frames explaining the outcome (a
# ConversationErrorEvent, the final messages) arrive moments later. How long a
# quiet socket waits before the tail is considered complete, and how many tail
# frames are accepted at most.
DEFAULT_GRACE_DRAIN_SECONDS: Final[float] = 2.0
DEFAULT_GRACE_DRAIN_MAX_FRAMES: Final[int] = 50
# ``after_seq`` cursor value that asks the server to replay the whole log
# before going live. Omitting the parameter entirely means live-only.
STREAM_FULL_HISTORY: Final[int] = -1

# Session-socket frame types (openhands/agent_server/session_protocol.py).
FRAME_SYNC: Final[str] = "sync"
FRAME_DURABLE: Final[str] = "durable"
FRAME_TRANSIENT: Final[str] = "transient"
FRAME_ITEM_STARTED: Final[str] = "item_started"
FRAME_DELTA: Final[str] = "delta"
FRAME_ITEM_ABORTED: Final[str] = "item_aborted"
FRAME_ERROR: Final[str] = "error"

# Conversational event kinds (the ``kind`` discriminator on events).
KIND_MESSAGE_EVENT: Final[str] = "MessageEvent"
KIND_ACTION_EVENT: Final[str] = "ActionEvent"
KIND_OBSERVATION_EVENT: Final[str] = "ObservationEvent"
KIND_AGENT_ERROR_EVENT: Final[str] = "AgentErrorEvent"
KIND_USER_REJECT_OBSERVATION: Final[str] = "UserRejectObservation"
KIND_STATE_UPDATE_EVENT: Final[str] = "ConversationStateUpdateEvent"
KIND_CONVERSATION_ERROR_EVENT: Final[str] = "ConversationErrorEvent"
KIND_SERVER_ERROR_EVENT: Final[str] = "ServerErrorEvent"

# Event sources we care about.
SOURCE_AGENT: Final[str] = "agent"
SOURCE_USER: Final[str] = "user"
SOURCE_ENVIRONMENT: Final[str] = "environment"

# The ``usage_id`` this backend registers on the LLM it asks the server to use.
# The server keys per-usage metrics by it (ConversationInfo.stats.usage_to_metrics),
# and reading our own id keeps other LLMs of the conversation out of the total.
LLM_USAGE_ID: Final[str] = "agenter"

# Where cumulative usage can appear on ConversationInfo. The ``metrics`` field is
# frequently null; ``stats.usage_to_metrics`` is the populated one.
PATH_STATS_USAGE_TO_METRICS: Final[tuple[str, str]] = ("stats", "usage_to_metrics")
PATH_METRICS: Final[str] = "metrics"

# Execution statuses that end a run (openhands ConversationExecutionStatus).
TERMINAL_EXECUTION_STATUSES: Final[frozenset[str]] = frozenset({"finished", "error", "stuck"})
STATUS_RUNNING: Final[str] = "running"
# ``full_state`` key on state-update events carries the whole conversation state.
STATE_KEY_FULL_STATE: Final[str] = "full_state"
STATE_KEY_EXECUTION_STATUS: Final[str] = "execution_status"

# WebSocket close codes used by the agent-server.
WS_CLOSE_AUTH_FAILED: Final[int] = 4001
WS_CLOSE_CONVERSATION_NOT_FOUND: Final[int] = 4004
WS_CLOSE_OVERLOADED: Final[int] = 1013

# Default tools requested from the server. Tool names are resolved from the
# server's registry (``register_default_tools`` at agent-server startup).
DEFAULT_SERVER_TOOLS: Final[tuple[dict[str, object], ...]] = (
    {"name": "terminal", "params": {}},
    {"name": "file_editor", "params": {}},
    {"name": "task_tracker", "params": {}},
)

# Tool names that write to the workspace. Used to track touched paths from
# ActionEvents; this is a fallback for workspaces that are not git repositories,
# because shell commands (``terminal``) can also modify files without naming them.
FILE_MODIFICATION_TOOLS: Final[frozenset[str]] = frozenset(
    {
        "file_editor",  # openhands.tools.file_editor
        "str_replace_editor",  # legacy name
        "create_file",  # legacy name
        "write_file",  # legacy name
    }
)

# Path argument keys used by file-writing tools (file_editor uses ``path``).
PATH_INPUT_KEYS: Final[tuple[str, ...]] = ("path", "file_path", "filename")

# REST paths (all served under the server root).
PATH_CONVERSATIONS: Final[str] = "/api/conversations"
PATH_SOCKETS_SESSION: Final[str] = "/sockets/session"

# System prompt for the REST backend. Unlike the SDK backend it does NOT append
# the Refusal-tool instructions: the tool set lives on the server, and v1 cannot
# register a Refusal tool there (that needs the server's client_tools protocol).
OPENHANDS_REST_PROMPT: Final[str] = """\
You are an autonomous coding agent. Your working directory is: {cwd}

You have a file editor, a terminal, and a task tracker at your disposal. Always:
1. Read existing files before modifying them
2. Make minimal, focused changes
3. Ensure code is syntactically correct
4. Prefer targeted edits over rewriting whole files

Work only inside the working directory shown above. \
Complete the task fully before stopping."""
