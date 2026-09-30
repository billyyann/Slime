"""OpenHands REST backend.

Drives a running OpenHands agent-server over HTTP + WebSocket, so the agent runs
in the server's workspace instead of this process.
"""

from .backend import OpenHandsRestBackend
from .client import BackendConflictError, BackendTransportError, EventStream, OpenHandsRestClient

__all__ = [
    "BackendConflictError",
    "BackendTransportError",
    "EventStream",
    "OpenHandsRestBackend",
    "OpenHandsRestClient",
]
