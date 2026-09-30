"""LangGraph adapter - create coding nodes for LangGraph workflows.

This adapter returns a RunnableLambda, giving you LangSmith tracing for free
and full compatibility with LangChain's Runnable interface (.batch(), .stream(), etc).

Example:
    from langgraph.graph import StateGraph
    from agenter.adapters.langgraph import create_coding_node, CodingState

    graph = StateGraph(CodingState)
    graph.add_node("coder", create_coding_node(cwd="./workspace"))

    # LangSmith tracing works automatically when LANGSMITH_API_KEY is set
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from ..data_models.types import Verbosity

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableLambda


class CodingState(TypedDict, total=False):
    """Minimal state for a coding node.

    Attributes:
        prompt: The coding task to perform.
        cwd: Working directory (optional, can be set at node creation).
        verbosity: Output verbosity level (optional, can be set at node creation).
        coding_result: Output field containing the result dict.
    """

    prompt: str
    cwd: str
    verbosity: Verbosity
    coding_result: dict[str, Any]


def create_coding_node(
    cwd: str | None = None,
    backend: str = "anthropic-sdk",
    verbosity: Verbosity = Verbosity.QUIET,
    **agent_kwargs: Any,
) -> RunnableLambda:
    """Create a LangGraph node that performs autonomous coding.

    Returns a RunnableLambda, which integrates with LangSmith tracing
    automatically and supports .batch(), .stream(), .ainvoke().

    Args:
        cwd: Working directory. Can be overridden by state["cwd"].
        backend: Backend to use ("anthropic-sdk", "claude-code", "codex", "openhands", "openhands-rest").
        verbosity: Output verbosity level.
        **agent_kwargs: Additional arguments passed to AutonomousCodingAgent.

    Returns:
        RunnableLambda compatible with StateGraph.add_node().

    Example:
        from langgraph.graph import StateGraph
        from agenter.adapters.langgraph import create_coding_node, CodingState

        graph = StateGraph(CodingState)
        graph.add_node("coder", create_coding_node(cwd="./workspace", verbosity=Verbosity.VERBOSE))

        # LangSmith tracing works automatically when LANGSMITH_API_KEY is set
        # The node reads state["prompt"] and writes to state["coding_result"]
    """
    from langchain_core.runnables import RunnableLambda

    from ..coding_agent import AutonomousCodingAgent
    from ..data_models import CodingRequest

    agent = AutonomousCodingAgent(backend=backend, **agent_kwargs)

    async def _run(state: dict[str, Any]) -> dict[str, Any]:
        """Execute a coding task based on state."""
        working_dir = state.get("cwd", cwd)
        if working_dir is None:
            raise ValueError("cwd must be provided either at node creation or in state")

        request = CodingRequest(
            prompt=state["prompt"],
            cwd=working_dir,
        )
        result = await agent.execute(request, verbosity=state.get("verbosity", verbosity))
        return {"coding_result": result.model_dump()}

    return RunnableLambda(_run, name="coding_agent")


__all__ = ["CodingState", "create_coding_node"]
