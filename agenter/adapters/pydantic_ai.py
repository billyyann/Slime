"""PydanticAI adapter - CodingAgent that subclasses pydantic_ai.Agent.

CodingAgent IS-A pydantic_ai.Agent, providing type compatibility with
pydantic-ai workflows while delegating execution to AutonomousCodingAgent.

Example:
    from agenter.adapters.pydantic_ai import CodingAgent

    agent = CodingAgent(cwd="./workspace")
    result = await agent.run("Add input validation to the form")
    print(result.summary)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Conditional import: subclass Agent if pydantic-ai is installed
try:
    from pydantic_ai import Agent as _BaseAgent
except ImportError:
    _BaseAgent = object  # type: ignore[misc, assignment]

if TYPE_CHECKING:
    from ..data_models import CodingResult


class CodingAgent(_BaseAgent):  # type: ignore[type-arg, misc]
    """PydanticAI Agent that performs autonomous coding.

    Subclasses pydantic_ai.Agent. Overrides .run() to call
    AutonomousCodingAgent.execute() directly - no extra LLM layer.

    Args:
        cwd: Working directory for code operations.
        backend: Agenter backend ("anthropic-sdk", "claude-code", "codex", "openhands", "openhands-rest").
        **agent_kwargs: Additional arguments passed to AutonomousCodingAgent.

    Example:
        agent = CodingAgent(cwd="./workspace")
        result = await agent.run("Refactor the auth module")
        print(result.summary)
    """

    def __init__(
        self,
        cwd: str,
        backend: str = "anthropic-sdk",
        **agent_kwargs: Any,
    ) -> None:
        # Initialize parent if pydantic-ai is installed
        if _BaseAgent is not object:
            super().__init__(model=None)

        from ..coding_agent import AutonomousCodingAgent

        self._cwd = cwd
        self._coding_agent = AutonomousCodingAgent(backend=backend, **agent_kwargs)

    async def run(self, prompt: str, **kwargs: Any) -> CodingResult:  # type: ignore[override]
        """Execute coding task directly via AutonomousCodingAgent.

        Overrides Agent.run() - no LLM-calls-tool indirection.

        Args:
            prompt: The coding task to perform.
            **kwargs: Additional arguments passed to CodingRequest.

        Returns:
            CodingResult with status, files modified, and summary.
        """
        from ..data_models import CodingRequest

        request = CodingRequest(prompt=prompt, cwd=self._cwd, **kwargs)
        return await self._coding_agent.execute(request)

    def run_sync(self, prompt: str, **kwargs: Any) -> CodingResult:  # type: ignore[override]
        """Synchronous version of run().

        Args:
            prompt: The coding task to perform.
            **kwargs: Additional arguments passed to CodingRequest.

        Returns:
            CodingResult with status, files modified, and summary.
        """
        import asyncio

        return asyncio.run(self.run(prompt, **kwargs))


def coding_tool(cwd: str, backend: str = "anthropic-sdk", **agent_kwargs: Any):
    """Create a coding tool for embedding in pydantic-ai agents.

    Use this when you want to add coding capability to an existing
    pydantic-ai Agent as a tool.

    Args:
        cwd: Working directory for code operations.
        backend: Agenter backend to use.
        **agent_kwargs: Additional arguments passed to AutonomousCodingAgent.

    Returns:
        Async function suitable for use as a pydantic-ai tool.

    Example:
        from pydantic_ai import Agent
        from agenter.adapters.pydantic_ai import coding_tool

        agent = Agent("anthropic:claude-sonnet-4-20250514")

        @agent.tool_plain
        async def write_code(task: str) -> str:
            '''Write or modify code autonomously.'''
            return await coding_tool("./workspace")(task)
    """
    from ..coding_agent import AutonomousCodingAgent
    from ..data_models import CodingRequest

    coding_agent = AutonomousCodingAgent(backend=backend, **agent_kwargs)

    async def _write_code(task: str) -> str:
        """Write or modify code autonomously."""
        request = CodingRequest(prompt=task, cwd=cwd)
        result = await coding_agent.execute(request)
        return f"{result.status.value}: {result.summary}"

    return _write_code


__all__ = ["CodingAgent", "coding_tool"]
