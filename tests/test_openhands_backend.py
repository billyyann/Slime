"""Tests for the OpenHands backend."""

from __future__ import annotations

import pytest

from agenter.coding_backends.openhands import OpenHandsBackend
from agenter.data_models import BackendError, ConfigurationError


class TestOpenHandsBackendFacade:
    """The facade must forward its configuration to the backend."""

    def test_facade_forwards_model(self) -> None:
        """Regression: the openhands branch used to drop ``model`` silently."""
        from agenter import AutonomousCodingAgent

        agent = AutonomousCodingAgent(backend="openhands", model="openai/gpt-4o-mini", sandbox=False)
        backend = agent._create_backend()

        assert isinstance(backend, OpenHandsBackend)
        assert backend.model == "openai/gpt-4o-mini"

    def test_facade_keeps_backend_default_without_model(self) -> None:
        from agenter import AutonomousCodingAgent

        agent = AutonomousCodingAgent(backend="openhands", sandbox=False)
        assert agent._create_backend().model == "openai/gpt-4o"


class TestOpenHandsBackend:
    """Behavior tests for OpenHandsBackend."""

    def test_requires_sandbox_false(self) -> None:
        """Backend requires sandbox=False (no sandboxing support)."""
        with pytest.raises(ConfigurationError, match="sandbox"):
            OpenHandsBackend()

    def test_accepts_sandbox_false(self) -> None:
        """Backend can be created with sandbox=False."""
        backend = OpenHandsBackend(sandbox=False)
        assert backend.model == "openai/gpt-4o"

    def test_custom_model(self) -> None:
        """Backend accepts custom model in litellm format."""
        backend = OpenHandsBackend(sandbox=False, model="openai/gpt-4")
        assert backend.model == "openai/gpt-4"

    @pytest.mark.anyio
    async def test_execute_requires_connect(self) -> None:
        """Execute raises error if not connected."""
        backend = OpenHandsBackend(sandbox=False)
        # Without calling connect(), execute should fail with "not connected"
        # or "openhands-sdk is required" if SDK not installed
        with pytest.raises(BackendError, match=r"not connected|openhands-sdk is required"):
            async for _ in backend.execute("test"):
                pass

    @pytest.mark.anyio
    async def test_disconnect_resets_state(self) -> None:
        """Disconnect resets all backend state."""
        backend = OpenHandsBackend(sandbox=False)
        await backend.connect("/tmp")

        # Set some state
        backend._files_modified = ["test.py"]
        backend._input_tokens = 100
        backend._output_tokens = 50
        backend._cost_usd = 0.01

        await backend.disconnect()

        assert backend.usage().input_tokens == 0
        assert backend.usage().output_tokens == 0
        assert backend.usage().cost_usd == 0.0
        assert backend.modified_files().file_paths == []
        assert backend.structured_output() is None
        assert backend.cwd is None

    @pytest.mark.anyio
    async def test_modified_files_returns_paths_only(self) -> None:
        """modified_files returns PathsModifiedFiles (no content)."""
        backend = OpenHandsBackend(sandbox=False)
        await backend.connect("/tmp")

        # Simulate file modifications
        backend._files_modified = ["test.py", "utils.py"]

        result = backend.modified_files()
        assert result.paths_only is True
        assert result.file_paths == ["test.py", "utils.py"]
        assert result.content("test.py") is None  # No content in paths_only mode

    @pytest.mark.anyio
    async def test_usage_returns_tracked_metrics(self) -> None:
        """usage returns tracked token counts and cost."""
        backend = OpenHandsBackend(sandbox=False)
        await backend.connect("/tmp")

        # Simulate usage tracking
        backend._input_tokens = 1000
        backend._output_tokens = 500
        backend._cost_usd = 0.05

        usage = backend.usage()
        assert usage.input_tokens == 1000
        assert usage.output_tokens == 500
        assert usage.cost_usd == 0.05
        assert usage.model == "openai/gpt-4o"
        assert usage.provider == "openhands"

    @pytest.mark.anyio
    async def test_connect_logs_warning_for_allowed_write_paths(self) -> None:
        """Connect logs warning when allowed_write_paths is set (not enforced)."""
        backend = OpenHandsBackend(sandbox=False)

        # Should not raise, just log warning
        await backend.connect("/tmp", allowed_write_paths=["*.py"])

        # Verify connection succeeded
        assert backend.cwd is not None

    @pytest.mark.anyio
    async def test_connect_logs_warning_for_resume_session_id(self) -> None:
        """Connect logs warning when resume_session_id is set (not supported)."""
        backend = OpenHandsBackend(sandbox=False)

        # Should not raise, just log warning
        await backend.connect("/tmp", resume_session_id="test-session")

        # Verify connection succeeded
        assert backend.cwd is not None
