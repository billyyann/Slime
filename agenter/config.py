"""Constants and defaults for the SDK."""

import os
from typing import Final

# Default models (can be overridden via environment variables)
# Model naming: claude-{capability}-{version}-{date}
# See: https://docs.anthropic.com/en/docs/about-claude/models
DEFAULT_MODEL_ANTHROPIC: Final = os.environ.get(
    "ACA_DEFAULT_MODEL",
    "claude-sonnet-4-20250514",
)
DEFAULT_MODEL_BEDROCK: Final = os.environ.get(
    "ACA_DEFAULT_MODEL_BEDROCK",
    "us.anthropic.claude-sonnet-4-20250514-v1:0",
)

# Limits
DEFAULT_MAX_ITERATIONS: Final = 5
DEFAULT_MAX_OUTPUT_TOKENS: Final = 65536  # 64K - Claude Sonnet 4 max output

# AWS
DEFAULT_AWS_REGION: Final = "us-east-1"

# Backend choices
BACKEND_ANTHROPIC_SDK: Final = "anthropic-sdk"  # Anthropic SDK with custom tool loop
BACKEND_CLAUDE_CODE: Final = "claude-code"  # Claude Code SDK (claude-agent-sdk)
BACKEND_CODEX: Final = "codex"  # OpenAI Codex CLI via MCP server
BACKEND_OPENHANDS: Final = "openhands"  # OpenHands SDK
BACKEND_OPENHANDS_REST: Final = "openhands-rest"  # OpenHands agent-server over REST/WebSocket
BACKEND_ACP: Final = "acp"  # Agent Client Protocol subprocess backend


def default_model() -> str:
    """Return appropriate default model based on environment."""
    if os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        return DEFAULT_MODEL_BEDROCK
    return DEFAULT_MODEL_ANTHROPIC


def is_bedrock() -> bool:
    """Check if Bedrock should be used."""
    return bool(os.environ.get("AWS_BEARER_TOKEN_BEDROCK"))


# Valid backends for spec parsing
VALID_BACKENDS: Final = frozenset(
    {
        BACKEND_ANTHROPIC_SDK,
        BACKEND_CLAUDE_CODE,
        BACKEND_CODEX,
        BACKEND_OPENHANDS,
        BACKEND_OPENHANDS_REST,
        BACKEND_ACP,
    }
)


def default_backend() -> str:
    """Return the default backend, configurable via ACA_DEFAULT_BACKEND env var."""
    env = os.environ.get("ACA_DEFAULT_BACKEND", BACKEND_ANTHROPIC_SDK)
    if env not in VALID_BACKENDS:
        from .data_models import ConfigurationError

        raise ConfigurationError(
            f"Invalid ACA_DEFAULT_BACKEND={env!r}. Must be one of: {', '.join(sorted(VALID_BACKENDS))}",
            parameter="ACA_DEFAULT_BACKEND",
            value=env,
        )
    return env


def parse_backend_spec(spec: str) -> tuple[str, str | None, str | None]:
    """Parse 'agenter:backend:model' notation.

    Args:
        spec: Specifier like 'agenter', 'agenter:codex', 'agenter:codex:o4-mini'

    Returns:
        Tuple of (generator_name, backend, model) where backend/model are None if not specified

    Raises:
        ValueError: If backend is unknown

    Examples:
        >>> parse_backend_spec('agenter')
        ('agenter', None, None)
        >>> parse_backend_spec('agenter:anthropic-sdk')
        ('agenter', 'anthropic-sdk', None)
        >>> parse_backend_spec('agenter:codex:o4-mini')
        ('agenter', 'codex', 'o4-mini')
    """
    parts = spec.split(":")
    generator = parts[0]
    backend = parts[1] if len(parts) > 1 else None
    model = parts[2] if len(parts) > 2 else None

    if backend and backend not in VALID_BACKENDS:
        raise ValueError(f"Unknown backend: {backend}. Must be one of: {', '.join(sorted(VALID_BACKENDS))}")
    return generator, backend, model
