# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `OpenHandsRestBackend` (`backend="openhands-rest"`): drives an already-running OpenHands agent-server over its REST API and session event socket, so the agent loop, the model credentials and the workspace all live on the server. Modified file contents are read back over HTTP, which keeps validation working when the client cannot see the server's filesystem. Configure it with `openhands_rest_base_url`, `openhands_rest_api_key`, `openhands_rest_llm_api_key`, `openhands_rest_llm_base_url`, `openhands_rest_max_iterations`, and `openhands_rest_delete_on_disconnect` (each falling back to an `ACA_OPENHANDS_REST_*` environment variable).
- `open_session()` now also supports `openhands-rest`: the persistent session owns one agent-server conversation, whose event log carries the agent's memory across follow-ups. `resume_session_id` resumes an existing conversation and treats its accumulated usage as the baseline.
- New optional dependency group `agenter[openhands-rest]` (`httpx`, `websockets`).

### Fixed
- `AutonomousCodingAgent(backend="openhands", model=...)` no longer silently ignores `model` (it was dropped when constructing `OpenHandsBackend`).

## [0.1.7] - 2026-08-18

### Added
- `ACPBackend` now transparently retries a turn that aborts with a *retryable* transport fault (provider stream disconnect, RPC error), re-sending the same prompt on the warm session before surfacing `CodingStatus.ERROR`. Bounded by the new `max_stream_disconnect_retries` (default 2) and `stream_retry_backoff_seconds` (default 1.0, linear) constructor arguments; set retries to 0 to restore fail-fast behavior. Non-retryable outcomes (refusal, cancellation, max_tokens) are never retried, and an exhausted retry still reports an honest error rather than a silent success.
- `AutonomousCodingAgent` exposes the retry policy via `acp_max_stream_disconnect_retries` and `acp_stream_retry_backoff_seconds` so callers can tune or disable it without constructing `ACPBackend` directly
- `CodingBackend.turn_error()` now has a `None`-returning default on `BaseBackend`, so non-ACP backends satisfy the protocol without change

## [0.1.6] - 2026-08-18

### Added
- `CodingStatus.ERROR` and `CodingEventType.ERROR` for turns that abort due to a transport/provider fault (stream disconnect, RPC error, cancellation, early stop) rather than a task-level failure
- `CodingResult.error`, `CodingResult.error_kind`, and `CodingResult.retryable` structured fields, plus a `TaskErrored` event, so callers can distinguish and react to transient turn failures
- `ACPBackend.turn_error()` side-channel (mirrors `refusal()`) and configurable `stream_abort_sentinels` constructor argument

### Fixed
- ACP backend no longer discards the `PromptResponse.stop_reason`; `refusal`, `cancelled`, and `max_tokens`/`max_turn_requests` stop reasons are now classified instead of being reported as success
- A codex-style mid-stream provider disconnect (emitted as ordinary `end_turn` assistant text such as `stream disconnected before completion: …`) with no tool calls and no file changes is now reported as `CodingStatus.ERROR` instead of `COMPLETED`
- ACP `prompt()` RPC failures (`RequestError`) are surfaced as a retryable turn error rather than crashing the session

## [0.1.5] - 2026-07-21

### Added
- Raw inbound and outbound ACP JSON-RPC stream observers for complete host-side protocol tracing
- Vendor-neutral ACP session metadata for `session/new`, `session/resume`, and `session/load`

### Changed
- Isolate stream-observer failures inside the ACP connection so observability cannot fail an agent session

## [0.1.4] - 2026-07-17

### Added
- Persistent ACP sessions through `AutonomousCodingAgent.open_session()` with serialized follow-ups
- Stable ACP session/request trace identifiers and per-request plus cumulative usage metadata
- Explicit usage availability so an ACP adapter's missing metrics are not mistaken for measured zero usage
- ACP `session/resume` support with legacy `session/load` fallback
- Active ACP prompt cancellation without discarding the persistent session
- Live ACP session updates through a host callback

### Changed
- Split backend connection lifetime from individual `CodingSession` requests while preserving one-shot `execute()` behavior
- Track ACP file changes separately for the latest turn, high-level request, and full session
- Require `agent-client-protocol` 0.10 or later so ACP transport frames are not arbitrarily truncated

### Fixed
- Type checking when the optional ACP dependency is not installed

## [0.1.3] - 2026-05-19

### Added
- Agent Client Protocol backend for ACP-compatible agent subprocesses

### Changed
- Pass model and thinking-token settings through to the Claude Code backend
- Support reasoning-effort configuration in the Codex backend

### Fixed
- Exclude bytecode caches from file-change results for path-only backends

## [0.1.2] - 2026-03-24

### Added
- Configurable default backend via `ACA_DEFAULT_BACKEND` environment variable
- OpenClaw skill integration (`integrations/openclaw/`) with CLI bridge for autonomous coding from any messaging channel
- Codex backend: filesystem diff to catch all modified files, not just event-parsed ones
- Codex backend: automatic `OPENAI_API_KEY` → codex auth sync on connect
- Codex backend: structured output support via prompt injection
- Codex backend: pre-flight check for codex CLI installation

### Fixed
- `AutonomousCodingAgent(backend=...)` now accepts `None` to use env var default instead of hardcoding `"anthropic-sdk"`

### Security
- Added warning about litellm 1.82.7–1.82.8 supply chain compromise in optional dependency comment

## [0.1.0] - 2025-01-22

### Added
- Initial release of Agenter SDK
- Core `Agent` abstraction with unified interface for coding agents
- Support for multiple backends:
  - Anthropic API (direct)
  - AWS Bedrock
  - Claude Code CLI
  - OpenAI Codex
- Budget controls with `BudgetConfig` for cost limits and token tracking
- Security validation framework with configurable validators
- Path validation to restrict file system access
- Streaming support with real-time output callbacks
- Framework adapters for PydanticAI and LangGraph integration
- Comprehensive type hints and Pydantic models
- Structured logging with structlog

[Unreleased]: https://github.com/3ive-ai/agenter/compare/v0.1.7...HEAD
[0.1.7]: https://github.com/3ive-ai/agenter/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/3ive-ai/agenter/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/3ive-ai/agenter/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/3ive-ai/agenter/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/3ive-ai/agenter/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/3ive-ai/agenter/compare/v0.1.0...v0.1.2
[0.1.0]: https://github.com/3ive-ai/agenter/releases/tag/v0.1.0
