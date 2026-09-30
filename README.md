# Agenter

<div align="center">

**Decode your intent into working code.**

*The backend-agnostic SDK for orchestrating autonomous AI coding agents.*

[![CI](https://github.com/3ive-ai/agenter/actions/workflows/ci.yml/badge.svg)](https://github.com/3ive-ai/agenter/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/agenter/badge/?version=latest)](https://agenter.readthedocs.io)
[![PyPI version](https://img.shields.io/pypi/v/agenter)](https://pypi.org/project/agenter/)
[![Python versions](https://img.shields.io/pypi/pyversions/agenter)](https://pypi.org/project/agenter/)
[![License](https://img.shields.io/github/license/3ive-ai/agenter)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked](https://img.shields.io/badge/mypy-checked-blue)](http://mypy-lang.org/)

[Documentation](https://agenter.readthedocs.io) • [Architecture](https://github.com/3ive-ai/agenter/blob/main/docs/ARCHITECTURE.md)

</div>

---

## 🚀 Why Agenter?

**The missing abstraction layer for embedding coding agents in your software.**

Other code agent wrapper SDKs let you swap the underlying **LLM** backend (e.g. Claude vs. GPT-4) - Agenter lets you swap the **entire Agent Runtime** (e.g. Claude Code vs. Codex vs. OpenHands or custom agents).

Why does this matter? Because the LLM "brain" is only half the battle. The "body" (tools, loops, environment, workflows, and graphs) is where the real engineering differentiation lives. While Anthropic has optimized Claude Code for enterprise workflows and OpenAI has fine-tuned Codex for code generation, **Agenter** does not tie you to any vendor.

**Agenter** is a universal interface for these specialized agent runtimes. It solves the **"N x M" integration problem** between your apps and a fragmented agent landscape. **Agenter** offers

1.  **Runtime Agnosticism**: Don't lock your platform into one vendor's agent architecture and its arbitrary conventions. Switch from a custom loop to **Claude Code** (or future runtimes) with a simple config change.
2.  **Unified Governance**: Apply the same **budget limits** (cost or tokens) and **safety policies** across *any* underlying coding agent regardless of how it executes code.
3.  **Workflow Portability**: Write your **LangGraph** or **PydanticAI** logic once. Run it on the best coding engine available today or tomorrow.

At the core of **Agenter** is `AutonomousCodingAgent` — a facade that handles all heavy backend selection, tool execution, validation loops, budget enforcement, event streaming, and other concerns. You give it a prompt and a directory, and it writes working code for you.

## 🏗️ Architecture

The SDK is built on a robust `CodingBackend` abstraction protocol that decouples agent logic from underlying backend providers.

```text
┌─────────────────────────────────────────────────────────────┐
│                        User Applications                    │
├─────────────────────────────┬───────────────────────────────┤
│      LangGraph Adapter      │      PydanticAI Adapter       │
└─────────────────────────────┴───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               AutonomousCodingAgent (Facade)                │
│       (Unified API, Configuration, Event Streaming)         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       CodingSession                         │
│   (Iteration Loop, Budget Enforcement, Error Recovery)      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                CodingBackend Protocol (Abstract)                                  │
├───────────────────────┬───────────────────────┬─────────────────────┬─────────────────┬───────────┤
│  AnthropicSDKBackend  │   ClaudeCodeBackend   │    CodexBackend     │ OpenHandsBackend│ ACPBackend│
│  (Custom Tool Loop)   │  (Claude Code SDK)    │   (OpenAI Models)   │  (Any Model)    │ (ACP CLI) │
└───────────────────────┴───────────────────────┴─────────────────────┴─────────────────┴───────────┘
```

## 📦 Installation

```bash
# Default installation (anthropic-sdk backend with Anthropic API or AWS Bedrock)
pip install agenter

# Installation with Claude Code support (claude-code-sdk)
pip install agenter[claude-code]

# Installation with Codex support (OpenAI models)
pip install agenter[codex]

# Installation with OpenHands SDK support (any model via litellm, in-process)
pip install agenter[openhands]

# Installation with OpenHands agent-server support (remote, over HTTP/WebSocket)
pip install agenter[openhands-rest]

# Installation with ACP support (Agent Client Protocol agents)
pip install agenter[acp]

# Installation with specific framework adapters
pip install agenter[langgraph,pydantic-ai]

# Installation with security scanning (Bandit)
pip install agenter[security]
```

## 📋 Backend Requirements

| Backend | Type | Requirements |
|---------|------|--------------|
| **anthropic-sdk** (default) | Pure library | `ANTHROPIC_API_KEY` or AWS credentials |
| **claude-code** | External CLI | [Claude Code CLI](https://github.com/anthropics/claude-code) installed |
| **codex** | External CLI | [Codex CLI](https://github.com/openai/codex) installed, `OPENAI_API_KEY` |
| **openhands** | In-process library | `openhands-sdk` (filesystem access, no isolation) |
| **openhands-rest** | External service | A running OpenHands agent-server (`agenter[openhands-rest]`), its URL and API key |
| **acp** | External CLI | Any ACP-compatible agent command, `agent-client-protocol` Python package |

The default `anthropic-sdk` backend works as a pure Python library - just set your API key and go.
Other backends wrap external CLIs/services and require additional setup.

## ⚡ Quick Start

The SDK manages the core coding loop for you. Just provide a prompt and a directory, and you are good to go.

```python
import asyncio
from agenter import AutonomousCodingAgent, CodingRequest, Verbosity

async def main():
    # 1. Initialize the coding agent (defaults to anthropic-sdk backend)
    agent = AutonomousCodingAgent(
        model="claude-sonnet-4-20250514",
    )

    # 2. Execute a task with full observability
    print("🤖 Coding agent starting...")
    result = await agent.execute(
        CodingRequest(
            prompt="Create a FastAPI app with a health check endpoint and a test file for it.",
            cwd="./workspace",
            allowed_write_paths=["*.py"]  # Security: Only allow Python files to be written
        ),
        verbosity=Verbosity.NORMAL  # Shows progress and tool calls in real time
    )

    # 3. Check results
    if result.status == "completed":
        print(f"✅ Success! Modified {len(result.files)} files.")
        print(f"💰 Cost: ${result.total_cost_usd:.4f}")
    else:
        print(f"❌ Failed: {result.summary}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🛡️ Unified Sandbox Mode

Sandboxing prevents the coding agent from modifying files outside the working directory, protecting your system against unintended changes.

All of the following coding agent backends default to `sandbox=True` for safe operation.

### Sandbox Enabled (Default)
```python
# All backends run in safe mode by default
agent = AutonomousCodingAgent(backend="anthropic-sdk")  # PathResolver enforces cwd
agent = AutonomousCodingAgent(
    backend="claude-code",
    model="claude-sonnet-4-5-20250929",
    claude_max_thinking_tokens=8192,
)  # Native OS-level sandbox
agent = AutonomousCodingAgent(
    backend="codex",
    model="gpt-5.4",
    codex_reasoning_effort="high",
)  # Workspace-write mode
agent = AutonomousCodingAgent(
    backend="acp",
    acp_command="codex-acp",
    acp_args=["-c", 'model="gpt-5.4"', "-c", 'model_reasoning_effort="high"'],
    acp_permission_policy="allow",
)
```

ACP prompts run with Agenter's autonomous backend contract by default, so
interactive agents do not pause for routine implementation confirmation. Set
`acp_autonomous=False` if you want the raw ACP agent behavior.

### OpenHands Agent Server (Remote Execution)

The `openhands-rest` backend drives an already-running OpenHands agent-server
over HTTP and WebSocket instead of linking the SDK into your process. The server
owns the agent loop, the model credentials and the workspace, so Agenter never
touches the filesystem the agent works in:

```python
# 1. Start an agent-server (this also creates its session API key):
#    pip install openhands-agent-server && agent-server --host 0.0.0.0 --port 18000
#    OH_SESSION_API_KEYS_0=<key> agent-server --host 0.0.0.0 --port 18000
agent = AutonomousCodingAgent(
    backend="openhands-rest",
    model="anthropic/claude-sonnet-4-5-20250929",
    openhands_rest_base_url="http://127.0.0.1:18000",  # or ACA_OPENHANDS_REST_BASE_URL
    openhands_rest_api_key="<server session key>",     # or ACA_OPENHANDS_REST_API_KEY
    openhands_rest_llm_api_key="<model key>",          # or ACA_OPENHANDS_REST_LLM_API_KEY
)
result = await agent.execute(CodingRequest(prompt="Fix the failing tests", cwd="/srv/project"))
```

`cwd` is a path on the **server**, not on the client: the conversation workspace
lives there. Modified file contents are read back over HTTP, so validation works
even when the client cannot see that filesystem.

Key options:

- `openhands_rest_llm_base_url` — custom model endpoint for the server (e.g. a
  local vLLM or an Anthropic-compatible gateway).
- `openhands_rest_max_iterations` — cap on agent steps per run, enforced by the
  server (default 500). This is not Agenter's retry budget.
- `openhands_rest_delete_on_disconnect` — delete the conversation afterwards.
  Off by default, so the conversation stays inspectable on the server.
- Custom tools (`tools=[...]`) are not supported yet: remote tool injection
  needs the server's `client_tools` protocol.
- The refusal tool is not advertised either, so `refusal()` is normally `None`.
  A Refusal-shaped tool call is still captured if the server exposes one.

### Persistent OpenHands Follow-ups

`open_session()` also supports `openhands-rest`. The session owns one
conversation, whose server-side event log is the agent's memory:

```python
agent = AutonomousCodingAgent(backend="openhands-rest", openhands_rest_base_url="http://127.0.0.1:18000")
session = await agent.open_session(cwd="/srv/project", resume_session_id=None)
try:
    first = await session.execute("Add a CLI flag for dry runs.")
    second = await session.execute("Now cover it with a test.")   # same conversation
    print(session.session_id)
finally:
    await session.close()   # the conversation stays on the server
```

### Persistent ACP Follow-ups

Use `open_session()` when an ACP coding agent should retain its own conversation
state across multiple prompts instead of receiving one large generated program:

```python
from agenter import AutonomousCodingAgent, Budget

agent = AutonomousCodingAgent(
    backend="acp",
    acp_command="codex-acp",
    acp_permission_policy="allow",
)
session = await agent.open_session(
    cwd="./workspace",
    session_budget=Budget(max_iterations=40, max_tokens=200_000),
)
try:
    first = await session.execute("Inspect the GLB and implement the extraction script.")
    followup = await session.execute("Run it, inspect the render, and fix the wall contacts.")
    print(session.session_id, followup.request_index, session.modified_files().paths())
finally:
    await session.close()
```

Follow-ups are serialized onto one ACP subprocess and `session/prompt` session.
Each result reports request-local files and usage plus stable `session_id`,
`request_index`, and cumulative session totals. `session.modified_files()` and
`session.usage()` expose cumulative state. Active prompts can be cancelled with
`session.cancel()` without discarding the session. Pass `resume_session_id` to
`open_session()` when the ACP agent advertises `session/resume` or legacy
`session/load` support. Check `result.usage_reported` (or
`session.usage().reported`) because some ACP adapters do not publish token usage.

| Backend | SDK | Providers | Sandbox |
|---------|-----|-----------|---------|
| **anthropic-sdk** | Anthropic SDK with custom tool loop | Anthropic API, AWS Bedrock | `PathResolver` path isolation |
| **claude-code** | Claude Code SDK (claude-code-sdk) | Anthropic API, AWS Bedrock, Google Vertex | Native OS-level sandbox |
| **codex** | OpenAI Codex CLI via MCP | OpenAI API | `workspace-write` mode |
| **openhands** | OpenHands SDK | Any provider via LiteLLM | ⚠️ **No sandbox** |
| **openhands-rest** | OpenHands agent-server (REST + WebSocket) | Any provider via LiteLLM, resolved server-side | ✅ Server-enforced workspace |
| **acp** | Agent Client Protocol subprocess | ACP-compatible agents | Depends on the launched ACP agent |

> **⚠️ OpenHands Warning:** The `openhands` backend requires `sandbox=False` and has **no filesystem isolation**. It can read and write anywhere on your system. Use it with caution and only in trusted environments. The `openhands-rest` backend is the isolated alternative: the agent runs on a separate agent-server, which confines every file and git operation to the conversation workspace.
>
> **⚠️ ACP Warning:** The ACP backend launches an external agent process. Agenter can observe changed files after the run, but filesystem isolation depends on the ACP agent and its own flags.

### Sandbox Disabled (Full File System Access)
```python
# Disable sandbox for full file system access
agent = AutonomousCodingAgent(backend="claude-code", sandbox=False)
```

## 🧩 Agentic Framework Integrations

Don't reinvent the wheel. Use our pre-built adapters to supercharge your agentic graphs. 

Adapters **subclass framework base classes** to include LangSmith/Logfire tracing and all framework features automatically.

### LangGraph Adapter

`create_coding_node()` returns a `RunnableLambda` (includes LangSmith tracing).

```python
from agenter.adapters.langgraph import create_coding_node

node = create_coding_node(cwd="./workspace", backend="claude-code")
result = await node.ainvoke({"prompt": "Create a hello world script"})

print(result["coding_result"]["summary"])
```

### PydanticAI Adapter

`CodingAgent` subclasses `pydantic_ai.Agent` (includes Logfire tracing).

```python
from agenter.adapters.pydantic_ai import CodingAgent

agent = CodingAgent(cwd="./workspace", backend="codex")
result = await agent.run("Add input validation")

print(result.summary)
```

## 🎯 Structured Outputs

Get typed and fully validated outputs from your coding tasks using Pydantic models.

### Basic Usage

```python
from pydantic import BaseModel
from agenter import AutonomousCodingAgent, CodingRequest

class AnalysisResult(BaseModel):
    summary: str
    issues_found: list[str]
    confidence: float

agent = AutonomousCodingAgent()
result = await agent.execute(
    CodingRequest(
        prompt="Analyze the security of auth.py",
        cwd="./workspace",
        output_type=AnalysisResult,  # Enables structured outputs
    )
)

# result.output is typed as AnalysisResult
print(result.output.summary)
print(result.output.confidence)
```

## 🔍 Observability

The SDK provides rich, structured event streaming for UIs or logs.

```python
async for event in agent.stream_execute(request):
    if event.type == "backend_message":
        print(f"🤖 AI: {event.data['content']}")
    elif event.type == "validation_start":
        print("🔍 Validating code...")
    elif event.type == "iteration_end":
        print(f"🔄 Iteration {event.data['iteration']} finished.")
```

## 📚 Documentation

- [Quickstart Notebook](examples/quickstart.ipynb) — Get started with an interactive tutorial.
- [Architecture](docs/ARCHITECTURE.md) — Dive deeper into the SDK's layered design, backend protocol, event system, and functionality.
- [Naming Conventions](docs/NAMING_CONVENTIONS.md) — Supplementary material about coding style and naming patterns.

## 📜 License

Copyright 2025 Moonsong Labs

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
