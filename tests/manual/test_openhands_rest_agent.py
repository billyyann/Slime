#!/usr/bin/env python3
"""Manual end-to-end test for the openhands-rest backend against a real agent-server.

Not collected by pytest (see tests/manual/conftest.py); run it directly:

    # 1. Start an agent-server (its session key lives under ~/.openhands/agent-canvas):
    #    bash run_slime.sh --bg
    # 2. Point this script at it (GLM example):
    export ACA_OPENHANDS_REST_BASE_URL=http://127.0.0.1:18000
    export ACA_OPENHANDS_REST_API_KEY="$(cat ~/.openhands/agent-canvas/api-key.txt)"
    export ACA_OPENHANDS_REST_LLM_API_KEY=...                    # bigmodel / z.ai key
    export ACA_OPENHANDS_REST_MODEL=openai/glm-5.3
    export ACA_OPENHANDS_REST_LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
    python tests/manual/test_openhands_rest_agent.py

It opens a persistent session on a throwaway workspace, asks the agent to write
a file, and prints the observable facts: the streamed messages, the terminal
status, the files reported back (contents fetched over HTTP), and the server's
metric keys.

Metric structures are printed as *keys and scalar values only*: a conversation
payload echoes the agent's LLM configuration, which includes the model API key.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

from agenter import AutonomousCodingAgent, Verbosity
from agenter.coding_backends.openhands_rest.client import OpenHandsRestClient
from agenter.data_models import CodingResult, CodingStatus

TASK = "Create a file named hello.py containing a function greet() that returns the string 'hello'."


def check_requirements() -> bool:
    missing = [
        name for name in ("ACA_OPENHANDS_REST_BASE_URL", "ACA_OPENHANDS_REST_LLM_API_KEY") if not os.environ.get(name)
    ]
    if missing:
        print(f"ERROR: missing environment variables: {', '.join(missing)}")
        return False
    return True


async def main() -> int:
    if not check_requirements():
        return 1

    base_url = os.environ["ACA_OPENHANDS_REST_BASE_URL"]
    api_key = os.environ.get("ACA_OPENHANDS_REST_API_KEY")

    with tempfile.TemporaryDirectory(prefix="agenter-oh-rest-") as workspace:
        print(f"workspace: {workspace}")
        agent = AutonomousCodingAgent(
            backend="openhands-rest",
            model=os.environ.get("ACA_OPENHANDS_REST_MODEL", "openai/gpt-4o"),
            openhands_rest_base_url=base_url,
            openhands_rest_api_key=api_key,
            openhands_rest_llm_api_key=os.environ["ACA_OPENHANDS_REST_LLM_API_KEY"],
            openhands_rest_llm_base_url=os.environ.get("ACA_OPENHANDS_REST_LLM_BASE_URL"),
            openhands_rest_max_iterations=12,
        )

        conversation_id: str | None = None
        result: CodingResult | None = None
        session = await agent.open_session(cwd=workspace, verbosity=Verbosity.QUIET)
        try:
            conversation_id = session.session_id
            print(f"conversation: {conversation_id}")
            print("--- streamed messages ---")
            async for event in session.stream_execute(TASK):
                if event.message is not None:
                    print(f"  [{event.message.type}] {event.message}")
                if event.result is not None:
                    result = event.result
        finally:
            await session.close()

        if result is None:
            print("ERROR: the session produced no result")
            return 2

        print("\n--- result ---")
        print(f"status          : {result.status.value}")
        print(f"summary         : {result.summary}")
        print(f"iterations      : {result.iterations}")
        print(f"total_tokens    : {result.total_tokens}")
        print(f"usage_reported  : {result.usage_reported}")
        print(f"error           : {result.error} (kind={result.error_kind}, retryable={result.retryable})")
        print(f"files           : {sorted(result.files)}")

        hello = Path(workspace) / "hello.py"
        print(f"workspace files : {sorted(p.name for p in Path(workspace).iterdir())}")
        if hello.exists():
            print("--- hello.py ---")
            print(hello.read_text())

        # The server's own view of the conversation: status and metric structure.
        client = OpenHandsRestClient(base_url=base_url, api_key=api_key)
        try:
            if conversation_id:
                info = await client.get_conversation(conversation_id)
                metrics = (info.get("stats") or {}).get("usage_to_metrics") or {}
                own = metrics.get("agenter") or {}
                tokens = own.get("accumulated_token_usage") or {}
                print("\n--- server view ---")
                print(f"execution_status  : {info.get('execution_status')}")
                print(f"usage ids         : {sorted(metrics)}")
                print(f"accumulated_cost  : {own.get('accumulated_cost')}")
                print(f"token usage keys  : {sorted(tokens)}")
                print(f"prompt/completion : {tokens.get('prompt_tokens')}/{tokens.get('completion_tokens')}")
        finally:
            await client.aclose()

        ok = result.status == CodingStatus.COMPLETED and hello.exists()
        print(f"\nSMOKE TEST {'PASSED' if ok else 'FAILED'}")
        return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
