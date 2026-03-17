"""
LangSmith Tracing Setup

Enables automatic tracing for all LangChain/LangGraph calls.
Every LLM invocation, tool call, and chain step is logged to LangSmith.

Call configure_tracing() once at app startup — nothing else needed.
All agents, workers, and chains automatically trace after that.

Usage:
    from tracing.setup import configure_tracing
    configure_tracing()  # call once at startup

    # Everything after this is traced automatically:
    agent = build_supervisor()
    result = agent.invoke(...)
"""

import os

def configure_tracing(project: str = "agent-patterns"):
    """Enable LangSmith tracing. Call once at app startup."""
    required = ["LANGCHAIN_API_KEY"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"[tracing] Disabled — missing env vars: {missing}")
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project

    print(f"[tracing] Enabled — project: {project}")
    return True


def tag_run(metadata: dict):
    """Add metadata tags to the current run (call from within a tool or chain).

    Example:
        tag_run({"customer_id": "user-123", "channel": "telegram", "version": "1.2"})

    Tags appear in LangSmith UI and are filterable — useful for:
    - Filtering traces by customer or session
    - Comparing agent versions in production
    - Building eval datasets from specific cohorts
    """
    from langsmith import RunTree
    try:
        run = RunTree.get_current_run()
        if run:
            run.add_metadata(metadata)
    except Exception:
        pass  # Never crash the agent due to tracing issues
