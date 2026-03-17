"""
Query Production Traces

Pull recent traces from LangSmith to find failures, slow runs, and patterns.
Run this manually or on a schedule to monitor production health.

Usage:
    python tracing/query_traces.py
    python tracing/query_traces.py --hours 24 --project my-project
"""

import argparse
import os
from datetime import datetime, timedelta
from langsmith import Client

def query_recent_traces(project: str, hours: int = 6):
    client = Client()
    start = datetime.utcnow() - timedelta(hours=hours)

    runs = list(client.list_runs(
        project_name=project,
        start_time=start,
        run_type="chain",
        filter='eq(is_root, true)',
    ))

    if not runs:
        print(f"No runs found in the last {hours}h in project '{project}'")
        return

    errors     = [r for r in runs if r.error]
    slow_runs  = [r for r in runs if r.total_tokens and r.total_tokens > 2000]
    total_cost = sum((r.total_cost or 0) for r in runs)

    print(f"\n── LangSmith trace report ({hours}h) ─────────────────────")
    print(f"Total runs  : {len(runs)}")
    print(f"Errors      : {len(errors)}")
    print(f"High-token  : {len(slow_runs)} runs over 2k tokens")
    print(f"Total cost  : ${total_cost:.4f}")

    if errors:
        print(f"\n── Errors ({'first 5'}) ──────────────────────────────────")
        for r in errors[:5]:
            print(f"  [{r.id}] {r.error[:120]}")

    if slow_runs:
        print(f"\n── High-token runs ──────────────────────────────────────")
        for r in slow_runs[:5]:
            print(f"  [{r.id}] {r.total_tokens} tokens | ${r.total_cost:.4f}")

    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours",   type=int, default=6)
    parser.add_argument("--project", default=os.getenv("LANGCHAIN_PROJECT", "agent-patterns"))
    args = parser.parse_args()
    query_recent_traces(args.project, args.hours)
