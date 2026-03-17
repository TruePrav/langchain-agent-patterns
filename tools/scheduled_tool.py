"""
Scheduled Tool — Cron-Triggered Agent Invocation Pattern

Demonstrates how to build a tool designed for scheduled/cron invocation:
- Reads a schedule config to determine what to run
- Invokes the agent with the scheduled task
- Logs results with timestamps for audit trails
- Handles failures gracefully (logs but doesn't crash)

Usage (from cron or task scheduler):
    python -m tools.scheduled_tool --task daily_report
    python -m tools.scheduled_tool --task inventory_check --dry-run

Production integration:
    # crontab
    0 8 * * * cd /app && python -m tools.scheduled_tool --task daily_report
    */30 * * * * cd /app && python -m tools.scheduled_tool --task inventory_check
"""

import json
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ─── Schedule config ─────────────────────────────────────────────────────────

SCHEDULED_TASKS = {
    "daily_report": {
        "description": "Generate daily sales and inventory report",
        "agent_prompt": "Generate a daily report covering: (1) yesterday's revenue, (2) current inventory alerts, (3) any items that need restocking.",
        "schedule": "0 8 * * *",  # 8 AM daily
        "timeout_seconds": 120,
        "notify_on_failure": True,
    },
    "inventory_check": {
        "description": "Check for low-stock items and alert",
        "agent_prompt": "Check inventory levels. List any items with low stock or out of stock. Format as an actionable alert.",
        "schedule": "*/30 * * * *",  # Every 30 minutes
        "timeout_seconds": 60,
        "notify_on_failure": True,
    },
    "weekly_summary": {
        "description": "Weekly performance summary for management",
        "agent_prompt": "Create a weekly summary: total revenue, top-selling items, inventory turnover, any operational issues flagged this week.",
        "schedule": "0 9 * * 1",  # 9 AM Monday
        "timeout_seconds": 180,
        "notify_on_failure": False,
    },
}

# ─── Execution log ───────────────────────────────────────────────────────────

LOG_DIR = Path(__file__).parent.parent / "logs"


def _log_execution(task_name: str, status: str, result: str = "", error: str = ""):
    """Append an execution record to the schedule log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "scheduled_runs.jsonl"

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": task_name,
        "status": status,
        "result_preview": result[:500] if result else "",
        "error": error[:500] if error else "",
    }

    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    logger.info(f"[{task_name}] {status}" + (f" — {error}" if error else ""))


# ─── Task runner ─────────────────────────────────────────────────────────────


def run_scheduled_task(
    task_name: str,
    agent_factory: Optional[Callable] = None,
    dry_run: bool = False,
) -> dict:
    """
    Execute a scheduled task by invoking the appropriate agent.

    Args:
        task_name: Key from SCHEDULED_TASKS
        agent_factory: Callable that returns a LangGraph agent (default: build_supervisor)
        dry_run: If True, log the task but don't invoke the agent

    Returns:
        dict with 'success', 'task', 'result', and optionally 'error'
    """
    if task_name not in SCHEDULED_TASKS:
        available = ", ".join(SCHEDULED_TASKS.keys())
        return {
            "success": False,
            "task": task_name,
            "error": f"Unknown task '{task_name}'. Available: {available}",
        }

    task_config = SCHEDULED_TASKS[task_name]
    prompt = task_config["agent_prompt"]

    logger.info(f"Running scheduled task: {task_name} — {task_config['description']}")

    if dry_run:
        _log_execution(task_name, "dry_run", result=prompt)
        return {
            "success": True,
            "task": task_name,
            "result": f"[DRY RUN] Would invoke agent with: {prompt}",
        }

    # Build the agent
    if agent_factory is None:
        try:
            from agents.supervisor import build_supervisor
            agent_factory = build_supervisor
        except ImportError as e:
            _log_execution(task_name, "error", error=str(e))
            return {"success": False, "task": task_name, "error": f"Import error: {e}"}

    try:
        agent = agent_factory()
        result = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"configurable": {"thread_id": f"scheduled-{task_name}"}},
        )
        output = result["messages"][-1].content
        _log_execution(task_name, "success", result=output)
        return {"success": True, "task": task_name, "result": output}

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        _log_execution(task_name, "error", error=error_msg)

        if task_config.get("notify_on_failure"):
            logger.error(f"ALERT: Scheduled task '{task_name}' failed — {error_msg}")
            # In production: send Telegram/Slack alert here

        return {"success": False, "task": task_name, "error": error_msg}


# ─── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run a scheduled agent task")
    parser.add_argument("--task", required=True, choices=list(SCHEDULED_TASKS.keys()))
    parser.add_argument("--dry-run", action="store_true", help="Log without invoking agent")
    parser.add_argument("--list", action="store_true", help="List all scheduled tasks")
    args = parser.parse_args()

    if args.list:
        for name, config in SCHEDULED_TASKS.items():
            print(f"  {name}: {config['description']} ({config['schedule']})")
    else:
        result = run_scheduled_task(args.task, dry_run=args.dry_run)
        print(json.dumps(result, indent=2))
