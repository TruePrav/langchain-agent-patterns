"""
Human Escalation Tool — Reusable Escalation Pattern

Extracts the human-escalation logic into a configurable, reusable tool class.
Supports multiple notification channels and structured escalation metadata.

Usage:
    from tools.human_escalation import HumanEscalation, create_escalation_tool

    # As a standalone tool
    escalation = HumanEscalation(channels=["telegram", "slack"])
    result = escalation.escalate(
        reason="Customer threatening legal action",
        customer_email="user@example.com",
        severity="high",
        context="3 failed refund attempts",
    )

    # As a LangChain tool (for agent use)
    tool = create_escalation_tool(channels=["telegram"])
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool as langchain_tool

logger = logging.getLogger(__name__)

LOG_DIR = Path(__file__).parent.parent / "logs"


class HumanEscalation:
    """
    Manages escalation from AI agents to human operators.

    Responsibilities:
    - Log every escalation with full context (audit trail)
    - Route notifications to configured channels (Telegram, Slack, email)
    - Track escalation status (open → acknowledged → resolved)
    - Provide escalation history for analytics

    In production: connect to your alerting stack (PagerDuty, Slack, Telegram bot).
    """

    SEVERITY_LEVELS = {"low": 1, "medium": 2, "high": 3, "critical": 4}

    def __init__(self, channels: Optional[list] = None, team_name: str = "ops"):
        """
        Args:
            channels: Notification channels ('telegram', 'slack', 'email')
            team_name: Name of the team being notified
        """
        self.channels = channels or ["telegram"]
        self.team_name = team_name
        self._escalation_count = 0

    def escalate(
        self,
        reason: str,
        customer_email: str = "",
        severity: str = "medium",
        context: str = "",
        conversation_id: str = "",
    ) -> dict:
        """
        Escalate a case to a human agent.

        Args:
            reason: Why escalation is needed (required)
            customer_email: Customer contact info
            severity: 'low', 'medium', 'high', or 'critical'
            context: Additional context (conversation summary, prior attempts)
            conversation_id: Thread/conversation ID for handoff

        Returns:
            dict with escalation ID, status, and confirmation message
        """
        self._escalation_count += 1
        escalation_id = f"ESC-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{self._escalation_count:03d}"

        escalation_record = {
            "id": escalation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "customer_email": customer_email,
            "severity": severity,
            "severity_level": self.SEVERITY_LEVELS.get(severity, 2),
            "context": context,
            "conversation_id": conversation_id,
            "team": self.team_name,
            "channels_notified": self.channels,
            "status": "open",
        }

        # Log the escalation
        self._log_escalation(escalation_record)

        # Notify channels
        for channel in self.channels:
            self._notify(channel, escalation_record)

        # Build response for the agent/customer
        severity_icon = {"low": "📋", "medium": "⚠️", "high": "🔴", "critical": "🚨"}.get(severity, "⚠️")

        response = (
            f"{severity_icon} Escalated to {self.team_name} team. "
            f"Reference: {escalation_id}. "
            f"Reason: {reason}. "
        )
        if customer_email:
            response += f"We'll follow up at {customer_email}. "
        response += "The team has been notified and will respond within 1 business day."

        if severity in ("high", "critical"):
            response += " This is marked as priority — expect a faster response."

        return {
            "success": True,
            "escalation_id": escalation_id,
            "severity": severity,
            "message": response,
            "channels_notified": self.channels,
        }

    def _log_escalation(self, record: dict):
        """Append escalation to the audit log."""
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOG_DIR / "escalations.jsonl"
        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(f"Escalation logged: {record['id']} ({record['severity']})")

    def _notify(self, channel: str, record: dict):
        """
        Send notification to a channel.
        In production: replace with actual API calls.
        """
        severity_icon = {"low": "📋", "medium": "⚠️", "high": "🔴", "critical": "🚨"}.get(
            record["severity"], "⚠️"
        )
        message = (
            f"{severity_icon} Escalation {record['id']}\n"
            f"Reason: {record['reason']}\n"
            f"Severity: {record['severity']}\n"
        )
        if record.get("customer_email"):
            message += f"Customer: {record['customer_email']}\n"
        if record.get("context"):
            message += f"Context: {record['context'][:200]}\n"

        # In production: dispatch to actual channel
        if channel == "telegram":
            logger.info(f"[Telegram] Would send: {message[:100]}...")
            # requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", ...)
        elif channel == "slack":
            logger.info(f"[Slack] Would send: {message[:100]}...")
            # requests.post(SLACK_WEBHOOK_URL, json={"text": message})
        elif channel == "email":
            logger.info(f"[Email] Would send: {message[:100]}...")
            # smtp.send(to=OPS_EMAIL, subject=f"Escalation {record['id']}", body=message)


# ─── LangChain tool factory ─────────────────────────────────────────────────


def create_escalation_tool(channels: Optional[list] = None, team_name: str = "ops"):
    """
    Create a LangChain-compatible escalation tool.

    Usage:
        tool = create_escalation_tool(channels=["telegram", "slack"])
        # Add to agent: create_react_agent(llm, tools=[..., tool])
    """
    escalation = HumanEscalation(channels=channels, team_name=team_name)

    @langchain_tool
    def transfer_to_human(reason: str, customer_email: str = "") -> str:
        """Escalate to a human agent. Use as a last resort when you cannot resolve
        the customer's issue through available tools and knowledge base.

        Args:
            reason: Clear explanation of why escalation is needed
            customer_email: Customer's email for follow-up (optional)
        """
        result = escalation.escalate(
            reason=reason,
            customer_email=customer_email,
            severity="medium",
        )
        return result["message"]

    return transfer_to_human


# ─── Quick test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    esc = HumanEscalation(channels=["telegram", "slack"])
    result = esc.escalate(
        reason="Customer requesting refund — 3 failed attempts",
        customer_email="customer@example.com",
        severity="high",
        context="Gift card code already redeemed, customer insists it wasn't them",
    )
    print(json.dumps(result, indent=2))
