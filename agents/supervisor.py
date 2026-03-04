"""
Multi-Agent Supervisor Pattern (LangGraph)

Production pattern: one supervisor routes to specialist worker agents.
Supervisor uses a stronger model for routing decisions.
Workers are stateless — supervisor owns conversation continuity.

Usage:
    agent = build_supervisor()
    result = agent.invoke({"messages": [{"role": "user", "content": "..."}]},
                          config={"configurable": {"thread_id": "user-123"}})
"""

import os
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from agents.support_worker import build_support_worker
from agents.data_worker import build_data_worker

# ─── Model setup ─────────────────────────────────────────────────────────────

def get_model(model_name: str = "claude-haiku-4-5-20241022"):
    return ChatAnthropic(
        model=model_name,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=1024,
    )

# ─── Worker agents (built once, reused across requests) ──────────────────────

_support_worker = None
_data_worker = None

def _get_workers():
    global _support_worker, _data_worker
    if _support_worker is None:
        _support_worker = build_support_worker()
    if _data_worker is None:
        _data_worker = build_data_worker()
    return _support_worker, _data_worker

# ─── Supervisor tools (workers exposed as tools) ─────────────────────────────

@tool
def ask_support_agent(query: str) -> str:
    """Route a customer support query to the support specialist.
    Use for: order issues, product questions, complaint handling, general FAQs.
    """
    support, _ = _get_workers()
    result = support.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"configurable": {"thread_id": "worker-ephemeral"}},
    )
    return result["messages"][-1].content

@tool
def ask_data_agent(query: str) -> str:
    """Route a data or reporting query to the data specialist.
    Use for: metrics, inventory status, reconciliation checks, reporting.
    """
    _, data = _get_workers()
    result = data.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"configurable": {"thread_id": "worker-ephemeral"}},
    )
    return result["messages"][-1].content

@tool
def transfer_to_human(reason: str, customer_email: str = "") -> str:
    """Escalate to a human agent. Use as last resort.
    Triggers an alert to the ops team and informs the customer.
    """
    # In production: send a Telegram/Slack alert here
    return (
        f"Escalated to human. Reason: {reason}. "
        f"{'Contact: ' + customer_email if customer_email else ''} "
        "The team has been notified and will follow up within 1 business day."
    )

# ─── Supervisor system prompt ─────────────────────────────────────────────────

SUPERVISOR_PROMPT = """You are the main orchestrator for a business AI assistant.

## Your role
You are the first point of contact. Route tasks to the right specialist or handle simple queries directly.

## Routing rules
- Customer queries, complaints, product questions → ask_support_agent
- Data requests, metrics, inventory, reports → ask_data_agent
- You've tried everything and can't resolve it → transfer_to_human

## Direct handling (no routing needed)
- Greetings and simple acknowledgements
- Clarifying what the business does
- Telling the user what you can help with

## Rules
- One action per turn — route OR respond, not both
- Never make up data — use the data agent
- Never promise outcomes the workers can't deliver
- Always sound like one seamless assistant — don't mention internal routing
"""

# ─── Build supervisor ─────────────────────────────────────────────────────────

def build_supervisor():
    """Build the supervisor agent with persistent memory (Supabase in production)."""
    llm = get_model("claude-sonnet-4-5-20241022")  # Stronger model for routing
    checkpointer = MemorySaver()  # Swap for SqliteSaver or Supabase in production

    agent = create_react_agent(
        llm,
        tools=[ask_support_agent, ask_data_agent, transfer_to_human],
        prompt=SUPERVISOR_PROMPT,
        checkpointer=checkpointer,
    )
    return agent

def chat(agent, message: str, thread_id: str = "default") -> str:
    """Send a message and get a response."""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].content
