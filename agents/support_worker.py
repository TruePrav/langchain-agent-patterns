"""
Support Worker Agent

Stateless worker — handles customer support queries without a checkpointer.
Called as a tool by the supervisor. Each call is self-contained.

Knows the business KB via RAG. Escalates back to supervisor when unsure.
"""

import os
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from rag.retriever import build_retriever

# ─── Tools ───────────────────────────────────────────────────────────────────

_retriever = None

def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = build_retriever()
    return _retriever

@tool
def search_knowledge_base(query: str) -> str:
    """Search the business knowledge base for answers.
    Use before answering any product, policy, or procedure question.
    """
    retriever = _get_retriever()
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in the knowledge base."
    return "\n\n".join(d.page_content for d in docs[:3])

@tool
def request_order_proof(customer_message: str) -> str:
    """Ask the customer for order proof before troubleshooting.
    Always call this before trying to resolve an order issue.
    """
    return (
        "To help you faster, could you share your order ID or "
        "confirmation email? That lets me pull up the exact details."
    )

# ─── System prompt ───────────────────────────────────────────────────────────

SUPPORT_PROMPT = """You are a customer support specialist.

## Your job
Resolve customer queries clearly and efficiently. One action per message.

## Process
1. For product/policy questions → search_knowledge_base first
2. For order issues → request_order_proof before anything else
3. If you genuinely can't resolve it → say so clearly (supervisor will escalate)

## Rules
- Never make up information — always search first
- One clear action per message — don't overwhelm the customer
- Warm but concise — customers are often frustrated
- If the KB has no answer, admit it: "I don't have that information on hand."

## Tone
Professional, empathetic, direct. No walls of text. No jargon.
"""

# ─── Build worker ─────────────────────────────────────────────────────────────

def build_support_worker():
    """Build the support worker (no checkpointer — stateless by design)."""
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=512,
    )
    return create_react_agent(llm, tools=[search_knowledge_base, request_order_proof], prompt=SUPPORT_PROMPT)
