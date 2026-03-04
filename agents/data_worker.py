"""
Data Worker Agent

Stateless worker — handles data queries, metrics, and reporting.
Called as a tool by the supervisor.

In production: connects to your POS, database, or spreadsheet.
Here: uses mock data to demonstrate the pattern.
"""

import os
from datetime import datetime, timedelta
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# ─── Mock data (replace with real API/DB calls in production) ────────────────

MOCK_INVENTORY = {
    "product-001": {"name": "Gift Card A", "stock": 45, "sold_today": 12},
    "product-002": {"name": "Gift Card B", "stock": 3,  "sold_today": 28},
    "product-003": {"name": "Gift Card C", "stock": 0,  "sold_today": 5},
}

MOCK_DAILY_REVENUE = {
    (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"): round(800 + i * 47.5, 2)
    for i in range(7)
}

# ─── Tools ───────────────────────────────────────────────────────────────────

@tool
def get_inventory_status(product_id: str = "") -> str:
    """Get current inventory levels. Pass a product_id for a specific item,
    or leave empty to get all items.
    """
    if product_id:
        item = MOCK_INVENTORY.get(product_id)
        if not item:
            return f"No product found with ID: {product_id}"
        status = "LOW STOCK" if item["stock"] < 5 else ("OUT OF STOCK" if item["stock"] == 0 else "OK")
        return f"{item['name']}: {item['stock']} in stock | Sold today: {item['sold_today']} | Status: {status}"

    lines = []
    for pid, item in MOCK_INVENTORY.items():
        status = "🔴 OUT" if item["stock"] == 0 else ("🟡 LOW" if item["stock"] < 5 else "🟢 OK")
        lines.append(f"{status} {item['name']}: {item['stock']} units")
    return "\n".join(lines)

@tool
def get_revenue_summary(days: int = 7) -> str:
    """Get revenue summary for the last N days (default: 7)."""
    days = min(max(1, days), 30)
    relevant = list(MOCK_DAILY_REVENUE.items())[:days]
    total = sum(v for _, v in relevant)
    daily_avg = total / len(relevant)
    lines = [f"  {date}: ${amount:,.2f}" for date, amount in relevant]
    return (
        f"Revenue — last {days} days:\n" +
        "\n".join(lines) +
        f"\n\nTotal: ${total:,.2f} | Daily avg: ${daily_avg:,.2f}"
    )

@tool
def get_low_stock_alerts() -> str:
    """Get all items with low or no stock that need restocking."""
    alerts = [
        f"{'🔴 OUT OF STOCK' if item['stock'] == 0 else '🟡 LOW STOCK'}: {item['name']} ({item['stock']} remaining)"
        for item in MOCK_INVENTORY.values()
        if item["stock"] < 10
    ]
    if not alerts:
        return "All items are sufficiently stocked."
    return "Restock alerts:\n" + "\n".join(alerts)

# ─── System prompt ───────────────────────────────────────────────────────────

DATA_PROMPT = """You are a data and reporting specialist.

## Your job
Answer data queries accurately. Pull from tools, never guess numbers.

## Tools available
- get_inventory_status — current stock levels for one or all products
- get_revenue_summary — revenue over the last N days
- get_low_stock_alerts — items needing urgent restock

## Rules
- Always use a tool before reporting numbers
- Format output cleanly — the supervisor will relay it to the user
- Flag anything that needs urgent attention (out of stock, revenue drops)
- Keep responses factual and concise
"""

# ─── Build worker ─────────────────────────────────────────────────────────────

def build_data_worker():
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=512,
    )
    return create_react_agent(
        llm,
        tools=[get_inventory_status, get_revenue_summary, get_low_stock_alerts],
        prompt=DATA_PROMPT,
    )
