"""
LangSmith Eval Suite — Customer Support Agent

Runs correctness + escalation evals against a dataset of production-style conversations.
Designed to run in CI — exits with code 1 if pass rate drops below threshold.

Setup:
    pip install langsmith langchain-anthropic
    export LANGCHAIN_API_KEY=your_key
    export ANTHROPIC_API_KEY=your_key

Usage:
    python evals/run_evals.py
    python evals/run_evals.py --dataset support-evals-v1 --threshold 0.95
"""

import argparse
import sys
import os
from langsmith import Client
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langchain_anthropic import ChatAnthropic

from agents.support_worker import build_support_worker

# ─── Eval dataset (inline — in production, load from LangSmith) ──────────────

EVAL_EXAMPLES = [
    {
        "input": "How do I redeem my PSN card?",
        "expected": "redemption steps for PSN",
        "category": "how-to",
    },
    {
        "input": "My code isn't working, I need a refund",
        "expected": "ask for order ID before troubleshooting",
        "category": "order-issue",
    },
    {
        "input": "Do you have Xbox gift cards?",
        "expected": "check knowledge base for availability",
        "category": "product-query",
    },
    {
        "input": "I want to speak to a real person",
        "expected": "escalate to human with empathy",
        "category": "escalation",
    },
    {
        "input": "What's the difference between a $25 and $50 PSN card?",
        "expected": "explain denominations from KB",
        "category": "product-query",
    },
]

# ─── Custom evaluators ────────────────────────────────────────────────────────

def escalation_evaluator(run, example) -> dict:
    """Check: does an escalation request actually trigger an escalation?"""
    if example.inputs.get("category") != "escalation":
        return {"key": "escalation_handled", "score": 1}  # N/A

    output = run.outputs.get("output", "").lower()
    triggered = any(word in output for word in ["human", "team", "follow up", "agent"])
    return {
        "key": "escalation_handled",
        "score": 1 if triggered else 0,
        "comment": "Escalation correctly triggered" if triggered else "FAILED: escalation not triggered",
    }

def order_proof_evaluator(run, example) -> dict:
    """Check: does an order issue response ask for proof before troubleshooting?"""
    if example.inputs.get("category") != "order-issue":
        return {"key": "order_proof_requested", "score": 1}  # N/A

    output = run.outputs.get("output", "").lower()
    asked = any(word in output for word in ["order id", "order number", "confirmation", "proof"])
    return {
        "key": "order_proof_requested",
        "score": 1 if asked else 0,
        "comment": "Order proof requested" if asked else "FAILED: troubleshot without asking for proof",
    }

# ─── Run evals ────────────────────────────────────────────────────────────────

def run_evals(dataset_name: str = "support-evals-v1", pass_threshold: float = 0.90):
    client = Client()
    agent = build_support_worker()

    def predict(inputs: dict) -> dict:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": inputs["input"]}]},
            config={"configurable": {"thread_id": "eval-ephemeral"}},
        )
        return {"output": result["messages"][-1].content}

    # LLM-based correctness eval
    correctness_evaluator = LangChainStringEvaluator(
        "labeled_criteria",
        config={
            "criteria": {
                "correctness": "Does the response correctly address what was expected? Score 1 if yes, 0 if no."
            },
            "llm": ChatAnthropic(model="claude-haiku-4-5-20241022", api_key=os.getenv("ANTHROPIC_API_KEY")),
        },
        prepare_data=lambda run, example: {
            "prediction": run.outputs["output"],
            "reference": example.outputs["expected"],
            "input": example.inputs["input"],
        },
    )

    results = evaluate(
        predict,
        data=EVAL_EXAMPLES,
        evaluators=[correctness_evaluator, escalation_evaluator, order_proof_evaluator],
        experiment_prefix="support-agent",
        metadata={"version": "1.0", "agent": "support_worker"},
    )

    # ── Pass/fail gate ────────────────────────────────────────────────────────
    scores = [r.get("feedback", {}).get("score", 0) for r in results._results]
    pass_rate = sum(scores) / len(scores) if scores else 0

    print(f"\nPass rate: {pass_rate:.1%} (threshold: {pass_threshold:.1%})")

    if pass_rate < pass_threshold:
        print(f"❌ FAILED — pass rate {pass_rate:.1%} below threshold {pass_threshold:.1%}")
        sys.exit(1)
    else:
        print(f"✅ PASSED — {len(scores)} evals, {pass_rate:.1%} pass rate")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="support-evals-v1")
    parser.add_argument("--threshold", type=float, default=0.90)
    args = parser.parse_args()
    run_evals(args.dataset, args.threshold)
