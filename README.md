# LangChain Agent Patterns

Production patterns from real multi-agent deployments. These are the architectural foundations behind live systems handling customer support, financial reconciliation, and ops automation.

## Structure

```
agents/          — Supervisor + worker agent patterns
evals/           — LangSmith eval suite setup
rag/             — RAG pipeline with KB loader
tools/           — Custom tool patterns (API wrappers, schedulers)
```

## What's here

### Multi-Agent Supervisor (LangGraph)
A production-tested supervisor pattern where a central orchestrator routes tasks to specialist workers. Each worker is stateless — the supervisor owns conversation continuity.

```
Supervisor (Sonnet)
├── Support Worker  (Haiku)  — resolves customer queries
├── Data Worker     (Haiku)  — queries external data sources
└── Content Worker  (Grok)   — research + generation tasks
```

### LangSmith Eval Suite
Eval pipeline for customer-facing agents. Template includes:
- Dataset builder from real production conversations
- Custom evaluators (correctness, escalation detection, tone)
- CI-ready runner — fails build if pass rate drops below threshold

### RAG Pipeline
KB loader + retriever pattern with:
- Chunking strategy for domain-specific knowledge
- Hybrid search (semantic + keyword fallback)
- Source attribution in responses

### Tool Patterns
- External API wrapper with retry + rate limit handling
- Scheduled tool (cron-triggered agent invocations)
- Human-in-the-loop escalation tool

## Production context

These patterns run across:
- **73+ LangSmith evals** on customer support agent — 100% pass rate in production
- **Multi-provider model routing** — Anthropic Claude, xAI Grok, Google Gemini with automatic fallback
- **Daily scheduled runs** via LangChain scheduler (reconciliation, content, evals)

## Stack
Python · LangChain · LangGraph · LangSmith · Anthropic · Groq · xAI

---
*Part of the AI agent systems at [ailevelup.ca](https://ailevelup.ca)*
