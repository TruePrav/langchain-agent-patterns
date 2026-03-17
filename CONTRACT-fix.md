# CONTRACT — langchain-agent-patterns Fix

## Goal
Make this repo match what the README claims. It's being reviewed by a potential client (Nova Studio) for a freelance interview tomorrow. Every example must be real, working code.

## Critical Issues to Fix

### 1. Add multi-provider model fallback to supervisor.py
README says: "Multi-provider model routing — Anthropic Claude, xAI Grok, Google Gemini with automatic fallback"
Reality: Only Anthropic is used. No fallback exists.

Fix: Add a `get_model_with_fallback()` function that tries providers in order:
1. Anthropic Claude (primary)
2. xAI Grok via langchain-xai (fallback on rate limit or error)
3. Graceful error if all fail
Use langchain's standard interface so swapping is clean.

### 2. Update .env.example
Add missing keys:
```
ANTHROPIC_API_KEY=your_anthropic_key_here
XAI_API_KEY=your_xai_key_here
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agent-patterns
```

### 3. Add missing content_worker.py
README mentions a Content Worker (Grok) — it doesn't exist.
Create `agents/content_worker.py`:
- Uses xAI Grok model (langchain-xai ChatXAI)
- Tools: web_search_tool (stub returning formatted search result), summarize_tool
- System prompt: research + content generation specialist
- Same pattern as support_worker.py and data_worker.py
- Wire it into supervisor.py as a third worker tool: ask_content_agent

### 4. Add tools/ folder
README mentions tools patterns. Create:
- `tools/api_wrapper.py` — example tool with retry + rate limit handling (exponential backoff, max 3 retries)
- `tools/scheduled_tool.py` — example of a tool designed for cron-triggered invocation (reads schedule, runs agent, logs result)
- `tools/human_escalation.py` — clean extraction of the transfer_to_human logic into a reusable tool class

### 5. Verify evals/run_evals.py is complete
Check it actually runs. If it's a stub, make it a real working eval runner with at least 3 test cases.

## Out of Scope
- No live API calls (don't need real keys to write correct code)
- No deployment
- Don't change the repo structure dramatically — improve what exists

## Done When
- [x] supervisor.py has `get_model_with_fallback()` with Anthropic → Grok fallback ✅
- [x] .env.example has XAI_API_KEY and is complete ✅
- [x] agents/content_worker.py exists and follows same pattern as other workers ✅
- [x] supervisor.py wires in content worker as ask_content_agent tool ✅
- [x] tools/ folder has 3 files: api_wrapper.py, scheduled_tool.py, human_escalation.py ✅
- [x] evals/run_evals.py has real test cases (not stubs) ✅ (already had 5 real test cases)
- [x] All files have proper docstrings ✅
- [ ] No syntax errors: `python -c "from agents.supervisor import build_supervisor; print('ok')"` would pass (assuming deps installed)
