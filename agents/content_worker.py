"""
Content Worker Agent (xAI Grok)

Stateless worker — handles content generation, research, and summarization.
Called as a tool by the supervisor. Each call is self-contained.

Uses xAI Grok for fast, creative content generation with web-aware reasoning.
Falls back to Anthropic Claude if xAI is unavailable.
"""

import os
import logging
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)

# ─── Tools ───────────────────────────────────────────────────────────────────


@tool
def web_search(query: str) -> str:
    """Search the web for current information on a topic.
    Use for: trending topics, recent news, current events, market data.
    Returns formatted search results with titles and snippets.
    """
    # In production: integrate with Brave Search, SerpAPI, or Tavily
    # Here: returns a structured stub that demonstrates the pattern
    return (
        f"Search results for: '{query}'\n\n"
        f"1. [Latest Analysis] Comprehensive overview of {query} — "
        f"Key findings indicate significant developments in the past 7 days.\n\n"
        f"2. [Industry Report] Market trends related to {query} — "
        f"Expert consensus suggests continued momentum through Q2 2026.\n\n"
        f"3. [Community Discussion] Reddit/X threads on {query} — "
        f"Mixed sentiment, leaning positive. Top concerns: scalability and adoption."
    )


@tool
def summarize_text(text: str, style: str = "brief") -> str:
    """Summarize a block of text into key takeaways.
    Use for: condensing research results, distilling articles, creating briefs.

    Args:
        text: The content to summarize
        style: 'brief' (3-5 bullets), 'detailed' (paragraph), or 'tweet' (280 chars)
    """
    # In production: this would use the LLM directly for summarization
    # Here: demonstrates the tool interface pattern
    if style == "tweet":
        return text[:277] + "..." if len(text) > 280 else text
    elif style == "detailed":
        return f"Detailed summary:\n{text[:500]}"
    else:
        lines = text.split(". ")[:5]
        return "Key takeaways:\n" + "\n".join(f"• {line.strip()}" for line in lines if line.strip())


@tool
def draft_content(
    topic: str,
    content_type: str = "social",
    tone: str = "professional",
) -> str:
    """Draft content based on a topic and specifications.
    Use for: social media posts, blog outlines, marketing copy, email drafts.

    Args:
        topic: What the content should be about
        content_type: 'social' (short post), 'blog' (outline), 'email' (draft), 'ad' (copy)
        tone: 'professional', 'casual', 'technical', or 'persuasive'
    """
    # In production: the LLM handles this via its system prompt
    # The tool exists to give the agent a structured action for content creation
    templates = {
        "social": f"📌 {topic}\n\nHere's what you need to know →\n\n[Key insight]\n[Supporting point]\n[Call to action]\n\n#relevant #hashtags",
        "blog": f"# {topic}\n\n## Introduction\n[Hook + context]\n\n## Key Points\n1. [Point 1]\n2. [Point 2]\n3. [Point 3]\n\n## Conclusion\n[Takeaway + CTA]",
        "email": f"Subject: {topic}\n\nHi [Name],\n\n[Opening — why this matters to them]\n\n[Core message — 2-3 sentences]\n\n[Clear next step]\n\nBest,\n[Sender]",
        "ad": f"🔥 {topic}\n\n[Pain point → Solution → Benefit]\n\n→ [CTA with urgency]",
    }
    return templates.get(content_type, templates["social"])


# ─── System prompt ───────────────────────────────────────────────────────────

CONTENT_PROMPT = """You are a content and research specialist.

## Your job
Generate high-quality content and conduct research efficiently.
You combine web research with creative writing to produce actionable output.

## Tools available
- web_search — find current information on any topic
- summarize_text — condense research into key takeaways
- draft_content — create structured content (social posts, blogs, emails, ads)

## Process
1. For research requests → web_search first, then summarize findings
2. For content creation → research the topic if needed, then draft
3. For trend analysis → search multiple angles, synthesize into insights

## Rules
- Always ground content in research — don't fabricate claims
- Match the requested tone and format exactly
- Keep social content punchy (<280 chars for tweets, <2200 for Instagram)
- Blog outlines should have 3-5 main sections minimum
- Flag if a topic needs human review (sensitive, legal, medical)

## Tone
Adaptable — match whatever the supervisor requests. Default: professional but engaging.
"""

# ─── Build worker ─────────────────────────────────────────────────────────────


def _get_content_llm():
    """Get the content worker's LLM — prefers xAI Grok, falls back to Anthropic."""
    xai_key = os.getenv("XAI_API_KEY")
    if xai_key:
        try:
            from langchain_xai import ChatXAI

            logger.info("Content worker using xAI Grok")
            return ChatXAI(
                model="grok-2-latest",
                api_key=xai_key,
                max_tokens=1024,
            )
        except ImportError:
            logger.warning("langchain-xai not installed, falling back to Anthropic")
        except Exception as e:
            logger.warning(f"xAI init failed ({e}), falling back to Anthropic")

    # Fallback to Anthropic
    from langchain_anthropic import ChatAnthropic

    logger.info("Content worker using Anthropic Claude (fallback)")
    return ChatAnthropic(
        model="claude-haiku-4-5-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=1024,
    )


def build_content_worker():
    """Build the content worker (no checkpointer — stateless by design)."""
    llm = _get_content_llm()
    return create_react_agent(
        llm,
        tools=[web_search, summarize_text, draft_content],
        prompt=CONTENT_PROMPT,
    )
