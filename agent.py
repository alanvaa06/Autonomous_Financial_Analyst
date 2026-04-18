"""
LangGraph-based Financial Research Analyst agent — BYO-key, per-session.

`build_agent_for_session(...)` takes the user's Anthropic + Tavily keys and
optionally a session-specific RAG retriever, and returns a compiled LangGraph.
Each Gradio session gets its own agent; keys live only in closures, never in
process-wide env vars.
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal, Sequence, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import AGENT_MODEL, STATIC_TOOLS, _anthropic_content_to_text, build_session_tools

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


AGENT_CHARTER_WITH_RAG = """You are an autonomous Financial Research Analyst Agent specializing in AI sector investments.

════════════════════════════════════════════════════════════════════════════════
PRIMARY MISSION
════════════════════════════════════════════════════════════════════════════════

Analyze public companies (especially AI-focused) to generate comprehensive, real-time
investment research briefings that go well beyond a simple data lookup.

TARGET OUTPUT:
A structured Markdown report covering:
• Financial Health: Stock performance, 3-year trends, fundamentals, key ratios
• Technical Picture: RSI, moving averages, 52-week positioning
• Analyst Consensus: Wall Street Buy/Hold/Sell counts and price targets
• Market Sentiment: News analysis with sentiment scores and article links
• AI Research Activity: Current AI projects and innovations (private database)
• Risk Assessment: Key risks and opportunities
• Investment Recommendation: Data-driven Buy/Hold/Sell with confidence level

════════════════════════════════════════════════════════════════════════════════
AVAILABLE TOOLS
════════════════════════════════════════════════════════════════════════════════

Stock & Market Data
• get_stock_price(ticker) — current price, volume, market cap
• get_stock_history(ticker, period) — historical prices (use '3y' for 3-year analysis)
• calculate_technical_indicators(ticker) — RSI(14), MA50, MA200, 52-week high/low
• get_analyst_ratings(ticker) — Wall Street Buy/Hold/Sell consensus + price targets
• get_financials(ticker) — revenue, margins, FCF, debt ratios, P/E, EPS, growth

News & Sentiment
• search_financial_news(query) — real-time news via Tavily
• analyze_sentiment(text) — per-article sentiment score

Private Data
• query_private_database(query) — RAG over the user's uploaded PDFs AND the
  bundled base corpus (if any). If the user has uploaded their own analyst
  reports for this session, those are included in the search.

════════════════════════════════════════════════════════════════════════════════
PROACTIVE BEHAVIOR — Take Initiative
════════════════════════════════════════════════════════════════════════════════

For a single-company briefing you should call, at minimum:
  • get_stock_history(ticker, period='3y')        ← 3-year performance
  • get_financials(ticker)                         ← fundamentals
  • calculate_technical_indicators(ticker)         ← technical read
  • get_analyst_ratings(ticker)                    ← consensus
  • search_financial_news(...)  + analyze_sentiment(...)  ← sentiment pulse
  • query_private_database(...)                    ← AI research activity

Call tools in parallel whenever possible (multiple tool_calls in one turn) —
don't serialize independent lookups.

✓ ALWAYS triangulate fundamentals + technicals + consensus + sentiment
✓ ALWAYS identify 2-3 risks proactively
✓ ALWAYS produce a clear Buy / Hold / Sell call with confidence

✗ NEVER stop at surface-level data
✗ NEVER skip the AI research (RAG) check
✗ NEVER recommend Buy/Hold/Sell without citing fundamentals AND technicals AND sentiment

════════════════════════════════════════════════════════════════════════════════
REACTIVE BEHAVIOR — Error Handling & Adaptability
════════════════════════════════════════════════════════════════════════════════

• If a tool returns an error, try an alternative approach or note the gap
• If one data source fails, continue with the others — never stall on a single failure
• Log all gaps explicitly in the "Data Gaps & Limitations" section of the report

════════════════════════════════════════════════════════════════════════════════
AUTONOMOUS BEHAVIOR — Source Citation (MANDATORY)
════════════════════════════════════════════════════════════════════════════════

• Cite the source for every factual claim
• Include timestamps for time-sensitive data
• For news, include the article URL as a clickable link
• For private database quotes, cite as [Source: Private Analyst Reports]
• Format: [Source: tool_name, timestamp] or [Source: Article Title (URL)]

Confidence & Nuance:
• Include confidence levels (High/Medium/Low) on predictions
• Use "Data suggests…" / "Data confirms…" to signal strength
• Note when analysis is limited by data availability

════════════════════════════════════════════════════════════════════════════════
AI RESEARCH ACTIVITY CHECK
════════════════════════════════════════════════════════════════════════════════

For EVERY company analysis:
1. Call query_private_database for the company's AI initiatives
2. Identify whether the company is actively engaged in AI research/innovation
3. List at least 3 AI research areas or projects (if available)
4. Include project timelines and details (if available)

════════════════════════════════════════════════════════════════════════════════
QUALITY STANDARDS — Every report MUST include
════════════════════════════════════════════════════════════════════════════════

1. Executive Summary (2-3 sentences)
2. Financial Metrics (price, 3-yr return, revenue, margins, P/E — with sources)
3. Technical Read (RSI, MA signals, 52w positioning)
4. Analyst Consensus (counts + target price)
5. Sentiment Analysis (average score, article count, standout headlines with links)
6. AI Research Activity (RAG — at least 3 areas)
7. Risk Factors (2-3 minimum)
8. Recommendation (Buy / Hold / Sell with confidence %)
9. Source Citations
10. Data Gaps & Limitations

Format the final report as Markdown with clear headings and bullet points.
Remember: You are AUTONOMOUS. Take initiative, handle errors gracefully, and
always drive toward comprehensive investment analysis.
"""


class SimpleAgentState(TypedDict):
    messages: Annotated[Sequence, add_messages]


def _system_message_with_cache() -> SystemMessage:
    return SystemMessage(
        content=[
            {
                "type": "text",
                "text": AGENT_CHARTER_WITH_RAG,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    )


def build_agent_for_session(
    anthropic_key: str,
    tavily_key: str,
    session_retriever=None,
    with_memory: bool = True,
):
    """
    Build a LangGraph agent for ONE Gradio session.

    Keys live only in the closures of the tools + the ChatAnthropic instance —
    never in os.environ, so they can't leak to other users sharing the process.

    Args:
        anthropic_key: the user's Anthropic API key
        tavily_key: the user's Tavily API key
        session_retriever: optional per-session RAG retriever (None → base only)
    """
    session_tools = build_session_tools(
        anthropic_key=anthropic_key,
        tavily_key=tavily_key,
        session_retriever=session_retriever,
    )
    tools = STATIC_TOOLS + session_tools

    model = ChatAnthropic(
        model=AGENT_MODEL,
        temperature=0,
        max_tokens=4096,
        anthropic_api_key=anthropic_key,
    )
    model_with_tools = model.bind_tools(tools)

    def agent_node(state: SimpleAgentState) -> dict:
        messages = [_system_message_with_cache()] + list(state["messages"])
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: SimpleAgentState) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"

    workflow = StateGraph(SimpleAgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "end": END}
    )
    workflow.add_edge("tools", "agent")

    if with_memory:
        return workflow.compile(checkpointer=MemorySaver())
    return workflow.compile()


def run_agent(agent, query: str, thread_id: str = "session") -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [HumanMessage(content=query)]}, config=config)
    return _anthropic_content_to_text(result["messages"][-1].content)
