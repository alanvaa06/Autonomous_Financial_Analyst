"""Sentiment specialist (v2.1): persona + source-quality methodology + few-shot + 3 tools."""

from __future__ import annotations

import logging

from tavily import TavilyClient

from agents import LLMClients, degraded_signal, run_with_tools
from agents.tools.sentiment_tools import build_sentiment_tools
from state import AgentSignal

logger = logging.getLogger("marketmind.sentiment_agent")


PERSONA = (
    "You are a senior buy-side equity analyst with a CFA Charter and a side "
    "specialty in behavioral finance and market-sentiment analysis. You "
    "weigh primary sources (issuer press releases, regulatory filings) over "
    "wire pickups, you discount social-media noise, and you separate "
    "fact-driven catalysts from narrative-driven momentum."
)

METHODOLOGY = """
Methodology you apply:
- Source-quality weighting (primary > tier-1 wire > tier-2 wire > aggregator/social)
- Catalyst classification (earnings, M&A, regulatory, product, insider, competitor, macro)
- Magnitude over count (one major filing > many recycled headlines)
- Recency decay (48h ×2, 7d ×1, >7d ×0.5)
- Narrative vs fundamental check (is the move grounded in disclosure or just chatter?)
- Contrarian flag (extreme one-sided coverage often precedes a fade)
""".strip()

FEWSHOT = """
Examples of correct reasoning:

Example 1 — strong fundamental catalyst:
12 articles. 8 positive (issuer 8-K announcing AI partnership + raised guidance,
2 wires + analyst upgrades), 3 neutral, 1 negative. Top catalyst: "Q3 guidance raise".
Primary sources present, recency 48h. Call: BULLISH 0.75. top_catalyst:
"raised Q3 guidance + AI partnership 8-K". Flag: catalyst_present, primary_source_present.

Example 2 — narrative-driven, fade:
9 articles. 7 positive (all aggregator headlines on "AI rally"), 2 neutral, 0 negative.
No issuer press release, no analyst action, no 10-Q/K. Recency 7d. Call: NEUTRAL 0.40.
top_catalyst: "general AI tape (no issuer-specific catalyst)". Flag: narrative_driven,
no_primary_source.

Example 3 — bearish overhang:
10 articles. 1 positive, 2 neutral, 7 negative (DOJ probe, FTC inquiry, customer
class action). Multiple primary regulatory sources, recency 48h. Call: BEARISH 0.78.
top_catalyst: "DOJ antitrust probe disclosed in 8-K". Flag: catalyst_present,
regulatory_overhang, primary_source_present.
""".strip()

OUTPUT_SCHEMA = """
Respond with a single JSON object (no markdown fences) with EXACTLY these keys:
{
  "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence": 0.0..1.0,
  "summary": "one sentence (≤25 words)",
  "section_markdown": "## News & Sentiment\\n... 120-180 words ...",
  "top_catalyst": "single specific event driving sentiment",
  "key_metrics": {
    "article_count": number,
    "positive_count": number,
    "neutral_count": number,
    "negative_count": number,
    "recency_24h_count": number | null,
    "source_mix": "primary_heavy" | "wire_heavy" | "aggregator_heavy" | "mixed"
  },
  "drivers_categorized": {
    "earnings": number, "m&a": number, "regulatory": number, "product": number,
    "insider": number, "competitor": number, "macro": number
  },
  "flags": ["string", ...]
}
""".strip()

GUARDRAILS = """
Constraints and guardrails:
- No "buy/sell" verbiage in section_markdown — use "supportive / mixed / unfavorable" framings.
- section_markdown must be 120-180 words.
- Confidence ≤ 0.4 if article_count < 3 (force `sparse_coverage` flag).
- Confidence ≤ 0.5 if 0 primary sources AND signal is BULLISH.
- top_catalyst must reference a specific event (not "general sentiment").
- Cite ≥1 specific headline in section_markdown.
""".strip()


def _build_system_prompt() -> str:
    return "\n\n".join([PERSONA, METHODOLOGY, FEWSHOT, OUTPUT_SCHEMA, GUARDRAILS])


def _gather_news(tavily_key: str, ticker: str, company: str) -> list[dict]:
    if not tavily_key:
        return []
    try:
        tav = TavilyClient(api_key=tavily_key)
        result = tav.search(
            query=f"{company} {ticker} stock news",
            max_results=15, search_depth="basic", days=7,
        )
    except Exception:
        return []
    raw_articles = result.get("results", []) or []
    seen, articles = set(), []
    for a in raw_articles:
        url = a.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        articles.append({
            "title": (a.get("title") or "").strip(),
            "snippet": (a.get("content") or "")[:280],
            "url": url,
        })
        if len(articles) >= 12:
            break
    return articles


def _build_user_prompt(ticker: str, company: str, articles: list[dict]) -> str:
    parts = [
        f"Ticker: {ticker}",
        f"Issuer: {company}",
        f"Article count: {len(articles)}",
        "",
        "Headlines (most recent first):",
    ]
    for i, a in enumerate(articles[:12]):
        parts.append(f"{i+1}. {a['title']} — {a['snippet']}")
    parts += [
        "",
        "Apply your methodology and the 3 examples above. Use tools "
        "(fetch_press_releases for primary sources, fetch_analyst_actions for "
        "consensus drift, categorize_drivers for taxonomy bucketing) only if "
        "a step needs data not in the always-on payload. Then output the "
        "final JSON.",
    ]
    return "\n".join(parts)


def sentiment_agent(state: dict, clients: LLMClients, tavily_key: str) -> dict:
    ticker = state["ticker"]
    company = state.get("company_name") or ticker

    if not tavily_key:
        return degraded_signal(
            "sentiment", "News & Sentiment", "No Tavily key provided",
        )

    articles = _gather_news(tavily_key, ticker, company)
    if not articles:
        return degraded_signal(
            "sentiment", "News & Sentiment", "No coverage found",
            raw={"article_count": 0},
        )

    api_key = clients.reasoning.anthropic_api_key.get_secret_value()
    tools = build_sentiment_tools(tavily_key=tavily_key, api_key=api_key)

    try:
        out = run_with_tools(
            api_key=api_key,
            system_prompt=_build_system_prompt(),
            user_prompt=_build_user_prompt(ticker, company, articles),
            tools=tools,
            max_iterations=3,
            max_tokens=1800,
        )
    except Exception as exc:
        logger.exception("sentiment: run_with_tools failed")
        return degraded_signal(
            "sentiment", "News & Sentiment", "Sentiment LLM error",
            raw={"article_count": len(articles)}, error=str(exc)[:200],
        )

    return {"agent_signals": [AgentSignal(
        agent="sentiment",
        signal=out.get("signal", "NEUTRAL"),
        confidence=float(out.get("confidence", 0.0) or 0.0),
        summary=out.get("summary", ""),
        section_markdown=out.get("section_markdown") or "## News & Sentiment\n_Section missing._",
        raw_data={
            "article_count": len(articles),
            "sample_headlines": [a["title"] for a in articles[:3]],
        },
        degraded=False,
        error=None,
        top_catalyst=out.get("top_catalyst"),
        key_metrics=out.get("key_metrics"),
        drivers_categorized=out.get("drivers_categorized"),
        flags=out.get("flags") or [],
    )]}
