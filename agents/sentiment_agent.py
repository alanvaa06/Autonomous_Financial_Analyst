"""Sentiment specialist: Tavily news -> Haiku per-article scoring -> Sonnet rollup."""

from __future__ import annotations

from tavily import TavilyClient

from agents import safe_parse_json
from state import AgentSignal


def _degraded(reason: str, raw: dict | None = None, error: str | None = None) -> dict:
    return {"agent_signals": [AgentSignal(
        agent="sentiment", signal="NEUTRAL", confidence=0.0,
        summary=reason,
        section_markdown=f"## News & Sentiment\n_Unavailable: {reason}_",
        raw_data=raw or {}, degraded=True, error=error,
    )]}


def sentiment_agent(state: dict, clients, tavily_key: str) -> dict:
    ticker = state["ticker"]
    company = state.get("company_name") or ticker

    if not tavily_key:
        return _degraded("No Tavily key provided")

    try:
        tav = TavilyClient(api_key=tavily_key)
        query = f"{company} {ticker} stock news"
        result = tav.search(query=query, max_results=15, search_depth="basic", days=7)
        raw_articles = result.get("results", []) or []

        # Dedupe by URL, cap to 12
        seen = set()
        articles = []
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

        if not articles:
            return _degraded("No coverage found", raw={"article_count": 0})

        haiku_prompt = (
            "Rate each headline as 'positive', 'neutral', or 'negative' for the issuer's stock, "
            "then list 2-5 short driver phrases.\n\n"
            "Headlines:\n"
            + "\n".join(f"{i+1}. {a['title']} — {a['snippet']}" for i, a in enumerate(articles))
            + "\n\nRespond with JSON ONLY:\n"
              '{"per_article": ["positive"|"neutral"|"negative", ...], "drivers": ["...", ...]}'
        )
        haiku_resp = clients.fast.invoke(haiku_prompt)
        haiku_out = safe_parse_json(haiku_resp.content)
        per = haiku_out.get("per_article", []) or []
        pos = sum(1 for x in per if x == "positive")
        neg = sum(1 for x in per if x == "negative")
        neu = sum(1 for x in per if x == "neutral")

        sonnet_prompt = (
            f"You are a market sentiment analyst. Synthesize the news rollup for {ticker} ({company}).\n\n"
            f"Article count: {len(articles)}; positive: {pos}, neutral: {neu}, negative: {neg}.\n"
            f"Top drivers: {haiku_out.get('drivers', [])}\n\n"
            "Sample headlines:\n"
            + "\n".join(f"- {a['title']}" for a in articles[:5])
            + "\n\nRespond with JSON ONLY (no fences):\n"
              '{"signal": "BULLISH"|"BEARISH"|"NEUTRAL", "confidence": 0.0..1.0, '
              '"summary": "one sentence", "section_markdown": "## News & Sentiment\\n... 120-180 word section ..."}'
        )
        sonnet_resp = clients.reasoning.invoke(sonnet_prompt)
        out = safe_parse_json(sonnet_resp.content)

        return {"agent_signals": [AgentSignal(
            agent="sentiment",
            signal=out["signal"],
            confidence=float(out["confidence"]),
            summary=out["summary"],
            section_markdown=out.get("section_markdown") or "## News & Sentiment\n_Section missing._",
            raw_data={
                "article_count": len(articles),
                "positive_count": pos,
                "neutral_count": neu,
                "negative_count": neg,
                "drivers": haiku_out.get("drivers", []),
                "sample_headlines": [a["title"] for a in articles[:3]],
            },
            degraded=False,
            error=None,
        )]}
    except Exception as exc:
        return _degraded("Sentiment agent error", error=str(exc)[:200])
