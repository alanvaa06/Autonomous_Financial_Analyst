from unittest.mock import MagicMock, patch

from agents.sentiment_agent import sentiment_agent


def _fake_tavily_results():
    return {
        "results": [
            {"title": "Microsoft Q1 beats estimates", "content": "Revenue rose 12%.", "url": "https://x.com/1"},
            {"title": "AI demand boosts cloud", "content": "Azure up 30%.", "url": "https://x.com/2"},
            {"title": "Regulator probes acquisition", "content": "EU launches inquiry.", "url": "https://x.com/3"},
        ]
    }


def test_sentiment_agent_happy():
    fake_tavily = MagicMock()
    fake_tavily.search.return_value = _fake_tavily_results()
    fake_haiku = MagicMock()
    fake_haiku.invoke.return_value = MagicMock(content='{"per_article": ["positive","positive","negative"], "drivers": ["earnings beat","cloud growth","regulatory"]}')
    fake_sonnet = MagicMock()
    fake_sonnet.invoke.return_value = MagicMock(content='{"signal": "BULLISH", "confidence": 0.65, "summary": "Mostly positive coverage", "section_markdown": "## News & Sentiment\\nLots of stuff."}')
    clients = MagicMock(reasoning=fake_sonnet, fast=fake_haiku)

    state = {"ticker": "MSFT", "company_name": "Microsoft Corp"}
    with patch("agents.sentiment_agent.TavilyClient", return_value=fake_tavily):
        out = sentiment_agent(state, clients, tavily_key="tvly-fake")

    sig = out["agent_signals"][0]
    assert sig["agent"] == "sentiment"
    assert sig["signal"] == "BULLISH"
    assert sig["raw_data"]["article_count"] == 3
    assert "positive_count" in sig["raw_data"]


def test_sentiment_agent_no_results():
    fake_tavily = MagicMock()
    fake_tavily.search.return_value = {"results": []}
    clients = MagicMock()

    with patch("agents.sentiment_agent.TavilyClient", return_value=fake_tavily):
        out = sentiment_agent({"ticker": "ZZZZ", "company_name": "Z"}, clients, tavily_key="tvly")

    sig = out["agent_signals"][0]
    assert sig["degraded"] is True
    assert sig["signal"] == "NEUTRAL"


def test_sentiment_agent_missing_key_degrades():
    out = sentiment_agent({"ticker": "MSFT", "company_name": "Microsoft"}, MagicMock(), tavily_key="")
    sig = out["agent_signals"][0]
    assert sig["degraded"] is True
