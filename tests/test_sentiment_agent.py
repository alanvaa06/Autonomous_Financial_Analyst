from unittest.mock import MagicMock, patch

from agents.sentiment_agent import sentiment_agent


def _clients(api_key="sk-ant-fake"):
    secret = MagicMock()
    secret.get_secret_value.return_value = api_key
    reasoning = MagicMock()
    reasoning.anthropic_api_key = secret
    return MagicMock(reasoning=reasoning)


def _fake_tav(articles):
    fake_tav = MagicMock()
    fake_tav.search.return_value = {"results": articles}
    return fake_tav


def test_sentiment_happy_path(monkeypatch):
    fake_tav = _fake_tav([
        {"title": "MSFT raises guidance", "url": "https://prnewswire.com/x",
         "content": "Microsoft Corp raised FY26 guidance..."},
        {"title": "MSFT analyst upgrade", "url": "https://reuters.com/y",
         "content": "Goldman raised PT to 500..."},
        {"title": "MSFT signs AI partnership", "url": "https://wsj.com/z",
         "content": "Cloud unit lands new enterprise deal..."},
    ])
    monkeypatch.setattr(
        "agents.sentiment_agent.TavilyClient",
        lambda api_key: fake_tav,
    )
    monkeypatch.setattr(
        "agents.sentiment_agent.run_with_tools",
        lambda **kw: {
            "signal": "BULLISH", "confidence": 0.75,
            "summary": "Issuer raise + upgrade.",
            "section_markdown": "## News & Sentiment\nDetails.",
            "top_catalyst": "raised guidance",
            "key_metrics": {"article_count": 3, "positive_count": 3,
                            "neutral_count": 0, "negative_count": 0,
                            "recency_24h_count": 1, "source_mix": "primary_heavy"},
            "drivers_categorized": {"earnings": 2, "product": 1,
                                    "m&a": 0, "regulatory": 0,
                                    "insider": 0, "competitor": 0, "macro": 0},
            "flags": ["catalyst_present", "primary_source_present"],
        },
    )
    out = sentiment_agent(
        {"ticker": "MSFT", "company_name": "Microsoft Corp"},
        _clients(), tavily_key="tvly-fake",
    )
    sig = out["agent_signals"][0]
    assert sig["agent"] == "sentiment"
    assert sig["signal"] == "BULLISH"
    assert sig["top_catalyst"] == "raised guidance"
    assert sig["drivers_categorized"]["earnings"] == 2
    assert sig["degraded"] is False


def test_sentiment_no_tavily_key_degrades():
    out = sentiment_agent(
        {"ticker": "MSFT"}, _clients(), tavily_key="",
    )
    sig = out["agent_signals"][0]
    assert sig["degraded"] is True
    assert "Tavily" in sig["summary"]


def test_sentiment_no_articles_degrades(monkeypatch):
    fake_tav = _fake_tav([])
    monkeypatch.setattr(
        "agents.sentiment_agent.TavilyClient",
        lambda api_key: fake_tav,
    )
    out = sentiment_agent(
        {"ticker": "ZZZZ"}, _clients(), tavily_key="tvly-x",
    )
    sig = out["agent_signals"][0]
    assert sig["degraded"] is True
    assert "No coverage" in sig["summary"]


def test_sentiment_llm_error_degrades(monkeypatch):
    fake_tav = _fake_tav([
        {"title": "x", "url": "https://a/1", "content": "y"},
    ])
    monkeypatch.setattr(
        "agents.sentiment_agent.TavilyClient",
        lambda api_key: fake_tav,
    )
    monkeypatch.setattr(
        "agents.sentiment_agent.run_with_tools",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    out = sentiment_agent(
        {"ticker": "MSFT"}, _clients(), tavily_key="tvly-x",
    )
    sig = out["agent_signals"][0]
    assert sig["degraded"] is True
    assert "Sentiment LLM error" in sig["summary"]
