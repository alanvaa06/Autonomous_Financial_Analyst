# tests/test_tools/test_sentiment_tools.py
from unittest.mock import MagicMock, patch

from agents.tools.sentiment_tools import build_sentiment_tools


def test_fetch_press_releases_uses_targeted_query():
    fake_tav = MagicMock()
    fake_tav.search.return_value = {"results": [
        {"title": "MSFT 8-K filed", "url": "https://prnewswire.com/x", "content": "..."}
    ]}
    with patch("agents.tools.sentiment_tools.TavilyClient", return_value=fake_tav):
        tools = build_sentiment_tools(tavily_key="tvly-fake", api_key="sk-fake")
    pr_tool = next(t for t in tools if t.name == "fetch_press_releases")
    out = pr_tool.handler({"ticker": "MSFT", "days": 14})
    assert "results" in out
    assert "site:prnewswire.com" in fake_tav.search.call_args.kwargs["query"]


def test_fetch_analyst_actions_query_shape():
    fake_tav = MagicMock()
    fake_tav.search.return_value = {"results": []}
    with patch("agents.tools.sentiment_tools.TavilyClient", return_value=fake_tav):
        tools = build_sentiment_tools(tavily_key="tvly-fake", api_key="sk-fake")
    aa_tool = next(t for t in tools if t.name == "fetch_analyst_actions")
    out = aa_tool.handler({"ticker": "NVDA"})
    assert "results" in out
    q = fake_tav.search.call_args.kwargs["query"]
    assert "NVDA" in q
    assert "analyst" in q.lower()


def test_categorize_drivers_calls_haiku(monkeypatch):
    fake_haiku = MagicMock()
    fake_haiku.invoke.return_value = MagicMock(
        content='{"earnings": 2, "regulatory": 1, "product": 0, "m&a": 0, '
                '"insider": 0, "competitor": 0, "macro": 0}'
    )
    monkeypatch.setattr(
        "agents.tools.sentiment_tools._build_haiku",
        lambda api_key: fake_haiku,
    )
    tools = build_sentiment_tools(tavily_key="x", api_key="sk-fake")
    cat_tool = next(t for t in tools if t.name == "categorize_drivers")
    out = cat_tool.handler({"drivers": ["earnings beat", "raised guidance", "EU probe"]})
    assert out["earnings"] == 2
    assert out["regulatory"] == 1


def test_build_sentiment_tools_returns_three():
    tools = build_sentiment_tools(tavily_key="x", api_key="y")
    names = [t.name for t in tools]
    assert names == ["fetch_press_releases", "fetch_analyst_actions", "categorize_drivers"]


def test_fetch_press_releases_no_key_returns_error():
    tools = build_sentiment_tools(tavily_key="", api_key="x")
    pr_tool = next(t for t in tools if t.name == "fetch_press_releases")
    out = pr_tool.handler({"ticker": "MSFT"})
    assert "error" in out
