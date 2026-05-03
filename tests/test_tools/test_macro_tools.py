# tests/test_tools/test_macro_tools.py
import responses
from unittest.mock import MagicMock

from agents.tools.macro_tools import build_macro_tools


@responses.activate
def test_fetch_fred_series_returns_observations():
    responses.add(
        responses.GET,
        "https://api.stlouisfed.org/fred/series/observations",
        json={"observations": [
            {"date": "2025-10-01", "value": "5.25"},
            {"date": "2025-09-01", "value": "5.30"},
        ]},
    )
    tools = build_macro_tools(fred_key="frd-fake")
    fred_tool = next(t for t in tools if t.name == "fetch_fred_series")
    out = fred_tool.handler({"series_id": "BAMLH0A0HYM2", "periods": 5})
    assert out["series_id"] == "BAMLH0A0HYM2"
    assert len(out["observations"]) == 2
    assert out["observations"][0]["value"] == 5.25


def test_fetch_fred_series_no_key_returns_error():
    tools = build_macro_tools(fred_key="")
    fred_tool = next(t for t in tools if t.name == "fetch_fred_series")
    out = fred_tool.handler({"series_id": "DFF"})
    assert "error" in out


def test_classify_ticker_sector(monkeypatch):
    import agents.tools.macro_tools as mt

    fake_ticker = MagicMock()
    fake_ticker.info = {"sector": "Technology", "industry": "Software—Infrastructure"}
    monkeypatch.setattr(mt, "yf", MagicMock(Ticker=lambda t: fake_ticker))

    tools = mt.build_macro_tools(fred_key="x")
    sec_tool = next(t for t in tools if t.name == "classify_ticker_sector")
    out = sec_tool.handler({"ticker": "MSFT"})
    assert out["sector"] == "Technology"
    assert out["industry"] == "Software—Infrastructure"


def test_fetch_credit_spreads(monkeypatch):
    import pandas as pd
    import agents.tools.macro_tools as mt

    hyg_df = pd.DataFrame({"Close": [78.0, 78.5, 79.0]})
    lqd_df = pd.DataFrame({"Close": [105.0, 105.2, 105.5]})

    def fake_dl(ticker, **kwargs):
        return hyg_df if ticker == "HYG" else lqd_df

    monkeypatch.setattr(mt, "download_with_retry", fake_dl)

    tools = mt.build_macro_tools(fred_key="x")
    cs_tool = next(t for t in tools if t.name == "fetch_credit_spreads")
    out = cs_tool.handler({})
    assert "hyg_close" in out
    assert "lqd_close" in out
    assert "hyg_lqd_ratio" in out


def test_build_macro_tools_returns_three():
    tools = build_macro_tools(fred_key="x")
    names = [t.name for t in tools]
    assert names == ["fetch_fred_series", "classify_ticker_sector", "fetch_credit_spreads"]
