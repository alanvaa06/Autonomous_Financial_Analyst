# tests/test_tools/test_fundamentals_tools.py
from unittest.mock import patch

from agents.tools.fundamentals_tools import (
    build_fundamentals_tools,
    _fetch_xbrl_tag,
    _fetch_segment_breakdown,
)
from edgar import EdgarBundle


def _bundle_with(facts: dict) -> EdgarBundle:
    return EdgarBundle(
        ticker="MSFT", cik="0000789019", company_name="MICROSOFT CORP",
        latest_10q=None, latest_10k=None,
        xbrl_facts=facts, mdna_text="", risk_factors_text="",
    )


def test_fetch_xbrl_tag_returns_recent_observations():
    bundle = _bundle_with({
        "facts": {"us-gaap": {"ResearchAndDevelopmentExpense": {"units": {"USD": [
            {"end": "2025-09-30", "val": 7_000_000_000, "form": "10-Q"},
            {"end": "2025-06-30", "val": 6_800_000_000, "form": "10-Q"},
            {"end": "2025-03-31", "val": 6_500_000_000, "form": "10-Q"},
        ]}}}}
    })
    out = _fetch_xbrl_tag(bundle, "ResearchAndDevelopmentExpense", periods=8)
    assert out["tag"] == "ResearchAndDevelopmentExpense"
    assert len(out["observations"]) == 3
    assert out["observations"][0]["val"] == 7_000_000_000


def test_fetch_xbrl_tag_unknown_tag_returns_empty():
    bundle = _bundle_with({"facts": {"us-gaap": {}}})
    out = _fetch_xbrl_tag(bundle, "MissingTag")
    assert out["observations"] == []


def test_fetch_segment_breakdown_returns_segments():
    bundle = _bundle_with({
        "facts": {"us-gaap": {"RevenueFromContractWithCustomerExcludingAssessedTax": {
            "units": {"USD": [
                {"end": "2025-06-30", "val": 50_000_000_000, "form": "10-K", "fp": "FY",
                 "frame": "CY2025", "label": "Cloud"},
                {"end": "2025-06-30", "val": 30_000_000_000, "form": "10-K", "fp": "FY",
                 "frame": "CY2025", "label": "Productivity"},
            ]}
        }}}
    })
    out = _fetch_segment_breakdown(bundle)
    assert "segments" in out


def test_build_fundamentals_tools_returns_three():
    tools = build_fundamentals_tools(bundle=_bundle_with({}), api_key="sk-fake")
    names = [t.name for t in tools]
    assert names == ["fetch_xbrl_tag", "fetch_segment_breakdown", "peer_multiples"]
    for t in tools:
        assert t.input_schema["type"] == "object"
        assert callable(t.handler)


def test_peer_multiples_handler_calls_yfinance(monkeypatch):
    from unittest.mock import MagicMock
    import agents.tools.fundamentals_tools as ft

    fake_info = {"trailingPE": 32.0, "enterpriseToEbitda": 22.5,
                 "priceToSalesTrailing12Months": 11.0, "priceToBook": 14.0}
    fake_ticker = MagicMock()
    fake_ticker.info = fake_info
    monkeypatch.setattr(ft, "yf", MagicMock(Ticker=lambda t: fake_ticker))

    tools = ft.build_fundamentals_tools(bundle=_bundle_with({}), api_key="sk-fake")
    peer_tool = next(t for t in tools if t.name == "peer_multiples")
    out = peer_tool.handler({"peer_tickers": ["AAPL", "GOOGL"]})
    assert "peers" in out
    assert "AAPL" in out["peers"]
    assert out["peers"]["AAPL"]["trailing_pe"] == 32.0
