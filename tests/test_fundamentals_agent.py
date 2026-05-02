from unittest.mock import MagicMock, patch

from agents.fundamentals_agent import (
    extract_latest_metric,
    yoy_delta_pct,
    fundamentals_agent,
)
from edgar import EdgarBundle, Filing, TickerNotFound


def _facts(values):
    return {"facts": {"us-gaap": {"Revenues": {"units": {"USD": values}}}}}


def test_extract_latest_metric_returns_most_recent_form10q():
    facts = _facts([
        {"end": "2025-09-30", "val": 70_000_000_000, "form": "10-Q"},
        {"end": "2024-09-30", "val": 65_000_000_000, "form": "10-Q"},
    ])
    val, end = extract_latest_metric(facts, "Revenues")
    assert val == 70_000_000_000 and end == "2025-09-30"


def test_yoy_delta_pct():
    assert yoy_delta_pct(110.0, 100.0) == 10.0
    assert yoy_delta_pct(90.0, 100.0) == -10.0
    assert yoy_delta_pct(100.0, 0.0) is None


def test_fundamentals_agent_happy():
    bundle = EdgarBundle(
        ticker="MSFT",
        cik="0000789019",
        company_name="MICROSOFT CORP",
        latest_10q=Filing("789019", "acc", "10-Q", "2025-10-30", "2025-09-30", "q.htm"),
        latest_10k=Filing("789019", "acc-k", "10-K", "2025-07-31", "2025-06-30", "k.htm"),
        xbrl_facts=_facts([
            {"end": "2025-09-30", "val": 70_000_000_000, "form": "10-Q"},
            {"end": "2024-09-30", "val": 65_000_000_000, "form": "10-Q"},
        ]),
        mdna_text="Revenue grew 12% YoY driven by Cloud and AI services.",
    )
    fake_sonnet = MagicMock()
    fake_sonnet.invoke.return_value = MagicMock(content='{"signal": "BULLISH", "confidence": 0.7, "summary": "Strong fundamentals", "section_markdown": "## Fundamentals\\nDetails."}')
    clients = MagicMock(reasoning=fake_sonnet)

    with patch("agents.fundamentals_agent.build_edgar_bundle", return_value=bundle):
        out = fundamentals_agent({"ticker": "MSFT"}, clients)

    sig = out["agent_signals"][0]
    assert sig["agent"] == "fundamentals"
    assert sig["signal"] == "BULLISH"
    assert sig["raw_data"]["revenue_yoy_pct"] is not None
    assert sig["degraded"] is False


def test_fundamentals_agent_ticker_not_in_sec():
    clients = MagicMock()
    with patch("agents.fundamentals_agent.build_edgar_bundle", side_effect=TickerNotFound("nope")):
        out = fundamentals_agent({"ticker": "FOREIGN"}, clients)
    sig = out["agent_signals"][0]
    assert sig["degraded"] is True
    assert "no sec filings" in sig["summary"].lower()
