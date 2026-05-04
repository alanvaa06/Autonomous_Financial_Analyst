from unittest.mock import MagicMock, patch

from agents.fundamentals_agent import fundamentals_agent
from edgar import EdgarBundle, TickerNotFound


def _bundle(facts=None, mdna="MD&A text.", rf="Risk text."):
    return EdgarBundle(
        ticker="MSFT", cik="0000789019", company_name="MICROSOFT CORP",
        latest_10q=None, latest_10k=None,
        xbrl_facts=facts or {},
        mdna_text=mdna, risk_factors_text=rf,
    )


def _clients(api_key="sk-ant-fake"):
    secret = MagicMock()
    secret.get_secret_value.return_value = api_key
    reasoning = MagicMock()
    reasoning.anthropic_api_key = secret
    return MagicMock(reasoning=reasoning)


def test_fundamentals_reads_bundle_from_state(monkeypatch):
    bundle = _bundle({"facts": {"us-gaap": {}}})

    def fake_run(api_key, system_prompt, user_prompt, tools, **kwargs):
        return {
            "signal": "BULLISH", "confidence": 0.7,
            "summary": "Margins expanding.",
            "section_markdown": "## Fundamentals\nDetails.",
            "key_metrics": {"op_margin_pct": 38.0},
            "flags": ["margin_expansion"],
        }

    monkeypatch.setattr("agents.fundamentals_agent.run_with_tools", fake_run)

    out = fundamentals_agent({"ticker": "MSFT", "edgar_bundle": bundle}, _clients())
    sig = out["agent_signals"][0]
    assert sig["agent"] == "fundamentals"
    assert sig["signal"] == "BULLISH"
    assert sig["key_metrics"]["op_margin_pct"] == 38.0
    assert "margin_expansion" in sig["flags"]
    assert sig["degraded"] is False


def test_fundamentals_falls_back_to_build_when_state_bundle_missing(monkeypatch):
    bundle = _bundle()
    with patch(
        "agents.fundamentals_agent.build_edgar_bundle",
        return_value=bundle,
    ), patch(
        "agents.fundamentals_agent.run_with_tools",
        return_value={"signal": "NEUTRAL", "confidence": 0.5,
                      "summary": "OK", "section_markdown": "## Fundamentals\nx",
                      "key_metrics": {}, "flags": []},
    ):
        out = fundamentals_agent({"ticker": "MSFT"}, _clients())
    assert out["agent_signals"][0]["signal"] == "NEUTRAL"


def test_fundamentals_ticker_not_in_sec_degrades():
    with patch("agents.fundamentals_agent.build_edgar_bundle",
               side_effect=TickerNotFound("nope")):
        out = fundamentals_agent({"ticker": "FOREIGN"}, _clients())
    sig = out["agent_signals"][0]
    assert sig["degraded"] is True
    assert "no sec filings" in sig["summary"].lower()


def test_fundamentals_llm_error_degrades(monkeypatch):
    bundle = _bundle()
    monkeypatch.setattr(
        "agents.fundamentals_agent.run_with_tools",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("anthropic 500")),
    )
    out = fundamentals_agent({"ticker": "MSFT", "edgar_bundle": bundle}, _clients())
    sig = out["agent_signals"][0]
    assert sig["degraded"] is True
    assert "LLM error" in sig["summary"]


def test_key_metrics_resolves_revenue_via_asc606_tag():
    """AMZN-style: revenue is filed under RevenueFromContractWithCustomerExcludingAssessedTax."""
    from agents.fundamentals_agent import _key_metrics_from_facts
    facts = {"facts": {"us-gaap": {
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
            {"end": "2026-03-31", "val": 187_000_000_000, "fp": "Q1", "form": "10-Q", "filed": "2026-05-01"},
            {"end": "2025-03-31", "val": 162_000_000_000, "fp": "Q1", "form": "10-Q", "filed": "2025-05-02"},
        ]}},
        "OperatingIncomeLoss": {"units": {"USD": [
            {"end": "2026-03-31", "val": 24_000_000_000, "fp": "Q1", "form": "10-Q", "filed": "2026-05-01"},
        ]}},
    }}}
    km = _key_metrics_from_facts(facts)
    assert km["revenue_latest_usd"] == 187_000_000_000
    # op_margin_pct should now compute (was None pre-fix because rev was None).
    assert km["op_margin_pct"] is not None
