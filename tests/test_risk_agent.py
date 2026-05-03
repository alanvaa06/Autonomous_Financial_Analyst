import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from agents.risk_agent import risk_agent
from edgar import EdgarBundle


def _price_df(n=120, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.012, n)
    prices = 100 * np.cumprod(1 + rets)
    return pd.DataFrame({"Close": prices})


def _bundle():
    return EdgarBundle(
        ticker="MSFT", cik="0000789019", company_name="MICROSOFT CORP",
        latest_10q=None, latest_10k=None,
        xbrl_facts={"facts": {"us-gaap": {
            "Revenues": {"units": {"USD": [
                {"end": "2025-09-30", "val": 70_000_000_000, "form": "10-Q"},
                {"end": "2024-09-30", "val": 65_000_000_000, "form": "10-Q"},
            ]}},
            "Liabilities": {"units": {"USD": [{"end": "2025-09-30", "val": 200e9, "form": "10-Q"}]}},
            "StockholdersEquity": {"units": {"USD": [{"end": "2025-09-30", "val": 250e9, "form": "10-Q"}]}},
        }}},
        mdna_text="", risk_factors_text="",
    )


def _clients(api_key="sk-ant-fake"):
    secret = MagicMock()
    secret.get_secret_value.return_value = api_key
    reasoning = MagicMock()
    reasoning.anthropic_api_key = secret
    return MagicMock(reasoning=reasoning)


def test_risk_forward_happy_path(monkeypatch):
    df = _price_df(120)
    monkeypatch.setattr(
        "agents.risk_agent.run_with_tools",
        lambda **kw: {
            "signal": "BULLISH", "confidence": 0.65,
            "summary": "Forward favorable; positioning medium.",
            "section_markdown": "## Risk Profile\nDetails.",
            "forward_risk_view": "favorable",
            "primary_risk_driver": "systemic_vol",
            "risk_decomposition": {"operating": "low", "balance_sheet": "low",
                                   "positioning": "medium", "systemic": "medium"},
            "vol_regime": "normal", "vix_regime": "normal",
            "key_metrics": {"annualized_vol_pct": 18.0, "max_drawdown_pct": -8.0,
                            "sharpe": 1.2, "vix": 16.0, "beta": 1.1,
                            "short_ratio": 2.5, "revenue_yoy_pct": 7.7,
                            "debt_to_equity": 0.8, "vol_percentile_1y": 50.0,
                            "trend_state": "above_sma50_and_sma200",
                            "drawdown_state": "near_peak", "fcf_margin_pct": 28.0,
                            "op_margin_yoy_bps": 50.0},
            "flags": ["forward_favorable"],
        },
    )
    out = risk_agent(
        {"ticker": "MSFT", "price_history": df, "edgar_bundle": _bundle(),
         "vix_history": pd.DataFrame({"Close": [16.0]})},
        _clients(),
    )
    sig = out["agent_signals"][0]
    assert sig["agent"] == "risk"
    assert sig["forward_risk_view"] == "favorable"
    assert sig["primary_risk_driver"] == "systemic_vol"
    assert sig["risk_decomposition"]["operating"] == "low"
    assert sig["raw_data"]["edgar_bundle_available"] is True
    assert sig["degraded"] is False


def test_risk_no_bundle_still_runs_with_price_only(monkeypatch):
    df = _price_df(120)
    monkeypatch.setattr(
        "agents.risk_agent.build_edgar_bundle",
        lambda t: (_ for _ in ()).throw(RuntimeError("no SEC")),
    )
    monkeypatch.setattr(
        "agents.risk_agent.run_with_tools",
        lambda **kw: {
            "signal": "NEUTRAL", "confidence": 0.45,
            "summary": "Forward fundamental data missing.",
            "section_markdown": "## Risk Profile\nDegraded forward dim.",
            "forward_risk_view": "mixed",
            "primary_risk_driver": "none",
            "risk_decomposition": {"operating": "medium", "balance_sheet": "medium",
                                   "positioning": "medium", "systemic": "medium"},
            "vol_regime": "normal", "vix_regime": "normal",
            "key_metrics": {}, "flags": ["forward_fundamentals_unavailable"],
        },
    )
    out = risk_agent(
        {"ticker": "FOREIGN", "price_history": df,
         "vix_history": pd.DataFrame({"Close": [16.0]})},
        _clients(),
    )
    sig = out["agent_signals"][0]
    assert sig["degraded"] is False
    assert sig["raw_data"]["edgar_bundle_available"] is False


def test_risk_no_price_history_degrades(monkeypatch):
    with patch(
        "agents.risk_agent.download_with_retry",
        return_value=pd.DataFrame(),
    ):
        out = risk_agent({"ticker": "ZZZZ"}, _clients())
    sig = out["agent_signals"][0]
    assert sig["degraded"] is True


def test_risk_llm_error_degrades(monkeypatch):
    df = _price_df(120)
    monkeypatch.setattr(
        "agents.risk_agent.run_with_tools",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    out = risk_agent(
        {"ticker": "MSFT", "price_history": df, "edgar_bundle": _bundle(),
         "vix_history": pd.DataFrame({"Close": [16.0]})},
        _clients(),
    )
    sig = out["agent_signals"][0]
    assert sig["degraded"] is True
    assert "Risk agent error" in sig["summary"]
