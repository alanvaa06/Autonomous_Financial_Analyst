# tests/test_tools/test_risk_tools.py
import numpy as np
import pandas as pd

from agents.tools.risk_tools import build_risk_tools


def _rets_df(n=120, mu=0.0005, sigma=0.012, seed=1):
    rng = np.random.default_rng(seed)
    rets = rng.normal(mu, sigma, n)
    prices = 100 * np.cumprod(1 + rets)
    return pd.DataFrame({"Close": prices})


def test_compute_var_es_returns_negative_var_and_es():
    df = _rets_df(120)
    tools = build_risk_tools(price_history=df, edgar_bundle=None)
    var_tool = next(t for t in tools if t.name == "compute_var_es")
    out = var_tool.handler({"confidence": 0.95})
    assert "var_pct" in out
    assert "es_pct" in out
    assert out["var_pct"] <= 0
    assert out["es_pct"] <= out["var_pct"]


def test_decompose_drawdown_returns_components():
    df = _rets_df(120)
    tools = build_risk_tools(price_history=df, edgar_bundle=None)
    dd_tool = next(t for t in tools if t.name == "decompose_drawdown")
    out = dd_tool.handler({})
    assert "current_drawdown_pct" in out
    assert "max_drawdown_pct" in out
    assert "days_since_peak" in out


def test_forward_risk_attribution_uses_bundle_when_present():
    from edgar import EdgarBundle
    bundle = EdgarBundle(
        ticker="MSFT", cik="0000789019", company_name="MICROSOFT CORP",
        latest_10q=None, latest_10k=None,
        xbrl_facts={"facts": {"us-gaap": {
            "Revenues": {"units": {"USD": [
                {"end": "2025-09-30", "val": 70_000_000_000, "form": "10-Q"},
                {"end": "2024-09-30", "val": 65_000_000_000, "form": "10-Q"},
            ]}}
        }}},
        mdna_text="", risk_factors_text="",
    )
    df = _rets_df(120)
    tools = build_risk_tools(price_history=df, edgar_bundle=bundle)
    fra_tool = next(t for t in tools if t.name == "forward_risk_attribution")
    out = fra_tool.handler({})
    assert set(out.keys()) >= {"operating", "balance_sheet", "positioning", "systemic"}
    for v in out.values():
        assert v in {"low", "medium", "high"}


def test_forward_risk_attribution_no_bundle_returns_unknown_for_fundamentals():
    df = _rets_df(120)
    tools = build_risk_tools(price_history=df, edgar_bundle=None)
    fra_tool = next(t for t in tools if t.name == "forward_risk_attribution")
    out = fra_tool.handler({})
    assert out["operating"] in {"low", "medium", "high", "unknown"}
    assert out["balance_sheet"] in {"low", "medium", "high", "unknown"}


def test_build_risk_tools_returns_three():
    df = _rets_df(60)
    tools = build_risk_tools(price_history=df, edgar_bundle=None)
    names = [t.name for t in tools]
    assert names == ["forward_risk_attribution", "decompose_drawdown", "compute_var_es"]
