"""On-demand tools for the Risk agent. All local computations on the
prefetched price_history DataFrame and edgar_bundle — no network calls."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from agents import ToolDef
from edgar import EdgarBundle


def _returns(df: pd.DataFrame) -> pd.Series:
    return df["Close"].squeeze().pct_change().dropna()


def _compute_var_es(df: pd.DataFrame, confidence: float = 0.95) -> dict:
    rets = _returns(df)
    if len(rets) < 20:
        return {"error": "insufficient return history"}
    quantile = float(np.quantile(rets, 1 - confidence))
    es = float(rets[rets <= quantile].mean())
    return {
        "confidence": confidence,
        "var_pct": round(quantile * 100, 3),
        "es_pct": round(es * 100, 3),
    }


def _decompose_drawdown(df: pd.DataFrame) -> dict:
    rets = _returns(df)
    if rets.empty:
        return {"error": "insufficient history"}
    cum = (1 + rets).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    current_dd = float(dd.iloc[-1])
    max_dd = float(dd.min())
    peak_idx = running_max.idxmax()
    last_idx = rets.index[-1]
    try:
        days_since_peak = int(last_idx - peak_idx)
    except Exception:
        days_since_peak = int(len(rets) - 1 - rets.index.get_loc(peak_idx))
    # Vol vs trend split: sum of negative returns since peak (trend) vs realized
    # vol over the same window (vol).
    since_peak = rets.loc[peak_idx:]
    trend_component = float(since_peak[since_peak < 0].sum())
    vol_component = float(since_peak.std() * np.sqrt(len(since_peak)))
    return {
        "current_drawdown_pct": round(current_dd * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "days_since_peak": days_since_peak,
        "trend_component_pct": round(trend_component * 100, 2),
        "vol_component_pct": round(vol_component * 100, 2),
    }


def _yoy_revenue_pct(bundle: EdgarBundle) -> Optional[float]:
    units = (
        (bundle.xbrl_facts or {})
        .get("facts", {}).get("us-gaap", {})
        .get("Revenues", {}).get("units", {}).get("USD") or []
    )
    if len(units) < 2:
        return None
    obs = sorted(units, key=lambda o: o.get("end", ""), reverse=True)
    latest = obs[0]
    latest_end = latest.get("end", "")
    if len(latest_end) < 10:
        return None
    target = f"{int(latest_end[:4]) - 1}{latest_end[4:]}"
    prior = next((o for o in obs if o.get("end") == target), None)
    if not prior or not prior.get("val"):
        return None
    return round((float(latest["val"]) - float(prior["val"])) / float(prior["val"]) * 100, 2)


def _forward_risk_attribution(
    df: pd.DataFrame, bundle: Optional[EdgarBundle]
) -> dict:
    out = {
        "operating": "unknown",
        "balance_sheet": "unknown",
        "positioning": "medium",
        "systemic": "medium",
    }
    if bundle is not None:
        # When a bundle is present, default fundamentals to "medium" so every
        # component has a concrete signal; specific data may sharpen the estimate.
        out["operating"] = "medium"
        out["balance_sheet"] = "medium"
        rev_yoy = _yoy_revenue_pct(bundle)
        if rev_yoy is not None:
            if rev_yoy < 0:
                out["operating"] = "high"
            elif rev_yoy < 5:
                out["operating"] = "medium"
            else:
                out["operating"] = "low"
        # Crude D/E read
        facts = (bundle.xbrl_facts or {}).get("facts", {}).get("us-gaap", {})
        liabilities_obs = facts.get("Liabilities", {}).get("units", {}).get("USD") or []
        equity_obs = facts.get("StockholdersEquity", {}).get("units", {}).get("USD") or []
        if liabilities_obs and equity_obs:
            try:
                d = float(liabilities_obs[0]["val"])
                e = float(equity_obs[0]["val"]) or 1e-9
                de = d / e
                if de < 1.0:
                    out["balance_sheet"] = "low"
                elif de < 2.5:
                    out["balance_sheet"] = "medium"
                else:
                    out["balance_sheet"] = "high"
            except Exception:
                pass
    # Positioning + systemic from price history (vol percentile proxy)
    if df is not None and not df.empty:
        rets = _returns(df)
        if len(rets) >= 30:
            ann_vol = float(rets.std() * np.sqrt(252)) * 100
            if ann_vol < 20:
                out["positioning"] = "low"
                out["systemic"] = "low"
            elif ann_vol < 40:
                out["positioning"] = "medium"
                out["systemic"] = "medium"
            else:
                out["positioning"] = "high"
                out["systemic"] = "high"
    return out


def build_risk_tools(
    *, price_history: Optional[pd.DataFrame], edgar_bundle: Optional[EdgarBundle]
) -> list[ToolDef]:

    def fra_handler(_args: dict) -> dict:
        return _forward_risk_attribution(price_history, edgar_bundle)

    def dd_handler(_args: dict) -> dict:
        if price_history is None or price_history.empty:
            return {"error": "price_history unavailable"}
        return _decompose_drawdown(price_history)

    def var_handler(args: dict) -> dict:
        if price_history is None or price_history.empty:
            return {"error": "price_history unavailable"}
        c = float(args.get("confidence", 0.95) or 0.95)
        return _compute_var_es(price_history, confidence=c)

    return [
        ToolDef(
            name="forward_risk_attribution",
            description=(
                "Decompose forward risk into {operating, balance_sheet, "
                "positioning, systemic} each in {low, medium, high, unknown}. "
                "Uses revenue YoY + leverage from EdgarBundle and vol regime "
                "from price history. Returns 'unknown' for fundamental "
                "components when EdgarBundle is missing."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=fra_handler,
        ),
        ToolDef(
            name="decompose_drawdown",
            description=(
                "Split current and max drawdown into trend vs volatility "
                "components and report days since peak. Local; no network."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=dd_handler,
        ),
        ToolDef(
            name="compute_var_es",
            description=(
                "Historical 1-day Value-at-Risk and Expected Shortfall on the "
                "price-history return series at the given confidence level "
                "(0.95 or 0.99 are typical). Both reported as negative "
                "percentages. Local; no network."
            ),
            input_schema={
                "type": "object",
                "properties": {"confidence": {"type": "number", "default": 0.95}},
            },
            handler=var_handler,
        ),
    ]
