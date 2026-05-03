"""Risk specialist (v2.1): forward-looking persona + few-shot + 3 tools.

Reads `state["price_history"]`, `state["vix_history"]`, and `state["edgar_bundle"]`
populated by data_prefetch. Falls back to `download_with_retry` and
`build_edgar_bundle` when missing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from agents import LLMClients, degraded_signal, run_with_tools
from agents.tools.risk_tools import build_risk_tools
from agents.yf_helpers import download_with_retry, last_close
from edgar import EdgarBundle, build_edgar_bundle
from state import AgentSignal


RISK_FREE_RATE = 0.04


PERSONA = (
    "You are an FRM Charterholder (Financial Risk Manager) and former PM at "
    "a long/short equity fund. You think in terms of risk-adjusted returns, "
    "drawdown discipline, and regime-conditional volatility — not absolute "
    "vol. You distinguish idiosyncratic risk (this name) from systemic risk "
    "(the tape) and you flag when correlation regimes are shifting."
)

METHODOLOGY = """
Methodology you apply (FORWARD-LOOKING drives the call; backward stats only calibrate confidence):

Forward (drives signal):
- Fundamental risk leading indicators: revenue YoY deceleration, op margin
  compression, leverage rising, FCF deterioration, accruals quality.
- Price-regime leading indicators: vol percentile (current vs trailing 1y),
  trend break (price vs SMA50/SMA200), drawdown state.

Backward (informational only):
- Sharpe, Sortino, Calmar, historical VaR, max drawdown, max 1-day drop.

Decomposition: operating / balance-sheet / positioning / systemic.
Distinguish idiosyncratic vs systemic in section_markdown.
""".strip()

FEWSHOT = """
Examples of correct reasoning:

Example 1 — forward-deteriorating despite OK trailing stats:
Trailing vol 22%, max DD -14%, Sharpe 0.8 (all benign). Forward: revenue
YoY -8% (decelerating), op margin -300bps YoY, D/E 1.8 rising, vol percentile
85th. Forward signals dominate; trailing OK is not a reprieve. Call: BEARISH 0.70.
forward_risk_view=deteriorating, primary_risk_driver=operating_deceleration.
risk_decomposition={operating: high, balance_sheet: medium, positioning: medium, systemic: medium}.

Example 2 — forward-favorable in noisy tape:
Trailing vol 35% (high), max DD -25% (large). Forward: revenue YoY +15%,
op margin +200bps YoY, D/E 0.6 stable, vol percentile dropping to 50th, price
above SMA50/SMA200. The tape is noisy but forward fundamentals + regime are
constructive. Call: BULLISH 0.65. forward_risk_view=favorable,
primary_risk_driver=systemic_vol (still elevated). risk_decomposition=
{operating: low, balance_sheet: low, positioning: medium, systemic: high}.

Example 3 — mixed, positioning is primary driver:
Trailing OK, fundamentals neutral (rev YoY +3%, margins flat). VIX 28 elevated,
beta 1.6, short_ratio 7.5 (squeeze risk). Call: NEUTRAL 0.55.
forward_risk_view=mixed, primary_risk_driver=positioning. risk_decomposition=
{operating: medium, balance_sheet: low, positioning: high, systemic: high}.
""".strip()

OUTPUT_SCHEMA = """
Respond with a single JSON object (no markdown fences) with EXACTLY these keys:
{
  "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence": 0.0..1.0,
  "summary": "one sentence stating forward_risk_view + primary_risk_driver",
  "section_markdown": "## Risk Profile\\n... 150-250 words distinguishing idio vs systemic ...",
  "forward_risk_view": "favorable" | "mixed" | "deteriorating" | "elevated",
  "primary_risk_driver": "operating_deceleration" | "margin_pressure" | "balance_sheet" | "systemic_vol" | "positioning" | "none",
  "risk_decomposition": {
    "operating": "low" | "medium" | "high",
    "balance_sheet": "low" | "medium" | "high",
    "positioning": "low" | "medium" | "high",
    "systemic": "low" | "medium" | "high"
  },
  "vol_regime": "compressed" | "normal" | "elevated" | "stress",
  "vix_regime": "low" | "normal" | "elevated" | "stress",
  "key_metrics": {
    "annualized_vol_pct": number,
    "max_drawdown_pct": number,
    "sharpe": number,
    "vix": number | null,
    "beta": number | null,
    "short_ratio": number | null,
    "revenue_yoy_pct": number | null,
    "op_margin_yoy_bps": number | null,
    "fcf_margin_pct": number | null,
    "debt_to_equity": number | null,
    "vol_percentile_1y": number | null,
    "trend_state": "above_sma50_and_sma200" | "below_sma50" | "below_both" | "unknown",
    "drawdown_state": "near_peak" | "moderate" | "deep" | "unknown"
  },
  "flags": ["string", ...]
}
""".strip()

GUARDRAILS = """
Constraints and guardrails:
- Forward-looking signals drive `signal`; backward stats only calibrate `confidence`.
- No "buy/sell" verbiage in section_markdown — use "constructive / cautious / negative" framings (risk-adjusted).
- section_markdown must be 150-250 words.
- Confidence ≤ 0.5 if forward fundamental data is missing (no edgar_bundle).
- Confidence ≤ 0.55 if returns history < 60 trading days.
- forward_risk_view AND primary_risk_driver MUST be set explicitly in summary.
- Distinguish idiosyncratic vs systemic risk in section_markdown.
""".strip()


def _build_system_prompt() -> str:
    return "\n\n".join([PERSONA, METHODOLOGY, FEWSHOT, OUTPUT_SCHEMA, GUARDRAILS])


def _trailing_stats(close: pd.Series) -> dict:
    rets = close.pct_change().dropna()
    if len(rets) < 5:
        return {}
    ann_vol = float(rets.std()) * np.sqrt(252) * 100
    cum = (1 + rets).cumprod()
    max_dd = float(((cum - cum.cummax()) / cum.cummax()).min()) * 100
    sharpe = float((rets.mean() * 252 - RISK_FREE_RATE) / (rets.std() * np.sqrt(252)))
    sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
    last = float(close.iloc[-1])
    if sma50 is not None and sma200 is not None and not pd.isna(sma50) and not pd.isna(sma200):
        if last >= sma50 and last >= sma200:
            trend = "above_sma50_and_sma200"
        elif last >= sma50:
            trend = "above_sma50_only"
        elif last < sma50 and last < sma200:
            trend = "below_both"
        else:
            trend = "below_sma50"
    else:
        trend = "unknown"
    # vol percentile vs rolling 60-day window of vol
    vol_pct_1y = None
    if len(rets) >= 60:
        rolling_vol = rets.rolling(20).std().dropna()
        if len(rolling_vol) > 0:
            current = float(rolling_vol.iloc[-1])
            vol_pct_1y = round(float((rolling_vol <= current).sum()) / len(rolling_vol) * 100, 1)
    # drawdown state from current vs max
    last_dd_pct = float(((cum.iloc[-1] - cum.cummax().iloc[-1]) / cum.cummax().iloc[-1])) * 100
    if last_dd_pct > -3:
        dd_state = "near_peak"
    elif last_dd_pct > -10:
        dd_state = "moderate"
    else:
        dd_state = "deep"
    return {
        "annualized_vol_pct": round(ann_vol, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "trend_state": trend,
        "drawdown_state": dd_state,
        "vol_percentile_1y": vol_pct_1y,
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
    end = latest.get("end", "")
    if len(end) < 10:
        return None
    target = f"{int(end[:4]) - 1}{end[4:]}"
    prior = next((o for o in obs if o.get("end") == target), None)
    if not prior or not prior.get("val"):
        return None
    return round((float(latest["val"]) - float(prior["val"])) / float(prior["val"]) * 100, 2)


def _forward_fundamentals(bundle: Optional[EdgarBundle]) -> dict:
    if bundle is None:
        return {"revenue_yoy_pct": None, "debt_to_equity": None}
    rev_yoy = _yoy_revenue_pct(bundle)
    facts = (bundle.xbrl_facts or {}).get("facts", {}).get("us-gaap", {})
    liab_obs = facts.get("Liabilities", {}).get("units", {}).get("USD") or []
    eq_obs = facts.get("StockholdersEquity", {}).get("units", {}).get("USD") or []
    de = None
    if liab_obs and eq_obs:
        try:
            d = float(liab_obs[0]["val"])
            e = float(eq_obs[0]["val"]) or 1e-9
            de = round(d / e, 3)
        except Exception:
            pass
    return {"revenue_yoy_pct": rev_yoy, "debt_to_equity": de}


def _build_user_prompt(ticker: str, raw: dict, has_bundle: bool, history_days: int) -> str:
    lines = [
        f"Ticker: {ticker}",
        f"Trading days of history: {history_days}",
        f"EdgarBundle present: {has_bundle}",
        "",
        "Pre-computed signals (always-on; forward-looking are PRIORITY):",
        *(f"- {k}: {v}" for k, v in raw.items()),
        "",
        "Apply your methodology and the 3 examples above. Forward-looking "
        "signals drive the call; backward stats calibrate confidence. Use "
        "tools (forward_risk_attribution, decompose_drawdown, compute_var_es) "
        "only if a step needs data not in the always-on payload. Then output "
        "the final JSON.",
    ]
    return "\n".join(lines)


def risk_agent(state: dict, clients: LLMClients) -> dict:
    ticker = state["ticker"]
    try:
        data = state.get("price_history")
        if data is None:
            data = download_with_retry(ticker, period="90d", interval="1d")
        if data is None or data.empty or "Close" not in data.columns:
            return degraded_signal(
                "risk", "Risk Profile", f"No price data for {ticker}",
            )

        close = data["Close"].squeeze()
        rets = close.pct_change().dropna()
        if len(rets) < 5:
            return degraded_signal("risk", "Risk Profile", "Insufficient return history")

        trailing = _trailing_stats(close)

        bundle = state.get("edgar_bundle")
        if bundle is None:
            try:
                bundle = build_edgar_bundle(ticker)
            except Exception:
                bundle = None

        forward = _forward_fundamentals(bundle)

        vix_df = state.get("vix_history")
        if vix_df is None:
            try:
                vix_df = download_with_retry("^VIX", period="5d", interval="1d")
            except Exception:
                vix_df = None
        vix_val = last_close(vix_df) if vix_df is not None else None
        vix = round(vix_val, 2) if vix_val is not None else None

        info = {}
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:
            info = {}

        raw = {
            **trailing,
            "vix": vix,
            "beta": info.get("beta"),
            "short_ratio": info.get("shortRatio"),
            **forward,
            "edgar_bundle_available": bundle is not None,
        }

        api_key = clients.reasoning.anthropic_api_key.get_secret_value()
        tools = build_risk_tools(price_history=data, edgar_bundle=bundle)

        out = run_with_tools(
            api_key=api_key,
            system_prompt=_build_system_prompt(),
            user_prompt=_build_user_prompt(ticker, raw, bundle is not None, len(close)),
            tools=tools,
            max_iterations=3,
            max_tokens=2000,
        )

        return {"agent_signals": [AgentSignal(
            agent="risk",
            signal=out.get("signal", "NEUTRAL"),
            confidence=float(out.get("confidence", 0.0) or 0.0),
            summary=out.get("summary", ""),
            section_markdown=out.get("section_markdown") or "## Risk Profile\n_Section missing._",
            raw_data=raw,
            degraded=False,
            error=None,
            forward_risk_view=out.get("forward_risk_view"),
            primary_risk_driver=out.get("primary_risk_driver"),
            risk_decomposition=out.get("risk_decomposition"),
            vol_regime=out.get("vol_regime"),
            vix_regime=out.get("vix_regime"),
            key_metrics=out.get("key_metrics"),
            flags=out.get("flags") or [],
        )]}
    except Exception as exc:
        return degraded_signal(
            "risk", "Risk Profile", "Risk agent error", error=str(exc)[:200],
        )
