"""Risk specialist: yfinance returns + VIX -> Sonnet."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yfinance as yf

from agents import safe_parse_json
from state import AgentSignal

RISK_FREE_RATE = 0.04


def _degraded(reason: str, raw: dict | None = None, error: str | None = None) -> dict:
    return {"agent_signals": [AgentSignal(
        agent="risk", signal="NEUTRAL", confidence=0.0,
        summary=reason,
        section_markdown=f"## Risk Profile\n_Unavailable: {reason}_",
        raw_data=raw or {}, degraded=True, error=error,
    )]}


def risk_agent(state: dict, clients) -> dict:
    ticker = state["ticker"]
    try:
        data = yf.download(ticker, period="90d", interval="1d", progress=False)
        if data.empty or "Close" not in data.columns:
            return _degraded(f"No price data for {ticker}")

        close = data["Close"].squeeze()
        rets = close.pct_change().dropna()
        if len(rets) < 5:
            return _degraded("Insufficient return history")

        ann_vol = float(rets.std()) * np.sqrt(252) * 100
        cum = (1 + rets).cumprod()
        max_dd = float(((cum - cum.cummax()) / cum.cummax()).min()) * 100
        sharpe = float((rets.mean() * 252 - RISK_FREE_RATE) / (rets.std() * np.sqrt(252)))

        vix_df = yf.download("^VIX", period="5d", interval="1d", progress=False)
        vix = (
            round(float(vix_df["Close"].iloc[-1]), 2)
            if not vix_df.empty else None
        )

        info = {}
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:
            info = {}

        raw = {
            "annualized_vol_pct": round(ann_vol, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 2),
            "vix": vix,
            "beta": info.get("beta"),
            "short_ratio": info.get("shortRatio"),
        }

        prompt = (
            f"You are a risk analyst. Assess risk-adjusted attractiveness for {ticker}.\n\n"
            f"Data:\n{json.dumps(raw, indent=2)}\n\n"
            "Reference: high vol + negative Sharpe = bearish risk; low vol + positive Sharpe = bullish risk; "
            "VIX > 25 = elevated stress; beta > 1.3 = high market sensitivity.\n\n"
            "Respond with JSON ONLY:\n"
            '{"signal": "BULLISH"|"BEARISH"|"NEUTRAL", "confidence": 0.0..1.0, '
            '"summary": "one sentence", "section_markdown": "## Risk Profile\\n... 150-250 words ..."}'
        )
        resp = clients.reasoning.invoke(prompt)
        out = safe_parse_json(resp.content)

        return {"agent_signals": [AgentSignal(
            agent="risk",
            signal=out["signal"],
            confidence=float(out["confidence"]),
            summary=out["summary"],
            section_markdown=out.get("section_markdown") or "## Risk Profile\n_Section missing._",
            raw_data=raw,
            degraded=False,
            error=None,
        )]}
    except Exception as exc:
        return _degraded("Risk agent error", error=str(exc)[:200])
