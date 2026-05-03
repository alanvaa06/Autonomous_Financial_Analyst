"""Macro specialist (v2.1): persona + cross-asset framework + CoT + 3 tools."""

from __future__ import annotations

import requests

from agents import LLMClients, degraded_signal, run_with_tools
from agents.tools.macro_tools import build_macro_tools
from state import AgentSignal


PERSONA = (
    "You are a senior global macro strategist, CFA Charterholder, with 15 "
    "years on the rates and FX desk of a major investment bank. You read "
    "cross-asset signals — DXY, yield curve, credit spreads, commodities, "
    "positioning — and synthesize them into a regime call (risk-on/off, "
    "reflation/disinflation, growth-scare) that you map to specific equity "
    "sector implications. You're skeptical of single-print headlines and "
    "prefer trend-confirmed moves."
)

METHODOLOGY = """
Methodology you apply:
- Yield curve (2s10s sign + slope; bull-steepening vs bear-flattening)
- Real rates impact (FF − headline CPI proxy)
- DXY transmission (rising DXY bearish for non-USD revenue exposure)
- Credit spreads (HY OAS / IG OAS ratio)
- Commodity regime (energy + base metals leading inflation)
- Sentiment positioning (Fear & Greed contrarian / confirming)
""".strip()

COT = """
Reason step-by-step. For each step, state the data point + inference.

1. Rates regime — Fed funds level, real rate proxy (FF − recent CPI)
2. Yield curve shape — 2s10s sign + magnitude; recession signal?
3. USD direction — DXY level, 5d trend; risk-asset implication
4. Credit / liquidity — HY spreads (use fetch_fred_series for BAMLH0A0HYM2 if needed)
5. Inflation signal — breakevens, commodities (tool if needed)
6. Regime classification — pick exactly one of {risk-on, risk-off, reflation, disinflation, stagflation, neutral}
7. Ticker sector mapping — call classify_ticker_sector once, map sector to regime impact
8. Integrated call — directional impact on this specific ticker

Tool-call rules:
- Use tools only when a step requires data not in always-on payload.
- Cap: max 3 tool calls per run.
- After 3 tool calls (or when you have enough), produce final JSON.
""".strip()

OUTPUT_SCHEMA = """
Respond with a single JSON object (no markdown fences) with EXACTLY these keys:
{
  "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence": 0.0..1.0,
  "summary": "one sentence stating regime + impact direction",
  "section_markdown": "## Macro Backdrop\\n... 150-250 words ...",
  "regime": "risk-on" | "risk-off" | "reflation" | "disinflation" | "stagflation" | "neutral",
  "yield_curve_state": "steep" | "flat" | "inverted",
  "ticker_exposure": "high" | "medium" | "low",
  "key_metrics": {
    "dxy_latest": number | null,
    "dxy_5d_change": number | null,
    "fed_funds_rate": number | null,
    "yield_curve_2s10s": number | null,
    "fear_greed_index": number | null,
    "real_rate_proxy": number | null
  },
  "flags": ["string", ...]
}
""".strip()

GUARDRAILS = """
Constraints and guardrails:
- No "buy/sell" verbiage in section_markdown — use "supportive / mixed / headwind" framings.
- section_markdown must be 150-250 words.
- Confidence ≤ 0.5 if FRED key absent (only Fear & Greed available).
- Confidence ≤ 0.6 if yield_curve_2s10s is null.
- regime MUST be one of the 6 enum values; default neutral on indeterminate.
""".strip()


def _fetch_fred_series(series_id: str, api_key: str, periods: int = 5) -> list[dict]:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id, "api_key": api_key,
        "file_type": "json", "sort_order": "desc", "limit": periods,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return [
        {"date": o["date"], "value": float(o["value"])}
        for o in resp.json().get("observations", [])
        if o.get("value") and o["value"] != "."
    ]


def _fetch_fear_greed() -> int | None:
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        r.raise_for_status()
        return int(r.json()["data"][0]["value"])
    except Exception:
        return None


def _gather_always_on(fred_key: str) -> tuple[dict, bool]:
    raw = {
        "dxy_latest": None, "dxy_5d_change": None,
        "fed_funds_rate": None, "treasury_10y": None, "treasury_2y": None,
        "yield_curve_2s10s": None, "fear_greed_index": _fetch_fear_greed(),
    }
    degraded = False
    if not fred_key:
        return raw, True
    try:
        dxy = _fetch_fred_series("DTWEXBGS", fred_key)
        ff = _fetch_fred_series("DFF", fred_key)
        t10 = _fetch_fred_series("DGS10", fred_key)
        t2 = _fetch_fred_series("DGS2", fred_key)
        if dxy:
            raw["dxy_latest"] = dxy[0]["value"]
            if len(dxy) >= 2:
                raw["dxy_5d_change"] = round(dxy[0]["value"] - dxy[-1]["value"], 2)
        if ff:
            raw["fed_funds_rate"] = ff[0]["value"]
        if t10:
            raw["treasury_10y"] = t10[0]["value"]
        if t2:
            raw["treasury_2y"] = t2[0]["value"]
        if raw["treasury_10y"] is not None and raw["treasury_2y"] is not None:
            raw["yield_curve_2s10s"] = round(raw["treasury_10y"] - raw["treasury_2y"], 3)
    except Exception:
        degraded = True
    return raw, degraded


def _build_system_prompt() -> str:
    return "\n\n".join([PERSONA, METHODOLOGY, OUTPUT_SCHEMA, GUARDRAILS, COT])


def _build_user_prompt(ticker: str, raw: dict, degraded: bool) -> str:
    parts = [
        f"Ticker: {ticker}",
        "",
        "Pre-fetched macro data (always-on):",
        *(f"- {k}: {v}" for k, v in raw.items()),
    ]
    if degraded:
        parts += ["", "NOTE: FRED data unavailable; only Fear & Greed reliable. Mark `degraded=true` upstream."]
    parts += [
        "",
        "Run your 8-step chain of thought, then output the final JSON.",
    ]
    return "\n".join(parts)


def macro_agent(state: dict, clients: LLMClients, fred_key: str) -> dict:
    ticker = state["ticker"]
    raw, degraded = _gather_always_on(fred_key)

    api_key = clients.reasoning.anthropic_api_key.get_secret_value()
    tools = build_macro_tools(fred_key=fred_key)

    try:
        out = run_with_tools(
            api_key=api_key,
            system_prompt=_build_system_prompt(),
            user_prompt=_build_user_prompt(ticker, raw, degraded),
            tools=tools,
            max_iterations=3,
            max_tokens=1800,
        )
    except Exception as exc:
        return degraded_signal(
            "macro", "Macro Backdrop", "Macro LLM error",
            raw=raw, error=str(exc)[:200],
        )

    return {"agent_signals": [AgentSignal(
        agent="macro",
        signal=out.get("signal", "NEUTRAL"),
        confidence=float(out.get("confidence", 0.0) or 0.0),
        summary=out.get("summary", ""),
        section_markdown=out.get("section_markdown") or "## Macro Backdrop\n_Section missing._",
        raw_data=raw,
        degraded=degraded,
        error=None,
        regime=out.get("regime"),
        yield_curve_state=out.get("yield_curve_state"),
        ticker_exposure=out.get("ticker_exposure"),
        key_metrics=out.get("key_metrics"),
        flags=out.get("flags") or [],
    )]}
