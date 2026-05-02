"""Macro specialist: FRED + Fear & Greed -> Sonnet."""

from __future__ import annotations

import requests

from agents import safe_parse_json
from state import AgentSignal


def _fetch_fred_series(series_id: str, api_key: str, limit: int = 5) -> list[dict]:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": limit,
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


def macro_agent(state: dict, clients, fred_key: str) -> dict:
    ticker = state["ticker"]
    raw = {
        "dxy_latest": None, "dxy_5d_change": None,
        "fed_funds_rate": None, "treasury_10y": None, "treasury_2y": None,
        "yield_curve_2s10s": None, "fear_greed_index": _fetch_fear_greed(),
    }
    degraded = False

    if not fred_key:
        degraded = True
    else:
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

    prompt = (
        f"You are a macro strategist analyzing {ticker}.\n\nData:\n"
        + "\n".join(f"- {k}: {v}" for k, v in raw.items())
        + ("\n\nFRED data unavailable; using only Fear & Greed.\n" if degraded else "")
        + "\nReference: rising DXY bearish for risk; high Fed funds bearish; "
          "inverted curve (2s10s<0) recessionary; Fear&Greed 0-25 extreme fear, 75-100 extreme greed.\n\n"
        "Respond with JSON ONLY:\n"
        '{"signal": "BULLISH"|"BEARISH"|"NEUTRAL", "confidence": 0.0..1.0, '
        '"summary": "one sentence", "section_markdown": "## Macro Backdrop\\n... 150-250 words ..."}'
    )
    try:
        resp = clients.reasoning.invoke(prompt)
        out = safe_parse_json(resp.content)
    except Exception as exc:
        return {"agent_signals": [AgentSignal(
            agent="macro", signal="NEUTRAL", confidence=0.0,
            summary="Macro LLM error",
            section_markdown="## Macro Backdrop\n_LLM unavailable._",
            raw_data=raw, degraded=True, error=str(exc)[:200],
        )]}

    return {"agent_signals": [AgentSignal(
        agent="macro",
        signal=out["signal"],
        confidence=float(out["confidence"]),
        summary=out["summary"],
        section_markdown=out.get("section_markdown") or "## Macro Backdrop\n_Section missing._",
        raw_data=raw,
        degraded=degraded,
        error=None,
    )]}
