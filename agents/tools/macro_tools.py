"""On-demand tools for the Macro agent."""

from __future__ import annotations

import requests
import yfinance as yf

from agents import ToolDef
from agents.yf_helpers import download_with_retry


def _fetch_fred(series_id: str, api_key: str, periods: int = 12) -> list[dict]:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": periods,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return [
        {"date": o["date"], "value": float(o["value"])}
        for o in resp.json().get("observations", [])
        if o.get("value") and o["value"] != "."
    ]


def _classify_sector(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as exc:
        return {"error": str(exc)[:200]}
    return {
        "ticker": ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
    }


def _credit_spreads() -> dict:
    try:
        hyg = download_with_retry("HYG", period="30d", interval="1d")
        lqd = download_with_retry("LQD", period="30d", interval="1d")
    except Exception as exc:
        return {"error": str(exc)[:200]}
    if hyg.empty or lqd.empty:
        return {"error": "no credit ETF data"}
    hyg_close = float(hyg["Close"].squeeze().iloc[-1])
    lqd_close = float(lqd["Close"].squeeze().iloc[-1])
    return {
        "hyg_close": round(hyg_close, 2),
        "lqd_close": round(lqd_close, 2),
        "hyg_lqd_ratio": round(hyg_close / lqd_close, 4),
        "interpretation": (
            "Higher HYG/LQD ratio = risk-on (HY outperforming IG); "
            "lower = risk-off (IG outperforming HY, credit deterioration)."
        ),
    }


def build_macro_tools(*, fred_key: str) -> list[ToolDef]:

    def fred_handler(args: dict) -> dict:
        if not fred_key:
            return {"error": "fred_key not configured"}
        series_id = args.get("series_id", "")
        periods = int(args.get("periods", 12) or 12)
        try:
            obs = _fetch_fred(series_id, fred_key, periods=periods)
        except Exception as exc:
            return {"error": str(exc)[:200]}
        return {"series_id": series_id, "observations": obs}

    def sector_handler(args: dict) -> dict:
        return _classify_sector(args.get("ticker", ""))

    def spreads_handler(_args: dict) -> dict:
        return _credit_spreads()

    return [
        ToolDef(
            name="fetch_fred_series",
            description=(
                "Fetch recent observations for any FRED economic data series. "
                "Useful series: BAMLH0A0HYM2 (HY OAS), T10YIE (10Y breakeven "
                "inflation), DCOILWTICO (WTI crude), UNRATE (unemployment), "
                "CPIAUCSL (headline CPI), DGS30 (30Y Treasury). Returns "
                "newest-first observations."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "series_id": {"type": "string"},
                    "periods": {"type": "integer", "default": 12},
                },
                "required": ["series_id"],
            },
            handler=fred_handler,
        ),
        ToolDef(
            name="classify_ticker_sector",
            description=(
                "Return GICS sector + sub-industry for a ticker via "
                "yfinance.Ticker.info. Use to map the current macro regime to "
                "the ticker's specific sector exposure."
            ),
            input_schema={
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
            handler=sector_handler,
        ),
        ToolDef(
            name="fetch_credit_spreads",
            description=(
                "Return current HYG and LQD ETF closes plus the HYG/LQD ratio "
                "as a risk-on/off gauge. Use when FRED HY series is unavailable "
                "or you want a market-priced credit signal."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=spreads_handler,
        ),
    ]
