"""Single-shot prefetch of all external data needed by specialists.

Runs once after the orchestrator and before the parallel fan-out:
  - ticker 90-day OHLC (yfinance, retried on rate limits)
  - ^VIX 5-day history (yfinance, retried on rate limits)
  - SEC EdgarBundle (CIK already resolved by orchestrator; one HTTP wave to SEC)

Both Price and Risk read `state["price_history"]`. Risk and Fundamentals read
`state["edgar_bundle"]`. Eliminates duplicate external requests and lets
specialists run as pure compute + LLM nodes.
"""

from __future__ import annotations

import time

import pandas as pd

from agents.yf_helpers import download_with_retry
from edgar import EdgarBundle, TickerNotFound, build_edgar_bundle


INTER_REQUEST_GAP_SECONDS = 1.0


def _safe_yf(ticker: str, period: str) -> pd.DataFrame:
    try:
        return download_with_retry(ticker, period=period, interval="1d")
    except Exception:
        return pd.DataFrame()


def _safe_edgar(ticker: str) -> EdgarBundle | None:
    try:
        return build_edgar_bundle(ticker)
    except TickerNotFound:
        return None
    except Exception:
        return None


def data_prefetch(state: dict) -> dict:
    ticker = state["ticker"]

    price_history = _safe_yf(ticker, period="90d")
    time.sleep(INTER_REQUEST_GAP_SECONDS)
    vix_history = _safe_yf("^VIX", period="5d")
    time.sleep(INTER_REQUEST_GAP_SECONDS)
    edgar_bundle = _safe_edgar(ticker)

    return {
        "price_history": price_history,
        "vix_history": vix_history,
        "edgar_bundle": edgar_bundle,
    }
