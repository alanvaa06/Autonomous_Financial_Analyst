"""Single-shot yfinance prefetch.

Runs after the orchestrator and before the parallel fan-out. Downloads the
ticker's 90-day OHLC and the VIX 5-day window once each, with a small gap
between the two requests to avoid hammering Yahoo. The Price and Risk
specialists then read from `state["price_history"]` / `state["vix_history"]`
instead of issuing their own downloads — this halves the per-run yfinance
call count and is the difference between green and degraded runs on
HF Space's shared (heavily-throttled) outbound IPs.

If both downloads fail, the state still advances with empty DataFrames, and
the downstream specialists will degrade cleanly.
"""

from __future__ import annotations

import time

import pandas as pd

from agents.yf_helpers import download_with_retry


INTER_REQUEST_GAP_SECONDS = 1.0


def _safe_download(ticker: str, period: str) -> pd.DataFrame:
    try:
        return download_with_retry(ticker, period=period, interval="1d")
    except Exception:
        # Caller treats empty DataFrame as "data unavailable".
        return pd.DataFrame()


def data_prefetch(state: dict) -> dict:
    ticker = state["ticker"]
    price_history = _safe_download(ticker, period="90d")
    # Don't immediately fire the next request; give Yahoo a moment.
    time.sleep(INTER_REQUEST_GAP_SECONDS)
    vix_history = _safe_download("^VIX", period="5d")
    return {
        "price_history": price_history,
        "vix_history": vix_history,
    }
