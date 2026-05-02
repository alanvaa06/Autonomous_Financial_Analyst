"""Shared yfinance helpers — retry-with-backoff on Yahoo rate limits.

Yahoo Finance throttles per IP. On HF Spaces (shared infrastructure) the IP is
often already near or at the limit when our request lands. yfinance 1.3 raises
`YFRateLimitError` on those responses.

Both `price_agent` and `risk_agent` issue `yf.download(...)` calls in the same
LangGraph superstep, so they hit the API in parallel. The retry layer here uses
jittered exponential backoff so the second attempt is staggered and likelier
to land outside the throttle window.
"""

from __future__ import annotations

import random
import time

import pandas as pd
import yfinance as yf

DEFAULT_MAX_RETRIES = 2
DEFAULT_BASE_BACKOFF = 2.0


def _is_rate_limit(exc: BaseException) -> bool:
    # Match by class name to avoid tight coupling to yfinance.exceptions module
    # path, which has shifted across yfinance versions.
    return "RateLimit" in type(exc).__name__


def download_with_retry(
    ticker: str,
    *,
    period: str = "90d",
    interval: str = "1d",
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_backoff: float = DEFAULT_BASE_BACKOFF,
) -> pd.DataFrame:
    """Wrap `yf.download` with retry + jittered exponential backoff on rate limits.

    Returns the (possibly empty) DataFrame on success or after all retries
    fail. Non-rate-limit exceptions propagate immediately.
    """
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            df = yf.download(
                ticker, period=period, interval=interval, progress=False
            )
            return df
        except Exception as exc:  # noqa: BLE001 — re-raised below or retried
            last_exc = exc
            if _is_rate_limit(exc) and attempt < max_retries:
                sleep_for = base_backoff * (2 ** attempt) + random.uniform(0, 0.75)
                time.sleep(sleep_for)
                continue
            raise
    # Unreachable: loop either returns or raises.
    raise last_exc  # type: ignore[misc]


def last_close(df: pd.DataFrame) -> float | None:
    """Return the last Close price as a plain float, or None if df is empty.

    yfinance 1.3 sometimes returns a MultiIndex DataFrame even for a single
    ticker. `.squeeze()` on the Close column collapses the 1-element second
    axis so that `.iloc[-1]` gives a scalar instead of a one-element Series
    (which `float(...)` warns about under future pandas).
    """
    if df is None or df.empty or "Close" not in df.columns:
        return None
    close = df["Close"].squeeze()
    return float(close.iloc[-1])
