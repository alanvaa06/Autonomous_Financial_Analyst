"""Shared yfinance helpers — retry-with-backoff + curl_cffi browser impersonation.

Yahoo Finance throttles per IP and blocks non-browser User-Agents. On HF Spaces
(shared infrastructure) the IP is often already near or at the limit when our
request lands, AND Yahoo's bot detection rejects vanilla `requests`-style UAs
with `401 Invalid Crumb`. yfinance 1.3 supports passing a `curl_cffi` session
that impersonates a real browser TLS fingerprint + UA — that's what gets us
past the bot wall on HF.

Both `price_agent` and `risk_agent` issue `yf.download(...)` calls in the same
LangGraph superstep, so they hit the API in parallel. The retry layer uses
jittered exponential backoff so the second attempt is staggered and likelier
to land outside the throttle window.
"""

from __future__ import annotations

import random
import time

import pandas as pd
import yfinance as yf

try:
    from curl_cffi import requests as curl_requests
    _HAS_CURL_CFFI = True
except Exception:  # noqa: BLE001 — defensive: fall back to vanilla yf
    _HAS_CURL_CFFI = False

DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_BACKOFF = 5.0


def _is_rate_limit(exc: BaseException) -> bool:
    # Match by class name to avoid tight coupling to yfinance.exceptions module
    # path, which has shifted across yfinance versions.
    return "RateLimit" in type(exc).__name__


def _build_session():
    """Return a curl_cffi Session impersonating Chrome, or None when unavailable.

    yfinance 1.3 accepts a `session=` kwarg on `Ticker(...)` and on `download(...)`.
    A curl_cffi Session reproduces a real browser's TLS fingerprint + UA, which
    is what bypasses Yahoo's 401 Invalid-Crumb / bot wall on HF Spaces.
    """
    if not _HAS_CURL_CFFI:
        return None
    try:
        return curl_requests.Session(impersonate="chrome")
    except Exception:  # noqa: BLE001
        return None


def download_with_retry(
    ticker: str,
    *,
    period: str = "90d",
    interval: str = "1d",
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_backoff: float = DEFAULT_BASE_BACKOFF,
) -> pd.DataFrame:
    """Wrap `yf.download` with retry + jittered exponential backoff + browser session.

    Returns the (possibly empty) DataFrame on success or after all retries
    fail. Non-rate-limit exceptions propagate immediately.
    """
    session = _build_session()
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            kwargs = {"period": period, "interval": interval, "progress": False}
            if session is not None:
                kwargs["session"] = session
            df = yf.download(ticker, **kwargs)
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
    # squeeze() collapses a 1-element Series to a numpy scalar; guard with
    # an explicit Series check so .iloc[-1] is only called when available.
    if isinstance(close, pd.Series):
        return float(close.iloc[-1])
    return float(close)
