# edgar.py
"""SEC EDGAR client for MarketMind v2 Fundamentals agent.

Replaces the deleted PDF/Chroma RAG pipeline. Provides:
  - resolve_ticker(ticker) -> (cik_padded, company_name)
  - fetch_company_facts(cik) -> dict (XBRL company facts)
  - fetch_latest_10q(cik) -> Filing | None
  - fetch_latest_10k(cik) -> Filing | None
  - extract_mdna(filing) -> str
  - build_edgar_bundle(ticker) -> EdgarBundle

All requests carry a `User-Agent` header per SEC policy. Per-process LRU cache
with TTL avoids hammering SEC; cache lives only in memory.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import requests

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

DEFAULT_USER_AGENT = "MarketMind/2.0 contact@marketmind.local"
SEC_POLITENESS_SLEEP = 0.1  # seconds between requests
HTTP_TIMEOUT = 15


class EdgarError(Exception):
    """Base for any EDGAR client failure."""


class TickerNotFound(EdgarError):
    """Ticker is not present in SEC's company tickers index."""


# -- ticker -> CIK -------------------------------------------------------------

# {ticker_upper: (cik_padded10, company_name, fetched_at)}
_CIK_CACHE: dict[str, tuple[str, str, float]] = {}
_CIK_TTL_SECONDS = 24 * 3600


def _now() -> float:
    return time.time()


def _user_agent() -> str:
    # Resolved at call time so an updated UA from the BYO-key panel is picked up.
    import os
    return os.environ.get("SEC_USER_AGENT", DEFAULT_USER_AGENT)


def _get(url: str, **kwargs) -> requests.Response:
    headers = kwargs.pop("headers", {}) or {}
    headers.setdefault("User-Agent", _user_agent())
    headers.setdefault("Accept", "application/json")
    time.sleep(SEC_POLITENESS_SLEEP)
    resp = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT, **kwargs)
    resp.raise_for_status()
    return resp


def _load_tickers_index() -> dict[str, tuple[str, str]]:
    """Return {ticker_upper: (cik_padded10, name)} from SEC's master file."""
    resp = _get(SEC_TICKERS_URL)
    out: dict[str, tuple[str, str]] = {}
    for entry in resp.json().values():
        ticker = str(entry["ticker"]).upper()
        cik_padded = str(entry["cik_str"]).zfill(10)
        out[ticker] = (cik_padded, entry["title"])
    return out


def resolve_ticker(ticker: str) -> tuple[str, str]:
    """Resolve a ticker to (CIK_padded_to_10, company_name).

    Raises TickerNotFound if SEC has no matching equity issuer (foreign issuers
    that file only as 20-F, ETFs, recent IPOs without CIKs, etc.).
    """
    key = ticker.strip().upper()
    cached = _CIK_CACHE.get(key)
    if cached and (_now() - cached[2]) < _CIK_TTL_SECONDS:
        return cached[0], cached[1]

    index = _load_tickers_index()
    if key not in index:
        raise TickerNotFound(f"{ticker} is not in SEC's company tickers index")
    cik, name = index[key]
    _CIK_CACHE[key] = (cik, name, _now())
    return cik, name


# -- XBRL company facts --------------------------------------------------------

# {cik: (data_dict, fetched_at)}
_FACTS_CACHE: dict[str, tuple[dict, float]] = {}
_FACTS_TTL_SECONDS = 6 * 3600


def fetch_company_facts(cik: str) -> dict:
    """Fetch XBRL company facts for a CIK. Cached per-process for 6h."""
    cached = _FACTS_CACHE.get(cik)
    if cached and (_now() - cached[1]) < _FACTS_TTL_SECONDS:
        return cached[0]
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    resp = _get(url)
    data = resp.json()
    _FACTS_CACHE[cik] = (data, _now())
    return data
