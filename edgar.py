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

DEFAULT_USER_AGENT = "MarketMind/2.1 alanvaa.06@gmail.com"
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


# -- Filings (10-Q / 10-K) -----------------------------------------------------


@dataclass
class Filing:
    cik: str
    accession: str
    form: str
    filing_date: str
    report_date: str
    primary_document: str

    @property
    def primary_url(self) -> str:
        accession_clean = self.accession.replace("-", "")
        return (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(self.cik)}/{accession_clean}/{self.primary_document}"
        )


_SUBMISSIONS_CACHE: dict[str, tuple[dict, float]] = {}
_SUBMISSIONS_TTL_SECONDS = 6 * 3600


def _fetch_submissions(cik: str) -> dict:
    cached = _SUBMISSIONS_CACHE.get(cik)
    if cached and (_now() - cached[1]) < _SUBMISSIONS_TTL_SECONDS:
        return cached[0]
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = _get(url).json()
    _SUBMISSIONS_CACHE[cik] = (data, _now())
    return data


def _latest_filing_of(cik: str, form_target: str) -> Optional[Filing]:
    data = _fetch_submissions(cik)
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    for i, form in enumerate(forms):
        if form == form_target:
            return Filing(
                cik=cik,
                accession=recent["accessionNumber"][i],
                form=form,
                filing_date=recent["filingDate"][i],
                report_date=recent["reportDate"][i],
                primary_document=recent["primaryDocument"][i],
            )
    return None


def fetch_latest_10q(cik: str) -> Optional[Filing]:
    return _latest_filing_of(cik, "10-Q")


def fetch_latest_10k(cik: str) -> Optional[Filing]:
    return _latest_filing_of(cik, "10-K")


# -- MD&A extraction -----------------------------------------------------------

import re
from bs4 import BeautifulSoup


_MDNA_HEADING_RE = re.compile(
    r"item\s*2\b.*?management.{0,30}discussion",
    re.IGNORECASE | re.DOTALL,
)
_NEXT_ITEM_RE = re.compile(r"item\s*[3-9]\b", re.IGNORECASE)


def extract_mdna_from_html(html: str, max_chars: int = 8000) -> str:
    """Extract Item 2 (MD&A) text from a 10-Q filing's primary HTML document.

    Strategy:
      1. Strip tags via BeautifulSoup, preserving paragraph spacing.
      2. Locate the Item 2 heading by regex.
      3. Slice from there to the next Item N heading (3..9) or end of document.
      4. Cap at `max_chars`.
    """
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    start = _MDNA_HEADING_RE.search(text)
    if not start:
        return ""

    after = text[start.end():]
    end_match = _NEXT_ITEM_RE.search(after)
    body = after[: end_match.start()] if end_match else after
    body = body.strip()
    if len(body) > max_chars:
        body = body[:max_chars]
    return body


def fetch_filing_html(filing: Filing) -> str:
    """Download the primary document HTML for a Filing."""
    return _get(filing.primary_url, headers={"Accept": "text/html"}).text


def extract_mdna(filing: Filing, max_chars: int = 8000) -> str:
    """Convenience: download and extract MD&A in one shot."""
    return extract_mdna_from_html(fetch_filing_html(filing), max_chars=max_chars)


# -- Risk Factors extraction (10-K Item 1A) ------------------------------------

_RISK_FACTORS_HEADING_RE = re.compile(
    r"item\s*1a\b.*?risk\s*factors?",
    re.IGNORECASE | re.DOTALL,
)
_AFTER_RISK_ITEM_RE = re.compile(r"item\s*(1b|2)\b", re.IGNORECASE)


def extract_risk_factors_from_html(html: str, max_chars: int = 8000) -> str:
    """Extract Item 1A (Risk Factors) text from a 10-K filing's primary HTML.

    Same strategy as `extract_mdna_from_html`: BeautifulSoup strip, locate
    `Item 1A. Risk Factors` heading, slice to the next item (1B or 2), cap.
    Returns empty string when the heading isn't present (issuers that file
    only 20-F or 40-F won't have this section).
    """
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    start = _RISK_FACTORS_HEADING_RE.search(text)
    if not start:
        return ""

    after = text[start.end():]
    end_match = _AFTER_RISK_ITEM_RE.search(after)
    body = after[: end_match.start()] if end_match else after
    body = body.strip()
    if len(body) > max_chars:
        body = body[:max_chars]
    return body


def extract_risk_factors(filing: Filing, max_chars: int = 8000) -> str:
    """Convenience: download a 10-K and extract Item 1A in one shot."""
    return extract_risk_factors_from_html(fetch_filing_html(filing), max_chars=max_chars)


# -- Aggregator ----------------------------------------------------------------


@dataclass
class EdgarBundle:
    ticker: str
    cik: str
    company_name: str
    latest_10q: Optional[Filing]
    latest_10k: Optional[Filing]
    xbrl_facts: dict
    mdna_text: str
    risk_factors_text: str = ""
    fetched_at: float = field(default_factory=_now)


def build_edgar_bundle(ticker: str) -> EdgarBundle:
    """One-shot bundle for the Fundamentals agent.

    Raises TickerNotFound if SEC has no record. Other errors propagate from the
    individual fetchers. The Fundamentals agent is responsible for catching
    those and producing a degraded AgentSignal.

    Risk Factors are pulled from the 10-K (annual). MD&A is pulled from the
    most recent 10-Q (quarterly). Either may be empty if the filing is missing
    or the section heading isn't present.
    """
    cik, name = resolve_ticker(ticker)
    f10q = fetch_latest_10q(cik)
    f10k = fetch_latest_10k(cik)
    facts = fetch_company_facts(cik)
    mdna = extract_mdna(f10q) if f10q else ""
    risk_factors = extract_risk_factors(f10k) if f10k else ""
    return EdgarBundle(
        ticker=ticker.strip().upper(),
        cik=cik,
        company_name=name,
        latest_10q=f10q,
        latest_10k=f10k,
        xbrl_facts=facts,
        mdna_text=mdna,
        risk_factors_text=risk_factors,
    )
