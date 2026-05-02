# agents/orchestrator.py
"""Orchestrator: validate input, resolve CIK, anchor the fan-out."""

from __future__ import annotations

import re

from edgar import TickerNotFound, resolve_ticker

VALID_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


def orchestrator(state: dict) -> dict:
    raw = (state.get("ticker") or "").strip().upper()
    if not VALID_TICKER_RE.match(raw):
        raise ValueError(f"invalid ticker: {raw!r}")

    update: dict = {"ticker": raw}
    try:
        cik, name = resolve_ticker(raw)
        update["cik"] = cik
        update["company_name"] = name
    except TickerNotFound:
        # Foreign issuer / ETF / pre-CIK IPO. Fundamentals agent will degrade.
        update["cik"] = None
        update["company_name"] = raw
    return update
