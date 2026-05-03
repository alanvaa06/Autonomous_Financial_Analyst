"""On-demand tools for the Fundamentals agent."""

from __future__ import annotations

from typing import Optional

import yfinance as yf

from agents import ToolDef
from edgar import EdgarBundle


# ---------------------------------------------------------------------------
# Internal helpers (testable)
# ---------------------------------------------------------------------------


def _fetch_xbrl_tag(bundle: EdgarBundle, tag_name: str, periods: int = 8) -> dict:
    """Pull recent observations for an arbitrary US-GAAP tag from the bundle."""
    units = (
        (bundle.xbrl_facts or {})
        .get("facts", {})
        .get("us-gaap", {})
        .get(tag_name, {})
        .get("units", {})
    )
    obs = units.get("USD") or units.get("USD/shares") or []
    obs_sorted = sorted(obs, key=lambda o: o.get("end", ""), reverse=True)[:periods]
    cleaned = [
        {"end": o.get("end"), "val": o.get("val"),
         "form": o.get("form"), "fp": o.get("fp")}
        for o in obs_sorted
    ]
    return {"tag": tag_name, "observations": cleaned}


def _fetch_segment_breakdown(bundle: EdgarBundle) -> dict:
    """Return segment-level RevenueFromContractWithCustomer observations.

    XBRL segment data is filed under several tag names; we try the most
    common: RevenueFromContractWithCustomerExcludingAssessedTax. If absent,
    return an explicit empty result so the LLM can adapt.
    """
    facts = (bundle.xbrl_facts or {}).get("facts", {}).get("us-gaap", {})
    candidates = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
    ]
    for tag in candidates:
        units = facts.get(tag, {}).get("units", {}).get("USD") or []
        if units:
            return {"tag": tag, "segments": units[:20]}
    return {"tag": None, "segments": []}


def _peer_multiples(peer_tickers: list[str]) -> dict:
    """Pull P/E, EV/EBITDA, P/S, P/B for a list of peers via yfinance.Ticker.info."""
    out: dict[str, dict] = {}
    for t in (peer_tickers or [])[:5]:
        try:
            info = yf.Ticker(t).info or {}
        except Exception:
            info = {}
        out[t] = {
            "trailing_pe": info.get("trailingPE"),
            "ev_to_ebitda": info.get("enterpriseToEbitda"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "price_to_book": info.get("priceToBook"),
        }
    return {"peers": out}


# ---------------------------------------------------------------------------
# Public: build ToolDef list bound to a session's bundle + api_key
# ---------------------------------------------------------------------------


def build_fundamentals_tools(
    *, bundle: Optional[EdgarBundle], api_key: str
) -> list[ToolDef]:
    """Construct the 3 ToolDef instances. `bundle` and `api_key` are bound
    via closures so the LLM-facing handlers take only their declared inputs."""

    def fetch_xbrl_tag_handler(args: dict) -> dict:
        if bundle is None:
            return {"error": "edgar_bundle unavailable"}
        return _fetch_xbrl_tag(bundle, args.get("tag_name", ""),
                               int(args.get("periods", 8) or 8))

    def fetch_segment_breakdown_handler(args: dict) -> dict:
        if bundle is None:
            return {"error": "edgar_bundle unavailable"}
        return _fetch_segment_breakdown(bundle)

    def peer_multiples_handler(args: dict) -> dict:
        peers = args.get("peer_tickers") or []
        if not isinstance(peers, list):
            return {"error": "peer_tickers must be a list"}
        return _peer_multiples(peers)

    return [
        ToolDef(
            name="fetch_xbrl_tag",
            description=(
                "Fetch recent quarterly observations for an arbitrary US-GAAP "
                "XBRL tag from the issuer's filings. Examples: "
                "ResearchAndDevelopmentExpense, OperatingCashFlowsContinuingOperations, "
                "CapitalExpenditures, ShareBasedCompensation. Returns up to "
                "`periods` most recent observations sourced from 10-Q/10-K."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "tag_name": {"type": "string"},
                    "periods": {"type": "integer", "default": 8},
                },
                "required": ["tag_name"],
            },
            handler=fetch_xbrl_tag_handler,
        ),
        ToolDef(
            name="fetch_segment_breakdown",
            description=(
                "Return segment-level revenue rows from XBRL when filed. Useful "
                "for mix-shift analysis. Returns empty list if the issuer does "
                "not file segment breakdowns under standard tags."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=fetch_segment_breakdown_handler,
        ),
        ToolDef(
            name="peer_multiples",
            description=(
                "Quick comparable multiples (P/E, EV/EBITDA, P/S, P/B) for up "
                "to 5 peer tickers via yfinance Ticker.info. Pick peers from "
                "MD&A or your own knowledge of the issuer's competitors."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "peer_tickers": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["peer_tickers"],
            },
            handler=peer_multiples_handler,
        ),
    ]
