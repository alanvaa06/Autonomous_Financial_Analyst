"""Fundamentals specialist: SEC EDGAR XBRL + MD&A -> Sonnet section."""

from __future__ import annotations

from typing import Optional

from agents import safe_parse_json
from edgar import EdgarBundle, TickerNotFound, build_edgar_bundle
from state import AgentSignal


KEY_TAGS = [
    "Revenues",
    "GrossProfit",
    "OperatingIncomeLoss",
    "NetIncomeLoss",
    "EarningsPerShareDiluted",
    "Assets",
    "Liabilities",
    "StockholdersEquity",
    "CashAndCashEquivalentsAtCarryingValue",
]


def extract_latest_metric(facts: dict, tag: str) -> tuple[Optional[float], Optional[str]]:
    """Most recent USD observation for an XBRL tag, preferring 10-Q then 10-K."""
    units = (
        facts.get("facts", {}).get("us-gaap", {}).get(tag, {}).get("units", {})
    )
    obs = units.get("USD") or units.get("USD/shares") or []
    obs_sorted = sorted(obs, key=lambda o: o.get("end", ""), reverse=True)
    for o in obs_sorted:
        if o.get("form") in ("10-Q", "10-K"):
            return float(o["val"]), o.get("end")
    return None, None


def find_yoy_pair(facts: dict, tag: str) -> tuple[Optional[float], Optional[float]]:
    """Find the latest value and the prior-year same-period value for a tag."""
    units = (
        facts.get("facts", {}).get("us-gaap", {}).get(tag, {}).get("units", {})
    )
    obs = units.get("USD") or []
    obs_sorted = sorted(obs, key=lambda o: o.get("end", ""), reverse=True)
    if not obs_sorted:
        return None, None
    latest = obs_sorted[0]
    latest_end = latest.get("end", "")
    if len(latest_end) < 10:
        return float(latest["val"]), None
    target_prior = f"{int(latest_end[:4]) - 1}{latest_end[4:]}"
    prior = next((o for o in obs_sorted if o.get("end") == target_prior), None)
    return float(latest["val"]), float(prior["val"]) if prior else None


def yoy_delta_pct(current: float, prior: float) -> Optional[float]:
    if not prior:
        return None
    return round((current - prior) / prior * 100, 2)


def _degraded(reason: str, raw: dict | None = None, error: str | None = None) -> dict:
    return {"agent_signals": [AgentSignal(
        agent="fundamentals", signal="NEUTRAL", confidence=0.0,
        summary=reason,
        section_markdown=f"## Fundamentals\n_Unavailable: {reason}_",
        raw_data=raw or {}, degraded=True, error=error,
    )]}


def fundamentals_agent(state: dict, clients) -> dict:
    ticker = state["ticker"]
    try:
        bundle: EdgarBundle = build_edgar_bundle(ticker)
    except TickerNotFound:
        return _degraded("Fundamentals unavailable — no SEC filings for this ticker")
    except Exception as exc:
        return _degraded("Fundamentals fetch error", error=str(exc)[:200])

    raw: dict = {"company_name": bundle.company_name, "cik": bundle.cik}
    for tag in KEY_TAGS:
        cur, end = extract_latest_metric(bundle.xbrl_facts, tag)
        raw[f"{tag}_latest"] = cur
        raw[f"{tag}_period_end"] = end
        latest, prior = find_yoy_pair(bundle.xbrl_facts, tag)
        if latest is not None and prior is not None:
            raw[f"{tag}_yoy_pct"] = yoy_delta_pct(latest, prior)

    raw["revenue_yoy_pct"] = raw.get("Revenues_yoy_pct")
    raw["mdna_excerpt"] = bundle.mdna_text[:2000]
    raw["latest_10q_filed"] = bundle.latest_10q.filing_date if bundle.latest_10q else None
    raw["latest_10k_filed"] = bundle.latest_10k.filing_date if bundle.latest_10k else None

    prompt = (
        f"You are a fundamentals analyst. Analyze {ticker} ({bundle.company_name}).\n\n"
        f"Latest XBRL metrics (USD):\n"
        + "\n".join(
            f"- {k}: {raw.get(f'{k}_latest')} as of {raw.get(f'{k}_period_end')} "
            f"(YoY {raw.get(f'{k}_yoy_pct')}%)"
            for k in KEY_TAGS
        )
        + f"\n\nMD&A excerpt (10-Q):\n{raw['mdna_excerpt'][:3000]}\n\n"
        "Cover: revenue trend, margin trajectory, balance-sheet health, MD&A signals.\n\n"
        "Respond with JSON ONLY:\n"
        '{"signal": "BULLISH"|"BEARISH"|"NEUTRAL", "confidence": 0.0..1.0, '
        '"summary": "one sentence", "section_markdown": "## Fundamentals\\n... 200-300 word section ..."}'
    )
    try:
        resp = clients.reasoning.invoke(prompt)
        out = safe_parse_json(resp.content)
    except Exception as exc:
        return _degraded("LLM error in fundamentals", raw=raw, error=str(exc)[:200])

    return {"agent_signals": [AgentSignal(
        agent="fundamentals",
        signal=out["signal"],
        confidence=float(out["confidence"]),
        summary=out["summary"],
        section_markdown=out.get("section_markdown") or "## Fundamentals\n_Section missing._",
        raw_data=raw,
        degraded=False,
        error=None,
    )]}
