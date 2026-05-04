"""Fundamentals specialist (v2.1): persona + DuPont + CoT + 3 tools.

Reads `state["edgar_bundle"]` populated by data_prefetch; falls back to
`build_edgar_bundle(ticker)` if missing (preserves single-call test paths).
"""

from __future__ import annotations

import logging

from typing import Optional

from agents import LLMClients, degraded_signal, run_with_tools
from agents.tools.fundamentals_tools import build_fundamentals_tools
from edgar import EdgarBundle, TickerNotFound, build_edgar_bundle
from state import AgentSignal

logger = logging.getLogger("marketmind.fundamentals_agent")


PERSONA = (
    "You are a CFA Charterholder Senior equity research analyst with deep "
    "expertise on US large-cap equities. You understand accounting "
    "conventions (US GAAP and IFRS) and extract key insights from 10-Q/10-K. "
    "Your judgment is objective and skeptic but flexible enough to identify "
    "where MD&A differ from fundamentals."
)

METHODOLOGY = """
Methodology you apply:
- DuPont decomposition (ROE = Net Margin × Asset Turnover × Equity Multiplier)
- Quality-of-earnings (operating cash flow vs net income; accruals ratio = (NI − CFO) / Avg Assets)
- Operating leverage (Δ%OpInc / Δ%Rev; >1 = positive leverage)
- Cash Conversion Cycle (DIO + DSO − DPO) for working-capital-heavy issuers
- Segment growth attribution when 10-K segment table is filed
""".strip()

COT = """
Reason step-by-step. For each step, state the metric value and the inference.
1. Revenue trajectory — YoY %, segment mix if available
2. Margin path — gross → operating → net, direction + drivers
3. Balance sheet health — leverage (D/E), liquidity (current ratio), working capital
4. Earnings quality — operating cash flow vs net income; accruals ratio
5. Operating leverage — Δ%OpInc / Δ%Rev
6. MD&A signals — growth drivers cited, risks acknowledged
7. Risk Factors — material new disclosures vs prior 10-K
8. Integrated call — bullish / bearish / neutral + confidence

Tool-call rules:
- Use tools only when a step requires data not in always-on payload.
- Cap: max 3 tool calls per run.
- After 3 tool calls (or when you have enough), produce final JSON.
""".strip()

OUTPUT_SCHEMA = """
Respond with a single JSON object (no markdown fences) with EXACTLY these keys:
{
  "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence": 0.0..1.0,
  "summary": "one sentence (≤25 words)",
  "section_markdown": "## Fundamentals\\n... 200-300 words ...",
  "key_metrics": {
    "roe_pct": number,
    "op_margin_pct": number,
    "op_margin_delta_yoy_bps": number,
    "fcf_margin_pct": number,
    "debt_to_equity": number,
    "current_ratio": number,
    "accruals_ratio_pct": number,
    "eps_yoy_pct": number
  },
  "flags": ["string", ...]
}
""".strip()

GUARDRAILS = """
Constraints and guardrails:
- No "buy/sell" verbiage in section_markdown — use "favorable / cautious / unfavorable" framings.
- section_markdown must be 200-300 words.
- Confidence ≤ 0.6 if latest 10-Q is older than 100 days, MD&A is empty, or
  fewer than 3 quarters of history are available.
- Confidence ≤ 0.4 if revenue YoY % is unavailable.
""".strip()


def _build_system_prompt() -> str:
    return "\n\n".join([
        PERSONA,
        METHODOLOGY,
        OUTPUT_SCHEMA,
        GUARDRAILS,
        COT,
    ])


def _key_metrics_from_facts(facts: dict) -> dict:
    """Compute the deterministic key_metrics block injected into the user prompt
    so the LLM has these numbers without using a tool call."""
    g = (facts or {}).get("facts", {}).get("us-gaap", {})

    def _latest(tag: str) -> tuple[Optional[float], Optional[str]]:
        units = g.get(tag, {}).get("units", {})
        obs = units.get("USD") or units.get("USD/shares") or []
        for o in sorted(obs, key=lambda x: x.get("end", ""), reverse=True):
            if o.get("form") in ("10-Q", "10-K"):
                return float(o["val"]), o.get("end")
        return None, None

    rev, _ = _latest("Revenues")
    op_inc, _ = _latest("OperatingIncomeLoss")
    net_inc, _ = _latest("NetIncomeLoss")
    eps, _ = _latest("EarningsPerShareDiluted")
    assets, _ = _latest("Assets")
    liab, _ = _latest("Liabilities")
    equity, _ = _latest("StockholdersEquity")

    def _pct(n, d):
        if n is None or d in (None, 0):
            return None
        return round(n / d * 100, 2)

    return {
        "revenue_latest_usd": rev,
        "op_margin_pct": _pct(op_inc, rev),
        "net_margin_pct": _pct(net_inc, rev),
        "eps_diluted": eps,
        "debt_to_equity": round(liab / equity, 3) if (liab and equity) else None,
    }


def _build_user_prompt(ticker: str, bundle: EdgarBundle) -> str:
    km = _key_metrics_from_facts(bundle.xbrl_facts or {})
    mdna = (bundle.mdna_text or "")[:8000]
    rf = (bundle.risk_factors_text or "")[:4000]
    parts = [
        f"Ticker: {ticker}",
        f"Issuer: {bundle.company_name} (CIK {bundle.cik})",
        f"Latest 10-Q filed: {bundle.latest_10q.filing_date if bundle.latest_10q else 'N/A'}",
        f"Latest 10-K filed: {bundle.latest_10k.filing_date if bundle.latest_10k else 'N/A'}",
        "",
        "Pre-computed key metrics (always-on):",
        *(f"- {k}: {v}" for k, v in km.items()),
    ]
    if mdna:
        parts += ["", "MD&A excerpt (10-Q):", mdna]
    if rf:
        parts += ["", "Risk Factors excerpt (10-K Item 1A):", rf]
    parts += [
        "",
        "Run your 8-step chain of thought, then output the final JSON.",
    ]
    return "\n".join(parts)


def fundamentals_agent(state: dict, clients: LLMClients) -> dict:
    ticker = state["ticker"]

    bundle = state.get("edgar_bundle")
    if bundle is None:
        try:
            bundle = build_edgar_bundle(ticker)
        except TickerNotFound:
            return degraded_signal(
                "fundamentals", "Fundamentals",
                "Fundamentals unavailable — no SEC filings for this ticker",
            )
        except Exception as exc:
            logger.exception("fundamentals: edgar fetch failed")
            return degraded_signal(
                "fundamentals", "Fundamentals",
                "Fundamentals fetch error", error=str(exc)[:200],
            )

    api_key = clients.reasoning.anthropic_api_key.get_secret_value()
    tools = build_fundamentals_tools(bundle=bundle, api_key=api_key)

    try:
        out = run_with_tools(
            api_key=api_key,
            system_prompt=_build_system_prompt(),
            user_prompt=_build_user_prompt(ticker, bundle),
            tools=tools,
            max_iterations=3,
            max_tokens=2000,
        )
    except Exception as exc:
        logger.exception("fundamentals: run_with_tools failed")
        return degraded_signal(
            "fundamentals", "Fundamentals",
            "LLM error in fundamentals", error=str(exc)[:200],
        )

    return {"agent_signals": [AgentSignal(
        agent="fundamentals",
        signal=out.get("signal", "NEUTRAL"),
        confidence=float(out.get("confidence", 0.0) or 0.0),
        summary=out.get("summary", ""),
        section_markdown=out.get("section_markdown") or "## Fundamentals\n_Section missing._",
        raw_data={"company_name": bundle.company_name, "cik": bundle.cik},
        degraded=False,
        error=None,
        key_metrics=out.get("key_metrics"),
        flags=out.get("flags") or [],
    )]}
