"""Shared state definitions for the MarketMind v2 LangGraph pipeline.

The `agent_signals` list uses `operator.add` as its LangGraph reducer so that
five specialist nodes running in the same superstep can each append their
result without races.

`price_history`, `vix_history`, and `edgar_bundle` are populated once by the
data_prefetch node before the parallel fan-out so the Price, Risk, and
Fundamentals specialists do not each issue their own external fetch. Yahoo
throttles HF Space IPs hard and SEC asks for politeness; deduplicating these
requests is the difference between green and degraded runs.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, List, Literal, Optional, TypedDict

Verdict = Literal["BUY", "HOLD", "SELL"]
Conviction = Literal["STRONG", "STANDARD", "CAUTIOUS"]
Signal = Literal["BULLISH", "BEARISH", "NEUTRAL"]


class AgentSignal(TypedDict, total=False):
    """v2.1: total=False — only the v2.0 required fields are populated by every
    specialist; the new optional fields are populated only by the agent that
    owns them. Supervisor and Synthesis must use `.get()` everywhere."""

    # v2.0 required
    agent: str
    signal: Signal
    confidence: float
    summary: str
    section_markdown: str
    raw_data: dict
    degraded: bool
    error: Optional[str]

    # v2.1 generic additions (every specialist populates these when not degraded)
    key_metrics: Optional[dict]
    flags: Optional[list[str]]

    # v2.1 agent-specific additions
    regime: Optional[str]                # price (trending_up/...), macro (risk-on/...)
    vol_regime: Optional[str]            # risk
    vix_regime: Optional[str]            # risk
    forward_risk_view: Optional[str]     # risk
    primary_risk_driver: Optional[str]   # risk
    risk_decomposition: Optional[dict]   # risk
    yield_curve_state: Optional[str]     # macro
    ticker_exposure: Optional[str]       # macro
    top_catalyst: Optional[str]          # sentiment
    drivers_categorized: Optional[dict]  # sentiment


class SupervisorReview(TypedDict):
    approved: bool
    critiques: dict
    retry_targets: list
    notes: str


class MarketMindState(TypedDict):
    ticker: str
    company_name: Optional[str]
    cik: Optional[str]
    # Prefetched market data (DataFrames + EdgarBundle dataclass).
    # `None` = prefetch did not run; empty/falsy = ran but failed.
    price_history: Optional[Any]
    vix_history: Optional[Any]
    edgar_bundle: Optional[Any]
    agent_signals: Annotated[List[AgentSignal], operator.add]
    retry_round: int
    supervisor_review: Optional[SupervisorReview]
    final_verdict: Optional[Verdict]
    final_conviction: Optional[Conviction]
    final_confidence: Optional[float]
    final_reasoning: Optional[str]
    final_report: Optional[str]
    # v2.1 synthesis additions
    key_drivers: Optional[list]
    dissenting_view: Optional[str]
    watch_items: Optional[list]
