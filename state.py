"""Shared state definitions for the MarketMind v2 LangGraph pipeline.

The `agent_signals` list uses `operator.add` as its LangGraph reducer so that
five specialist nodes running in the same superstep can each append their
result without races.

`price_history` and `vix_history` are populated once by the data_prefetch node
before the parallel fan-out so the Price and Risk specialists do not each
issue their own `yf.download(...)` call. Yahoo throttles HF Space IPs hard;
deduplicating these requests is the difference between green and degraded runs.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, List, Literal, Optional, TypedDict

Verdict = Literal["BUY", "HOLD", "SELL"]
Conviction = Literal["STRONG", "STANDARD", "CAUTIOUS"]
Signal = Literal["BULLISH", "BEARISH", "NEUTRAL"]


class AgentSignal(TypedDict):
    agent: str
    signal: Signal
    confidence: float
    summary: str
    section_markdown: str
    raw_data: dict
    degraded: bool
    error: Optional[str]


class SupervisorReview(TypedDict):
    approved: bool
    critiques: dict
    retry_targets: list
    notes: str


class MarketMindState(TypedDict):
    ticker: str
    company_name: Optional[str]
    cik: Optional[str]
    # Prefetched market data (DataFrames). Both Optional — `None` means the
    # prefetch step did not run; an empty DataFrame means it ran but failed.
    price_history: Optional[Any]
    vix_history: Optional[Any]
    agent_signals: Annotated[List[AgentSignal], operator.add]
    retry_round: int
    supervisor_review: Optional[SupervisorReview]
    final_verdict: Optional[Verdict]
    final_conviction: Optional[Conviction]
    final_confidence: Optional[float]
    final_reasoning: Optional[str]
    final_report: Optional[str]
