"""Shared state definitions for the MarketMind v2 LangGraph pipeline.

The `agent_signals` list uses `operator.add` as its LangGraph reducer so that
five specialist nodes running in the same superstep can each append their
result without races.
"""

from __future__ import annotations

import operator
from typing import Annotated, List, Literal, Optional, TypedDict

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
    agent_signals: Annotated[List[AgentSignal], operator.add]
    retry_round: int
    supervisor_review: Optional[SupervisorReview]
    final_verdict: Optional[Verdict]
    final_conviction: Optional[Conviction]
    final_confidence: Optional[float]
    final_reasoning: Optional[str]
    final_report: Optional[str]
