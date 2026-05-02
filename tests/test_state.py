# tests/test_state.py
import operator
from state import MarketMindState, AgentSignal, SupervisorReview


def test_state_has_required_keys():
    s: MarketMindState = {
        "ticker": "MSFT",
        "company_name": None,
        "cik": None,
        "agent_signals": [],
        "retry_round": 0,
        "supervisor_review": None,
        "final_verdict": None,
        "final_conviction": None,
        "final_confidence": None,
        "final_reasoning": None,
        "final_report": None,
    }
    assert s["ticker"] == "MSFT"
    assert s["agent_signals"] == []


def test_agent_signals_reducer_appends():
    a: list = []
    b = a + [{"agent": "price"}]
    c = b + [{"agent": "risk"}]
    assert len(c) == 2 and c[0]["agent"] == "price" and c[1]["agent"] == "risk"


def test_agent_signal_shape():
    sig: AgentSignal = {
        "agent": "price",
        "signal": "BULLISH",
        "confidence": 0.7,
        "summary": "uptrend",
        "section_markdown": "## Technical\nDetails...",
        "raw_data": {"rsi": 55.0},
        "degraded": False,
        "error": None,
    }
    assert sig["signal"] == "BULLISH"


def test_supervisor_review_shape():
    rev: SupervisorReview = {
        "approved": True,
        "critiques": {},
        "retry_targets": [],
        "notes": "All sections complete.",
    }
    assert rev["approved"] is True
