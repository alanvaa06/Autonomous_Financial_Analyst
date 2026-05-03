# tests/test_state.py
import operator
from state import MarketMindState, AgentSignal, SupervisorReview  # noqa: F401


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


def test_state_carries_edgar_bundle_field():
    s: MarketMindState = {
        "ticker": "MSFT", "company_name": None, "cik": None,
        "price_history": None, "vix_history": None, "edgar_bundle": None,
        "agent_signals": [], "retry_round": 0, "supervisor_review": None,
        "final_verdict": None, "final_conviction": None,
        "final_confidence": None, "final_reasoning": None, "final_report": None,
        "key_drivers": None, "dissenting_view": None, "watch_items": None,
    }
    assert s["edgar_bundle"] is None
    assert s["key_drivers"] is None


def test_agent_signal_v2_1_optional_fields_can_be_omitted():
    # total=False — only the legacy required fields need be present.
    sig: AgentSignal = {
        "agent": "price",
        "signal": "BULLISH",
        "confidence": 0.7,
        "summary": "uptrend",
        "section_markdown": "## Technical\nBody",
        "raw_data": {},
        "degraded": False,
        "error": None,
    }
    # No KeyError when accessing new fields with .get()
    assert sig.get("key_metrics") is None
    assert sig.get("flags") is None
    assert sig.get("regime") is None


def test_agent_signal_v2_1_optional_fields_round_trip():
    sig: AgentSignal = {
        "agent": "price", "signal": "BULLISH", "confidence": 0.7,
        "summary": "x", "section_markdown": "## H\nb",
        "raw_data": {}, "degraded": False, "error": None,
        "key_metrics": {"rsi": 58.2},
        "flags": ["trend_confirmation"],
        "regime": "trending_up",
    }
    assert sig["key_metrics"]["rsi"] == 58.2
    assert sig["flags"] == ["trend_confirmation"]
    assert sig["regime"] == "trending_up"
