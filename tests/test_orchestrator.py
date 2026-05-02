# tests/test_orchestrator.py
import pytest
from unittest.mock import patch

from agents.orchestrator import orchestrator


def test_orchestrator_uppercases_and_validates():
    state = {
        "ticker": "  msft  ",
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
    with patch("agents.orchestrator.resolve_ticker") as mock_resolve:
        mock_resolve.return_value = ("0000789019", "MICROSOFT CORP")
        update = orchestrator(state)
    assert update["ticker"] == "MSFT"
    assert update["cik"] == "0000789019"
    assert update["company_name"] == "MICROSOFT CORP"


def test_orchestrator_rejects_invalid_ticker():
    state = {"ticker": "!!", "agent_signals": [], "retry_round": 0,
             "company_name": None, "cik": None, "supervisor_review": None,
             "final_verdict": None, "final_conviction": None,
             "final_confidence": None, "final_reasoning": None, "final_report": None}
    with pytest.raises(ValueError, match="invalid ticker"):
        orchestrator(state)


def test_orchestrator_unknown_to_sec_still_proceeds():
    state = {"ticker": "FOREIGN", "agent_signals": [], "retry_round": 0,
             "company_name": None, "cik": None, "supervisor_review": None,
             "final_verdict": None, "final_conviction": None,
             "final_confidence": None, "final_reasoning": None, "final_report": None}
    from edgar import TickerNotFound
    with patch("agents.orchestrator.resolve_ticker", side_effect=TickerNotFound("nope")):
        update = orchestrator(state)
    # Pipeline continues — fundamentals will degrade.
    assert update["ticker"] == "FOREIGN"
    assert update["cik"] is None
    assert update["company_name"] == "FOREIGN"
