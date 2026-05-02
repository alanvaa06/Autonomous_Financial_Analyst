from unittest.mock import patch

from graph import build_graph
from state import MarketMindState


class _StubAgent:
    def __init__(self, name, signal="BULLISH", confidence=0.7, **kw):
        self.name = name
        self.signal = signal
        self.confidence = confidence

    def __call__(self, state, *args, **kwargs):
        return {"agent_signals": [{
            "agent": self.name, "signal": self.signal, "confidence": self.confidence,
            "summary": f"{self.name} stub", "section_markdown": "## H\n" + "x" * 250,
            "raw_data": {}, "degraded": False, "error": None,
        }]}


def test_graph_builds_and_runs_end_to_end():
    # Patch each specialist + orchestrator + supervisor + synthesis at the graph wiring layer.
    with patch("graph.orchestrator", return_value={"ticker": "MSFT", "cik": "0000789019", "company_name": "Microsoft Corp"}), \
         patch("graph.price_agent", _StubAgent("price")), \
         patch("graph.sentiment_agent", _StubAgent("sentiment")), \
         patch("graph.fundamentals_agent", _StubAgent("fundamentals")), \
         patch("graph.macro_agent", _StubAgent("macro")), \
         patch("graph.risk_agent", _StubAgent("risk")):
        g = build_graph(
            llm_clients=object(),
            tavily_key="tvly", fred_key="frd",
        )
        init: MarketMindState = {
            "ticker": "MSFT", "company_name": None, "cik": None,
            "agent_signals": [], "retry_round": 0, "supervisor_review": None,
            "final_verdict": None, "final_conviction": None,
            "final_confidence": None, "final_reasoning": None, "final_report": None,
        }
        result = g.invoke(init)

    assert result["final_verdict"] == "BUY"
    assert result["final_report"].startswith("# MSFT")
    # All five specialists wrote.
    agents_seen = {s["agent"] for s in result["agent_signals"]}
    assert agents_seen == {"price", "sentiment", "fundamentals", "macro", "risk"}
