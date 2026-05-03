import responses
from unittest.mock import MagicMock

from agents.macro_agent import macro_agent


def _clients(api_key="sk-ant-fake"):
    secret = MagicMock()
    secret.get_secret_value.return_value = api_key
    reasoning = MagicMock()
    reasoning.anthropic_api_key = secret
    return MagicMock(reasoning=reasoning)


@responses.activate
def test_macro_full_data_with_tool_use(monkeypatch):
    fk = "frd-fake"
    for series in ("DTWEXBGS", "DFF", "DGS10", "DGS2"):
        responses.add(
            responses.GET,
            "https://api.stlouisfed.org/fred/series/observations",
            json={"observations": [
                {"date": "2025-10-30", "value": "104.2"},
                {"date": "2025-10-25", "value": "103.8"},
            ]},
            match=[responses.matchers.query_param_matcher({
                "series_id": series, "api_key": fk, "file_type": "json",
                "sort_order": "desc", "limit": "5",
            })],
        )
    responses.add(
        responses.GET,
        "https://api.alternative.me/fng/?limit=1",
        json={"data": [{"value": "62", "value_classification": "Greed"}]},
    )

    monkeypatch.setattr(
        "agents.macro_agent.run_with_tools",
        lambda **kw: {
            "signal": "BEARISH", "confidence": 0.55,
            "summary": "Risk-off regime; high ticker exposure.",
            "section_markdown": "## Macro Backdrop\nDetails.",
            "regime": "risk-off",
            "yield_curve_state": "inverted",
            "ticker_exposure": "high",
            "key_metrics": {"dxy_latest": 104.2, "fed_funds_rate": 5.25,
                            "yield_curve_2s10s": -0.45, "fear_greed_index": 62},
            "flags": ["dxy_rising", "curve_inverted"],
        },
    )

    out = macro_agent({"ticker": "MSFT"}, _clients(), fred_key=fk)
    sig = out["agent_signals"][0]
    assert sig["agent"] == "macro"
    assert sig["regime"] == "risk-off"
    assert sig["ticker_exposure"] == "high"
    assert sig["raw_data"]["dxy_latest"] == 104.2
    assert sig["raw_data"]["fear_greed_index"] == 62
    assert sig["degraded"] is False


@responses.activate
def test_macro_no_fred_key_degrades(monkeypatch):
    responses.add(
        responses.GET,
        "https://api.alternative.me/fng/?limit=1",
        json={"data": [{"value": "30", "value_classification": "Fear"}]},
    )
    monkeypatch.setattr(
        "agents.macro_agent.run_with_tools",
        lambda **kw: {
            "signal": "NEUTRAL", "confidence": 0.3,
            "summary": "Limited macro data.",
            "section_markdown": "## Macro Backdrop\nDegraded.",
            "regime": "neutral", "yield_curve_state": "flat",
            "ticker_exposure": "medium",
            "key_metrics": {"fear_greed_index": 30},
            "flags": ["fred_unavailable"],
        },
    )
    out = macro_agent({"ticker": "MSFT"}, _clients(), fred_key="")
    sig = out["agent_signals"][0]
    assert sig["degraded"] is True
    assert sig["raw_data"]["fear_greed_index"] == 30
    assert sig["raw_data"]["dxy_latest"] is None


@responses.activate
def test_macro_llm_error_degrades(monkeypatch):
    responses.add(
        responses.GET,
        "https://api.alternative.me/fng/?limit=1",
        json={"data": [{"value": "50"}]},
    )
    monkeypatch.setattr(
        "agents.macro_agent.run_with_tools",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    out = macro_agent({"ticker": "MSFT"}, _clients(), fred_key="")
    sig = out["agent_signals"][0]
    assert sig["degraded"] is True
    assert "Macro LLM error" in sig["summary"]
