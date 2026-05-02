import responses
from unittest.mock import MagicMock

from agents.macro_agent import macro_agent


def _fred_series(observations):
    return {"observations": [{"date": d, "value": v} for d, v in observations]}


@responses.activate
def test_macro_agent_full_data():
    fk = "frd-fake"
    for series in ("DTWEXBGS", "DFF", "DGS10", "DGS2"):
        responses.add(
            responses.GET,
            "https://api.stlouisfed.org/fred/series/observations",
            json=_fred_series([("2025-10-30", "104.2"), ("2025-10-25", "103.8")]),
            match=[responses.matchers.query_param_matcher({"series_id": series, "api_key": fk, "file_type": "json", "sort_order": "desc", "limit": "5"})],
        )
    responses.add(
        responses.GET,
        "https://api.alternative.me/fng/?limit=1",
        json={"data": [{"value": "62", "value_classification": "Greed"}]},
    )

    fake_sonnet = MagicMock()
    fake_sonnet.invoke.return_value = MagicMock(content='{"signal": "BEARISH", "confidence": 0.55, "summary": "DXY rising", "section_markdown": "## Macro Backdrop\\nText."}')
    clients = MagicMock(reasoning=fake_sonnet)

    out = macro_agent({"ticker": "MSFT"}, clients, fred_key=fk)
    sig = out["agent_signals"][0]
    assert sig["agent"] == "macro"
    assert sig["signal"] == "BEARISH"
    assert sig["raw_data"]["dxy_latest"] == 104.2
    assert sig["raw_data"]["fear_greed_index"] == 62


@responses.activate
def test_macro_agent_degraded_without_fred_key():
    responses.add(
        responses.GET,
        "https://api.alternative.me/fng/?limit=1",
        json={"data": [{"value": "30", "value_classification": "Fear"}]},
    )
    fake_sonnet = MagicMock()
    fake_sonnet.invoke.return_value = MagicMock(content='{"signal": "NEUTRAL", "confidence": 0.3, "summary": "Limited macro data", "section_markdown": "## Macro Backdrop\\nText."}')
    clients = MagicMock(reasoning=fake_sonnet)

    out = macro_agent({"ticker": "MSFT"}, clients, fred_key="")
    sig = out["agent_signals"][0]
    assert sig["degraded"] is True
    assert sig["raw_data"]["fear_greed_index"] == 30
    assert sig["raw_data"]["dxy_latest"] is None
