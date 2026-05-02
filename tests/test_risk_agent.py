from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from agents.risk_agent import risk_agent


def _series(n=90):
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0008, 0.015, n)
    prices = 100 * np.cumprod(1 + rets)
    return pd.DataFrame({"Close": prices})


def test_risk_agent_happy():
    fake_sonnet = MagicMock()
    fake_sonnet.invoke.return_value = MagicMock(content='{"signal": "NEUTRAL", "confidence": 0.5, "summary": "Moderate vol", "section_markdown": "## Risk Profile\\nText."}')
    clients = MagicMock(reasoning=fake_sonnet)

    fake_ticker = MagicMock()
    fake_ticker.info = {"beta": 1.1, "shortRatio": 2.5}

    with patch("agents.risk_agent.yf.download", side_effect=[_series(), pd.DataFrame({"Close": [22.1, 22.5, 21.9, 22.0, 22.3]})]), \
         patch("agents.risk_agent.yf.Ticker", return_value=fake_ticker):
        out = risk_agent({"ticker": "MSFT"}, clients)

    sig = out["agent_signals"][0]
    assert sig["agent"] == "risk"
    assert "annualized_vol_pct" in sig["raw_data"]
    assert sig["raw_data"]["vix"] == 22.3


def test_risk_agent_empty_data_degrades():
    clients = MagicMock()
    with patch("agents.risk_agent.yf.download", return_value=pd.DataFrame()):
        out = risk_agent({"ticker": "ZZZZ"}, clients)
    sig = out["agent_signals"][0]
    assert sig["degraded"] is True
