import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from agents.price_agent import (
    compute_rsi,
    compute_macd,
    compute_bollinger_pctb,
    price_agent,
)


def _flat_close(n=90, start=100.0, drift=0.5):
    return pd.Series(np.linspace(start, start + drift * n, n))


def test_compute_rsi_neutral_for_steady_uptrend():
    s = _flat_close()
    rsi = compute_rsi(s)
    assert 50 < rsi <= 100


def test_compute_macd_returns_two_floats():
    s = _flat_close()
    line, signal = compute_macd(s)
    assert isinstance(line, float) and isinstance(signal, float)


def test_compute_bollinger_at_midpoint_for_constant_series():
    s = pd.Series([100.0] * 50)
    pctb = compute_bollinger_pctb(s)
    assert pctb == 0.5


def test_price_agent_happy_path():
    fake_close = _flat_close(90, 100, 0.3)
    fake_df = pd.DataFrame({"Close": fake_close})

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content='{"signal": "BULLISH", "confidence": 0.7, "summary": "Trend up"}')
    fake_clients = MagicMock(reasoning=fake_llm)

    with patch("agents.price_agent.yf.download", return_value=fake_df):
        signals = price_agent({"ticker": "MSFT"}, fake_clients)

    sig = signals["agent_signals"][0]
    assert sig["agent"] == "price"
    assert sig["signal"] == "BULLISH"
    assert sig["confidence"] == 0.7
    assert "rsi" in sig["raw_data"]
    assert sig["section_markdown"].startswith("## Technical Analysis")


def test_price_agent_empty_data_degrades():
    fake_clients = MagicMock()
    with patch("agents.price_agent.yf.download", return_value=pd.DataFrame()):
        signals = price_agent({"ticker": "ZZZZ"}, fake_clients)
    sig = signals["agent_signals"][0]
    assert sig["degraded"] is True
    assert sig["signal"] == "NEUTRAL"
    assert sig["confidence"] == 0.0
