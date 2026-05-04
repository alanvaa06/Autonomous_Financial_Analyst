import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from agents.price_agent import (
    compute_rsi, compute_macd, compute_bollinger_pctb, price_agent,
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


def _clients(api_key="sk-ant-fake"):
    secret = MagicMock()
    secret.get_secret_value.return_value = api_key
    reasoning = MagicMock()
    reasoning.anthropic_api_key = secret
    return MagicMock(reasoning=reasoning)


def test_price_agent_happy_path_reads_state_history(monkeypatch):
    fake_close = _flat_close(90, 100, 0.3)
    fake_df = pd.DataFrame({"Close": fake_close})

    monkeypatch.setattr(
        "agents.price_agent.run_with_tools",
        lambda **kw: {
            "signal": "BULLISH", "confidence": 0.7,
            "summary": "Trend up, RSI mid.",
            "section_markdown": "## Technical Analysis\nDetails.",
            "regime": "trending_up",
            "key_metrics": {"rsi": 58.2, "macd_state": "positive_crossover",
                            "bollinger_pctb": 0.62, "atr_pct": 1.8,
                            "sma50_vs_sma200": "above"},
            "flags": ["trend_confirmation"],
        },
    )

    out = price_agent({"ticker": "MSFT", "price_history": fake_df}, _clients())
    sig = out["agent_signals"][0]
    assert sig["agent"] == "price"
    assert sig["signal"] == "BULLISH"
    assert sig["regime"] == "trending_up"
    assert sig["key_metrics"]["rsi"] == 58.2


def test_price_agent_falls_back_to_download(monkeypatch):
    fake_close = _flat_close(90, 100, 0.3)
    fake_df = pd.DataFrame({"Close": fake_close})

    with patch(
        "agents.price_agent.download_with_retry", return_value=fake_df,
    ), patch(
        "agents.price_agent.run_with_tools",
        return_value={"signal": "NEUTRAL", "confidence": 0.5,
                      "summary": "x", "section_markdown": "## Technical Analysis\nx",
                      "regime": "ranging", "key_metrics": {}, "flags": []},
    ):
        out = price_agent({"ticker": "MSFT"}, _clients())
    assert out["agent_signals"][0]["signal"] == "NEUTRAL"


def test_price_agent_empty_data_degrades():
    with patch(
        "agents.price_agent.download_with_retry", return_value=pd.DataFrame(),
    ):
        out = price_agent({"ticker": "ZZZZ"}, _clients())
    sig = out["agent_signals"][0]
    assert sig["degraded"] is True
    assert sig["confidence"] == 0.0


def test_price_agent_llm_error_degrades(monkeypatch):
    fake_close = _flat_close(90, 100, 0.3)
    fake_df = pd.DataFrame({"Close": fake_close})
    monkeypatch.setattr(
        "agents.price_agent.run_with_tools",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    out = price_agent({"ticker": "MSFT", "price_history": fake_df}, _clients())
    assert out["agent_signals"][0]["degraded"] is True


def test_compute_raw_change_90d_pct_uses_minus_90_index_when_history_is_long():
    """When prefetch returns 1y, change_90d_pct must reflect 90-day move,
    not the full 1y move."""
    import pandas as pd
    from agents.price_agent import _compute_raw

    # 252 days, slow uptrend overall, but the most recent 90 days are flat.
    n = 252
    prices = []
    for i in range(n):
        if i < n - 90:
            prices.append(50.0 + i * 0.1)  # ramps from 50 to ~66
        else:
            prices.append(66.0)            # flat at 66 for the last 90 days
    close = pd.Series(prices)

    raw = _compute_raw(close)
    # 90d change should be ~0 (last 90 days flat), not ~32% (1y change).
    assert abs(raw["change_90d_pct"]) < 1.0
