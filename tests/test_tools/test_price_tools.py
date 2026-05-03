# tests/test_tools/test_price_tools.py
import numpy as np
import pandas as pd

from agents.tools.price_tools import build_price_tools


def _price_df(n=200, start=100, drift=0.05, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.012, n)
    prices = start * np.cumprod(1 + rets)
    return pd.DataFrame({"Close": prices, "Volume": rng.integers(1e6, 5e6, n)})


def test_compute_indicator_atr():
    df = _price_df(120)
    tools = build_price_tools(price_history=df)
    indicator_tool = next(t for t in tools if t.name == "compute_indicator")
    out = indicator_tool.handler({"name": "ATR"})
    assert "name" in out and out["name"] == "ATR"
    assert "value" in out


def test_compute_indicator_sma():
    df = _price_df(120)
    tools = build_price_tools(price_history=df)
    indicator_tool = next(t for t in tools if t.name == "compute_indicator")
    out = indicator_tool.handler({"name": "SMA50"})
    assert out["value"] is not None


def test_compute_indicator_unknown():
    df = _price_df(120)
    tools = build_price_tools(price_history=df)
    indicator_tool = next(t for t in tools if t.name == "compute_indicator")
    out = indicator_tool.handler({"name": "BOGUS"})
    assert "error" in out


def test_detect_chart_pattern_returns_dict():
    df = _price_df(120)
    tools = build_price_tools(price_history=df)
    pattern_tool = next(t for t in tools if t.name == "detect_chart_pattern")
    out = pattern_tool.handler({})
    assert "patterns" in out


def test_volume_profile_summary_returns_buckets():
    df = _price_df(120)
    tools = build_price_tools(price_history=df)
    vp_tool = next(t for t in tools if t.name == "volume_profile_summary")
    out = vp_tool.handler({"n_buckets": 8})
    assert "buckets" in out
    assert len(out["buckets"]) <= 8
    for b in out["buckets"]:
        assert "price_range" in b
        assert "volume" in b


def test_build_price_tools_returns_three():
    df = _price_df(60)
    tools = build_price_tools(price_history=df)
    names = [t.name for t in tools]
    assert names == ["compute_indicator", "detect_chart_pattern", "volume_profile_summary"]


def test_compute_indicator_empty_df_returns_error():
    tools = build_price_tools(price_history=pd.DataFrame())
    out = tools[0].handler({"name": "ATR"})
    assert "error" in out
