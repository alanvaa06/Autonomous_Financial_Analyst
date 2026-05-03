"""On-demand tools for the Price agent. All local computations on the
prefetched price_history DataFrame — no network calls."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from agents import ToolDef


VALID_INDICATORS = {"ATR", "ADX", "STOCH", "SMA50", "SMA200", "OBV"}


def _atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    if "High" not in df.columns or "Low" not in df.columns:
        # Approximate ATR with abs daily Close % range
        close = df["Close"].squeeze()
        return round(float(close.pct_change().abs().rolling(period).mean().iloc[-1] * 100), 3)
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    close = df["Close"].squeeze()
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return round(float(atr), 3) if pd.notna(atr) else None


def _sma(df: pd.DataFrame, period: int) -> Optional[float]:
    close = df["Close"].squeeze()
    if len(close) < period:
        return None
    return round(float(close.rolling(period).mean().iloc[-1]), 4)


def _obv(df: pd.DataFrame) -> Optional[float]:
    if "Volume" not in df.columns:
        return None
    close = df["Close"].squeeze()
    vol = df["Volume"].squeeze()
    direction = np.sign(close.diff().fillna(0))
    obv = (direction * vol).cumsum().iloc[-1]
    return round(float(obv), 0)


def _stoch(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Optional[dict]:
    close = df["Close"].squeeze()
    if len(close) < k_period:
        return None
    if "High" in df.columns and "Low" in df.columns:
        high_n = df["High"].squeeze().rolling(k_period).max()
        low_n = df["Low"].squeeze().rolling(k_period).min()
    else:
        high_n = close.rolling(k_period).max()
        low_n = close.rolling(k_period).min()
    k = 100 * (close - low_n) / (high_n - low_n).replace(0, 1e-9)
    d = k.rolling(d_period).mean()
    return {"k": round(float(k.iloc[-1]), 2), "d": round(float(d.iloc[-1]), 2)}


def _adx(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    # Lightweight: returns rolling absolute % move as a proxy if no High/Low.
    close = df["Close"].squeeze()
    if len(close) < period + 1:
        return None
    move = close.pct_change().abs().rolling(period).mean().iloc[-1] * 100
    return round(float(move), 3) if pd.notna(move) else None


def _detect_patterns(df: pd.DataFrame) -> list[dict]:
    """Heuristic, very lightweight chart-pattern detection."""
    close = df["Close"].squeeze()
    if len(close) < 30:
        return []
    last30 = close.iloc[-30:]
    out: list[dict] = []
    # Ascending triangle: rising lows + flat highs
    highs = last30.rolling(5).max()
    lows = last30.rolling(5).min()
    if highs.iloc[-1] - highs.iloc[0] < highs.iloc[0] * 0.01 and lows.iloc[-1] > lows.iloc[0]:
        out.append({"name": "ascending_triangle", "confidence": 0.5})
    # Double-top: two peaks within 1% over the last 60 bars
    if len(close) >= 60:
        last60 = close.iloc[-60:]
        peak = last60.max()
        peaks = last60[last60 > peak * 0.99]
        if len(peaks) >= 2 and peaks.index[-1] - peaks.index[0] >= 10:
            out.append({"name": "double_top", "confidence": 0.4})
    # Trend break: latest close vs SMA20 cross
    sma20 = close.rolling(20).mean()
    if pd.notna(sma20.iloc[-2]) and pd.notna(sma20.iloc[-1]):
        if close.iloc[-2] > sma20.iloc[-2] and close.iloc[-1] < sma20.iloc[-1]:
            out.append({"name": "sma20_break_down", "confidence": 0.6})
        elif close.iloc[-2] < sma20.iloc[-2] and close.iloc[-1] > sma20.iloc[-1]:
            out.append({"name": "sma20_break_up", "confidence": 0.6})
    return out


def _volume_profile(df: pd.DataFrame, n_buckets: int = 10) -> list[dict]:
    if "Volume" not in df.columns:
        return []
    close = df["Close"].squeeze()
    vol = df["Volume"].squeeze()
    lo, hi = float(close.min()), float(close.max())
    if lo == hi:
        return []
    bins = np.linspace(lo, hi, n_buckets + 1)
    bucket = np.digitize(close, bins) - 1
    rows: list[dict] = []
    for i in range(n_buckets):
        mask = bucket == i
        rows.append({
            "price_range": [round(float(bins[i]), 2), round(float(bins[i + 1]), 2)],
            "volume": int(vol[mask].sum()) if mask.any() else 0,
        })
    return rows


def build_price_tools(*, price_history: Optional[pd.DataFrame]) -> list[ToolDef]:

    def indicator_handler(args: dict) -> dict:
        name = (args.get("name") or "").upper()
        if price_history is None or price_history.empty:
            return {"error": "price_history unavailable"}
        if name not in VALID_INDICATORS:
            return {"error": f"unknown indicator '{name}'; valid: {sorted(VALID_INDICATORS)}"}
        if name == "ATR":
            value = _atr(price_history)
        elif name == "ADX":
            value = _adx(price_history)
        elif name == "STOCH":
            value = _stoch(price_history)
        elif name == "SMA50":
            value = _sma(price_history, 50)
        elif name == "SMA200":
            value = _sma(price_history, 200)
        elif name == "OBV":
            value = _obv(price_history)
        else:
            value = None
        return {"name": name, "value": value}

    def pattern_handler(_args: dict) -> dict:
        if price_history is None or price_history.empty:
            return {"error": "price_history unavailable"}
        return {"patterns": _detect_patterns(price_history)}

    def vp_handler(args: dict) -> dict:
        if price_history is None or price_history.empty:
            return {"error": "price_history unavailable"}
        n = int(args.get("n_buckets", 10) or 10)
        return {"buckets": _volume_profile(price_history, n_buckets=n)}

    return [
        ToolDef(
            name="compute_indicator",
            description=(
                "Compute one of: ATR, ADX, STOCH, SMA50, SMA200, OBV on the "
                "prefetched price history. All local; no network. Use to drill "
                "into volatility, trend strength, momentum, or volume trend."
            ),
            input_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            handler=indicator_handler,
        ),
        ToolDef(
            name="detect_chart_pattern",
            description=(
                "Heuristic pattern detection on recent price history: "
                "ascending_triangle, double_top, sma20_break_up/down. Returns "
                "list of {name, confidence}. Local; no network."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=pattern_handler,
        ),
        ToolDef(
            name="volume_profile_summary",
            description=(
                "Bucket price range into n_buckets and return total traded "
                "volume per bucket. High-volume buckets identify support / "
                "resistance levels. Local; no network."
            ),
            input_schema={
                "type": "object",
                "properties": {"n_buckets": {"type": "integer", "default": 10}},
            },
            handler=vp_handler,
        ),
    ]
