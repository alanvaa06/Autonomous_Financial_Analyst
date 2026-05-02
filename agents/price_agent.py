"""Technical analysis specialist: yfinance OHLC -> RSI/MACD/Bollinger -> LLM."""

from __future__ import annotations

import json

import pandas as pd
import yfinance as yf

from agents import degraded_signal, safe_parse_json
from state import AgentSignal


def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2)


def compute_macd(prices: pd.Series) -> tuple[float, float]:
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    return round(float(line.iloc[-1]), 4), round(float(signal.iloc[-1]), 4)


def compute_bollinger_pctb(prices: pd.Series, period: int = 20) -> float:
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    band = float(upper.iloc[-1]) - float(lower.iloc[-1])
    if band <= 0 or pd.isna(band):
        return 0.5
    return round((float(prices.iloc[-1]) - float(lower.iloc[-1])) / band, 3)


def _degraded(reason: str, raw: dict | None = None, error: str | None = None) -> dict:
    return degraded_signal("price", "Technical Analysis", reason, raw=raw, error=error)


def price_agent(state: dict, clients) -> dict:
    ticker = state["ticker"]
    try:
        data = yf.download(ticker, period="90d", interval="1d", progress=False)
        if data.empty or "Close" not in data.columns:
            return _degraded(f"No price data for {ticker}")

        close = data["Close"].squeeze()
        current = round(float(close.iloc[-1]), 4)
        prev_7d = float(close.iloc[-7]) if len(close) >= 7 else float(close.iloc[0])
        change_7d = round((current - prev_7d) / prev_7d * 100, 2) if prev_7d else 0.0
        change_30d = (
            round((current - float(close.iloc[-30])) / float(close.iloc[-30]) * 100, 2)
            if len(close) >= 30 else 0.0
        )
        change_90d = round((current - float(close.iloc[0])) / float(close.iloc[0]) * 100, 2)

        rsi = compute_rsi(close)
        macd_line, macd_signal = compute_macd(close)
        pctb = compute_bollinger_pctb(close)

        raw = {
            "current_price": current,
            "change_7d_pct": change_7d,
            "change_30d_pct": change_30d,
            "change_90d_pct": change_90d,
            "rsi": rsi,
            "macd_line": macd_line,
            "macd_signal": macd_signal,
            "macd_crossover": "positive" if macd_line > macd_signal else "negative",
            "bollinger_pctb": pctb,
        }

        prompt = (
            f"You are a technical analyst. Interpret these indicators for {ticker}:\n\n"
            f"{json.dumps(raw, indent=2)}\n\n"
            "Reference: RSI<30 oversold, RSI>70 overbought; positive MACD crossover bullish; "
            "Bollinger %B near 0 = lower band, near 1 = upper band.\n\n"
            "Respond with JSON ONLY (no markdown fences) with EXACTLY these keys:\n"
            "{\n"
            '  "signal": "BULLISH" | "BEARISH" | "NEUTRAL",\n'
            '  "confidence": 0.0..1.0,\n'
            '  "summary": "one sentence under 25 words",\n'
            '  "section_markdown": "## Technical Analysis\\n... 120-200 word section discussing RSI, MACD, Bollinger position, and 7/30/90d trend ..."\n'
            "}"
        )
        resp = clients.reasoning.invoke(prompt)
        out = safe_parse_json(resp.content)

        return {"agent_signals": [AgentSignal(
            agent="price",
            signal=out["signal"],
            confidence=float(out["confidence"]),
            summary=out["summary"],
            section_markdown=out.get("section_markdown") or "## Technical Analysis\n_Section missing._",
            raw_data=raw,
            degraded=False,
            error=None,
        )]}
    except Exception as exc:
        return _degraded(f"Price agent error", error=str(exc)[:200])
