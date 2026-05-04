"""Price/technical specialist (v2.1): persona + methodology + few-shot + 3 tools.

Reads `state["price_history"]` populated by data_prefetch; falls back to
`download_with_retry` when missing (preserves single-call test paths).
Pure compute helpers (compute_rsi/compute_macd/compute_bollinger_pctb) are
preserved from v2.0 so existing unit tests still cover them.
"""

from __future__ import annotations

import logging

import pandas as pd

from agents import LLMClients, degraded_signal, run_with_tools
from agents.tools.price_tools import build_price_tools
from agents.yf_helpers import download_with_retry
from state import AgentSignal

logger = logging.getLogger("marketmind.price_agent")


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


PERSONA = (
    "You are a CMT Charterholder (Chartered Market Technician) with 12 years "
    "on a quant systematic trading desk. You read momentum, mean-reversion, "
    "and volatility regimes from price action — RSI/MACD/Bollinger are "
    "starting points, not gospel. You're explicit about whether the current "
    "setup is trend-confirming or mean-reverting, and you flag when "
    "indicators conflict."
)

METHODOLOGY = """
Methodology you apply:
- Trend regime first (trending vs ranging vs trending-down)
- RSI in context (in strong uptrend, RSI 70+ can persist; not auto-overbought)
- MACD line vs signal + histogram divergence
- Bollinger %B with mid-band as trend filter
- Multi-timeframe alignment (7d / 30d / 90d)
- Volatility regime (ATR-based)
""".strip()

FEWSHOT = """
Examples of correct reasoning:

Example 1 — momentum override of overbought:
RSI 76, MACD line above signal with rising histogram, %B 0.92, 7d +6%, 30d +14%, 90d +28%.
Indicators are extreme but in a confirmed uptrend with volume. Call: BULLISH 0.7,
regime=trending_up. Flag: trend_confirmation, momentum_persistent. Do NOT fade
overbought when trend is intact.

Example 2 — conflicting signals → wait:
RSI 52, MACD line crosses below signal, %B 0.48, 7d +1%, 30d -2%, 90d +5%.
Mixed timeframes, %B at midpoint, momentum reversing. Call: NEUTRAL 0.45,
regime=ranging. Flag: mid_band, mtf_divergence.

Example 3 — oversold bounce setup:
RSI 28, MACD line crosses above signal (positive cross), %B 0.06, 7d -8%,
30d -12%, 90d -3%. Capitulation print + early MACD turn near lower band. Call:
BULLISH 0.65, regime=ranging (transition). Flag: oversold_reversal, lower_band.
""".strip()

OUTPUT_SCHEMA = """
Respond with a single JSON object (no markdown fences) with EXACTLY these keys:
{
  "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence": 0.0..1.0,
  "summary": "one sentence stating regime + trend bias (≤25 words)",
  "section_markdown": "## Technical Analysis\\n... 120-200 words ...",
  "regime": "trending_up" | "trending_down" | "ranging" | "volatility_expansion",
  "key_metrics": {
    "rsi": number,
    "macd_state": "positive_crossover" | "negative_crossover" | "above" | "below",
    "bollinger_pctb": number,
    "atr_pct": number | null,
    "sma50_vs_sma200": "above" | "below" | "unknown"
  },
  "flags": ["string", ...]
}
""".strip()

GUARDRAILS = """
Constraints and guardrails:
- No "buy/sell" verbiage in section_markdown — use "constructive / cautious / negative" framings.
- section_markdown must be 120-200 words.
- Confidence ≤ 0.5 if any indicator (RSI, MACD, %B) is NaN.
- Confidence ≤ 0.6 if price_history < 60 trading days.
- regime MUST be set explicitly in summary.
""".strip()


def _build_system_prompt() -> str:
    return "\n\n".join([PERSONA, METHODOLOGY, FEWSHOT, OUTPUT_SCHEMA, GUARDRAILS])


def _compute_raw(close: pd.Series) -> dict:
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
    return {
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


def _build_user_prompt(ticker: str, raw: dict, history_len: int) -> str:
    lines = [
        f"Ticker: {ticker}",
        f"Trading days of history: {history_len}",
        "",
        "Pre-computed indicators (always-on):",
        *(f"- {k}: {v}" for k, v in raw.items()),
        "",
        "Apply your methodology and the 3 examples above. Use tools "
        "(compute_indicator for ATR/SMA/etc., detect_chart_pattern, "
        "volume_profile_summary) only if a step needs data not in the "
        "always-on payload. Then output the final JSON.",
    ]
    return "\n".join(lines)


def price_agent(state: dict, clients: LLMClients) -> dict:
    ticker = state["ticker"]
    try:
        data = state.get("price_history")
        if data is None:
            data = download_with_retry(ticker, period="90d", interval="1d")
        if data is None or data.empty or "Close" not in data.columns:
            return degraded_signal(
                "price", "Technical Analysis", f"No price data for {ticker}",
            )

        close = data["Close"].squeeze()
        raw = _compute_raw(close)

        api_key = clients.reasoning.anthropic_api_key.get_secret_value()
        tools = build_price_tools(price_history=data)

        out = run_with_tools(
            api_key=api_key,
            system_prompt=_build_system_prompt(),
            user_prompt=_build_user_prompt(ticker, raw, len(close)),
            tools=tools,
            max_iterations=3,
            max_tokens=1500,
        )

        return {"agent_signals": [AgentSignal(
            agent="price",
            signal=out.get("signal", "NEUTRAL"),
            confidence=float(out.get("confidence", 0.0) or 0.0),
            summary=out.get("summary", ""),
            section_markdown=out.get("section_markdown") or "## Technical Analysis\n_Section missing._",
            raw_data=raw,
            degraded=False,
            error=None,
            regime=out.get("regime"),
            key_metrics=out.get("key_metrics"),
            flags=out.get("flags") or [],
        )]}
    except Exception as exc:
        logger.exception("price: agent failed")
        return degraded_signal(
            "price", "Technical Analysis", "Price agent error",
            error=str(exc)[:200],
        )
