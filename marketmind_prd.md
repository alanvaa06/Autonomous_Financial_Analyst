# MarketMind — Product Requirements Document

**A Parallel Multi-Agent Stock & Crypto Analyst**

| Field | Value |
|---|---|
| Document type | Product Requirements Document (PRD) |
| Version | 1.0 |
| Date | May 2026 |
| Source | Tutorial by @datasciencebrain (Instagram), code verified April 22, 2026 |
| Status | Reference / Build Specification |

---

## 1. Executive Summary

MarketMind is a multi-agent AI system that analyzes any stock or cryptocurrency ticker by running **five specialist analyst agents in parallel**, then synthesizes their signals through a sixth agent into a final **BUY / HOLD / SELL** verdict with a confidence score and natural-language reasoning.

The system is built on **LangGraph's fan-out → fan-in parallel execution pattern** — the same architecture used by production AI systems at hedge funds in 2026. It is explicitly **not a chatbot**: it is a deterministic, structured analytical pipeline that mirrors the workflow of a real investment committee, where each member contributes a domain-specific opinion before a chair synthesizes a final call.

**Core value proposition:**
- 5x faster than sequential agent execution (~4 seconds vs ~20 seconds for 5 agents).
- Combines five orthogonal analytical dimensions (technicals, sentiment, on-chain/microstructure, macro, risk).
- Built entirely on free-tier APIs and a free-tier inference provider (Groq + Llama 3.3 70B).
- Race-condition-safe state management via LangGraph's `operator.add` reducer.

---

## 2. Goals and Non-Goals

### 2.1 Goals

1. Deliver a **single-ticker analytical verdict** (BUY / HOLD / SELL) backed by quantitative evidence and natural-language reasoning.
2. Demonstrate the **fan-out → fan-in multi-agent pattern** in LangGraph as a reusable production architecture.
3. Run end-to-end on **free APIs and free LLM inference** so the system has no cost barrier for development or evaluation.
4. Provide a **Streamlit dashboard** that surfaces both the final verdict and each specialist agent's individual contribution and confidence.
5. Be **completable in under 60 minutes** by a developer comfortable with Python.

### 2.2 Non-Goals

1. **Not** a portfolio manager. The system analyzes one ticker at a time and does not size positions, manage risk across a book, or track P&L.
2. **Not** a trading execution system. There is no broker integration, order management, or live trading.
3. **Not** a chatbot. There is no conversational memory, follow-up turns, or freeform Q&A.
4. **Not** a backtester. The agents analyze current state only, with no historical signal validation framework.
5. **Not** financial advice. Outputs are structured analytical opinions, not regulated investment recommendations.

---

## 3. System Architecture

### 3.1 High-Level Flow

```
                    User Input (Ticker Symbol)
                              │
                              ▼
                  Orchestrator (Validate + Normalize)
                              │
        ┌──────────┬──────────┼──────────┬──────────┐
        ▼          ▼          ▼          ▼          ▼
    Price      Sentiment   On-Chain    Macro       Risk
    Agent       Agent       Agent      Agent       Agent
   (yfinance) (NewsAPI    (CoinGecko   (FRED API  (Volatility,
    RSI/MACD/ + Llama)   /yfinance)  + Fear&Greed) VIX, Sharpe)
    Bollinger)
        │          │          │          │          │
        └──────────┴──────────┼──────────┴──────────┘
                              ▼
                    Synthesis Agent
                  (Weighted Signal Merge)
                              │
                              ▼
                    Final Verdict
              (BUY / HOLD / SELL + Confidence + Reasoning)
```

### 3.2 Why Fan-Out / Fan-In

Most LangGraph tutorials demonstrate **chains** — agent A calls agent B calls agent C. That is *not* a multi-agent system; it is sequential delegation.

The real power of LangGraph is the **fan-out → fan-in pattern**:

- One **orchestrator node fans out** to N agents simultaneously.
- All agents run in **the same superstep** — LangGraph's term for a parallel execution unit.
- A **synthesis node fans in** only after all agents complete.

For MarketMind:

| Mode | Time |
|---|---|
| Sequential (5 agents × ~4s each) | ~20 seconds |
| Parallel (5 agents in same superstep) | ~4 seconds |
| **Speedup** | **5x** |

### 3.3 Component Responsibilities

| Component | Responsibility |
|---|---|
| Orchestrator | Validate the ticker symbol, classify asset type (stock vs. crypto), normalize state, fan out to specialists. |
| Price Agent | Compute RSI, MACD, Bollinger position, 7-day price change; ask Llama to interpret. |
| Sentiment Agent | Pull last 3 days of news headlines via NewsAPI; ask Llama to score market sentiment. |
| On-Chain Agent | Pull crypto market microstructure from CoinGecko OR stock microstructure from yfinance; interpret volume/flow. |
| Macro Agent | Pull DXY, Fed funds rate, yield curve via FRED API + Fear & Greed index; interpret macro regime. |
| Risk Agent | Compute annualized volatility, max drawdown, Sharpe ratio; pull VIX. |
| Synthesis Agent | Read all five `AgentSignal` outputs and produce final BUY/HOLD/SELL with confidence and reasoning. |

---

## 4. Specialist Agents

| Agent | Data Source | What It Analyzes |
|---|---|---|
| **Price Agent** | yfinance (free) | RSI, MACD, Bollinger Bands, 7-day trend |
| **Sentiment Agent** | NewsAPI free tier | Last 3 days of headlines, sentiment score |
| **On-Chain Agent** | CoinGecko free API / yfinance | Market cap rank, volume ratios, momentum / volume / beta / short ratio |
| **Macro Agent** | FRED API (free) + Fear & Greed | DXY, Fed funds rate, yield curve, Fear & Greed index |
| **Risk Agent** | yfinance (free) | Annualized volatility, max drawdown, VIX, Sharpe ratio |

All five agents run simultaneously. The Synthesis Agent reads all five `AgentSignal` results and makes the final call.

---

## 5. Tech Stack

```
Python 3.12+
LangGraph 0.3+              parallel agent orchestration
langchain-groq              Groq LLM integration for LangChain
langchain-core              LangChain base types
llama-3.3-70b-versatile     Groq hosted model (free, 128K context)
yfinance                    price and risk data
newsapi-python              news headlines
pycoingecko                 crypto market data
fredapi                     macro economic data
streamlit                   dashboard UI
python-dotenv               env variable management
requests                    HTTP calls (Fear & Greed index)
uv                          package manager
```

### 5.1 Why Groq

Groq (not Grok by xAI) is a hardware company that built custom **LPU chips** specifically for LLM inference. The free tier delivers **300–500 tokens/second** — 5–10x faster than GPU-based providers. Sign up at `console.groq.com` with just an email — no credit card.

`llama-3.3-70b-versatile` is their flagship free model with GPT-4o-level quality, **128K context window**, and reliable structured JSON output.

**Free tier limits:** 30 requests/minute, 1,000 requests/day on the 70B model. More than enough for this system.

### 5.2 API Keys (all free)

| Service | URL | Notes |
|---|---|---|
| Groq | `console.groq.com` | Email signup, no credit card. Takes 2 minutes. |
| NewsAPI | `newsapi.org` | Free tier: 100 requests/day. Plenty for tutorials. |
| FRED API | `fred.stlouisfed.org/docs/api/api_key.html` | Completely free, no credit card. |
| CoinGecko | (no key) | Free tier rate-limited but works fine for this use case. |

---

## 6. Project Setup

### 6.1 Project Initialization

```bash
uv init marketmind
cd marketmind
```

### 6.2 Install Dependencies

```bash
uv add langgraph langchain-groq langchain-core \
       yfinance newsapi-python pycoingecko fredapi \
       streamlit python-dotenv pandas numpy requests
```

### 6.3 Environment Variables (`.env`)

```
# .env — add this to .gitignore, never commit it
GROQ_API_KEY=gsk_your_key_here
NEWS_API_KEY=your_newsapi_key_here
FRED_API_KEY=your_fred_key_here
```

### 6.4 Project Structure

```
marketmind/
├── .env
├── state.py                 # shared TypedDict state
├── graph.py                 # LangGraph wiring (parallel fan-out)
├── agents/
│   ├── __init__.py          # shared Groq client + safe_parse_json
│   ├── price_agent.py
│   ├── sentiment_agent.py
│   ├── onchain_agent.py
│   ├── macro_agent.py
│   ├── risk_agent.py
│   └── synthesis_agent.py
└── app.py                   # Streamlit dashboard
```

```bash
mkdir agents
touch agents/__init__.py
```

---

## 7. Shared State Definition

In LangGraph, all agents communicate through a single shared **state object**. Defining it correctly is the single most important step in the project — a misconfigured state object will silently lose data when agents write in parallel.

### `state.py`

```python
import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict


class AgentSignal(TypedDict):
    """Output from a single specialist agent."""
    agent: str          # "price", "sentiment", "onchain", "macro", "risk"
    signal: str         # "BULLISH", "BEARISH", or "NEUTRAL"
    confidence: float   # 0.0 to 1.0
    summary: str        # one-line explanation
    raw_data: dict      # the numbers behind the call


class MarketMindState(TypedDict):
    """Shared state that flows through the entire graph."""
    ticker: str
    asset_type: str     # "stock" or "crypto"

    # operator.add is the reducer — it APPENDS each agent's result
    # instead of overwriting. This prevents race conditions when all
    # 5 agents try to write to the same field simultaneously.
    agent_signals: Annotated[list, operator.add]

    # Synthesis agent writes these after all 5 agents complete
    final_verdict: Optional[str]
    final_confidence: Optional[float]
    final_reasoning: Optional[str]
```

### 7.1 The Critical Reducer

The line `Annotated[list, operator.add]` is the most important line in the entire project.

When five agents all write to `agent_signals` at the same time, LangGraph uses the `operator.add` reducer to **append** each result to the list rather than overwrite. Without this reducer, only the last agent's result would survive — the other four would be silently lost.

This is the difference between a working parallel system and a system that appears to run but produces incomplete output.

---

## 8. Shared LLM Client

All six agents share **one Groq client** and **one JSON parse helper**. This avoids duplicated configuration and ensures every agent uses identical inference settings.

### `agents/__init__.py`

```python
import os
import json
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Shared Groq client — used by all agents
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1,    # low = consistent, structured outputs
    max_tokens=500,
)


def safe_parse_json(content: str) -> dict:
    """Parse LLM JSON response, handling markdown code fences.

    Llama models sometimes wrap JSON in ```json ... ``` even when asked
    not to. This function strips those fences before parsing.
    """
    content = content.strip()
    if content.startswith("```"):
        for part in content.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                content = part
                break
    return json.loads(content.strip())
```

**Design principle:** One file. One client. One helper. Imported by every agent.

---

## 9. Agent Implementations

Every agent follows the same canonical pattern:

> **fetch data → compute numbers → ask LLM to interpret → return structured `AgentSignal`**

This consistency is what makes the synthesis step trivial — the synthesis agent doesn't need to understand each agent's domain, only the standardized signal format.

### 9.1 Price Agent — `agents/price_agent.py`

The price agent fetches 90 days of price history, computes three technical indicators, and asks Llama to interpret them.

```python
# agents/price_agent.py
import yfinance as yf
import pandas as pd
from agents import llm, safe_parse_json
from state import MarketMindState, AgentSignal


def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss
    return round(float((100 - (100 / (1 + rs))).iloc[-1]), 2)


def compute_macd(prices: pd.Series):
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return (
        round(float(macd_line.iloc[-1]), 4),
        round(float(signal_line.iloc[-1]), 4),
    )


def compute_bollinger(prices: pd.Series, period: int = 20) -> float:
    """Price position within Bollinger Bands: 0 = lower band, 1 = upper band."""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    band_range = float(upper.iloc[-1]) - float(lower.iloc[-1])
    if band_range == 0:
        return 0.5
    return round((prices.iloc[-1] - float(lower.iloc[-1])) / band_range, 3)


def price_agent(state: MarketMindState) -> dict:
    ticker = state["ticker"]
    # yfinance needs -USD suffix for crypto tickers
    yfin_ticker = f"{ticker}-USD" if state["asset_type"] == "crypto" else ticker

    try:
        data = yf.download(yfin_ticker, period="90d", interval="1d", progress=False)

        if data.empty:
            return {"agent_signals": [AgentSignal(
                agent="price", signal="NEUTRAL", confidence=0.0,
                summary=f"No price data for {ticker}", raw_data={}
            )]}

        close = data["Close"].squeeze()
        current_price = round(float(close.iloc[-1]), 4)
        price_7d_ago = round(float(close.iloc[-7]), 4)
        change_7d = round(((current_price - price_7d_ago) / price_7d_ago) * 100, 2)

        rsi = compute_rsi(close)
        macd_line, signal_line = compute_macd(close)
        bollinger = compute_bollinger(close)

        raw_data = {
            "current_price": current_price,
            "price_change_7d_pct": change_7d,
            "rsi": rsi,
            "macd_line": macd_line,
            "macd_signal": signal_line,
            "macd_crossover": "positive" if macd_line > signal_line else "negative",
            "bollinger_position": bollinger,
        }

        prompt = f"""You are a technical analyst. Analyze these indicators for {ticker}:

RSI: {rsi} (below 30 = oversold, above 70 = overbought)
MACD crossover: {raw_data['macd_crossover']} (positive = bullish momentum)
Bollinger position: {bollinger} (0 = at lower band, 1 = at upper band)
7-day price change: {change_7d}%

Return ONLY valid JSON with exactly these fields, no other text:
{{"signal": "BULLISH" or "BEARISH" or "NEUTRAL", "confidence": 0.0 to 1.0, "summary": "one sentence under 20 words"}}"""

        response = llm.invoke(prompt)
        result = safe_parse_json(response.content)

        return {"agent_signals": [AgentSignal(
            agent="price",
            signal=result["signal"],
            confidence=float(result["confidence"]),
            summary=result["summary"],
            raw_data=raw_data,
        )]}

    except Exception as e:
        return {"agent_signals": [AgentSignal(
            agent="price", signal="NEUTRAL", confidence=0.0,
            summary=f"Price agent error: {str(e)[:60]}", raw_data={}
        )]}
```

### 9.2 Sentiment Agent — `agents/sentiment_agent.py`

Uses NewsAPI to pull the last 3 days of relevant headlines, then asks Llama to rate aggregate sentiment.

```python
# agents/sentiment_agent.py
import os
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from agents import llm, safe_parse_json
from state import MarketMindState, AgentSignal

CRYPTO_NAMES = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana",
    "BNB": "Binance Coin", "DOGE": "Dogecoin", "ADA": "Cardano",
    "AVAX": "Avalanche", "DOT": "Polkadot", "MATIC": "Polygon",
    "LINK": "Chainlink", "XRP": "Ripple", "LTC": "Litecoin",
}


def sentiment_agent(state: MarketMindState) -> dict:
    ticker = state["ticker"]
    asset_type = state["asset_type"]

    try:
        newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))

        # Full name gets better results than ticker symbol
        query = CRYPTO_NAMES.get(ticker, ticker) if asset_type == "crypto" else ticker

        # 3-day window — free tier has a ~24h delay on some regions
        from_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        response = newsapi.get_everything(
            q=query, from_param=from_date,
            language="en", sort_by="relevancy", page_size=10,
        )

        headlines = [
            a["title"] for a in response.get("articles", [])
            if a.get("title") and a["title"] != "[Removed]"
        ][:10]

        if not headlines:
            return {"agent_signals": [AgentSignal(
                agent="sentiment", signal="NEUTRAL", confidence=0.3,
                summary="No recent news found", raw_data={"headline_count": 0}
            )]}

        headlines_text = "\n".join(f"- {h}" for h in headlines)

        prompt = f"""You are a financial sentiment analyst. Rate the market sentiment from these recent headlines about {ticker}:

{headlines_text}

Return ONLY valid JSON with exactly these fields, no other text:
{{"signal": "BULLISH" or "BEARISH" or "NEUTRAL", "confidence": 0.0 to 1.0, "summary": "one sentence under 20 words", "positive_count": number, "negative_count": number}}"""

        result_raw = llm.invoke(prompt)
        result = safe_parse_json(result_raw.content)

        return {"agent_signals": [AgentSignal(
            agent="sentiment",
            signal=result["signal"],
            confidence=float(result["confidence"]),
            summary=result["summary"],
            raw_data={
                "headline_count": len(headlines),
                "positive_count": result.get("positive_count", 0),
                "negative_count": result.get("negative_count", 0),
                "sample_headline": headlines[0],
            }
        )]}

    except Exception as e:
        return {"agent_signals": [AgentSignal(
            agent="sentiment", signal="NEUTRAL", confidence=0.0,
            summary=f"Sentiment agent error: {str(e)[:60]}", raw_data={}
        )]}
```

### 9.3 On-Chain Agent — `agents/onchain_agent.py`

Branches by asset type: CoinGecko for crypto, yfinance for stocks. Both produce a comparable microstructure signal.

```python
# agents/onchain_agent.py
import json
import yfinance as yf
from pycoingecko import CoinGeckoAPI
from agents import llm, safe_parse_json
from state import MarketMindState, AgentSignal

COINGECKO_IDS = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
    "BNB": "binancecoin", "DOGE": "dogecoin", "ADA": "cardano",
    "AVAX": "avalanche-2", "DOT": "polkadot", "MATIC": "matic-network",
    "LINK": "chainlink", "XRP": "ripple", "LTC": "litecoin",
}


def onchain_agent(state: MarketMindState) -> dict:
    ticker = state["ticker"]
    asset_type = state["asset_type"]

    try:
        if asset_type == "crypto":
            coin_id = COINGECKO_IDS.get(ticker.upper())
            if not coin_id:
                return {"agent_signals": [AgentSignal(
                    agent="onchain", signal="NEUTRAL", confidence=0.3,
                    summary=f"{ticker} not in supported crypto list", raw_data={}
                )]}

            cg = CoinGeckoAPI()
            data = cg.get_coin_by_id(
                coin_id, localization=False, tickers=False,
                market_data=True, community_data=False, developer_data=False,
            )
            market = data.get("market_data", {})
            market_cap = market.get("market_cap", {}).get("usd", 0)
            total_vol = market.get("total_volume", {}).get("usd", 0)

            raw_data = {
                "market_cap_rank": data.get("market_cap_rank", "N/A"),
                "price_change_24h_pct": round(market.get("price_change_percentage_24h", 0), 2),
                "price_change_7d_pct": round(market.get("price_change_percentage_7d", 0), 2),
                "volume_to_market_cap_ratio": round(total_vol / market_cap, 4) if market_cap > 0 else 0,
                "ath_change_pct": round(market.get("ath_change_percentage", {}).get("usd", 0), 2),
            }

        else:
            stock = yf.Ticker(ticker)
            info = stock.info
            avg_vol = max(info.get("averageVolume", 1), 1)
            raw_data = {
                "volume_ratio": round(info.get("volume", 0) / avg_vol, 3),
                "market_cap": info.get("marketCap", 0),
                "short_ratio": info.get("shortRatio", 0),
                "beta": info.get("beta", 1.0),
            }

        prompt = f"""You are a market microstructure analyst. Analyze this data for {ticker} ({asset_type}):

{json.dumps(raw_data, indent=2)}

Focus on: volume patterns, market activity, structural signals.

Return ONLY valid JSON with exactly these fields, no other text:
{{"signal": "BULLISH" or "BEARISH" or "NEUTRAL", "confidence": 0.0 to 1.0, "summary": "one sentence under 20 words"}}"""

        result_raw = llm.invoke(prompt)
        result = safe_parse_json(result_raw.content)

        return {"agent_signals": [AgentSignal(
            agent="onchain",
            signal=result["signal"],
            confidence=float(result["confidence"]),
            summary=result["summary"],
            raw_data=raw_data,
        )]}

    except Exception as e:
        return {"agent_signals": [AgentSignal(
            agent="onchain", signal="NEUTRAL", confidence=0.0,
            summary=f"On-chain agent error: {str(e)[:60]}", raw_data={}
        )]}
```

### 9.4 Macro Agent — `agents/macro_agent.py`

Macro conditions affect everything. Rising DXY (US Dollar Index) is bearish for risk assets. Fed rate hikes slow spending. This agent fetches real economic data from the Federal Reserve's FRED API — completely free.

```python
# agents/macro_agent.py
import os
import json
import requests
from agents import llm, safe_parse_json
from state import MarketMindState, AgentSignal


def fetch_fred(series_id: str, api_key: str, limit: int = 5) -> list:
    """Fetch recent values from a FRED economic data series."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id, "api_key": api_key,
        "file_type": "json", "sort_order": "desc", "limit": limit,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    # FRED uses "." as placeholder for missing/unreleased data — filter those out
    return [
        {"date": o["date"], "value": round(float(o["value"]), 4)}
        for o in resp.json().get("observations", [])
        if o.get("value") and o["value"] != "."
    ]
```

> **Note on coverage:** the source tutorial provides the FRED helper above through page 19/20. The full `macro_agent` function body, the Risk Agent, the Synthesis Agent, the LangGraph wiring (`graph.py`), and the Streamlit dashboard (`app.py`) are referenced in the tutorial's index (sections 11–15) but their full code listings are not present in the extracted slides 1–19. The implementations below are **inferred reference implementations** consistent with the patterns established by the Price, Sentiment, and On-Chain agents.

### 9.5 Macro Agent (inferred completion)

```python
def macro_agent(state: MarketMindState) -> dict:
    ticker = state["ticker"]

    try:
        api_key = os.getenv("FRED_API_KEY")

        # DXY (Dollar Index), Fed Funds Rate, 10Y Treasury yield
        dxy = fetch_fred("DTWEXBGS", api_key, limit=5)
        fed_funds = fetch_fred("DFF", api_key, limit=5)
        treasury_10y = fetch_fred("DGS10", api_key, limit=5)

        # Fear & Greed index (alternative.me, free, no key)
        fg_resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        fear_greed = int(fg_resp.json()["data"][0]["value"])

        raw_data = {
            "dxy_latest": dxy[0]["value"] if dxy else None,
            "dxy_5d_change": round(dxy[0]["value"] - dxy[-1]["value"], 2) if len(dxy) >= 2 else 0,
            "fed_funds_rate": fed_funds[0]["value"] if fed_funds else None,
            "treasury_10y_yield": treasury_10y[0]["value"] if treasury_10y else None,
            "fear_greed_index": fear_greed,
        }

        prompt = f"""You are a macro strategist. Analyze the macro regime for {ticker}:

{json.dumps(raw_data, indent=2)}

Rules of thumb:
- Rising DXY is bearish for risk assets (stocks, crypto)
- High Fed funds rate is bearish for risk assets
- Steepening yield curve is generally pro-risk
- Fear & Greed: 0-25 = extreme fear, 75-100 = extreme greed

Return ONLY valid JSON:
{{"signal": "BULLISH" or "BEARISH" or "NEUTRAL", "confidence": 0.0 to 1.0, "summary": "one sentence under 20 words"}}"""

        result_raw = llm.invoke(prompt)
        result = safe_parse_json(result_raw.content)

        return {"agent_signals": [AgentSignal(
            agent="macro",
            signal=result["signal"],
            confidence=float(result["confidence"]),
            summary=result["summary"],
            raw_data=raw_data,
        )]}

    except Exception as e:
        return {"agent_signals": [AgentSignal(
            agent="macro", signal="NEUTRAL", confidence=0.0,
            summary=f"Macro agent error: {str(e)[:60]}", raw_data={}
        )]}
```

### 9.6 Risk Agent (inferred reference implementation)

```python
# agents/risk_agent.py
import numpy as np
import yfinance as yf
from agents import llm, safe_parse_json
from state import MarketMindState, AgentSignal


def risk_agent(state: MarketMindState) -> dict:
    ticker = state["ticker"]
    yfin_ticker = f"{state['ticker']}-USD" if state["asset_type"] == "crypto" else ticker

    try:
        data = yf.download(yfin_ticker, period="90d", interval="1d", progress=False)
        close = data["Close"].squeeze()
        returns = close.pct_change().dropna()

        annualized_vol = float(returns.std() * np.sqrt(252)) * 100
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = float(drawdown.min()) * 100

        # Risk-free rate proxy: ~4% for 2026
        sharpe = float((returns.mean() * 252 - 0.04) / (returns.std() * np.sqrt(252)))

        # VIX
        vix_data = yf.download("^VIX", period="5d", interval="1d", progress=False)
        vix = round(float(vix_data["Close"].iloc[-1]), 2)

        raw_data = {
            "annualized_vol_pct": round(annualized_vol, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe, 2),
            "vix": vix,
        }

        prompt = f"""You are a risk analyst. Assess risk-adjusted attractiveness for {ticker}:

Annualized volatility: {raw_data['annualized_vol_pct']}%
Max drawdown (90d): {raw_data['max_drawdown_pct']}%
Sharpe ratio: {raw_data['sharpe_ratio']}
VIX (market fear gauge): {vix}

High vol + negative Sharpe = BEARISH. Low vol + positive Sharpe = BULLISH. VIX > 25 = elevated stress.

Return ONLY valid JSON:
{{"signal": "BULLISH" or "BEARISH" or "NEUTRAL", "confidence": 0.0 to 1.0, "summary": "one sentence under 20 words"}}"""

        result_raw = llm.invoke(prompt)
        result = safe_parse_json(result_raw.content)

        return {"agent_signals": [AgentSignal(
            agent="risk",
            signal=result["signal"],
            confidence=float(result["confidence"]),
            summary=result["summary"],
            raw_data=raw_data,
        )]}

    except Exception as e:
        return {"agent_signals": [AgentSignal(
            agent="risk", signal="NEUTRAL", confidence=0.0,
            summary=f"Risk agent error: {str(e)[:60]}", raw_data={}
        )]}
```

### 9.7 Synthesis Agent (inferred reference implementation)

```python
# agents/synthesis_agent.py
import json
from agents import llm, safe_parse_json
from state import MarketMindState


def synthesis_agent(state: MarketMindState) -> dict:
    signals = state["agent_signals"]

    summary_block = "\n".join(
        f"- {s['agent'].upper()}: {s['signal']} (confidence {s['confidence']:.2f}) — {s['summary']}"
        for s in signals
    )

    prompt = f"""You are the chief investment strategist. Synthesize the following five specialist signals for {state['ticker']} into a final verdict.

{summary_block}

Weighting heuristics:
- Strong consensus across agents → high confidence
- Macro and Risk signals override Price/Sentiment when they conflict and confidence is high
- Mixed signals with no clear majority → HOLD

Return ONLY valid JSON:
{{"verdict": "BUY" or "HOLD" or "SELL", "confidence": 0.0 to 1.0, "reasoning": "two to three sentences explaining the call"}}"""

    response = llm.invoke(prompt)
    result = safe_parse_json(response.content)

    return {
        "final_verdict": result["verdict"],
        "final_confidence": float(result["confidence"]),
        "final_reasoning": result["reasoning"],
    }
```

---

## 10. Graph Wiring (inferred reference implementation)

```python
# graph.py
from langgraph.graph import StateGraph, END
from state import MarketMindState
from agents.price_agent import price_agent
from agents.sentiment_agent import sentiment_agent
from agents.onchain_agent import onchain_agent
from agents.macro_agent import macro_agent
from agents.risk_agent import risk_agent
from agents.synthesis_agent import synthesis_agent


def orchestrator(state: MarketMindState) -> dict:
    # Pure pass-through node — its job is to anchor the fan-out edges
    return {}


def build_graph():
    g = StateGraph(MarketMindState)

    g.add_node("orchestrator", orchestrator)
    g.add_node("price", price_agent)
    g.add_node("sentiment", sentiment_agent)
    g.add_node("onchain", onchain_agent)
    g.add_node("macro", macro_agent)
    g.add_node("risk", risk_agent)
    g.add_node("synthesis", synthesis_agent)

    g.set_entry_point("orchestrator")

    # FAN-OUT: orchestrator → all 5 specialists in the same superstep
    for agent_name in ["price", "sentiment", "onchain", "macro", "risk"]:
        g.add_edge("orchestrator", agent_name)

    # FAN-IN: all 5 specialists → synthesis
    for agent_name in ["price", "sentiment", "onchain", "macro", "risk"]:
        g.add_edge(agent_name, "synthesis")

    g.add_edge("synthesis", END)

    return g.compile()


graph = build_graph()
```

---

## 11. Streamlit Dashboard (inferred reference implementation)

```python
# app.py
import streamlit as st
from graph import graph

st.set_page_config(page_title="MarketMind", layout="wide")
st.title("MarketMind")
st.caption("Parallel Multi-Agent Stock & Crypto Analyst | LangGraph + Groq Llama 3.3 70B")

ticker = st.text_input("Enter ticker symbol", value="BTC").upper().strip()
asset_type = st.radio("Asset type", ["crypto", "stock"], horizontal=True)

if st.button("Analyze", type="primary"):
    with st.spinner("All 5 agents firing in parallel..."):
        result = graph.invoke({
            "ticker": ticker,
            "asset_type": asset_type,
            "agent_signals": [],
            "final_verdict": None,
            "final_confidence": None,
            "final_reasoning": None,
        })

    col1, col2, col3 = st.columns(3)
    col1.metric("Verdict", result["final_verdict"])
    col2.metric("Confidence", f"{result['final_confidence']*100:.0f}%")
    col3.progress(result["final_confidence"])

    st.subheader("Reasoning")
    st.write(result["final_reasoning"])

    st.subheader("Agent Breakdown")
    cols = st.columns(5)
    for col, signal in zip(cols, result["agent_signals"]):
        with col:
            st.markdown(f"**{signal['agent'].title()}**")
            st.markdown(f"`{signal['signal']}`")
            st.caption(f"Confidence: {signal['confidence']*100:.0f}%")
            st.caption(signal["summary"])
```

Run with:

```bash
streamlit run app.py
```

---

## 12. Expected Output

Sample terminal output for `BTC`:

```
Analyzing: BTC
[PARALLEL EXECUTION — all 5 agents firing simultaneously]

Price Agent      → BULLISH  (RSI: 58, MACD: positive crossover)
Sentiment Agent  → BULLISH  (8 positive headlines found)
On-Chain Agent   → NEUTRAL  (moderate market activity)
Macro Agent      → BEARISH  (DXY rising, Fed hawkish tone)
Risk Agent       → NEUTRAL  (30-day vol: 42%, VIX: 22.1)

Synthesis Agent  → HOLD (62% confidence)
Reasoning: Technicals and sentiment lean bullish but macro headwinds
and elevated volatility suggest waiting for confirmation.

Total execution time: ~4 seconds. Sequential execution would take ~20 seconds.
```

---

## 13. Functional Requirements

| ID | Requirement |
|---|---|
| FR-1 | The system accepts a single ticker symbol (string, uppercase) and asset type (`stock` or `crypto`). |
| FR-2 | The system returns a verdict in `{BUY, HOLD, SELL}`, a confidence in `[0.0, 1.0]`, and a free-text reasoning. |
| FR-3 | The five specialist agents must execute concurrently in a single LangGraph superstep. |
| FR-4 | Each specialist agent must produce an `AgentSignal` with `agent`, `signal`, `confidence`, `summary`, and `raw_data` fields. |
| FR-5 | The shared state's `agent_signals` field must use `Annotated[list, operator.add]` to prevent race conditions. |
| FR-6 | The system must continue to produce a verdict even if individual agents fail (graceful degradation via try/except returning a NEUTRAL signal with confidence 0.0). |
| FR-7 | The Streamlit dashboard must display the final verdict, the confidence, the reasoning, and each individual agent's contribution. |
| FR-8 | All API keys must be loaded from `.env` via `python-dotenv`, never hard-coded. |

## 14. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-1 | End-to-end latency target: ≤ 6 seconds at the p50 for a single ticker analysis. |
| NFR-2 | The system must operate entirely on free API tiers under typical evaluation usage. |
| NFR-3 | The Groq client must use `temperature=0.1` and `max_tokens=500` for consistent structured output. |
| NFR-4 | All LLM responses must be parsed through `safe_parse_json` to handle markdown fence wrapping. |
| NFR-5 | The system must handle empty data responses (e.g., delisted ticker, no news headlines) without raising. |
| NFR-6 | `.env` must be added to `.gitignore` and never committed. |

---

## 15. Common Issues and Fixes

| Issue | Cause | Fix |
|---|---|---|
| Only one agent's signal in `agent_signals` | Missing `operator.add` reducer in state | Ensure `agent_signals: Annotated[list, operator.add]` |
| `JSONDecodeError` on LLM response | Llama wrapped JSON in ` ```json ... ``` ` fences | Use `safe_parse_json` helper |
| `yfinance` returns empty for crypto | Missing `-USD` suffix | Append `-USD` for crypto tickers (`BTC-USD`, `ETH-USD`) |
| `KeyError: 'GROQ_API_KEY'` | `.env` not loaded | Confirm `load_dotenv()` runs at import time in `agents/__init__.py` |
| Agents run sequentially, not in parallel | Linear edges in graph (A → B → C) | Use parallel `add_edge` from orchestrator to each agent |
| FRED API returns `"."` values | FRED uses `.` as placeholder for missing data | Filter out before float conversion |
| NewsAPI returns "[Removed]" titles | Free-tier scrubbing | Filter out in list comprehension |

---

## 16. Future Extensions

1. **Multi-ticker batch mode** — accept a basket of tickers and run the full pipeline for each, then rank by confidence.
2. **Historical backtesting harness** — replay the agents over past dates using point-in-time data to measure verdict accuracy.
3. **Persistent verdict log** — store every analysis in SQLite/Postgres with timestamp, ticker, agent signals, and verdict for later evaluation.
4. **Agent weight tuning** — replace the synthesis LLM call with a learned weighted ensemble fitted on historical verdict accuracy per agent.
5. **Streaming output** — surface each agent's signal in the UI as it completes, rather than waiting for synthesis.
6. **Additional specialists** — options-flow agent (unusual options activity), insider-trading agent (Form 4 filings), ESG agent.
7. **Production hardening** — add observability (LangSmith / OpenTelemetry), distributed tracing, retries with exponential backoff, structured logging, and a circuit breaker per data source.

---

## 17. Source Attribution

This PRD is based on the tutorial **"Build a Parallel Multi-Agent Stock & Crypto Analyst In 60 Minutes"** by **@datasciencebrain** (Instagram), code verified April 22, 2026. Sections 1–10 reflect the source tutorial verbatim where extracted. Sections covering the Macro Agent body, Risk Agent, Synthesis Agent, graph wiring, and Streamlit dashboard are reference implementations consistent with the patterns established in the source, since the corresponding source slides were not present in the provided extraction.
