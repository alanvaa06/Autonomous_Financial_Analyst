"""
Tools exposed to the Financial Research Analyst agent (BYO-key edition).

Split into two families:

**STATIC_TOOLS** — don't need any API key, share safely across sessions:
    1. get_stock_price                — yfinance current price
    2. get_stock_history              — yfinance 3-year performance
    3. get_analyst_ratings            — yfinance Buy/Hold/Sell consensus
    4. get_financials                 — yfinance income / balance / cashflow
    5. calculate_technical_indicators — RSI(14), MA50, MA200, 52w high/low

**build_session_tools(...)** — rebuilt per Gradio session, closes over
    the user's keys and their session-scoped RAG retriever:
    6. search_financial_news          — Tavily (uses session's tavily_key)
    7. analyze_sentiment              — Claude Haiku (uses session's anthropic_key)
    8. query_private_database         — RAG over session + base corpora,
                                         answered with Claude Sonnet

Every tool that hits an API receives its key via closure — we NEVER write
user-supplied keys into os.environ, because the process is shared across all
concurrent users of the public Space.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

import rag


# Models (per the environment's active Claude family)
AGENT_MODEL = "claude-sonnet-4-6"
SENTIMENT_MODEL = "claude-haiku-4-5-20251001"
RAG_MODEL = "claude-sonnet-4-6"


# ===========================================================================
#                        STATIC TOOLS (no API key needed)
# ===========================================================================

@tool
def get_stock_price(ticker: str) -> Dict:
    """
    Returns the current stock price and basic information for a given ticker.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL', 'MSFT').
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        current_price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )
        if current_price is None:
            return {
                "ticker": ticker.upper(),
                "status": "error",
                "error": f"Could not retrieve price data for {ticker}.",
            }
        return {
            "ticker": ticker.upper(),
            "current_price": round(current_price, 2),
            "currency": info.get("currency", "USD"),
            "day_high": info.get("dayHigh", info.get("regularMarketDayHigh")),
            "day_low": info.get("dayLow", info.get("regularMarketDayLow")),
            "volume": info.get("volume", info.get("regularMarketVolume")),
            "market_cap": info.get("marketCap"),
            "company_name": info.get("longName", info.get("shortName")),
            "timestamp": datetime.now().isoformat(),
            "status": "success",
        }
    except Exception as e:
        return {
            "ticker": ticker.upper(),
            "status": "error",
            "error": f"Error fetching stock data: {e}",
            "timestamp": datetime.now().isoformat(),
        }


@tool
def get_stock_history(ticker: str, period: str = "3y") -> Dict:
    """
    Returns historical price data (default 3y) for trend analysis.

    Args:
        ticker: Stock ticker symbol.
        period: '1mo', '3mo', '6mo', '1y', '2y', '3y', '5y', '10y'.
    """
    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period=period)
        if hist.empty:
            return {
                "ticker": ticker.upper(),
                "status": "error",
                "error": f"No history for {ticker} over {period}.",
            }
        start_price = hist["Close"].iloc[0]
        end_price = hist["Close"].iloc[-1]
        return {
            "ticker": ticker.upper(),
            "period": period,
            "start_date": hist.index[0].strftime("%Y-%m-%d"),
            "end_date": hist.index[-1].strftime("%Y-%m-%d"),
            "start_price": round(start_price, 2),
            "end_price": round(end_price, 2),
            "return_pct": round((end_price - start_price) / start_price * 100, 2),
            "high": round(hist["High"].max(), 2),
            "low": round(hist["Low"].min(), 2),
            "avg_volume": int(hist["Volume"].mean()),
            "data_points": len(hist),
            "status": "success",
        }
    except Exception as e:
        return {"ticker": ticker.upper(), "status": "error", "error": str(e)}


@tool
def get_analyst_ratings(ticker: str) -> Dict:
    """
    Wall Street analyst consensus (strong buy / buy / hold / sell / strong sell
    counts) plus mean / high / low price target. Use to ground Buy/Hold/Sell.
    """
    try:
        stock = yf.Ticker(ticker.upper())
        rec = stock.recommendations
        info = stock.info

        if rec is None or rec.empty:
            counts = {"note": "No recommendations table returned by yfinance."}
        else:
            latest = rec.iloc[0].to_dict()
            counts = {
                "period": str(latest.get("period", "current")),
                "strong_buy": int(latest.get("strongBuy", 0) or 0),
                "buy": int(latest.get("buy", 0) or 0),
                "hold": int(latest.get("hold", 0) or 0),
                "sell": int(latest.get("sell", 0) or 0),
                "strong_sell": int(latest.get("strongSell", 0) or 0),
            }
            counts["total_analysts"] = sum(
                v for k, v in counts.items() if k != "period" and isinstance(v, int)
            )

        return {
            "ticker": ticker.upper(),
            **counts,
            "recommendation_key": info.get("recommendationKey"),
            "recommendation_mean": info.get("recommendationMean"),
            "target_mean_price": info.get("targetMeanPrice"),
            "target_high_price": info.get("targetHighPrice"),
            "target_low_price": info.get("targetLowPrice"),
            "number_of_analyst_opinions": info.get("numberOfAnalystOpinions"),
            "currency": info.get("currency", "USD"),
            "timestamp": datetime.now().isoformat(),
            "status": "success",
        }
    except Exception as e:
        return {"ticker": ticker.upper(), "status": "error", "error": str(e)}


def _safe_latest(df, key):
    try:
        if df is None or df.empty or key not in df.index:
            return None
        val = df.loc[key].iloc[0]
        if pd.isna(val):
            return None
        return float(val)
    except Exception:
        return None


@tool
def get_financials(ticker: str) -> Dict:
    """
    Key fundamentals from the most recent annual filing: revenue, net income,
    margins, FCF, debt/assets, P/E, EPS, growth. Supports recommendations with
    fundamentals, not just price trend.
    """
    try:
        stock = yf.Ticker(ticker.upper())
        income, balance, cashflow = stock.income_stmt, stock.balance_sheet, stock.cashflow
        info = stock.info

        total_revenue = _safe_latest(income, "Total Revenue")
        net_income = _safe_latest(income, "Net Income")
        operating_income = _safe_latest(income, "Operating Income")
        gross_profit = _safe_latest(income, "Gross Profit")
        total_assets = _safe_latest(balance, "Total Assets")
        total_debt = _safe_latest(balance, "Total Debt")
        free_cash_flow = _safe_latest(cashflow, "Free Cash Flow")

        def pct(n, d):
            return round(100 * n / d, 2) if (n is not None and d) else None

        return {
            "ticker": ticker.upper(),
            "fiscal_year_revenue": total_revenue,
            "fiscal_year_net_income": net_income,
            "operating_income": operating_income,
            "gross_profit": gross_profit,
            "net_margin_pct": pct(net_income, total_revenue),
            "operating_margin_pct": pct(operating_income, total_revenue),
            "gross_margin_pct": pct(gross_profit, total_revenue),
            "free_cash_flow": free_cash_flow,
            "fcf_margin_pct": pct(free_cash_flow, total_revenue),
            "total_assets": total_assets,
            "total_debt": total_debt,
            "debt_to_assets_pct": pct(total_debt, total_assets),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "eps_trailing": info.get("trailingEps"),
            "eps_forward": info.get("forwardEps"),
            "revenue_growth_yoy": info.get("revenueGrowth"),
            "earnings_growth_yoy": info.get("earningsGrowth"),
            "profit_margins": info.get("profitMargins"),
            "return_on_equity": info.get("returnOnEquity"),
            "currency": info.get("currency", "USD"),
            "timestamp": datetime.now().isoformat(),
            "status": "success",
        }
    except Exception as e:
        return {"ticker": ticker.upper(), "status": "error", "error": str(e)}


@tool
def calculate_technical_indicators(ticker: str) -> Dict:
    """
    RSI(14), MA50, MA200, 52-week high/low, and price-vs-MA signals. Provides a
    technical (chart-based) read to complement fundamentals.
    """
    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period="2y")
        if hist.empty:
            return {"ticker": ticker.upper(), "status": "error", "error": "No history."}

        close = hist["Close"]
        current = float(close.iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

        last_year = close.iloc[-252:] if len(close) >= 252 else close
        week52_high = float(last_year.max())
        week52_low = float(last_year.min())

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, pd.NA)
        rsi_series = 100 - (100 / (1 + rs))
        rsi_val = (
            float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else None
        )

        def ma_signal(price, ma):
            return None if ma is None else ("above (bullish)" if price > ma else "below (bearish)")

        if rsi_val is None:
            rsi_sig = None
        elif rsi_val > 70:
            rsi_sig = "overbought (>70)"
        elif rsi_val < 30:
            rsi_sig = "oversold (<30)"
        else:
            rsi_sig = "neutral (30-70)"

        cross = None
        if ma50 is not None and ma200 is not None:
            cross = "golden cross (MA50 > MA200)" if ma50 > ma200 else "death cross (MA50 < MA200)"

        return {
            "ticker": ticker.upper(),
            "current_price": round(current, 2),
            "ma50": round(ma50, 2) if ma50 is not None else None,
            "ma200": round(ma200, 2) if ma200 is not None else None,
            "ma50_signal": ma_signal(current, ma50),
            "ma200_signal": ma_signal(current, ma200),
            "moving_average_cross": cross,
            "rsi_14": round(rsi_val, 2) if rsi_val is not None else None,
            "rsi_signal": rsi_sig,
            "week52_high": round(week52_high, 2),
            "week52_low": round(week52_low, 2),
            "pct_from_52w_high": round(100 * (current - week52_high) / week52_high, 2),
            "pct_from_52w_low": round(100 * (current - week52_low) / week52_low, 2),
            "timestamp": datetime.now().isoformat(),
            "status": "success",
        }
    except Exception as e:
        return {"ticker": ticker.upper(), "status": "error", "error": str(e)}


STATIC_TOOLS = [
    get_stock_price,
    get_stock_history,
    get_analyst_ratings,
    get_financials,
    calculate_technical_indicators,
]


# ===========================================================================
#                 PER-SESSION TOOL FACTORY (keys via closure)
# ===========================================================================


def _anthropic_content_to_text(content) -> str:
    """Claude responses may come back as a list of content blocks — collapse to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


def build_session_tools(
    anthropic_key: str,
    tavily_key: str,
    session_retriever=None,
):
    """
    Build the 3 API-key-dependent tools for a single Gradio session.

    The tools close over the user's keys and the session's RAG retriever.
    Keys are NEVER written to os.environ.

    Returns: list of 3 LangChain tools, ready to pass to `bind_tools`.
    """
    base_retriever = rag.get_base_retriever()
    # Capture at build time — stable reference for the life of this session
    _session_retriever = session_retriever

    @tool
    def search_financial_news(query: str) -> List[Dict]:
        """
        Searches real-time financial news via Tavily. Use for recent developments,
        market sentiment context, and company news.
        """
        try:
            tool_instance = TavilySearchResults(
                max_results=5,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=False,
                include_images=False,
                tavily_api_key=tavily_key,
            )
            return tool_instance.invoke(query)
        except Exception as e:
            return [{"status": "error", "error": f"Error searching news: {e}"}]

    @tool
    def analyze_sentiment(text: str) -> Dict:
        """
        Sentiment score (positive / negative / neutral) for a piece of financial
        text. Returns {sentiment, score, confidence, reasoning}.
        """
        try:
            model = ChatAnthropic(
                model=SENTIMENT_MODEL,
                temperature=0,
                max_tokens=512,
                anthropic_api_key=anthropic_key,
            )
            prompt = f"""Analyze the sentiment of this financial text and provide:
1. Sentiment label: positive, negative, or neutral
2. Score: 0.0 (very negative) to 1.0 (very positive), 0.5 is neutral
3. Confidence: 0.0 to 1.0
4. Brief reasoning

Text: {text}

Respond with JSON only (no prose, no code fences):
{{"sentiment": "positive|negative|neutral", "score": 0.0-1.0, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""
            response = model.invoke(prompt)
            content = _anthropic_content_to_text(response.content).strip()
            if content.startswith("```"):
                content = content.strip("`")
                if content.lower().startswith("json"):
                    content = content[4:].strip()
            result = json.loads(content)
            result["status"] = "success"
            return result
        except Exception as e:
            # Keyword fallback
            pos = ["growth", "profit", "gain", "success", "up", "positive", "strong"]
            neg = ["loss", "decline", "down", "weak", "risk", "concern", "negative"]
            t = text.lower()
            p, n = sum(w in t for w in pos), sum(w in t for w in neg)
            if p > n:
                sentiment, score = "positive", 0.6 + p * 0.05
            elif n > p:
                sentiment, score = "negative", 0.4 - n * 0.05
            else:
                sentiment, score = "neutral", 0.5
            return {
                "sentiment": sentiment,
                "score": max(0.0, min(1.0, score)),
                "confidence": 0.6,
                "reasoning": "Fallback keyword-based analysis",
                "status": "success (fallback)",
                "note": f"LLM sentiment failed: {e}",
            }

    @tool
    def query_private_database(query: str) -> str:
        """
        Query the private database of analyst reports about AI initiatives.

        Merges two sources:
        - User-uploaded PDFs for this session (if any)
        - The bundled Space base corpus (if any)

        Use this for: AI projects, research areas, innovation timelines, roadmaps.
        """
        try:
            docs = []
            seen_hashes = set()

            for retr, tag in (
                (_session_retriever, "user-uploaded"),
                (base_retriever, "bundled"),
            ):
                if retr is None:
                    continue
                try:
                    for d in retr.invoke(query):
                        h = hash(d.page_content[:200])
                        if h in seen_hashes:
                            continue
                        seen_hashes.add(h)
                        d.metadata = {**(d.metadata or {}), "corpus": tag}
                        docs.append(d)
                except Exception:
                    continue

            if not docs:
                return (
                    "No private analyst reports available. Upload PDFs in the "
                    "'Your Private Data' panel, or the bundled corpus is empty."
                )

            context_parts = []
            for d in docs[:12]:
                src = d.metadata.get("source", "unknown") if d.metadata else "unknown"
                tag = d.metadata.get("corpus", "") if d.metadata else ""
                context_parts.append(
                    f"[{tag} | {src}]\n{d.page_content}"
                )
            context_for_query = "\n\n---\n\n".join(context_parts)

            system_msg = """You are an assistant specialized in reviewing AI initiatives of companies and providing accurate answers based on the provided context.

User input will include all the context you need to answer their question.
This context will always begin with the token: ###Context.
The context contains references to specific AI initiatives, projects, or programs of companies relevant to the user's query.

User questions will begin with the token: ###Question.

Answer only using the context provided. Do not add external information or mention the context in your answer.
Always cite which company the information comes from.
If the answer cannot be found in the context, respond with "I don't know - this information is not available in our analyst reports."
"""
            user_msg = f"""###Context
Here are some documents that are relevant to the question mentioned below.
{context_for_query}

###Question
{query}
"""
            model = ChatAnthropic(
                model=RAG_MODEL,
                temperature=0,
                max_tokens=2048,
                anthropic_api_key=anthropic_key,
            )
            response = model.invoke(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ]
            )
            return _anthropic_content_to_text(response.content)
        except Exception as e:
            return f"Error querying private database: {e}"

    return [search_financial_news, analyze_sentiment, query_private_database]
