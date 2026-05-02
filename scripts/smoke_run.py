"""Manual smoke test for MarketMind v2. Runs against real services.

Usage:
    ANTHROPIC_API_KEY=... TAVILY_API_KEY=... FRED_API_KEY=... \
    python scripts/smoke_run.py MSFT
"""
from __future__ import annotations

import os
import sys

from agents import build_llm_clients
from graph import build_graph


def main(ticker: str) -> None:
    anthropic_key = os.environ["ANTHROPIC_API_KEY"]
    tavily_key = os.environ["TAVILY_API_KEY"]
    fred_key = os.environ.get("FRED_API_KEY", "")
    clients = build_llm_clients(anthropic_key)
    g = build_graph(clients, tavily_key=tavily_key, fred_key=fred_key)
    init = {
        "ticker": ticker, "company_name": None, "cik": None,
        "agent_signals": [], "retry_round": 0, "supervisor_review": None,
        "final_verdict": None, "final_conviction": None,
        "final_confidence": None, "final_reasoning": None, "final_report": None,
    }
    out = g.invoke(init)
    print("VERDICT:", out["final_verdict"], out["final_conviction"], "conf", out["final_confidence"])
    print("------ REPORT ------")
    print(out["final_report"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "MSFT")
