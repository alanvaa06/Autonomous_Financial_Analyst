"""LangGraph wiring for MarketMind v2.

  orchestrator -> [price, sentiment, fundamentals, macro, risk]  (parallel)
                              |
                           supervisor
                              |
              ┌───────────────┴───────────────┐
              ▼                               ▼
      retry_targets nonempty            approved or forced
              │                               │
       Send(target_agents)                synthesis
              │                               │
            (back to supervisor)              END

After one retry round, supervisor force-approves.
"""

from __future__ import annotations

from typing import Optional

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from agents.orchestrator import orchestrator
from agents.price_agent import price_agent
from agents.sentiment_agent import sentiment_agent
from agents.fundamentals_agent import fundamentals_agent
from agents.macro_agent import macro_agent
from agents.risk_agent import risk_agent
from agents.supervisor_agent import supervisor_agent
from agents.synthesis_agent import synthesis_agent
from state import MarketMindState


SPECIALIST_NODES = ("price", "sentiment", "fundamentals", "macro", "risk")


def build_graph(llm_clients, tavily_key: str, fred_key: Optional[str] = ""):
    """Compile the MarketMind graph with the session's bound clients/keys."""

    def _price(state):
        return price_agent(state, llm_clients)

    def _sentiment(state):
        return sentiment_agent(state, llm_clients, tavily_key=tavily_key)

    def _fundamentals(state):
        return fundamentals_agent(state, llm_clients)

    def _macro(state):
        return macro_agent(state, llm_clients, fred_key=fred_key or "")

    def _risk(state):
        return risk_agent(state, llm_clients)

    def _synthesis(state):
        return synthesis_agent(state, llm_clients)

    def _bump_retry_round(state):
        # Increments retry_round before re-running flagged specialists.
        return {"retry_round": int(state.get("retry_round", 0)) + 1}

    g = StateGraph(MarketMindState)
    g.add_node("orchestrator", orchestrator)
    g.add_node("price", _price)
    g.add_node("sentiment", _sentiment)
    g.add_node("fundamentals", _fundamentals)
    g.add_node("macro", _macro)
    g.add_node("risk", _risk)
    g.add_node("supervisor", supervisor_agent)
    g.add_node("retry_bump", _bump_retry_round)
    g.add_node("synthesis", _synthesis)

    g.set_entry_point("orchestrator")
    for name in SPECIALIST_NODES:
        g.add_edge("orchestrator", name)
        g.add_edge(name, "supervisor")

    def _after_supervisor(state) -> str:
        review = state.get("supervisor_review") or {}
        if review.get("approved") or not review.get("retry_targets"):
            return "synthesis"
        return "retry_bump"

    g.add_conditional_edges("supervisor", _after_supervisor, {
        "synthesis": "synthesis",
        "retry_bump": "retry_bump",
    })

    def _fan_out_retries(state):
        targets = ((state.get("supervisor_review") or {}).get("retry_targets")) or []
        return [Send(t, state) for t in targets if t in SPECIALIST_NODES]

    g.add_conditional_edges("retry_bump", _fan_out_retries, list(SPECIALIST_NODES))
    g.add_edge("synthesis", END)
    return g.compile()
