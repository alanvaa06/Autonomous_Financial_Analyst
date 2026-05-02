# app.py
"""Gradio shell for MarketMind v2 — multi-agent equity analyst.

BYO-key, per-session isolation, streaming agent updates.
"""
from __future__ import annotations

import os
import time
import traceback
from typing import Any, Dict, Generator, Tuple

# gradio_client 1.3.0 schema-introspection workaround. Must run before `import gradio`.
import gradio_client.utils as _gc_utils

_orig_json_schema = _gc_utils._json_schema_to_python_type


def _safe_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_json_schema(schema, defs)


_gc_utils._json_schema_to_python_type = _safe_json_schema_to_python_type

import gradio as gr

from agents import build_llm_clients
from graph import build_graph
from ratelimit import SessionRateLimiter

ALLOW_ENV_KEYS = os.environ.get("ALLOW_ENV_KEYS", "").lower() in ("1", "true", "yes")

PILL_LABELS = {
    "price": "Technicals",
    "sentiment": "Sentiment",
    "fundamentals": "Fundamentals",
    "macro": "Macro",
    "risk": "Risk",
}
SPECIALISTS = list(PILL_LABELS.keys())


def _new_state() -> Dict[str, Any]:
    s = {
        "anthropic_key": "", "tavily_key": "", "fred_key": "",
        "sec_user_agent": "MarketMind/2.0 contact@marketmind.local",
        "rate_limiter": SessionRateLimiter(),
    }
    if ALLOW_ENV_KEYS:
        s["anthropic_key"] = os.environ.get("ANTHROPIC_API_KEY", "")
        s["tavily_key"] = os.environ.get("TAVILY_API_KEY", "")
        s["fred_key"] = os.environ.get("FRED_API_KEY", "")
    return s


def _save_keys(state, anthropic, tavily, fred, ua):
    state = dict(state)
    state["anthropic_key"] = (anthropic or "").strip()
    state["tavily_key"] = (tavily or "").strip()
    state["fred_key"] = (fred or "").strip()
    state["sec_user_agent"] = (ua or "").strip() or "MarketMind/2.0 contact@marketmind.local"
    os.environ["SEC_USER_AGENT"] = state["sec_user_agent"]
    return state, _format_status(state)


def _format_status(state) -> str:
    def dot(b: bool) -> str:
        return "🟢" if b else "🔴"
    rows = [
        f"{dot(bool(state.get('anthropic_key')))} Anthropic",
        f"{dot(bool(state.get('tavily_key')))} Tavily",
        f"{dot(bool(state.get('fred_key')))} FRED (optional)",
    ]
    return " · ".join(rows)


def _pill(name: str, status: str) -> str:
    color = {"pending": "#888", "running": "#3a7", "done": "#27a", "degraded": "#a82", "error": "#a33"}.get(status, "#888")
    return f'<span style="background:{color};color:#fff;padding:4px 10px;border-radius:12px;margin:2px;display:inline-block;">{PILL_LABELS[name]}: {status}</span>'


def _pills_html(status_map: Dict[str, str]) -> str:
    return " ".join(_pill(n, status_map.get(n, "pending")) for n in SPECIALISTS)


def analyze(state, ticker: str) -> Generator[Tuple[str, str], None, None]:
    """Streaming generator: yields (status_html, report_markdown) tuples."""
    state = state or _new_state()

    # Validate keys
    if not state.get("anthropic_key"):
        yield _pills_html({}), "**Missing Anthropic API key.** Configure it in the keys panel."
        return
    if not state.get("tavily_key"):
        yield _pills_html({}), "**Missing Tavily API key.** Configure it in the keys panel."
        return

    # Rate limit
    rl: SessionRateLimiter = state.get("rate_limiter") or SessionRateLimiter()
    state["rate_limiter"] = rl
    allowed, reason = rl.check("analyze")
    if not allowed:
        yield _pills_html({}), f"**Rate limited.** {reason}"
        return

    ticker = (ticker or "").strip().upper()
    if not ticker:
        yield _pills_html({}), "**Enter a ticker.**"
        return

    # SEC_USER_AGENT is a contact string (not a secret) — written to os.environ
    # so the edgar.py module's _user_agent() helper can pick it up at call time.
    os.environ["SEC_USER_AGENT"] = state.get("sec_user_agent") or "MarketMind/2.0 contact@marketmind.local"

    clients = build_llm_clients(state["anthropic_key"])
    graph = build_graph(clients, tavily_key=state["tavily_key"], fred_key=state.get("fred_key", ""))

    init = {
        "ticker": ticker, "company_name": None, "cik": None,
        "price_history": None, "vix_history": None,
        "agent_signals": [], "retry_round": 0, "supervisor_review": None,
        "final_verdict": None, "final_conviction": None,
        "final_confidence": None, "final_reasoning": None, "final_report": None,
    }

    status_map = {n: "pending" for n in SPECIALISTS}
    partial_sections: Dict[str, str] = {}
    yield _pills_html(status_map), f"_Running MarketMind on **{ticker}**..._"

    try:
        last_state = init
        for event in graph.stream(init, stream_mode="values"):
            last_state = event
            for s in event.get("agent_signals") or []:
                a = s.get("agent")
                if a in status_map:
                    if s.get("error"):
                        status_map[a] = "error"
                    elif s.get("degraded"):
                        status_map[a] = "degraded"
                    else:
                        status_map[a] = "done"
                    if s.get("section_markdown"):
                        partial_sections[a] = s["section_markdown"]
            interim = "\n\n".join(partial_sections[n] for n in SPECIALISTS if n in partial_sections) or "_Agents running..._"
            yield _pills_html(status_map), interim

        final = last_state.get("final_report") or "_(no report produced)_"
        yield _pills_html(status_map), final
    except Exception:
        yield _pills_html(status_map), f"**Run failed.**\n\n```\n{traceback.format_exc()[-1500:]}\n```"


CSS = """
.report { padding: 8px 14px; }
"""

with gr.Blocks(title="MarketMind v2", css=CSS) as demo:
    state = gr.State(_new_state())

    gr.Markdown("# MarketMind v2 — Multi-Agent Equity Analyst")
    gr.Markdown(
        "Five specialists run in parallel (Technicals, Sentiment, Fundamentals, Macro, Risk), "
        "a Supervisor performs QA, and a Synthesis layer assembles the final report."
    )

    with gr.Accordion("API keys (BYO)", open=False):
        anthropic_box = gr.Textbox(label="Anthropic API key", type="password")
        tavily_box = gr.Textbox(label="Tavily API key", type="password")
        fred_box = gr.Textbox(label="FRED API key (optional — enables full Macro)", type="password")
        ua_box = gr.Textbox(label="SEC User-Agent email", value="MarketMind/2.0 contact@marketmind.local")
        save_btn = gr.Button("Save keys")
        status_label = gr.Markdown(_format_status(_new_state()))
        save_btn.click(_save_keys, [state, anthropic_box, tavily_box, fred_box, ua_box], [state, status_label])

    with gr.Row():
        ticker_box = gr.Textbox(label="Ticker", placeholder="MSFT", scale=3)
        analyze_btn = gr.Button("Analyze (~$0.30/run on BYO key)", variant="primary", scale=1)

    pills = gr.HTML()
    report = gr.Markdown(elem_classes="report")

    analyze_btn.click(analyze, [state, ticker_box], [pills, report])

if __name__ == "__main__":
    demo.queue(max_size=8).launch()
