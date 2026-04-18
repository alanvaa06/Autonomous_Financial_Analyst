"""
Gradio interface for the Autonomous Financial Research Analyst.

Designed for **public Hugging Face Space deployment** with BYO-key + per-session
isolation:

- Each Gradio session gets its own `gr.State` holding the user's Anthropic +
  Tavily keys, their uploaded PDFs, a session-scoped Chroma retriever, and a
  freshly built LangGraph agent.
- User-supplied API keys are NEVER written to `os.environ` — they live only in
  the closures of that session's tools and model, so one user's key cannot
  leak to another user's request on the same process.
- Uploaded PDFs are indexed into an in-memory Chroma collection (no
  persist_directory) — not written to disk, not visible across sessions, and
  GC'd when the session ends.

Three tabs:
    1. 🔑  API Keys           — REQUIRED. Each user pastes their own keys.
    2. 📊  Analyze Company    — includes an optional "Your private PDFs" accordion.
    3. 🏆  Rank Companies

Env-var keys are ignored by default. Set `ALLOW_ENV_KEYS=1` for single-user
local development (so you don't have to paste on every reload).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Workaround for a gradio_client 1.3.0 bug where schema introspection crashes
# on `additionalProperties: True` (a bool, not a dict) — surfaces as
# `TypeError: argument of type 'bool' is not iterable` in get_type(), which
# Gradio's launch() then misreports as "localhost not accessible".
# Patch MUST be applied before `import gradio`.
# ---------------------------------------------------------------------------
import gradio_client.utils as _gc_utils

_orig_json_schema = _gc_utils._json_schema_to_python_type


def _safe_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_json_schema(schema, defs)


_gc_utils._json_schema_to_python_type = _safe_json_schema_to_python_type

import gradio as gr

import rag
from agent import build_agent_for_session, run_agent
from ratelimit import SessionRateLimiter


APP_DIR = Path(__file__).resolve().parent

ALLOW_ENV_KEYS = os.environ.get("ALLOW_ENV_KEYS", "").lower() in ("1", "true", "yes")

# Build the shared base corpus once at import time — local embeddings, no key
# needed, safe to share read-only across all sessions.
rag.build_base_pipeline()


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------
def _new_state() -> Dict[str, Any]:
    """Initial per-session state."""
    state = {
        "anthropic_key": "",
        "tavily_key": "",
        "uploaded_files": [],   # list[str]: filenames the user has indexed
        "session_retriever": None,
        "agent": None,
        "rate_limiter": SessionRateLimiter(),
    }
    if ALLOW_ENV_KEYS:
        state["anthropic_key"] = os.environ.get("ANTHROPIC_API_KEY", "") or ""
        state["tavily_key"] = os.environ.get("TAVILY_API_KEY", "") or ""
    return state


def _rate_check(state: Dict[str, Any], action: str) -> Tuple[bool, str]:
    """Thin wrapper that tolerates old sessions without a limiter."""
    rl = state.get("rate_limiter")
    if rl is None:
        rl = SessionRateLimiter()
        state["rate_limiter"] = rl
    return rl.check(action)


def _format_status(state: Dict[str, Any]) -> str:
    def dot(b: bool) -> str:
        cls = "ready" if b else "missing"
        return f'<span class="status-dot {cls}"></span>'

    def val(b: bool) -> str:
        return "Configured" if b else "Pending"

    files = state.get("uploaded_files", [])
    base = rag.base_status().replace("✅ ", "").replace("⚠️ ", "").replace("ℹ️ ", "")
    rows = [
        f'<tr><td class="status-label">Anthropic API key</td><td>{dot(bool(state.get("anthropic_key")))}{val(bool(state.get("anthropic_key")))}</td></tr>',
        f'<tr><td class="status-label">Tavily API key</td><td>{dot(bool(state.get("tavily_key")))}{val(bool(state.get("tavily_key")))}</td></tr>',
        f'<tr><td class="status-label">Agent</td><td>{dot(state.get("agent") is not None)}{"Ready" if state.get("agent") else "Not built"}</td></tr>',
        f'<tr><td class="status-label">Your PDFs</td><td>{dot(len(files) > 0) if files else dot(False)}{len(files)} indexed{(" — " + ", ".join(files)) if files else ""}</td></tr>',
        f'<tr><td class="status-label">Base corpus</td><td>{dot("chunks" in base.lower() or "loaded" in base.lower())}{base}</td></tr>',
    ]
    return '<table class="status-table"><tbody>' + "".join(rows) + "</tbody></table>"


def _try_build_agent(state: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Rebuild this session's agent from its current keys + retriever."""
    if not state.get("anthropic_key") or not state.get("tavily_key"):
        state["agent"] = None
        return state, "Waiting for API keys."
    try:
        state["agent"] = build_agent_for_session(
            anthropic_key=state["anthropic_key"],
            tavily_key=state["tavily_key"],
            session_retriever=state.get("session_retriever"),
            with_memory=True,
        )
        return state, "Agent built."
    except Exception as e:
        state["agent"] = None
        return state, f"Agent build failed: {e}"


# ---------------------------------------------------------------------------
# Tab 1 — API Keys
# ---------------------------------------------------------------------------
def save_keys(
    anthropic_key: str, tavily_key: str, state: Dict[str, Any]
) -> Tuple[str, str, Dict[str, Any]]:
    anthropic_key = (anthropic_key or "").strip()
    tavily_key = (tavily_key or "").strip()

    if not anthropic_key or not tavily_key:
        return (
            "### Missing key\nBoth **Anthropic** and **Tavily** keys are required.",
            _format_status(state),
            state,
        )

    state["anthropic_key"] = anthropic_key
    state["tavily_key"] = tavily_key
    state, msg = _try_build_agent(state)

    if state.get("agent") is None:
        return (
            f"### Could not build the agent\n`{msg}`\nDouble-check your keys.",
            _format_status(state),
            state,
        )
    return (
        "### Ready\nCredentials saved **for this session only**. "
        "Switch to **Analyze** to run your first briefing.",
        _format_status(state),
        state,
    )


def clear_keys(state: Dict[str, Any]) -> Tuple[str, str, str, Dict[str, Any]]:
    state["anthropic_key"] = ""
    state["tavily_key"] = ""
    state["agent"] = None
    return (
        "",
        "",
        _format_status(state),
        state,
    )


# ---------------------------------------------------------------------------
# Private-data panel — PDF upload
# ---------------------------------------------------------------------------
def index_uploaded_pdfs(
    files: List[Any], state: Dict[str, Any]
) -> Tuple[str, str, Dict[str, Any]]:
    if not state.get("anthropic_key"):
        return (
            "Save your API keys first (API Keys tab).",
            _format_status(state),
            state,
        )
    if not files:
        return ("No files selected.", _format_status(state), state)

    ok, reason = _rate_check(state, "upload")
    if not ok:
        return (reason, _format_status(state), state)

    paths: List[str] = []
    for f in files:
        if isinstance(f, str):
            paths.append(f)
        elif hasattr(f, "name"):
            paths.append(f.name)
        else:
            paths.append(str(f))

    try:
        retriever, info = rag.build_session_retriever(paths)
    except ValueError as e:
        return (f"{e}", _format_status(state), state)
    except Exception as e:
        return (f"Upload failed: {e}", _format_status(state), state)

    state["session_retriever"] = retriever
    state["uploaded_files"] = info["indexed"]
    state, _ = _try_build_agent(state)

    parts = [
        f"Indexed **{len(info['indexed'])} document(s)** — {info['chunks']} chunks.",
        "",
        "**Documents indexed:**",
        "\n".join(f"- {n}" for n in info["indexed"]) or "(none)",
    ]
    if info["skipped"]:
        parts += ["", "**Skipped:**", "\n".join(f"- {s}" for s in info["skipped"])]
    return "\n".join(parts), _format_status(state), state


def clear_uploaded_pdfs(state: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    state["session_retriever"] = None
    state["uploaded_files"] = []
    state, _ = _try_build_agent(state)
    return (
        "Your uploaded documents have been cleared from this session.",
        _format_status(state),
        state,
    )


# ---------------------------------------------------------------------------
# Tab 2 — Analyze Company
# ---------------------------------------------------------------------------
DEFAULT_SINGLE_PROMPT = (
    "Generate a comprehensive investment research briefing for {ticker}.\n\n"
    "Call the tools in parallel where possible. Include: "
    "(1) Executive summary, "
    "(2) 3-year stock performance with numbers, "
    "(3) Fundamentals (revenue, margins, FCF, P/E, EPS, growth), "
    "(4) Technical read (RSI, MA50 vs MA200, 52w positioning), "
    "(5) Wall Street analyst consensus + price targets, "
    "(6) Recent news with sentiment scores and article URLs, "
    "(7) AI research activity from the private analyst database (minimum 3 areas), "
    "(8) Top 2-3 risks, "
    "(9) Buy/Hold/Sell recommendation with confidence level, "
    "(10) Source citations, "
    "(11) Any data gaps."
)


def analyze_company(
    ticker: str,
    custom_query: str,
    state: Dict[str, Any],
    progress=gr.Progress(),
):
    if state is None or state.get("agent") is None:
        yield "**Agent not ready.** Please configure your API keys in the **API Keys** tab first."
        return

    ticker = (ticker or "").strip().upper()
    if not ticker:
        yield "**Please enter a ticker symbol** (e.g. `MSFT`, `NVDA`, `AAPL`)."
        return

    ok, reason = _rate_check(state, "analyze")
    if not ok:
        yield f"**{reason}**"
        return

    query = (custom_query or "").strip() or DEFAULT_SINGLE_PROMPT.format(ticker=ticker)
    thread_id = f"single-{ticker}-{int(time.time())}"

    progress(0.1, desc="Calling the agent…")
    try:
        result = run_agent(state["agent"], query, thread_id=thread_id)
    except Exception as e:
        yield f"### Agent error\n```\n{e}\n```"
        return
    progress(1.0, desc="Done")
    yield result


# ---------------------------------------------------------------------------
# Tab 3 — Rank Companies
# ---------------------------------------------------------------------------
DEFAULT_RANK_PROMPT = (
    "Analyze and rank the following AI-focused companies from best to worst "
    "investment opportunity.\n\n"
    "Companies to analyze: {companies}\n\n"
    "For EACH company, gather (in parallel tool calls where possible): 3-year stock "
    "performance, fundamentals, technical indicators, analyst consensus, recent news "
    "sentiment, AI research activity from the private analyst database (≥3 areas), "
    "and risk factors. Then produce:\n"
    "1. A ranked Markdown table (rank, ticker, score 0-100, Buy/Hold/Sell, one-line justification)\n"
    "2. Per-company notes grounded in data\n"
    "3. Source citations for all factual claims"
)


def rank_companies(tickers_csv: str, state: Dict[str, Any], progress=gr.Progress()):
    if state is None or state.get("agent") is None:
        yield "**Agent not ready.** Please configure your API keys in the **API Keys** tab first."
        return

    tickers = [t.strip().upper() for t in (tickers_csv or "").split(",") if t.strip()]
    if len(tickers) < 2:
        yield "**Please enter at least 2 tickers**, comma-separated (e.g. `MSFT, NVDA, GOOGL`)."
        return
    if len(tickers) > 8:
        yield "**Max 8 tickers per ranking request** (to keep latency + cost reasonable)."
        return

    ok, reason = _rate_check(state, "rank")
    if not ok:
        yield f"**{reason}**"
        return

    query = DEFAULT_RANK_PROMPT.format(companies=", ".join(tickers))
    thread_id = f"rank-{'-'.join(tickers)}-{int(time.time())}"

    progress(0.1, desc=f"Ranking {len(tickers)} companies…")
    try:
        result = run_agent(state["agent"], query, thread_id=thread_id)
    except Exception as e:
        yield f"### Agent error\n```\n{e}\n```"
        return
    progress(1.0, desc="Done")
    yield result


# ---------------------------------------------------------------------------
# Theme & CSS  — institutional, editorial, no-emoji aesthetic
# ---------------------------------------------------------------------------
# Palette: warm cream background, near-black text, oxblood accent.
# Typography: Fraunces (serif display) + Instrument Sans (body) + JetBrains Mono.
# Layout: editorial width (960px), 1px borders, flat surfaces, no shadows.
THEME = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#f5eeee", c100="#ebdcdc", c200="#d7b9b9", c300="#bf9393",
        c400="#9c6666", c500="#764141", c600="#5b2e2e", c700="#482525",
        c800="#361c1c", c900="#261313", c950="#170b0b",
    ),
    secondary_hue="stone",
    neutral_hue="stone",
    radius_size="sm",
    font=[
        gr.themes.GoogleFont("Instrument Sans"),
        "system-ui",
        "-apple-system",
        "sans-serif",
    ],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#faf9f4",
    background_fill_primary="#faf9f4",
    background_fill_secondary="#ffffff",
    block_background_fill="#ffffff",
    block_border_color="#e8e5dc",
    block_border_width="1px",
    block_shadow="none",
    block_label_background_fill="transparent",
    block_title_background_fill="transparent",
    input_background_fill="#ffffff",
    input_border_color="#d8d3c4",
    input_border_color_focus="#5b2e2e",
    button_primary_background_fill="#2d1f1f",
    button_primary_background_fill_hover="#0f0808",
    button_primary_text_color="#faf9f4",
    button_primary_border_color="#2d1f1f",
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_hover="#f5f1e8",
    button_secondary_text_color="#1a1a1a",
    button_secondary_border_color="#d8d3c4",
    body_text_color="#1a1a1a",
    body_text_color_subdued="#686561",
    border_color_primary="#e8e5dc",
)

CUSTOM_CSS = """
/* ============================================================
   Layout & global
   ============================================================ */
.gradio-container {
    max-width: 960px !important;
    font-variant-numeric: tabular-nums;
    background: #faf9f4;
}
.main, .block { box-shadow: none !important; }

/* ============================================================
   Typography — Fraunces for display, Instrument Sans for body
   ============================================================ */
.wrap h1, .wrap h2, .gradio-container h1, .gradio-container h2 {
    font-family: "Fraunces", Georgia, serif !important;
    font-weight: 400 !important;
    letter-spacing: -0.015em;
    color: #0d0d0d;
}
.gradio-container h1 {
    font-size: 2.6rem !important;
    line-height: 1.1 !important;
    margin: 0 0 4px 0 !important;
}
.gradio-container h2 {
    font-size: 1.6rem !important;
    line-height: 1.2 !important;
    margin: 28px 0 10px 0 !important;
}
.gradio-container h3, .gradio-container h4 {
    font-family: "Instrument Sans", sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0 !important;
    color: #0d0d0d;
}
.gradio-container p, .gradio-container li {
    color: #3a3833;
    line-height: 1.6;
}
code, .mono { font-family: "JetBrains Mono", monospace !important; }

/* Small-caps tracked kicker label */
.kicker {
    display: block;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #8a7060;
    margin-bottom: 8px;
}

/* Masthead (app title block) */
.masthead {
    border-bottom: 1px solid #e8e5dc;
    padding: 28px 0 24px 0;
    margin-bottom: 32px;
}
.masthead .tagline {
    font-size: 14px;
    color: #686561;
    margin-top: 4px;
    font-style: italic;
}

/* ============================================================
   Banner (top of Keys tab) — lighter, institutional
   ============================================================ */
#keys-banner {
    background: #ffffff;
    border: 1px solid #e8e5dc;
    border-left: 3px solid #8a5a2b;
    padding: 22px 26px;
    border-radius: 2px;
    margin-bottom: 20px;
}
#keys-banner.ready {
    border-left-color: #3a5a3a;
    background: #f5f2e8;
}
#keys-banner h3 {
    margin: 0 0 6px 0 !important;
    font-size: 1.05rem !important;
    font-family: "Instrument Sans", sans-serif !important;
    font-weight: 600 !important;
    color: #0d0d0d;
}

/* ============================================================
   Privacy note — warmer, less chipper
   ============================================================ */
.privacy-note {
    font-size: 13px;
    background: transparent;
    padding: 14px 18px;
    border-left: 2px solid #d8d3c4;
    color: #4a4741;
    line-height: 1.55;
    margin: 8px 0 12px 0;
}
.privacy-note b { color: #0d0d0d; font-weight: 600; }

/* ============================================================
   Status panel — two-column table, colored dots, no emojis
   ============================================================ */
.status-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-top: 10px;
}
.status-table tr { border-bottom: 1px solid #efece3; }
.status-table tr:last-child { border-bottom: none; }
.status-table td {
    padding: 10px 4px;
    vertical-align: middle;
    color: #3a3833;
}
.status-table td.status-label {
    width: 38%;
    font-weight: 500;
    color: #0d0d0d;
    letter-spacing: 0.01em;
}
.status-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    margin-right: 10px;
    vertical-align: middle;
    transform: translateY(-1px);
}
.status-dot.ready   { background: #3a5a3a; }
.status-dot.missing { background: #b85c38; }
.status-dot.pending { background: #8a5a2b; }

/* ============================================================
   Buttons — rectangular, thin border, no shadow
   ============================================================ */
button.primary, button.secondary {
    border-radius: 2px !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
    transition: all 120ms ease;
}
button.lg { padding: 10px 18px !important; }

/* Quick-pick ticker buttons — compact mono */
.quick-pick button {
    font-family: "JetBrains Mono", monospace !important;
    font-size: 12px !important;
    padding: 6px 10px !important;
    background: transparent !important;
    border: 1px solid #d8d3c4 !important;
    color: #1a1a1a !important;
    border-radius: 2px !important;
}
.quick-pick button:hover {
    background: #f5f1e8 !important;
    border-color: #8a5a2b !important;
}

/* ============================================================
   Tabs
   ============================================================ */
.tab-nav {
    border-bottom: 1px solid #e8e5dc !important;
    background: transparent !important;
    padding: 0 !important;
    margin-bottom: 24px !important;
}
.tab-nav button {
    font-family: "Instrument Sans", sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    color: #686561 !important;
    padding: 14px 22px !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
.tab-nav button.selected {
    color: #0d0d0d !important;
    border-bottom-color: #5b2e2e !important;
}

/* ============================================================
   Inputs  — force dark text on cream surfaces
   ============================================================ */
input[type="text"], input[type="password"], textarea,
.gr-input, .gr-textbox textarea, .gr-textbox input {
    border-radius: 2px !important;
    font-family: "Instrument Sans", sans-serif !important;
    color: #1a1a1a !important;
    background: #ffffff !important;
}
input::placeholder, textarea::placeholder,
.gr-input::placeholder {
    color: #a8a39a !important;
    opacity: 1 !important;
}
label, label span, .gr-form label, .label, .block-label {
    color: #4a4741 !important;
    font-weight: 500 !important;
}
/* Gradio helper/info text */
.gr-info, .info, .hint, .gr-textbox .tip, small, .gradio-container small {
    color: #686561 !important;
}
/* Markdown paragraphs and list text */
.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .prose strong,
.gradio-container .markdown p,
.gradio-container .markdown li {
    color: #1a1a1a !important;
}
/* Links stay oxblood */
.gradio-container a { color: #5b2e2e !important; text-decoration: underline; }
.gradio-container a:hover { color: #2d1f1f !important; }
/* File-upload drop zone text */
.file-preview, .gr-file, .file_block, .upload-container {
    color: #1a1a1a !important;
}
/* Accordion header text */
.gr-accordion > .label-wrap, .gr-accordion button span {
    color: #0d0d0d !important;
    font-weight: 500 !important;
}

/* ============================================================
   Output area
   ============================================================ */
.output-markdown {
    background: #ffffff;
    border: 1px solid #e8e5dc;
    padding: 24px 28px;
    border-radius: 2px;
    min-height: 240px;
}

/* ============================================================
   Footer
   ============================================================ */
.app-footer {
    font-size: 10px;
    color: #8a8680;
    text-align: center;
    padding: 24px 0 8px 0;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    border-top: 1px solid #e8e5dc;
    margin-top: 48px;
}

/* Accordion */
.gr-accordion {
    border: 1px solid #e8e5dc !important;
    border-radius: 2px !important;
    background: #fdfcf8 !important;
}
"""


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------
with gr.Blocks(theme=THEME, css=CUSTOM_CSS, title="Autonomous Financial Analyst") as demo:
    # Session state — isolated per browser session by Gradio
    session_state = gr.State(value=None)

    gr.HTML(
        """
        <div class="masthead">
            <span class="kicker">Investment Research · Beta</span>
            <h1>Autonomous Financial Analyst</h1>
            <div class="tagline">
                Institutional-grade briefings — fundamentals, technicals, analyst
                consensus, news sentiment, and AI research activity — synthesized
                by a multi-tool LangGraph agent.
            </div>
        </div>
        """
    )

    with gr.Tabs():

        # =====================================================================
        # Tab 1 — API Keys
        # =====================================================================
        with gr.Tab("API Keys"):
            banner = gr.Markdown(
                "### Setup required\nEach visitor brings their own API keys. "
                "Keys live only in your session — not written to disk, not shared "
                "with anyone else.",
                elem_id="keys-banner",
            )

            gr.HTML('<span class="kicker">Credentials</span>')
            gr.Markdown(
                """
                1. **Anthropic** — billed to you directly for LLM usage.
                   Create one at [console.anthropic.com](https://console.anthropic.com/settings/keys).
                2. **Tavily** — free tier is enough for casual use.
                   [tavily.com](https://tavily.com).
                """
            )

            gr.HTML(
                """
                <div class="privacy-note">
                <b>Privacy and cost model.</b> Public Space, BYO-key. Your credentials
                are held only in this Gradio session — never in environment variables,
                never in logs, never shared across sessions. You are billed directly by
                Anthropic and Tavily for your own usage. When you close the tab, your
                keys, uploaded documents, and conversation history are discarded.
                </div>
                """
            )

            with gr.Group():
                anthropic_key_in = gr.Textbox(
                    label="Anthropic API key  (required)",
                    placeholder="sk-ant-…",
                    type="password",
                    lines=1,
                )
                tavily_key_in = gr.Textbox(
                    label="Tavily API key  (required)",
                    placeholder="tvly-…",
                    type="password",
                    lines=1,
                )

            with gr.Row():
                save_btn = gr.Button("Save & Verify", variant="primary", size="lg")
                clear_keys_btn = gr.Button("Clear from this session", variant="secondary")

            gr.HTML('<span class="kicker" style="margin-top:28px;">Session status</span>')
            status_md = gr.HTML('<div class="status-table">Not initialized.</div>')

        # =====================================================================
        # Tab 2 — Analyze
        # =====================================================================
        with gr.Tab("Analyze"):
            gr.HTML('<span class="kicker">Single-company briefing</span>')
            gr.Markdown(
                "Enter a ticker to generate a complete investment briefing. "
                "The agent calls fundamentals, technicals, analyst consensus, "
                "news, sentiment, and the private RAG database in parallel."
            )

            with gr.Accordion(
                "Private documents — optional (session-only)",
                open=False,
            ):
                gr.Markdown(
                    "Upload up to 10 PDFs (≤25 MB each). They are indexed into an "
                    "in-memory vector store for **this session only** — never "
                    "written to disk, never visible to other users. The agent's RAG "
                    "tool searches them alongside the bundled base corpus."
                )

                pdf_upload = gr.File(
                    label="Drop PDFs",
                    file_count="multiple",
                    file_types=[".pdf"],
                    type="filepath",
                )
                with gr.Row():
                    upload_btn = gr.Button("Index documents", variant="primary")
                    clear_pdfs_btn = gr.Button("Clear documents", variant="secondary")
                upload_status = gr.Markdown("")

            with gr.Row():
                with gr.Column(scale=1):
                    ticker_in = gr.Textbox(
                        label="Ticker symbol",
                        placeholder="MSFT",
                        lines=1,
                        max_lines=1,
                    )
                    gr.HTML('<span class="kicker">Quick picks</span>')
                    with gr.Row(elem_classes=["quick-pick"]):
                        for sym in ["MSFT", "GOOGL", "NVDA", "AMZN", "IBM"]:
                            b = gr.Button(sym, size="sm")
                            b.click(lambda s=sym: s, outputs=ticker_in)

                    custom_q = gr.Textbox(
                        label="Custom question  (optional)",
                        placeholder="Focus on MSFT's Copilot revenue and its 2025 AI capex.",
                        lines=3,
                    )
                    analyze_btn = gr.Button(
                        "Generate briefing", variant="primary", size="lg"
                    )

                with gr.Column(scale=2):
                    single_output = gr.Markdown(
                        value="*Your investment briefing will appear here.*",
                        elem_classes=["output-markdown"],
                    )

        # =====================================================================
        # Tab 3 — Rank
        # =====================================================================
        with gr.Tab("Rank"):
            gr.HTML('<span class="kicker">Comparative ranking</span>')
            gr.Markdown(
                "Enter 2+ tickers separated by commas. The agent analyzes each and "
                "produces a ranked table with Buy / Hold / Sell ratings. Any "
                "documents uploaded in the **Analyze** tab are used here too."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    tickers_in = gr.Textbox(
                        label="Tickers  (comma-separated)",
                        value="MSFT, GOOGL, NVDA, AMZN, IBM",
                        lines=2,
                    )
                    rank_btn = gr.Button(
                        "Rank companies", variant="primary", size="lg"
                    )

                with gr.Column(scale=2):
                    rank_output = gr.Markdown(
                        value="*Your ranked recommendation will appear here.*",
                        elem_classes=["output-markdown"],
                    )

    gr.HTML(
        '<div class="app-footer">'
        "LangGraph · Claude Sonnet 4.6 · Gradio · Public Space · BYO-Key"
        "</div>"
    )

    # ==== Wire up event handlers (all take/return session_state) ====

    save_btn.click(
        save_keys,
        inputs=[anthropic_key_in, tavily_key_in, session_state],
        outputs=[banner, status_md, session_state],
    )
    clear_keys_btn.click(
        clear_keys,
        inputs=[session_state],
        outputs=[anthropic_key_in, tavily_key_in, status_md, session_state],
    )

    upload_btn.click(
        index_uploaded_pdfs,
        inputs=[pdf_upload, session_state],
        outputs=[upload_status, status_md, session_state],
    )
    clear_pdfs_btn.click(
        clear_uploaded_pdfs,
        inputs=[session_state],
        outputs=[upload_status, status_md, session_state],
    )

    analyze_btn.click(
        analyze_company,
        inputs=[ticker_in, custom_q, session_state],
        outputs=single_output,
    )
    rank_btn.click(
        rank_companies,
        inputs=[tickers_in, session_state],
        outputs=rank_output,
    )

    # ---- Initialize per-session state on page load ----
    def _init_state():
        st = _new_state()
        # If ALLOW_ENV_KEYS=1 and env keys were picked up, try to auto-build
        if st["anthropic_key"] and st["tavily_key"]:
            st, _ = _try_build_agent(st)
        msg = (
            "### Development mode\nEnvironment keys detected — agent pre-built for this session."
            if st.get("agent") is not None
            else "### Setup required\nEach visitor brings their own API keys."
        )
        return st, msg, _format_status(st)

    demo.load(_init_state, outputs=[session_state, banner, status_md])


if __name__ == "__main__":
    # show_api=False — avoids a gradio_client 1.3.0 bug where /info schema
    # introspection recurses through gr.State holding the compiled LangGraph
    # agent (non-JSON-serializable). Safe to disable — not used by the UI.
    #
    # max_size + default_concurrency_limit — Space-wide backpressure so one
    # chatty user can't starve the 2-vCPU HF free tier.
    demo.queue(max_size=20, default_concurrency_limit=2).launch(show_api=False)
