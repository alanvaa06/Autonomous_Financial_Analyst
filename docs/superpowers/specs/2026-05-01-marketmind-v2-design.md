# MarketMind v2 — Design Spec

**Date:** 2026-05-01
**Status:** Draft for review
**Owner:** alanvaa06
**Supersedes:** Current single-agent `Autonomous_Financial_Analyst` (LangGraph chat agent + RAG)
**Related:** `marketmind_prd.md` (reference architecture, partially adopted)

---

## 1. Summary

Replace the current single-agent chat application with a **multi-agent parallel equity analysis pipeline** that takes a single stock ticker as input and produces a **comprehensive sectioned investment report** with a final verdict, conviction, and confidence score.

The pipeline runs five domain specialists in parallel (fan-out/fan-in via LangGraph), gates the result through a lightweight Supervisor for QA, and emits a synthesized markdown report rendered live in the existing Gradio UI on Hugging Face Spaces.

The current PDF-based RAG corpus is removed and replaced with a live SEC EDGAR connector that pulls the most recent 10-Q and 10-K for fundamentals analysis.

---

## 2. Goals and Non-Goals

### 2.1 Goals
1. Single-ticker, end-to-end automated analysis with a sectioned markdown report as output.
2. Five orthogonal specialists running in parallel via LangGraph fan-out/fan-in, plus Supervisor and Synthesis nodes.
3. Replace PDF RAG with live EDGAR 10-Q / 10-K data per ticker.
4. Preserve BYO-key model, per-session isolation, and Hugging Face Spaces deployment.
5. Stream each agent's section to the UI as it completes.
6. Produce a verdict with both 3-state (BUY/HOLD/SELL) and a conviction qualifier (Strong / Standard / Cautious).

### 2.2 Non-Goals (v2.0)
1. No crypto support. (Stocks only. Crypto code paths and `-USD` ticker handling are removed.)
2. No multi-ticker batch mode. (One ticker per run; future v2.2.)
3. No persistent verdict log / database. (Future v2.2.)
4. No backtesting harness.
5. No conversational chat. (Replaced by the deterministic pipeline.)
6. No portfolio sizing, broker integration, or order routing.
7. No per-agent UI toggles. (Future v2.1; v2.0 always runs all five.)

---

## 3. Architecture

### 3.1 High-level flow

```
User input: stock ticker
        │
        ▼
   Orchestrator
   (validate ticker, resolve CIK, classify equity, init state)
        │
        ├────────┬────────┬────────┬────────┐
        ▼        ▼        ▼        ▼        ▼
      Price  Sentiment  Fundamentals  Macro  Risk     ← parallel superstep
        │        │        │            │      │
        └────────┴────────┴────────────┴──────┘
                          │
                          ▼
                    Supervisor
                  (light QA review)
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
        retry_targets ≠ ∅        approve / force
              │                       │
              ▼                       ▼
        Re-run flagged          Synthesis Agent
        agents (max 1 round)    (writes sectioned report)
              │                       │
              └──────────┬────────────┘
                         ▼
                       END
```

### 3.2 Why fan-out/fan-in plus Supervisor

- Independent specialists analyze orthogonal data sources, so they have no sequential dependency. Running them in the same LangGraph superstep reduces wall-clock latency from ~5× single-agent to ~1× single-agent (bounded by the slowest specialist).
- The Supervisor is a deliberate, lightweight QA layer. It does **not** override verdicts or rewrite content. It checks for missing data, contradictions, and weak confidence, and may request **at most one** retry per agent. After one retry round it forces synthesis with caveats. This keeps worst-case latency bounded.
- Synthesis happens after the Supervisor signs off, so the final report always reflects either approved specialist output or output explicitly flagged as degraded.

### 3.3 LangGraph wiring

- Entry: `orchestrator`
- Edges from `orchestrator` → each of `price`, `sentiment`, `fundamentals`, `macro`, `risk` (fan-out, all in one superstep).
- Edges from each specialist → `supervisor` (fan-in barrier).
- Conditional edge from `supervisor`:
  - if `retry_targets` is non-empty AND `retry_round == 0` → route to each named specialist (parallel re-run), then back to `supervisor`.
  - otherwise → `synthesis`.
- Edge `synthesis` → `END`.

The supervisor → specialist re-routing uses LangGraph's `Send` API (or per-target conditional edges) to fan out only to flagged agents.

---

## 4. Shared State

```python
# state.py
import operator
from typing import Annotated, Optional, TypedDict, Literal


Verdict = Literal["BUY", "HOLD", "SELL"]
Conviction = Literal["STRONG", "STANDARD", "CAUTIOUS"]
Signal = Literal["BULLISH", "BEARISH", "NEUTRAL"]


class AgentSignal(TypedDict):
    agent: str                       # "price" | "sentiment" | "fundamentals" | "macro" | "risk"
    signal: Signal
    confidence: float                # 0.0 .. 1.0
    summary: str                     # one-line headline
    section_markdown: str            # the agent's full report section (markdown)
    raw_data: dict                   # numbers / structured payload behind the call
    degraded: bool                   # True if data was partially missing / fallback used
    error: Optional[str]             # populated on hard failure


class SupervisorReview(TypedDict):
    approved: bool
    critiques: dict                  # {agent_name: critique_text}
    retry_targets: list              # [agent_name, ...]
    notes: str                       # supervisor-level commentary appended to final report


class MarketMindState(TypedDict):
    # Inputs
    ticker: str
    company_name: Optional[str]
    cik: Optional[str]

    # Specialist outputs (parallel-safe append)
    agent_signals: Annotated[list, operator.add]

    # Retry bookkeeping (incremented by orchestrator before re-running flagged agents)
    retry_round: int                 # 0 or 1; bounded

    # Supervisor output (the conditional edge reads supervisor_review["retry_targets"])
    supervisor_review: Optional[SupervisorReview]

    # Final synthesis
    final_verdict: Optional[Verdict]
    final_conviction: Optional[Conviction]
    final_confidence: Optional[float]
    final_reasoning: Optional[str]
    final_report: Optional[str]      # full sectioned markdown
```

The `Annotated[list, operator.add]` reducer on `agent_signals` is load-bearing: it allows all five agents in the same superstep to write without a race. Without it, only the last writer survives.

---

## 5. Agents

All agents follow the same canonical contract:

> **fetch data → compute / extract → ask Claude to interpret → return `{"agent_signals": [AgentSignal]}` (and `section_markdown`)**

All LLM calls are routed through a shared `agents/__init__.py` that exposes a **factory function**:

```python
def build_llm_clients(anthropic_key: str) -> LLMClients:
    """Returns a NamedTuple/dataclass with .reasoning and .fast clients,
    bound to the per-session key. Never reads os.environ on the public path."""
```

- `clients.reasoning` — `claude-sonnet-4-6`, temperature 0.1, prompt caching enabled on the system prompt
- `clients.fast` — `claude-haiku-4-5-20251001`, temperature 0.1
- `safe_parse_json(content)` — module-level pure helper that strips markdown fences before `json.loads`

The factory is invoked once per analysis run (per Gradio session). The clients live inside the LangGraph state's runtime config, never as module globals constructed from `os.environ` on the public deployment.

### 5.1 Price Agent

- **Data:** `yfinance` 90-day OHLC.
- **Computes:** RSI(14), MACD(12/26/9), Bollinger %B, 7-day / 30-day / 90-day price change.
- **LLM (Sonnet):** Interpret indicator stack, return `{signal, confidence, summary}` + a 100–200 word `section_markdown` formatted as a "Technical Analysis" section.
- **Errors:** empty data → degraded `AgentSignal` with `signal=NEUTRAL`, `confidence=0.0`, `degraded=True`.

### 5.2 Sentiment Agent

- **Data:** Tavily search (existing key in BYO-key panel) for the last 7 days of news on `{ticker} stock {company_name}`.
- **Compute:** dedupe by URL, cap at 15 articles, extract title + snippet + source.
- **LLM (Haiku):** score sentiment per article (`positive | neutral | negative`) and aggregate; produce drivers list.
- **LLM (Sonnet):** write `section_markdown` "News & Sentiment" section: headline rollup, top drivers, 3–5 cited headlines.
- **Errors:** Tavily failure or zero results → degraded section with explicit "No coverage found".

### 5.3 Fundamentals Agent (replaces RAG)

- **Data:** SEC EDGAR via `edgar.py` module (see §6).
  - Latest 10-Q (most recent quarterly).
  - Latest 10-K (most recent annual, for YoY comparisons and full segment breakdown).
- **Compute:**
  - Pull XBRL company facts: `Revenues`, `GrossProfit`, `OperatingIncomeLoss`, `NetIncomeLoss`, `EarningsPerShareDiluted`, `Assets`, `Liabilities`, `StockholdersEquity`, `CashAndCashEquivalentsAtCarryingValue`, `CommonStockSharesOutstanding`. Compute QoQ and YoY deltas where data available.
  - Extract MD&A narrative text from the 10-Q filing (Item 2 of 10-Q). Cap at 8K characters.
  - Extract Risk Factors highlights from the 10-K (Item 1A) only if 10-Q's MD&A is short.
- **LLM (Sonnet):** synthesize fundamentals view: revenue trend, margin trajectory, balance-sheet health, MD&A signals. Return `{signal, confidence, summary}` + `section_markdown` "Fundamentals" section.
- **Errors:**
  - Ticker has no SEC filings (foreign issuer, recent IPO without filings yet, ETF) → return degraded signal with `signal=NEUTRAL`, `confidence=0.0`, `degraded=True`, summary `"Fundamentals unavailable — no SEC filings"`. Supervisor flags. Synthesis notes the gap.
  - 10-Q parse failure → fall back to XBRL facts only; `section_markdown` notes MD&A unavailable.

### 5.4 Macro Agent

- **Data:**
  - FRED API (BYO-key, optional): `DTWEXBGS` (DXY), `DFF` (Fed funds), `DGS10` (10Y), `DGS2` (2Y) → compute 2s10s spread.
  - Fear & Greed index from `alternative.me` (no key).
- **Compute:** latest values, 5-day deltas, yield curve sign.
- **LLM (Sonnet):** macro regime interpretation + how it bears on the ticker's sector. Return signal + `section_markdown` "Macro Backdrop".
- **Degraded mode:** if FRED key missing → run Fear & Greed only; mark `degraded=True`; section explicitly states reduced macro coverage.

### 5.5 Risk Agent

- **Data:** `yfinance` 90-day returns + `yfinance` `info` for beta and short ratio + `^VIX`.
- **Compute:** annualized volatility, max drawdown, Sharpe (rf = 4%, 2026 proxy), beta, short ratio, VIX level.
- **LLM (Sonnet):** risk-adjusted view. Return signal + `section_markdown` "Risk Profile".

### 5.6 Supervisor Agent (light QA)

**Scope:** quality assurance only. Cannot rewrite agent content, cannot override verdicts.

**Inputs:** all five `AgentSignal`s.

**Checks (deterministic, then LLM-confirmed):**
1. Any agent returned `error` or `confidence == 0.0` with `degraded=True`?
2. Any direct contradiction (one agent BULLISH and another BEARISH, both with `confidence ≥ 0.7`)? → flag the lower-confidence side (or the side with `degraded=True`) for retry. Never flag both.
3. Any `section_markdown` shorter than 200 characters? (proxy for empty section)
4. Any obvious data sanity violations (e.g., RSI > 100, negative volatility, vol > 500%, NaN/Inf in any numeric)?

**Output:** `SupervisorReview { approved, critiques, retry_targets, notes }`.
- `retry_targets`: agent names eligible for one retry.
- `notes`: 2–4 sentences appended to the final report's Executive Summary as a "Data Quality" line.

**Retry budget:** maximum **one retry round** for the entire run. If any flagged agent fails again or `retry_round >= 1`, supervisor sets `approved=True` (forced) and synthesis proceeds with caveats.

### 5.7 Synthesis Agent

**Inputs:** all `AgentSignal`s + `SupervisorReview`.

**Compute:**
- Map each `AgentSignal` to a numeric score: `BULLISH = +1`, `NEUTRAL = 0`, `BEARISH = -1`.
- Per-agent weighted contribution: `score_i × confidence_i × (0 if degraded_i else 1)`.
- **Net score** = `sum(contributions) / 5` (denominator is the fixed agent count, so degraded agents reduce the magnitude rather than being excluded).
- **Verdict mapping** on net score:
  - `> +0.20` → BUY
  - `< -0.20` → SELL
  - otherwise → HOLD
- **Agreeing set** = agents whose mapped score has the same sign as the verdict (HOLD's agreeing set = agents with score = 0).
- **Conviction mapping** on (agreeing_count, avg_confidence_of_agreeing_set):
  - `agreeing_count ≥ 4 AND avg_conf ≥ 0.75` → STRONG
  - `agreeing_count ≥ 3 AND avg_conf ≥ 0.55` → STANDARD
  - otherwise → CAUTIOUS
- **Final confidence** = `mean(confidence_i for non-degraded i) × (non_degraded_count / 5)`. If all agents degraded, `final_confidence = 0.0` and verdict is forced to HOLD.

**Verdict label rendering (UI string):**

| Verdict | Conviction | Label |
|---|---|---|
| BUY | STRONG | "Strong Buy" |
| BUY | STANDARD | "Buy" |
| BUY | CAUTIOUS | "Cautious Buy" |
| HOLD | STRONG | "Hold (High Conviction)" |
| HOLD | STANDARD | "Hold" |
| HOLD | CAUTIOUS | "Hold (Mixed Signals)" |
| SELL | CAUTIOUS | "Cautious Sell" |
| SELL | STANDARD | "Sell" |
| SELL | STRONG | "Strong Sell" |

**LLM (Sonnet):** Given the deterministic vote, generate the `final_reasoning` (3–5 sentences) and assemble the full `final_report` in this section order:
1. **Executive Summary** — verdict label (e.g. "Buy — Strong Conviction"), confidence %, 3-bullet thesis, data-quality line from supervisor.
2. **Technical Analysis** — Price agent's `section_markdown`.
3. **News & Sentiment** — Sentiment agent's `section_markdown`.
4. **Fundamentals** — Fundamentals agent's `section_markdown`.
5. **Macro Backdrop** — Macro agent's `section_markdown`.
6. **Risk Profile** — Risk agent's `section_markdown`.
7. **Synthesis & Final Verdict** — LLM-written reasoning that integrates the five views, calls out conflicts, and states the verdict + conviction + confidence explicitly.
8. **Disclaimers** — static block: not financial advice, data freshness, agent confidence caveats, links to data sources.

The verdict label rendered in the UI combines verdict and conviction, e.g. `"Strong Buy"`, `"Buy"`, `"Cautious Buy"`, `"Hold"`, `"Cautious Sell"`, `"Sell"`, `"Strong Sell"`.

---

## 6. EDGAR Module (`edgar.py`)

New module. Replaces all RAG / Chroma / PDF code.

### 6.1 Responsibilities
- Resolve `ticker → CIK` via SEC's `https://www.sec.gov/files/company_tickers.json` (cached in process for 24h).
- Fetch the latest 10-Q and 10-K filings for a CIK via `https://data.sec.gov/submissions/CIK{cik}.json`.
- Fetch XBRL company facts via `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json` (one request per company, all tags inline).
- Extract MD&A and Risk Factors text from filing HTML by SEC accession number.
- Return a normalized `EdgarBundle` with: `{ company_name, cik, latest_10q, latest_10k, xbrl_facts, mdna_text, risk_factors_text, fetched_at }`.

### 6.2 Politeness and rate limits
- All requests carry a `User-Agent: "MarketMind/2.0 contact@example.com"` header; the email is taken from the BYO-key panel (or a configured default).
- Per-process LRU cache, key = `(cik, "10q" | "10k" | "facts")`, TTL = 6 hours. Cache lives only in memory; disk cache is out of scope for v2.0.
- Politeness sleep between SEC requests: 100ms (SEC's published rate limit is 10 req/s).

### 6.3 Error handling
- `ticker_not_in_sec_universe` → caller returns degraded Fundamentals signal.
- `no_recent_10q` (e.g. issuer reports only 20-F) → use 10-K only; mark degraded.
- HTTP 4xx/5xx → 1 retry with exponential backoff (1s then 3s); after that, raise and caller degrades.

### 6.4 Public surface (initial)

```python
def resolve_ticker(ticker: str) -> tuple[str, str]:
    """Return (cik_padded_10, company_name). Raises TickerNotFound."""

def fetch_company_facts(cik: str) -> dict: ...
def fetch_latest_10q(cik: str) -> Filing | None: ...
def fetch_latest_10k(cik: str) -> Filing | None: ...
def extract_mdna(filing: Filing) -> str: ...
def extract_risk_factors(filing: Filing) -> str: ...

def build_edgar_bundle(ticker: str) -> EdgarBundle:
    """Top-level entry used by the Fundamentals agent."""
```

---

## 7. UI (`app.py`, Gradio)

### 7.1 Layout
- **Header:** "MarketMind — Multi-Agent Equity Analyst" + version tag.
- **BYO-key panel** (collapsible, existing pattern):
  - `ANTHROPIC_API_KEY` (required)
  - `TAVILY_API_KEY` (required for sentiment)
  - `FRED_API_KEY` (optional; enables full Macro)
  - `SEC_USER_AGENT_EMAIL` (optional; defaults to repo contact)
- **Input row:** ticker textbox (auto-uppercase, trimmed) + Analyze button + cost hint badge `"~$0.30 per run (BYO key)"`.
- **Live status panel:** five status pills, one per agent: `pending → running → done | degraded | error`.
- **Output area:** rendered markdown of `final_report`, streaming section-by-section as agents complete. The Synthesis section appears last after Supervisor sign-off.
- **Footer:** disclaimer + version + run latency.

### 7.2 Streaming
- Uses Gradio's generator/`yield` pattern (already in use in current `app.py`).
- The graph runs with `astream` so each node update yields a partial state update to the UI.
- On each yield, the UI renders the most up-to-date `agent_signals` and any completed `section_markdown` blocks. The Executive Summary is rendered last, after Synthesis writes `final_report`.

### 7.3 Removed UI
- PDF upload control (RAG corpus replacement).
- Chat textbox / conversation history.
- Free-text question input.

### 7.4 Preserved
- BYO-key plumbing and per-session isolation.
- Rate limits (`ratelimit.py`) — re-tuned to the analysis-run cadence: 1 run per 60s per session, queue cap 1.
- `gradio_client` 1.3.0 monkey-patch at top of `app.py`.

---

## 8. File Layout (Target)

```
app.py                       Gradio shell (rewritten — ticker input, streaming)
graph.py                     LangGraph build (NEW)
state.py                     MarketMindState + AgentSignal + reducers (NEW)
edgar.py                     SEC EDGAR client (NEW)
agents/
  __init__.py                shared LLM clients (factory) + safe_parse_json (NEW)
  orchestrator.py            (NEW)
  price_agent.py             (NEW — rewritten from current tools.py helpers)
  sentiment_agent.py         (NEW — uses Tavily)
  fundamentals_agent.py      (NEW — uses edgar.py)
  macro_agent.py             (NEW — FRED + Fear&Greed)
  risk_agent.py              (NEW)
  supervisor_agent.py        (NEW — light QA)
  synthesis_agent.py         (NEW — assembles report)
ratelimit.py                 KEPT (re-tuned thresholds)
tasks/
  todo.md                    KEPT (CLAUDE.md workflow)
  lessons.md                 KEPT (CLAUDE.md workflow)
docs/superpowers/specs/
  2026-05-01-marketmind-v2-design.md   (this doc)
requirements.txt             UPDATED — see §10
README.md                    UPDATED for v2
.env.example                 UPDATED keys list
```

**Deleted:**
- `agent.py` (old single-agent factory)
- `tools.py` (its yfinance helpers are rewritten as agent-internal helpers per Q4 decision)
- `rag.py`
- `data/Companies-AI-Initiatives/` (PDF corpus)
- Any chroma persistence dirs

---

## 9. Functional and Non-Functional Requirements

### 9.1 Functional
- **FR-1.** Accept a single uppercase stock ticker. Reject obviously invalid inputs (length, characters) before invoking the graph.
- **FR-2.** Run all five specialists concurrently in a single LangGraph superstep.
- **FR-3.** Each specialist returns an `AgentSignal` with all required fields populated, including `section_markdown`.
- **FR-4.** Supervisor approves or requests at most one retry round; never rewrites agent content.
- **FR-5.** Synthesis emits a sectioned markdown report with sections in the order defined in §5.7 and a labeled verdict combining verdict + conviction.
- **FR-6.** UI streams updates: each agent's status pill changes as it transitions; sections render as they arrive.
- **FR-7.** All API keys (Anthropic, Tavily, FRED, SEC user-agent email) are session-scoped; never module-global; never logged.
- **FR-8.** Graceful degradation: any single specialist failure does not abort the run; report is produced with `degraded=True` markings on affected sections.
- **FR-9.** EDGAR module never hits SEC without a `User-Agent` header.

### 9.2 Non-Functional
- **NFR-1.** End-to-end p50 latency target: ≤ 25 seconds (bounded by slowest specialist + Sonnet inference + supervisor + synthesis). Acceptable upper bound: 60s with one retry round.
- **NFR-2.** Cost target: ≤ $0.50 per run at list price using prompt caching on system prompts.
- **NFR-3.** All Anthropic calls use prompt caching on the static system prompt portion.
- **NFR-4.** Per-session rate limit: 1 analysis per 60s, queue cap 1. Per-process EDGAR cache TTL 6h.
- **NFR-5.** Environment: Python 3.12 pinned (HF Space). Existing `audioop-lts` shim retained.
- **NFR-6.** No secrets logged in any path. UI never reflects keys back into the rendered output.
- **NFR-7.** Pinned dependencies in `requirements.txt` retained where load-bearing (gradio 4.44.1, fastapi/starlette, hf_hub, transformers, yfinance).

---

## 10. Dependencies

**Add:**
- `fredapi` (or use `requests` directly to FRED; prefer raw `requests` to avoid extra dep — decided: raw `requests`)
- No new EDGAR client lib — use `requests` + `beautifulsoup4` (already transitively present; pin if not)

**Remove:**
- `chromadb`
- `pypdf` / `pymupdf` if only RAG used them
- Anything tied to the deleted RAG path

Final `requirements.txt` is regenerated during implementation; no new heavy deps expected.

---

## 11. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Anthropic concurrency 429s on fan-out (5 simultaneous Sonnet calls per run) | Per-key concurrency semaphore in `agents/__init__.py` factory (cap = 3 concurrent in-flight per key; remaining specialists queue briefly). Plus jittered exponential backoff on 429 with up to 2 retries. Synthesis runs after barrier so contends only with retry traffic. Documented in README. |
| EDGAR rate limiting / IP blocks on HF Space | Strict 100ms inter-request delay; retries with backoff; cache TTL 6h; document SEC's 10 req/s policy. |
| Tavily quota exhaustion | Sentiment agent degrades to "no coverage" rather than failing run. |
| FRED key absent for many users | Macro agent runs in Fear&Greed-only degraded mode by design. |
| MD&A extraction fragile across 10-Q HTML formats | Use multiple parser strategies (Item 2 anchor, then heading regex); on failure fall back to XBRL-only fundamentals. |
| Foreign issuers / ETFs have no 10-Q | Fundamentals returns degraded signal; supervisor flags; synthesis notes "Fundamentals unavailable" in Executive Summary. |
| User pastes ticker for a recently-delisted stock | yfinance returns empty → Price + Risk agents degrade; pipeline still completes with reduced confidence. |
| Spec scope creep (multi-ticker, backtest, batch) | Explicitly out of scope per §2.2; tracked in §13. |

---

## 12. Migration Plan (high-level only; full plan lives in implementation plan doc)

1. Delete RAG/PDF/chroma surface and `agent.py`/`tools.py` in a single migration commit (work happens on a feature branch; main stays runnable until merge).
2. Land `state.py`, `agents/__init__.py`, `edgar.py` as foundations.
3. Implement specialists one at a time with unit-level smoke tests against a known-good ticker (e.g., `MSFT`).
4. Implement supervisor + synthesis.
5. Implement `graph.py` wiring.
6. Rewrite `app.py` UI.
7. End-to-end smoke run against `MSFT`, `AAPL`, `NVDA`, plus an edge case (`SHEL` ADR for foreign-issuer behavior, `SPY` for ETF behavior).
8. Update README, `.env.example`, and `requirements.txt`.
9. Deploy to HF Space.

The detailed task breakdown belongs in `tasks/todo.md` and the implementation plan generated by `writing-plans` after this spec is approved.

---

## 13. Future Work (post-v2.0)
- **v2.1:** Per-agent UI toggles. EDGAR caching to disk. Retry-budget tuning. Add Insider-Trading agent (Form 4) and Options-Flow agent.
- **v2.2:** Multi-ticker batch mode. Persistent verdict log (SQLite). Backtesting harness.
- **v2.3:** Crypto support reintroduced as a parallel pipeline behind asset-type detection.

---

## 14. Open Decisions Locked In This Round

- LLM provider: **Anthropic only** (Sonnet 4.6 + Haiku 4.5). No Groq.
- Sentiment provider: **Tavily** (existing key).
- Fundamentals: **EDGAR 10-Q + 10-K** (replaces PDF RAG entirely).
- Supervisor: **Light QA** (cannot override verdicts; max 1 retry round).
- Verdict format: **3-state verdict + conviction qualifier** (Strong / Standard / Cautious).
- Crypto: **out of scope for v2.0**.
- UI: **Gradio**, streaming, single-ticker only.
- Cost hint surfaced in UI: **yes**.
- Existing tools.py helpers: **rewritten** (not reused) for cleanliness.
