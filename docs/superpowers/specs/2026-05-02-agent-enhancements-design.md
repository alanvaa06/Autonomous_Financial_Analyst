# MarketMind v2.1 — Agent Enhancements Design Spec

**Date:** 2026-05-02
**Status:** Draft for review
**Owner:** alanvaa06
**Supersedes parts of:** `docs/superpowers/specs/2026-05-01-marketmind-v2-design.md` §5 (specialist agents) and §3 (graph wiring)
**Builds on:** v2.0 shipped at GitHub `1a8465a` / HF Space build of same.

---

## 1. Summary

Upgrade every specialist agent in MarketMind v2 from a single-shot generic prompt
to a **persona-driven, methodology-grounded, tool-calling agent** with deeper
domain framing, structured output schemas, and explicit guardrails.

Three orthogonal enhancements per agent:

1. **A — Persona + methodology + system prompt overhaul.** Every agent gets a
   credentialed persona (CFA, CMT, FRM, etc.), a named methodology block
   (DuPont, Yield Curve framework, Wyckoff/regime, source-quality weighting,
   forward-looking risk decomposition), explicit constraints, and explicit
   guardrails.
2. **B — Selectively richer always-on data.** Per-agent decisions on what
   raw inputs are pre-injected. Heavier data on Fundamentals and Macro;
   surgical additions on Price/Sentiment/Risk.
3. **C — Tool-calling loop.** Each specialist becomes a Claude tool-use
   agent with 2-4 on-demand tools, capped at **3 tool calls per run**, with
   prompt caching on the persona block.

Reasoning style per agent (locked):

| Agent | Mode |
|---|---|
| Fundamentals | **CoT** |
| Macro | **CoT** |
| Synthesis | **CoT** (promoted from few-shot) |
| Price | Few-shot (3 examples) |
| Sentiment | Few-shot (3 examples) |
| Risk | Few-shot (3 examples) |

---

## 2. Goals and Non-Goals

### 2.1 Goals
1. Replace generic single-shot prompts with credentialed personas and named
   methodologies.
2. Enrich `AgentSignal` outputs with `key_metrics` (numeric ratios) and
   `flags` (named conditions) so the Supervisor and Synthesis layers have
   structured context, not just prose.
3. Give each agent 2-4 on-demand tools so it can drill into specific data
   when its analysis demands it — without making every run pay for every
   tool call.
4. Promote `data_prefetch` from "price + VIX only" to a single ingestion
   layer that also fetches the SEC EdgarBundle, so both Fundamentals and
   Risk can read forward-looking fundamental signals from shared state.
5. Promote Synthesis from few-shot to CoT and add structured outputs
   (`key_drivers`, `dissenting_view`, `watch_items`).
6. Maintain v2.0 contracts: deterministic `final_verdict` / `final_conviction` /
   `final_confidence` math, supervisor light-QA semantics, BYO-key model,
   per-session isolation.

### 2.2 Non-Goals (v2.1)
1. No new specialist agents (Insider/Options/ESG remain v2.2 backlog).
2. No multi-ticker batch mode.
3. No agent-to-agent direct communication. Specialists still run in parallel
   superstep; coordination remains via shared state and the Supervisor/Synthesis
   nodes.
4. No streaming UI changes (already shipped in v2.0).
5. No model bumps; stay on Sonnet 4.6 + Haiku 4.5.
6. No backtesting harness or persistent log.

---

## 3. Cross-Cutting Architecture

### 3.1 Tool-calling loop (shared helper)

A shared utility in `agents/__init__.py` exposes:

```python
def run_with_tools(
    clients: LLMClients,
    system_prompt: str,
    user_prompt: str,
    tools: list[ToolDef],
    max_iterations: int = 3,
    model: str = "reasoning",  # or "fast"
) -> dict:
    """Execute a Claude tool-use loop with hard iteration cap.

    - Uses prompt caching on system_prompt (cache_control on the system block)
    - Each iteration: Claude either emits tool_use blocks (executed locally,
      results fed back) or emits a final text block with JSON.
    - After max_iterations tool-use rounds, the final iteration is forced
      with `tool_choice={"type": "any", "disable_parallel_tool_use": true}`
      replaced by `tool_choice={"type": "none"}` so Claude must produce text.
    - Returns the parsed JSON dict from the final text block.
    """
```

`ToolDef` is a small dataclass:
```python
@dataclass
class ToolDef:
    name: str
    description: str          # used in tool definition for Claude
    input_schema: dict        # JSONSchema
    handler: Callable[[dict], dict]  # local executor
```

The handler receives the parsed tool-input dict and returns a JSON-serializable
dict (or string) sent back to Claude as the `tool_result` content.

### 3.2 Prompt caching strategy

System prompt for each agent is split into two sections:

- **Cached block** (cache_control set): persona + methodology + tool catalogue
  + few-shot examples + output schema. This is identical for every run, so
  it caches across requests for the same Anthropic key.
- **Per-run block**: ticker, date, always-on data payload (price metrics,
  XBRL facts, news rollup, etc.). This changes every run — not cached.

Anthropic's prompt cache TTL is 5 minutes. Within a single multi-turn agent
run (3 tool-use iterations), the cache stays warm for the entire conversation.

### 3.3 EdgarBundle promotion to data_prefetch

Currently `agents/fundamentals_agent.py` calls `build_edgar_bundle(ticker)`
itself. After this change:

- `agents/data_prefetch.py` calls `build_edgar_bundle(ticker)` and stores
  the result in `state["edgar_bundle"]` (`Optional[EdgarBundle]`).
- `fundamentals_agent` reads from state; falls back to fetching itself if
  `state["edgar_bundle"]` is `None` (preserves test compatibility).
- `risk_agent` also reads from state to pull forward-looking fundamental
  signals (revenue YoY, margin YoY, FCF margin, D/E).

### 3.4 State additions

```python
# state.py
class MarketMindState(TypedDict):
    # ... (existing fields unchanged)
    edgar_bundle: Optional[Any]   # NEW: prefetched EdgarBundle dataclass
```

### 3.5 AgentSignal schema additions

```python
# state.py
class AgentSignal(TypedDict, total=False):
    # Existing required fields:
    agent: str
    signal: Signal
    confidence: float
    summary: str
    section_markdown: str
    raw_data: dict
    degraded: bool
    error: Optional[str]
    # NEW optional fields (per-agent contributions):
    key_metrics: Optional[dict]      # numeric ratios; agent-defined keys
    flags: Optional[list[str]]       # named conditions
    # Agent-specific extension fields (typed loosely as Optional[Any]):
    regime: Optional[str]            # price (trending_up/...), macro (risk-on/...)
    vol_regime: Optional[str]        # risk
    vix_regime: Optional[str]        # risk
    forward_risk_view: Optional[str] # risk
    primary_risk_driver: Optional[str]  # risk
    risk_decomposition: Optional[dict]  # risk
    yield_curve_state: Optional[str] # macro
    ticker_exposure: Optional[str]   # macro
    top_catalyst: Optional[str]      # sentiment
    drivers_categorized: Optional[dict]  # sentiment
```

`total=False` makes all fields optional (the v2.0 required fields are still
populated by every specialist; the new fields are populated only by the
agent that owns them).

### 3.6 Supervisor changes

The Supervisor remains deterministic light-QA. Two additions:

1. **Sanity checks read `key_metrics`** — e.g., `key_metrics.roe_pct > 200`
   flags Fundamentals; `key_metrics.annualized_vol_pct > 500` flags Risk
   (existing check, just sourced from `key_metrics` instead of `raw_data`).
2. **Cross-signal consistency check** — if Fundamentals' `key_metrics.revenue_yoy_pct < -10`
   and Macro's `regime == "risk-on"` and ticker_exposure == "high", flag
   Macro for re-examination. Light cross-checks only; not a verdict override.

### 3.7 Synthesis changes

- LLM call promoted to CoT.
- LLM produces `key_drivers`, `dissenting_view`, `watch_items` in addition
  to `final_reasoning`.
- `final_report` markdown assembly extended to render these new sections:
  - Below the verdict label in the Executive Summary, render `key_drivers`
    as bullets.
  - At the end of "Synthesis & Final Verdict", render `dissenting_view`
    italicized and `watch_items` as a bulleted "What to watch" subsection.
- `final_verdict`, `final_conviction`, `final_confidence` math unchanged
  (deterministic upstream).

### 3.8 Cost envelope

v2.0 ran ~$0.30 per analysis (BYO key, list price, with caching). v2.1
expected envelope:

- Fundamentals: 1 base call + up to 3 tool calls = ~4 Sonnet calls (~$0.10–0.18)
- Macro: 1 base + up to 3 tool calls = ~4 Sonnet calls (~$0.08–0.15)
- Synthesis CoT promotion: ~$0.05 increase
- Other agents: small change
- Estimated v2.1 per-run cost: **$0.40–0.60** with caching. Document this in
  the cost-hint badge in `app.py` ("~$0.50/run on BYO key").

---

## 4. Per-Agent Specifications

Each agent below has been individually approved during brainstorming; the
spec captures the locked decisions verbatim.

### 4.1 Fundamentals (CoT)

**Persona:**
> "You are a CFA Charterholder Senior equity research analyst with deep
> expertise on US large-cap equities. You understand accounting conventions
> (US GAAP and IFRS) and extract key insights from 10-Q/10-K. Your judgment
> is objective and skeptic but flexible enough to identify where MD&A differ
> from fundamentals."

**Methodology:**
- DuPont decomposition (ROE = Net Margin × Asset Turnover × Equity Multiplier)
- Quality-of-earnings checks (operating cash flow vs net income; accruals ratio)
- Operating leverage (Δ%OpInc / Δ%Rev)
- Cash Conversion Cycle (DIO + DSO − DPO)
- Segment growth attribution

**Always-on data (from `state["edgar_bundle"]`):**
- Latest XBRL facts: Revenues, GrossProfit, OperatingIncomeLoss, NetIncomeLoss,
  EarningsPerShareDiluted, Assets, Liabilities, StockholdersEquity, Cash,
  SharesOutstanding (and YoY deltas, computed before LLM call)
- MD&A excerpt (full 8K)
- 10-K Risk Factors first 4K

**On-demand tools (3 max):**
1. `fetch_xbrl_tag(tag_name: str, periods: int = 8)` — arbitrary GAAP tag
   with quarterly history. Examples in tool description: `ResearchAndDevelopmentExpense`,
   `OperatingCashFlowsContinuingOperations`, `CapitalExpenditures`. Reads
   from `state["edgar_bundle"].xbrl_facts` (no new network).
2. `fetch_segment_breakdown()` — `RevenuesFromExternalCustomers` by
   reportable segment when filed in XBRL.
3. `peer_multiples(peer_tickers: list[str])` — quick comp via yfinance
   (P/E, EV/EBITDA, P/S, P/B). Agent picks 3-5 peers from MD&A or its own
   knowledge.

**Output schema additions:**
- `key_metrics`: `{roe_pct, op_margin_pct, op_margin_delta_yoy_bps, fcf_margin_pct, debt_to_equity, current_ratio, accruals_ratio_pct, eps_yoy_pct}`
- `flags`: e.g. `["operating_leverage_positive", "margin_expansion", "low_accruals"]`

**CoT block (in system prompt):** 8-step chain — revenue → margin → balance
sheet → earnings quality → operating leverage → MD&A → risk factors →
integrated call. Tool-call rules included.

**Constraints:**
- No "buy/sell" verbiage; "favorable / cautious / unfavorable" framings
- section_markdown 200-300 words
- Confidence ≤ 0.6 if `latest_10q_filed > 100 days old` OR MD&A empty OR
  < 3 quarters of history
- Confidence ≤ 0.4 if `Revenues_yoy_pct is None`
- Tool budget: 3 calls

### 4.2 Macro (CoT)

**Persona:**
> "You are a senior global macro strategist, CFA Charterholder, with 15
> years on the rates and FX desk of a major investment bank. You read
> cross-asset signals — DXY, yield curve, credit spreads, commodities,
> positioning — and synthesize them into a regime call (risk-on/off,
> reflation/disinflation, growth-scare) that you map to specific equity
> sector implications. You're skeptical of single-print headlines and
> prefer trend-confirmed moves."

**Methodology:**
- Yield curve framework (2s10s sign + slope; bull-steepening vs bear-flattening)
- Real rates impact (FF − headline CPI proxy)
- DXY transmission (rising DXY bearish for non-USD revenue exposure)
- Credit spreads (HY OAS / IG OAS ratio)
- Commodity regime
- Sentiment positioning (Fear & Greed contrarian / confirming)

**Always-on data:**
- DXY (DTWEXBGS) latest + 5d change
- Fed funds rate (DFF), 10Y (DGS10), 2Y (DGS2), 2s10s spread
- Fear & Greed index

**On-demand tools (3 max):**
1. `fetch_fred_series(series_id: str, periods: int = 12)` — arbitrary FRED
   series. Tool description includes catalogue: `BAMLH0A0HYM2` (HY OAS),
   `T10YIE` (10Y breakeven inflation), `DCOILWTICO` (WTI), `UNRATE`,
   `CPIAUCSL`.
2. `classify_ticker_sector(ticker: str)` — returns GICS sector + sub-industry
   from yfinance `Ticker.info`.
3. `fetch_credit_spreads()` — HYG/LQD ratio + 30d trend (yfinance proxy).
   Risk-on/off gauge when FRED HY series unavailable.

**Output schema additions:**
- `regime` ∈ `{risk-on, risk-off, reflation, disinflation, stagflation, neutral}`
- `yield_curve_state` ∈ `{steep, flat, inverted}`
- `ticker_exposure` ∈ `{high, medium, low}`
- `key_metrics`: `{dxy_latest, dxy_5d_change, fed_funds_rate, yield_curve_2s10s, fear_greed_index, real_rate_proxy}`

**CoT block (in system prompt):** 8-step chain — rates regime → curve →
USD → credit/liquidity → inflation → regime classification → sector mapping
→ integrated call.

**Constraints:**
- "supportive / mixed / headwind" framings
- section_markdown 150-250 words
- Confidence ≤ 0.5 if FRED key absent
- Confidence ≤ 0.6 if `yield_curve_2s10s is None`
- `regime` MUST be one of the 6 enum values; default `neutral` on indeterminate
- Tool budget: 3 calls

### 4.3 Synthesis (CoT, no tools)

**Persona:**
> "You are the Chief Investment Officer and Director of Research at a
> multi-strategy fund. You chair the daily investment committee — your
> job is to integrate five specialist signals into a single coherent thesis,
> name the strongest argument for the call AND the strongest argument
> against, and identify the leading indicators that would force a
> re-evaluation. You never overstate confidence and you always make dissent
> visible."

**Methodology:**
- Weight-of-evidence integration (verdict already deterministic upstream)
- Investment-committee-chair pattern (bull case, bear case, integrated thesis, dissent)
- Cite-the-source discipline (specialist + metric)
- Regime-change watch list (leading indicators that flip the call)

**Always-on data:** all five `AgentSignal`s (with new `key_metrics`, `flags`,
agent-specific fields), plus `supervisor_review`, plus deterministic verdict
already computed.

**On-demand tools:** **none**. Synthesis only narrates over existing state.

**Output schema additions:**
- `key_drivers`: list[str], 2-4 entries, each "Specialist: metric/observation"
- `dissenting_view`: str, 1 sentence, ≤25 words
- `watch_items`: list[str], 2-3 entries, each ≤20 words

`final_verdict`, `final_conviction`, `final_confidence` remain deterministic.

**CoT block:** 8-step chain — consensus → dissent → strongest FOR → strongest
AGAINST → integrated thesis → key drivers → dissenting view → watch items.

**Constraints:**
- Cannot change verdict/conviction/confidence
- ≥3 specialists referenced by name in `final_reasoning`
- Lengths: reasoning 80-150 words; key_drivers entries ≤15 words; dissenting
  ≤25 words; watch_items ≤20 words
- If supervisor flagged degraded data, `dissenting_view` MUST acknowledge it

### 4.4 Price (few-shot)

**Persona:**
> "You are a CMT Charterholder (Chartered Market Technician) with 12 years
> on a quant systematic trading desk. You read momentum, mean-reversion,
> and volatility regimes from price action — RSI/MACD/Bollinger are
> starting points, not gospel. You're explicit about whether the current
> setup is trend-confirming or mean-reverting, and you flag when indicators
> conflict."

**Methodology:**
- Trend regime first (trending vs ranging vs trending-down)
- RSI in context (in strong uptrend, RSI 70+ can persist)
- MACD line vs signal + histogram divergence
- Bollinger %B with mid-band as trend filter
- Multi-timeframe alignment (7d / 30d / 90d)
- Volatility regime (ATR-based)

**Always-on data:** existing — current price, 7/30/90d % change, RSI(14),
MACD line + signal + crossover, Bollinger %B.

**On-demand tools (3 max — all local, NO new network):**
1. `compute_indicator(name: str)` — ATR(14), ADX(14), Stochastic(14,3,3),
   SMA(50), SMA(200), OBV. Computed on `state["price_history"]`.
2. `detect_chart_pattern()` — heuristic detection: ascending/descending
   triangle, double-top/bottom, head-and-shoulders.
3. `volume_profile_summary(n_buckets: int = 10)` — volume by price bucket.

(Excluded: peer-relative-strength, sector-ETF-relative — both would need
new yfinance hits and re-trigger Yahoo throttle.)

**Output schema additions:**
- `regime` ∈ `{trending_up, trending_down, ranging, volatility_expansion}`
- `key_metrics`: `{rsi, macd_state, bollinger_pctb, atr_pct, sma50_vs_sma200}`
- `flags`: e.g. `["trend_confirmation", "mid_band", "momentum_persistent"]`

**Few-shot examples (3 in system prompt):**
- Momentum override of overbought (RSI 76 in confirmed uptrend → BULLISH)
- Conflicting signals → wait (NEUTRAL ranging)
- Oversold bounce setup (RSI 28 + MACD positive cross + lower band → BULLISH)

**Constraints:**
- "constructive / cautious / negative" framings
- section_markdown 120-200 words
- Confidence ≤ 0.5 if any indicator NaN
- Confidence ≤ 0.6 if price_history < 60 trading days
- `regime` MUST be set explicitly in summary
- Tool budget: 3 calls

### 4.5 Sentiment (few-shot)

**Persona:**
> "You are a senior buy-side equity analyst with a CFA Charter and a side
> specialty in behavioral finance and market-sentiment analysis. You weigh
> primary sources (issuer press releases, regulatory filings) over wire
> pickups, you discount social-media noise, and you separate fact-driven
> catalysts from narrative-driven momentum."

**Methodology:**
- Source-quality weighting (primary > tier-1 > tier-2 > aggregator/social)
- Catalyst classification (earnings, M&A, regulatory, product, insider,
  competitor, macro)
- Magnitude over count
- Recency decay (48h ×2, 7d ×1, >7d ×0.5)
- Narrative vs fundamental check
- Contrarian flag

**Always-on data:** existing — Tavily news (last 7 days, deduped, max 12),
Haiku per-article classification, drivers list, pos/neu/neg counts, sample
headlines.

**On-demand tools (3 max — Tavily-only, no yfinance):**
1. `fetch_press_releases(ticker: str, days: int = 14)` — Tavily targeted
   query (`site:prnewswire.com OR site:businesswire.com OR site:globenewswire.com`).
2. `fetch_analyst_actions(ticker: str)` — Tavily query for "{ticker} analyst
   upgrade downgrade price target" last 14 days.
3. `categorize_drivers(drivers: list[str])` — Haiku call with fixed taxonomy.

**Output schema additions:**
- `top_catalyst`: str — single specific event driving sentiment
- `key_metrics`: `{article_count, positive_count, neutral_count, negative_count, recency_24h_count, source_mix}`
- `drivers_categorized`: dict — taxonomy bucket counts
- `flags`: e.g. `["catalyst_present", "narrative_driven", "primary_source_present"]`

**Few-shot examples (3 in system prompt):**
- Strong fundamental catalyst (BULLISH 0.75)
- Narrative-driven, fade (NEUTRAL 0.40)
- Bearish overhang (BEARISH 0.78)

**Constraints:**
- "supportive / mixed / unfavorable" framings
- section_markdown 120-180 words
- Confidence ≤ 0.4 if `article_count < 3` (forces `sparse_coverage` flag)
- Confidence ≤ 0.5 if 0 primary sources AND signal is BULLISH
- `top_catalyst` must reference a specific event
- Cite ≥1 specific headline in section_markdown
- Tool budget: 3 calls

### 4.6 Risk (few-shot, forward-looking)

**Persona:**
> "You are an FRM Charterholder (Financial Risk Manager) and former PM at
> a long/short equity fund. You think in terms of risk-adjusted returns,
> drawdown discipline, and regime-conditional volatility — not absolute vol.
> You distinguish idiosyncratic risk (this name) from systemic risk
> (the tape) and you flag when correlation regimes are shifting."

**Methodology:**
- **Forward-looking risk evaluation drives the call.** Backward stats
  (Sharpe, Sortino, Calmar, historical VaR) are *information*, not verdict.
- Fundamental risk leading indicators — revenue YoY deceleration, op margin
  compression, leverage rising, FCF deterioration, accruals quality
  deterioration. From `state["edgar_bundle"]`.
- Price regime leading indicators — vol percentile (current vs trailing 1y),
  trend break (price vs SMA50/SMA200), drawdown state.
- Trailing quant calibrates *confidence*, not direction.
- Positioning risk (beta in current VIX regime, short squeeze potential).
- Risk decomposition: operating / balance-sheet / positioning / systemic.

**Always-on data (after edgar_bundle promotion):**
- Forward-looking (priority): revenue_yoy_pct, revenue_qoq_pct, op_margin_yoy_bps,
  fcf_margin_pct, debt_to_equity, debt_to_equity_yoy_delta, vol_percentile_1y,
  trend_state, drawdown_state
- Backward-looking (informational): annualized_vol_pct, max_drawdown_pct,
  sharpe, sortino, calmar, var_95_1d, max_1d_drop
- Regime/positioning: VIX, beta, short_ratio

**On-demand tools (3 max — all local):**
1. `forward_risk_attribution()` — returns `{operating: low/med/high,
   balance_sheet: ..., positioning: ..., systemic: ...}` based on always-on
   signals + fixed thresholds.
2. `decompose_drawdown()` — splits current/max drawdown into trend vs vol vs
   gap-down components.
3. `compute_var_es(confidence: float = 0.95)` — historical VaR + ES. Informational.

**Output schema additions:**
- `forward_risk_view` ∈ `{favorable, mixed, deteriorating, elevated}`
- `primary_risk_driver` ∈ `{operating_deceleration, margin_pressure, balance_sheet, systemic_vol, positioning, none}`
- `risk_decomposition`: `{operating, balance_sheet, positioning, systemic}` each ∈ `{low, medium, high}`
- `vol_regime` ∈ `{compressed, normal, elevated, stress}`
- `vix_regime` ∈ `{low, normal, elevated, stress}`
- `key_metrics`: full forward + backward dict (see §4.6 brainstorm proposal for full list)
- `flags`: e.g. `["operating_deceleration", "margin_compression", "vol_expansion", "high_beta_in_stress"]`

**Few-shot examples (3 in system prompt, rewritten for forward-first emphasis):**
- Forward-deteriorating despite OK trailing stats (BEARISH 0.70)
- Forward-favorable in noisy tape (BULLISH 0.65)
- Mixed with positioning as primary driver (NEUTRAL 0.55)

**Constraints:**
- Forward-looking signals drive `signal`; backward stats only calibrate `confidence` (explicit in prompt)
- "constructive / cautious / negative" framings (risk-adjusted)
- section_markdown 150-250 words
- Confidence ≤ 0.5 if forward fundamental data missing
- Confidence ≤ 0.55 if returns history < 60 trading days
- `forward_risk_view` AND `primary_risk_driver` MUST be set explicitly in summary
- Distinguish idio vs systemic in section_markdown
- Tool budget: 3 calls

---

## 5. File Layout (Target)

```
agents/
  __init__.py            MODIFIED — add run_with_tools(), ToolDef, prompt-cache helper
  data_prefetch.py       MODIFIED — also fetch EdgarBundle
  orchestrator.py        UNCHANGED
  price_agent.py         REWRITTEN — persona + methodology + few-shot + tools
  sentiment_agent.py     REWRITTEN — persona + methodology + few-shot + tools
  fundamentals_agent.py  REWRITTEN — persona + methodology + CoT + tools
  macro_agent.py         REWRITTEN — persona + methodology + CoT + tools
  risk_agent.py          REWRITTEN — persona + forward-looking methodology + few-shot + tools
  supervisor_agent.py    MODIFIED — sanity reads from key_metrics; new cross-signal check
  synthesis_agent.py     REWRITTEN — CoT + key_drivers + dissenting_view + watch_items
  yf_helpers.py          UNCHANGED
  tools/
    __init__.py          NEW — shared tool registry
    fundamentals_tools.py NEW — fetch_xbrl_tag, fetch_segment_breakdown, peer_multiples
    macro_tools.py       NEW — fetch_fred_series, classify_ticker_sector, fetch_credit_spreads
    price_tools.py       NEW — compute_indicator, detect_chart_pattern, volume_profile_summary
    sentiment_tools.py   NEW — fetch_press_releases, fetch_analyst_actions, categorize_drivers
    risk_tools.py        NEW — forward_risk_attribution, decompose_drawdown, compute_var_es

state.py                 MODIFIED — add edgar_bundle field; AgentSignal becomes total=False with new optional fields

graph.py                 UNCHANGED (data_prefetch already wired)

app.py                   MODIFIED — initial state includes edgar_bundle: None; cost-hint badge updated to "$0.50/run"

tests/
  test_agents_init.py        MODIFIED — add tests for run_with_tools()
  test_data_prefetch.py      MODIFIED — covers EdgarBundle fetch
  test_state.py              MODIFIED — covers edgar_bundle + new AgentSignal fields
  test_price_agent.py        REWRITTEN — covers tool-use loop + new outputs
  test_sentiment_agent.py    REWRITTEN
  test_fundamentals_agent.py REWRITTEN
  test_macro_agent.py        REWRITTEN
  test_risk_agent.py         REWRITTEN — forward-looking emphasis
  test_supervisor_agent.py   MODIFIED — new cross-signal check tests
  test_synthesis_agent.py    MODIFIED — covers CoT outputs
  test_tools/                NEW directory — one test file per tool module

docs/superpowers/specs/2026-05-02-agent-enhancements-design.md   THIS DOC
```

---

## 6. Functional and Non-Functional Requirements

### 6.1 Functional
- **FR-1.** Each specialist agent uses a credentialed persona + named methodology in its system prompt.
- **FR-2.** Each specialist agent supports a tool-use loop with hard cap of 3 tool calls per run; after the cap, Claude is forced to produce final JSON.
- **FR-3.** Each specialist returns `AgentSignal` with the v2.0 required fields plus its agent-specific `key_metrics`, `flags`, and any agent-specific schema additions (regime, vol_regime, etc.).
- **FR-4.** `data_prefetch` populates `state["edgar_bundle"]` in addition to `price_history` and `vix_history`.
- **FR-5.** Fundamentals reads from `state["edgar_bundle"]` (falls back to fetching itself when state field is None).
- **FR-6.** Risk reads forward-looking fundamental signals from `state["edgar_bundle"]` AND price-regime signals from `state["price_history"]`. Trailing stats are computed but explicitly demoted in the prompt.
- **FR-7.** Synthesis emits `key_drivers`, `dissenting_view`, `watch_items` in addition to `final_reasoning`. Verdict math stays deterministic.
- **FR-8.** Supervisor reads `key_metrics` (not just `raw_data`) for sanity checks; adds a cross-signal consistency check between Fundamentals and Macro.
- **FR-9.** All agents render their `section_markdown` per their length budget and use the prescribed framings (no "buy/sell" verbiage).

### 6.2 Non-Functional
- **NFR-1.** Per-run cost target: ≤ $0.60 with prompt caching on persona blocks (≤ $1.20 worst-case with all tools used and no cache).
- **NFR-2.** Per-run latency target: ≤ 45s p50 (up from v2.0's 25s — tool-use loops add iterations).
- **NFR-3.** Prompt cache MUST be applied to each agent's static system block (persona + methodology + tool catalogue + few-shot examples + output schema).
- **NFR-4.** Tool handlers MUST be local (no new network) wherever possible. Where a network call is unavoidable (yfinance peer multiples, Tavily targeted queries), the tool docstring MUST flag the cost.
- **NFR-5.** Backward compatibility: every agent must produce the v2.0 required `AgentSignal` fields so that the existing supervisor + synthesis + UI continue to work even if the new optional fields are missing (degraded fallback path).
- **NFR-6.** Tests: one test file per agent + one per tool module + integration test exercising the full graph with mocked tool handlers.

---

## 7. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Tool-use loop runs away (more iterations than expected) | Hard cap at 3; force `tool_choice=none` on the 4th iteration. |
| Anthropic 429s on tool-use loops (more requests per run) | Existing per-key concurrency already absent; rely on Anthropic's per-key burst limit. Prompt caching reduces token volume. If 429s observed, add `tenacity` retry around `clients.reasoning.invoke`. |
| Tool handlers crash silently | Each handler wrapped in try/except; on error, returns `{"error": "..."}` to Claude as tool_result so the model can recover or skip. |
| Schema drift breaks Supervisor / Synthesis | `AgentSignal` becomes `total=False` so old fields stay required, new fields are optional. Supervisor and Synthesis use `.get()` everywhere. |
| EdgarBundle promotion breaks Fundamentals tests | Fundamentals retains the fallback `build_edgar_bundle(ticker)` call when `state["edgar_bundle"]` is None. Existing unit tests still pass. |
| Forward-risk view depends on EdgarBundle which may be missing (foreign issuer / ETF) | Risk degrades that dimension (sets `forward_fundamentals_unavailable` flag), uses price/vol-only call, caps confidence at 0.5. |
| Cross-signal supervisor check creates false-positive retries | Check is conservative (only when explicit conflict + high confidence on both sides); supervisor force-approves after one retry round per existing v2.0 contract. |
| Cost overrun (more than $0.60/run) | Prompt caching is the primary lever. If costs exceed budget in production, tighten per-agent `max_iterations` from 3 to 2. |

---

## 8. Phases

- **v2.1.0 (this spec):** All 6 agent enhancements + EdgarBundle promotion + AgentSignal schema additions + Supervisor sanity-check upgrade + Synthesis CoT.
- **v2.1.1 (post-merge):** Tune per-agent few-shot examples based on production observations. Add 1-2 more on-demand tools per agent if specific gaps emerge.
- **v2.2 (backlog, separate spec):** Insider Trading agent, Options Flow agent, ESG agent. Multi-ticker batch.

---

## 9. Open Decisions Locked This Round

- Persona format: 2 lines, credentialed (CFA / CMT / FRM specific where applicable).
- Reasoning style per agent locked (CoT for Fundamentals + Macro + Synthesis; few-shot for Price + Sentiment + Risk).
- Tool budget: 3 per run, all agents.
- Tools where possible are local-compute (no new network). Network tools allowed but flagged.
- Verdict + conviction + confidence stay deterministic upstream — Synthesis cannot override.
- EdgarBundle promoted to data_prefetch; shared by Fundamentals + Risk.
- Risk agent pivots to forward-looking; trailing quant demoted.
- AgentSignal schema becomes `total=False` so new fields are optional.
- Cost target: ≤ $0.60/run with caching.
