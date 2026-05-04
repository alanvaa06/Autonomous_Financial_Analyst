# MarketMind v2.1 — Data & Synthesis Fixes Design

**Date:** 2026-05-04
**Status:** Design draft — pending user review
**Related:** AMZN run 2026-05-04 14:27 UTC surfaced three defects

## Problem statement

The AMZN report run produced a usable but visibly degraded artifact. Three defects compound:

1. **Synthesis LLM call returns empty content** — `safe_parse_json("")` raises "Expecting value: line 1 column 1 (char 0)" → narrative falls back to a static "Synthesis LLM call failed" line. Verdict math still works, but the integrated thesis, key drivers, dissenting view, and watch items are absent from the final report.
2. **SMA200 unavailable** — the price agent and risk agent both depend on SMA200 for trend-state classification. Prefetch fetches only 90 trading days, so SMA200 is never computable. Risk agent reports `trend_state=unknown` and price agent caps conviction.
3. **XBRL Revenues tag returns no observations for AMZN** — issuers that adopted ASC 606 file under `RevenueFromContractWithCustomerExcludingAssessedTax`, not the legacy `Revenues` tag. Both `_key_metrics_from_facts` (fundamentals) and `_yoy_revenue_pct` (risk) hardcode `Revenues` and return None. Cascades into missing margin and YoY growth.

## Goals

- Restore the synthesis narrative section.
- Compute SMA200 (and 1-year vol percentile, drawdown) for any ticker with sufficient listing history.
- Resolve revenue observations for ASC 606 issuers (AMZN, MSFT, GOOG, MS-class large caps).

## Non-goals

- Atomic period-matched key_metrics builder (margin denominator/numerator from same accession). Tracked in spawned task.
- `Liabilities` tag fallback for issuers that file only `LiabilitiesAndStockholdersEquity`. Tracked in spawned task.
- Foreign filer support (non-USD `units`). Out of scope.
- Multi-tag historical merge across `Revenues` ↔ ASC 606 boundary. Out of scope; we look at last 8 quarters where boundary doesn't bite.

## Architecture

Three orthogonal changes, no cross-coupling.

### A1 — Synthesis migrates to `run_with_tools(tools=[])`

**Current:** `agents/synthesis_agent.py` calls `clients.reasoning.invoke([...])` (LangChain `ChatAnthropic`) → `safe_parse_json(resp.content)` with no retry. When Claude writes the 8-step CoT as prose, hits `max_tokens=1500` before producing the JSON object, content is empty / truncated → parse fails.

**Change:** Replace the LangChain invoke with `agents.run_with_tools(api_key, system_prompt, user_prompt, tools=[], max_iterations=2, max_tokens=2000)`. Empty `tools` list means no tool schema is sent on any iteration; the loop becomes a pure JSON-with-retry path.

**Why this works:** `run_with_tools` already handles:
- Empty text response → corrective user nudge "Please reply with the final JSON object now."
- JSON parse failure → corrective user nudge "That response was not valid JSON. Reply ONLY with a single JSON object..."
- Tries up to `max_iterations + 1` times before raising `ValueError`.

**Verify path:** `run_with_tools` builds the `tools` kwarg only when `tool_specs and iteration < max_iterations`. With `tools=[]`, `tool_specs` is falsy → the kwarg is never set on any iteration. Anthropic API accepts this. Cache control on system prompt still applies. Stop reason will be `end_turn` (never `tool_use`), so the loop falls through to the text-parse branch every iteration.

**Token bump:** `max_tokens` 1500 → 2000. Synthesis JSON has ≤4 fields with bounded length (reasoning ≤150 words, key_drivers ≤4×15 words, dissenting_view ≤25 words, watch_items ≤3×20 words). 2000 tokens is comfortably above the worst case and absorbs prose-then-JSON behavior on iteration 1.

**Prompt tightening:** add one explicit line at end of `OUTPUT_SCHEMA`: "Reason step-by-step internally; emit ONLY the JSON object as your final response." Keeps the CoT mental scaffold but discourages writing it as visible prose.

**Failure mode:** if all 3 iterations fail, `run_with_tools` raises `ValueError`. Synthesis catches `Exception` and falls back to the existing static line. Net behavior: same-or-better than today.

### B1 — Revenue tag fallback helper

**Current:** Hardcoded `Revenues` reads in three call sites:
- `agents/fundamentals_agent.py:109` — `_key_metrics_from_facts._latest("Revenues")`
- `agents/risk_agent.py:182` — `_yoy_revenue_pct` reads `facts...Revenues...units.USD`
- `agents/tools/fundamentals_tools.py:48` — `_fetch_segment_breakdown` already has a 2-tag fallback chain

**Change:** Add two helpers to `edgar.py`:

```python
REVENUE_TAGS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
)
VALID_FP = {"Q1", "Q2", "Q3", "FY"}

def latest_revenue_observations(facts: dict, *, periods: int = 8) -> tuple[str | None, list[dict]]:
    """Pick first revenue tag with >=2 clean observations.
    Returns (tag_used, sorted_observations) or (None, []).
    Filters: form in {10-Q, 10-K}, fp in {Q1, Q2, Q3, FY}.
    Sort: (end DESC, filed DESC) — newest period first; restated values win.
    """

def yoy_revenue_pct(observations: list[dict]) -> float | None:
    """Compute YoY % from latest observation against same-fp prior year.
    Returns None when fewer than 2 obs OR no matching prior fp."""
```

**Pre-existing function `_fetch_xbrl_tag`:** unchanged. Generic; consumers pass tag explicitly.

**Call site updates:**
- `agents/risk_agent.py::_yoy_revenue_pct` deletes its inline implementation, calls `edgar.latest_revenue_observations` then `edgar.yoy_revenue_pct`.
- `agents/fundamentals_agent.py::_key_metrics_from_facts` — replace `rev, _ = _latest("Revenues")` with `tag, obs = latest_revenue_observations(facts); rev = float(obs[0]["val"]) if obs else None`. (Still subject to period-mismatch bug; that fix lives in spawned task.)
- `agents/tools/fundamentals_tools.py::_fetch_segment_breakdown` — replace local 2-tag fallback with shared helper for consistency.

**Filter rationale:**
- `fp ∈ {Q1, Q2, Q3, FY}` excludes synthetic "FY-Q4" computed periods and Q4 (which XBRL elides — Q4 = FY − Q1 − Q2 − Q3 implicitly).
- Sort by `filed DESC` as secondary key picks restated values over original filings.
- Picking the first tag with ≥2 obs (not the first non-empty tag) avoids edge cases where issuer filed `Revenues` once historically and then switched to ASC 606.

### C1 — Prefetch period 90d → 1y

**Current:** `agents/data_prefetch.py:45` calls `_safe_yf(ticker, period="90d")`. Fallbacks in `agents/price_agent.py:169` and `agents/risk_agent.py:239` also use `period="90d"`.

**Change:** Replace `"90d"` with `"1y"` in all three locations.

- Same number of yfinance HTTP calls; same rate-limit profile.
- Payload is ~3× larger (~250 rows vs ~90 rows). Negligible — well under 1MB.
- Risk agent's `_trailing_stats` already guards SMA200 with `if len(close) >= 200`. No code change needed.
- Price agent's `_compute_raw` uses `iloc[-7]`, `iloc[-30]`, and `iloc[0]` for change calcs. With 1y of data:
  - `change_7d_pct` and `change_30d_pct` unchanged.
  - `change_90d_pct` is currently computed as `(current - close.iloc[0]) / close.iloc[0] * 100`. With `iloc[0]` = ~1y ago, this becomes a 1y change, not a 90d change. **Bug introduced unless we fix.** Fix: change `close.iloc[0]` to `close.iloc[-90] if len(close) >= 90 else close.iloc[0]`. Or rename to `change_1y_pct`. Recommend the index fix to preserve existing field semantics.
- VIX prefetch stays at 5d. Macro snapshot only needs latest level.

**Side effects:**
- Risk agent will now compute `trend_state` correctly for any ticker with ≥200 trading days listed (i.e., almost all). `vol_percentile_1y` becomes truly 1y instead of trailing-window approximation.
- Price agent gets SMA200 via tool calls when it asks. The persona prompt already mentions multi-timeframe alignment; with SMA200 available the LLM can use it.

## Data flow

No structural change. State keys (`price_history`, `vix_history`, `edgar_bundle`) keep their shapes. Consumers handle longer DataFrames idempotently.

## Error handling

- A1: `run_with_tools` raises `ValueError` after exhausting iterations; existing `try/except Exception` in synthesis catches it; falls back to deterministic-only narrative (same as current behavior on failure).
- B1: helpers return `None` / `(None, [])` when no tag matches; existing call sites already check for `None`. No new exception paths.
- C1: yfinance behavior unchanged (same retry layer). Tickers with <200 days listing history (recent IPOs) degrade exactly as today via the existing `if len(close) >= 200` guard.

## Testing

Per CLAUDE.md "Verification Before Done":

- **A1**: unit test in `tests/` — patch `Anthropic` client to return (a) empty text on iter 0 and valid JSON on iter 1; assert synthesis returns the iter-1 JSON. Live smoke run on AMZN to confirm narrative section is populated.
- **B1**: unit test using a fixture XBRL `companyfacts` dict with only `RevenueFromContractWithCustomerExcludingAssessedTax` populated; assert `latest_revenue_observations` resolves it and `yoy_revenue_pct` computes against same-fp prior year. Second fixture with both tags present asserts ASC 606 wins. Third fixture with Q1 + YTD double-counted observations confirms the `fp` filter excludes ambiguous duplicates.
- **C1**: live smoke run on AMZN — confirm `key_metrics.sma50_vs_sma200` is populated (price agent) and `trend_state` is `above_sma50_and_sma200` or similar (risk agent), not `unknown`. Confirm `change_90d_pct` still represents 90 trading days.

## Migration / rollout

Single PR / single branch. No data migration. No state schema change. No breaking change to public state shape. Deploy to HF Space after smoke run.

## Open questions

None. Scope locked.

## Out of scope (tracked elsewhere)

- Atomic period-matched key_metrics builder — spawned task `Atomic period-match + Liabilities tag cleanup`.
- `Liabilities` tag fallback — same spawned task.
- Foreign filer (non-USD) revenue resolution.
- Tag-switch reconciliation for >2-year history backtests.
