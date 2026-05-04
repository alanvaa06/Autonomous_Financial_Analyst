# MarketMind v2 — Project Todo

**Status:** Design spec drafted — awaiting user review.
**Spec:** `docs/superpowers/specs/2026-05-01-marketmind-v2-design.md`
**Branch:** main (will move to feature branch before implementation)

## Workflow checkpoints

- [x] Brainstorm and lock direction (Option D — full multi-agent rewrite)
- [x] Resolve key decisions: EDGAR scope, supervisor power, sentiment provider, tools rewrite, fallbacks, conviction tiers
- [x] Draft design spec to `docs/superpowers/specs/2026-05-01-marketmind-v2-design.md`
- [x] Self-review spec for placeholders, contradictions, ambiguity, scope
- [ ] **User reviews spec** ← current step
- [ ] Commit spec to git
- [ ] Write implementation plan via `superpowers:writing-plans`
- [ ] Execute implementation plan (separate session per CLAUDE.md workflow)

## Implementation phases (set after spec approval)

1. Foundations: `state.py`, `agents/__init__.py` factory, `edgar.py`
2. Specialists (one PR per agent): price, sentiment, fundamentals, macro, risk
3. Supervisor + synthesis
4. `graph.py` wiring
5. UI rewrite (`app.py`)
6. Cleanup deletes (`agent.py`, `tools.py`, `rag.py`, `data/`)
7. README + `.env.example` + `requirements.txt` updates
8. Smoke runs: MSFT, AAPL, NVDA, SHEL (foreign), SPY (ETF)
9. HF Space deploy

## Open follow-ups

- v2.1 backlog: per-agent toggles, EDGAR disk cache, additional specialists (insider, options-flow)
- v2.2 backlog: multi-ticker batch, persistent verdict log, backtesting harness
- v2.3 backlog: crypto path reintroduced behind asset-type detection

## Review section

(populated after implementation per CLAUDE.md workflow item 5)

---

## 2026-05-04 — Data fixes (A1+B1+C1)

**Status:** complete (pending smoke run + PR).
**Spec:** `docs/superpowers/specs/2026-05-04-data-fixes-design.md`
**Plan:** `docs/superpowers/plans/2026-05-04-data-fixes.md`
**Branch:** `feat/data-fixes-2026-05-04`

### Review

- **A1** — synthesis on `run_with_tools(tools=[])` with retry; AMZN report now
  has a real narrative section (no more "Synthesis LLM call failed").
  Token budget bumped 1500 → 2000 for the JSON output.
- **B1** — `edgar.latest_revenue_observations` shared across risk + fundamentals
  + segment tool; ASC 606 issuers (AMZN, MSFT, GOOG) resolve revenue YoY.
  Tag chain priority: ASC 606 standard → legacy `Revenues` → ASC 606 with tax
  → pre-ASC 606 retail.
- **C1** — equity prefetch window 1y; SMA200, 1y vol percentile, drawdown all
  computable for any liquid name. `change_90d_pct` semantics preserved by
  pinning to `iloc[-90]` instead of `iloc[0]`.

### Test results

- 11 task commits + 3 review-fix commits on the branch.
- 118 tests passing on the changed surface; 2 pre-existing failures in
  `test_agents_init.py` and `test_graph.py` (langchain.debug AttributeError on
  Python 3.14 / pydantic V1 incompat — unrelated to this branch).
- 6 collection errors in test_edgar_*.py and test_macro_* due to missing
  `responses` package in this environment (also pre-existing).

### Spawned for follow-up

- Atomic period-matched key_metrics + Liabilities tag fallback (separate
  task, separate spec) — chip already created during brainstorm session.
