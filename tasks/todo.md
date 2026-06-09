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

---

## 2026-06-09 — Security remediation (dependency bumps + prompt hardening)

**Source:** `docs/security/2026-06-09-vulnerability-scan.md`
**Branch:** `claude/repo-vulnerability-scan-jdmvuv`

### Plan

- [x] D1 — Bump `langchain-core` 0.3.79 → 0.3.86 (CVE-2025-65106, CVE-2025-68664, CVE-2026-40087, CVE-2026-44843)
- [x] D2 — Bump `langgraph` 0.3.7 → 1.0.10+ (CVE-2026-28277; pulls langgraph-checkpoint ≥3 fixing CVE-2025-64439/CVE-2026-27794)
- [x] D3 — Bump `gradio` to `>=6.7,<7.0` (PYSEC-2026-63/64/65/66); fix version-coupled gradio_client monkey-patch in app.py
- [x] P1 — Shared `sanitize_external_text()` helper in `agents/__init__.py` (strip md images/links/fences from untrusted text)
- [x] P2 — Sentiment: sanitize Tavily titles/snippets, delimit external block, untrusted-data guardrail in system prompt
- [x] P3 — Sentiment tools: sanitize `_tav_search` results
- [x] P4 — Fundamentals: delimit + guardrail for MD&A / Risk Factors excerpts
- [x] P5 — Output side: strip markdown images from report markdown before rendering in app.py
- [x] V1 — pytest green (vs baseline), `pip-audit` clean for bumped packages, app imports under gradio 6

### Review

- Dependency bumps required moving the whole langchain stack to 1.x:
  `langgraph` ≥1.0.10 transitively needs `langchain-core` ≥1.0 (via
  `langgraph-prebuilt`), so a 0.3-line-only fix was impossible. Final set:
  langchain-core 1.4.2, langchain-anthropic 1.4.4, langgraph 1.2.4,
  gradio >=6.7,<7.0 (resolves 6.17.3), pillow >=12.2; anthropic 0.97.0 unchanged.
- Gradio 6 changes: `css=` moved from Blocks constructor to `launch()`;
  gradio_client monkey-patch now guarded with getattr (helper still present
  in gradio_client 2.5.0, patch still applies).
- Prompt hardening: shared `sanitize_external_text()` strips md images/links/
  fences from Tavily titles/snippets (agent + tools) and EDGAR MD&A/Risk
  Factors; external blocks wrapped in <external_data> tags with an
  EXTERNAL_DATA_GUARDRAIL line in sentiment/fundamentals system prompts;
  `strip_markdown_images()` applied to interim + final report in app.py.

### Test results

- Baseline before changes: 138 passed. After all changes: 138 passed.
- `pip-audit -r requirements.txt` (full transitive resolution): no known
  vulnerabilities.
- `import app` clean under gradio 6.17.3 with `-W error::UserWarning`
  (version-check warning ignored); ChatAnthropic `anthropic_api_key`
  secret access verified on langchain-anthropic 1.4.4.
