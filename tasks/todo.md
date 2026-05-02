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
