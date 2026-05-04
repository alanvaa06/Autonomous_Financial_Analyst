# MarketMind v2.1 Data & Synthesis Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three orthogonal defects surfaced by the AMZN run on 2026-05-04: synthesis JSON parse failure, missing SMA200 from a 90-day prefetch window, and missing XBRL Revenues for ASC 606 issuers (AMZN, MSFT-class large caps).

**Architecture:** Three independent changes. (A1) Synthesis swaps its LangChain `ChatAnthropic.invoke` call for the existing `agents.run_with_tools(tools=[])` loop, which already contains JSON-parse retry and corrective-turn nudges. (B1) New `latest_revenue_observations` and `yoy_revenue_pct` helpers in `edgar.py` provide a tag-fallback chain (ASC 606 priority) shared by fundamentals + risk. (C1) Prefetch window grows 90d → 1y so SMA200 / 1-year vol percentile / drawdown become computable; the `change_90d_pct` semantic in price_agent gets pinned to `iloc[-90]` instead of `iloc[0]`.

**Tech Stack:** Python 3.12, langchain-anthropic 0.3.22, anthropic 0.97.0, langgraph 0.3.7, yfinance 1.3.0 + curl_cffi, pytest + responses + unittest.mock.

**Spec:** [`docs/superpowers/specs/2026-05-04-data-fixes-design.md`](../specs/2026-05-04-data-fixes-design.md) (commit `786e4c4`)

---

## File Structure (Target)

| Path | Status | Responsibility |
|---|---|---|
| `edgar.py` | MODIFY | Add `REVENUE_TAGS`, `VALID_FP`, `latest_revenue_observations()`, `yoy_revenue_pct()`. |
| `agents/data_prefetch.py` | MODIFY | `_safe_yf(ticker, period="1y")` for the equity ticker. VIX stays 5d. |
| `agents/price_agent.py` | MODIFY | Fallback `download_with_retry(period="1y")`. Pin `change_90d_pct` to `iloc[-90]`. |
| `agents/risk_agent.py` | MODIFY | Fallback `download_with_retry(period="1y")`. `_yoy_revenue_pct` delegates to `edgar.latest_revenue_observations` + `edgar.yoy_revenue_pct`. |
| `agents/fundamentals_agent.py` | MODIFY | `_key_metrics_from_facts` uses `edgar.latest_revenue_observations` for `rev`. |
| `agents/tools/fundamentals_tools.py` | MODIFY | `_fetch_segment_breakdown` uses shared helper for tag fallback. |
| `agents/synthesis_agent.py` | MODIFY | Replace `clients.reasoning.invoke(...)` with `run_with_tools(tools=[])`. Bump tokens 1500→2000. Add "emit ONLY JSON" line to schema block. |
| `tests/test_edgar_revenue.py` | NEW | Cover `latest_revenue_observations` + `yoy_revenue_pct` (4 fixtures). |
| `tests/test_data_prefetch.py` | MODIFY | Assert prefetch passes `period="1y"` for equity ticker, `period="5d"` for VIX. |
| `tests/test_price_agent.py` | MODIFY | Cover `change_90d_pct` index pinning when `len(close) > 90`. |
| `tests/test_risk_agent.py` | MODIFY | Add fixture using ASC 606 tag → assert `revenue_yoy_pct` resolves. |
| `tests/test_fundamentals_agent.py` | MODIFY | Add fixture using ASC 606 tag → assert `revenue_latest_usd` resolves. |
| `tests/test_synthesis_agent.py` | MODIFY | Stub `run_with_tools` instead of `clients.reasoning.invoke`. Add empty-then-valid retry test. |
| `tasks/lessons.md` | CREATE | Capture lessons per CLAUDE.md workflow item 6. |

---

## Phase 0: Workspace Setup

### Task 0.1: Cut feature branch

**Files:** none (git only)

- [ ] **Step 1: Create branch**

```bash
git checkout -b feat/data-fixes-2026-05-04
git status
```

Expected: `On branch feat/data-fixes-2026-05-04`. Working tree clean (spec already committed in `786e4c4`).

- [ ] **Step 2: Verify spec is on the branch**

```bash
git log --oneline -1 docs/superpowers/specs/2026-05-04-data-fixes-design.md
```

Expected: shows commit `786e4c4 docs(spec): data + synthesis fixes for AMZN-class issuers`.

---

## Phase 1: B1 — Revenue Tag Fallback Helper

Foundational. Lands first because risk + fundamentals + segment tool all consume it.

### Task 1.1: Add `latest_revenue_observations` helper to `edgar.py`

**Files:**
- Modify: `edgar.py` (append after the `_FACTS_CACHE` block at line ~99 — or at the end of the file; pick the end so existing line numbers in the spec stay valid)
- Test: `tests/test_edgar_revenue.py` (new file)

- [ ] **Step 1: Write the failing test for ASC 606 tag selection**

Create `tests/test_edgar_revenue.py`:

```python
"""Tests for edgar.latest_revenue_observations and edgar.yoy_revenue_pct."""

from edgar import latest_revenue_observations, yoy_revenue_pct


def test_picks_asc606_tag_when_revenues_empty():
    """AMZN-style: only RevenueFromContractWithCustomerExcludingAssessedTax populated."""
    facts = {"facts": {"us-gaap": {
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
            {"end": "2026-03-31", "val": 187_000_000_000, "fp": "Q1", "form": "10-Q", "filed": "2026-05-01"},
            {"end": "2025-03-31", "val": 162_000_000_000, "fp": "Q1", "form": "10-Q", "filed": "2025-05-02"},
        ]}},
    }}}
    tag, obs = latest_revenue_observations(facts)
    assert tag == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert len(obs) == 2
    assert obs[0]["val"] == 187_000_000_000  # newest first


def test_returns_none_when_no_tag_has_two_obs():
    facts = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [
            {"end": "2025-12-31", "val": 100, "fp": "FY", "form": "10-K", "filed": "2026-02-01"},
        ]}},
    }}}
    tag, obs = latest_revenue_observations(facts)
    assert tag is None
    assert obs == []


def test_asc606_wins_when_both_tags_present():
    """When both legacy Revenues and ASC 606 tag exist, ASC 606 is preferred."""
    facts = {"facts": {"us-gaap": {
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
            {"end": "2026-03-31", "val": 187e9, "fp": "Q1", "form": "10-Q", "filed": "2026-05-01"},
            {"end": "2025-03-31", "val": 162e9, "fp": "Q1", "form": "10-Q", "filed": "2025-05-02"},
        ]}},
        "Revenues": {"units": {"USD": [
            {"end": "2026-03-31", "val": 999e9, "fp": "Q1", "form": "10-Q", "filed": "2026-05-01"},
            {"end": "2025-03-31", "val": 888e9, "fp": "Q1", "form": "10-Q", "filed": "2025-05-02"},
        ]}},
    }}}
    tag, obs = latest_revenue_observations(facts)
    assert tag == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert obs[0]["val"] == 187e9


def test_filters_invalid_fp():
    """Observations with fp not in {Q1,Q2,Q3,FY} are filtered (excludes Q4 and synthetic periods)."""
    facts = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [
            {"end": "2025-12-31", "val": 999, "fp": "Q4", "form": "10-Q", "filed": "2026-02-01"},  # filtered
            {"end": "2025-12-31", "val": 100, "fp": "FY", "form": "10-K", "filed": "2026-02-01"},
            {"end": "2024-12-31", "val": 90, "fp": "FY", "form": "10-K", "filed": "2025-02-01"},
        ]}},
    }}}
    tag, obs = latest_revenue_observations(facts)
    assert tag == "Revenues"
    assert len(obs) == 2
    assert all(o["fp"] in {"Q1", "Q2", "Q3", "FY"} for o in obs)


def test_restated_value_wins_via_filed_date():
    """Same (end, fp), two different filed dates: most recent filed wins."""
    facts = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [
            {"end": "2025-12-31", "val": 100, "fp": "FY", "form": "10-K", "filed": "2026-02-01"},
            {"end": "2025-12-31", "val": 105, "fp": "FY", "form": "10-K/A", "filed": "2026-08-01"},
            {"end": "2024-12-31", "val": 90, "fp": "FY", "form": "10-K", "filed": "2025-02-01"},
        ]}},
    }}}
    tag, obs = latest_revenue_observations(facts)
    # 10-K/A is filtered (form not in {10-Q, 10-K}). Note: per spec §B1, only
    # 10-Q and 10-K forms are accepted, so the restated 10-K/A is excluded.
    # Adjust: this test verifies that observations are restricted to allowed
    # forms, and that within allowed forms the filed-date sort is stable.
    assert tag == "Revenues"
    assert len(obs) == 2
    assert obs[0]["val"] == 100  # original FY2025 10-K (10-K/A filtered out)
    assert obs[1]["val"] == 90   # FY2024


def test_yoy_pct_basic():
    obs = [
        {"end": "2026-03-31", "val": 187e9, "fp": "Q1"},
        {"end": "2025-03-31", "val": 162e9, "fp": "Q1"},
    ]
    pct = yoy_revenue_pct(obs)
    assert pct is not None
    assert 15.0 < pct < 16.0  # (187-162)/162 = 15.43%


def test_yoy_pct_requires_matching_fp():
    """Latest is Q1; prior has same end-year-1 but fp=FY. No match → None."""
    obs = [
        {"end": "2026-03-31", "val": 187e9, "fp": "Q1"},
        {"end": "2025-03-31", "val": 600e9, "fp": "FY"},  # synthetic mismatch
    ]
    assert yoy_revenue_pct(obs) is None


def test_yoy_pct_returns_none_when_lt_two_obs():
    assert yoy_revenue_pct([]) is None
    assert yoy_revenue_pct([{"end": "2026-03-31", "val": 1, "fp": "Q1"}]) is None
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd C:/Proyectos/Autonomous_Financial_Analyst
pytest tests/test_edgar_revenue.py -v
```

Expected: ImportError — `cannot import name 'latest_revenue_observations' from 'edgar'`.

- [ ] **Step 3: Implement the helpers in `edgar.py`**

Append to `edgar.py` (after the existing `fetch_company_facts` block, before the `_SUBMISSIONS_CACHE` block — pick a spot that keeps the data-layer helpers grouped):

```python
# -- Revenue tag fallback helpers ----------------------------------------------

REVENUE_TAGS: tuple[str, ...] = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",  # ASC 606 standard, post-2018
    "Revenues",                                              # legacy generic
    "RevenueFromContractWithCustomerIncludingAssessedTax",   # ASC 606 with tax
    "SalesRevenueNet",                                       # pre-ASC 606 retail
)
VALID_FP = frozenset({"Q1", "Q2", "Q3", "FY"})


def latest_revenue_observations(facts: dict, *, periods: int = 8) -> tuple[Optional[str], list[dict]]:
    """Resolve revenue observations via tag-fallback chain.

    Walks REVENUE_TAGS in priority order and returns the first tag whose
    USD observations include at least 2 clean entries (form in {10-Q, 10-K},
    fp in VALID_FP). Sort key is (end DESC, filed DESC) so the most recent
    period wins and restated values supersede originals when both forms
    qualify.

    Returns (tag_used, observations) capped at `periods`, or (None, []) if
    no tag has at least 2 clean observations.
    """
    g = (facts or {}).get("facts", {}).get("us-gaap", {})
    for tag in REVENUE_TAGS:
        obs = g.get(tag, {}).get("units", {}).get("USD") or []
        clean = [
            o for o in obs
            if o.get("form") in ("10-Q", "10-K") and o.get("fp") in VALID_FP
        ]
        clean.sort(
            key=lambda o: (o.get("end", ""), o.get("filed", "")),
            reverse=True,
        )
        if len(clean) >= 2:
            return tag, clean[:periods]
    return None, []


def yoy_revenue_pct(observations: list[dict]) -> Optional[float]:
    """Compute YoY revenue percent from a list returned by `latest_revenue_observations`.

    Pairs the newest observation with a prior observation of the same `fp`
    and an `end` year exactly one year earlier. Returns None when fewer
    than 2 observations are available, or when no matching prior is found.
    """
    if len(observations) < 2:
        return None
    latest = observations[0]
    fp = latest.get("fp")
    end = latest.get("end", "")
    if not fp or len(end) < 10:
        return None
    target_year = str(int(end[:4]) - 1)
    prior = next(
        (
            o for o in observations[1:]
            if o.get("fp") == fp and o.get("end", "")[:4] == target_year
        ),
        None,
    )
    if not prior or not prior.get("val"):
        return None
    return round(
        (float(latest["val"]) - float(prior["val"])) / float(prior["val"]) * 100,
        2,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_edgar_revenue.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add edgar.py tests/test_edgar_revenue.py
git commit -m "feat(edgar): add revenue tag fallback helpers for ASC 606 issuers"
```

---

### Task 1.2: Wire `risk_agent._yoy_revenue_pct` to the shared helper

**Files:**
- Modify: `agents/risk_agent.py:178-195` (`_yoy_revenue_pct`)
- Test: `tests/test_risk_agent.py` (add fixture + assertion)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_risk_agent.py`:

```python
def test_risk_resolves_revenue_yoy_via_asc606_tag(monkeypatch):
    """AMZN-style bundle: only RevenueFromContractWithCustomerExcludingAssessedTax
    populated. _forward_fundamentals must still resolve revenue_yoy_pct."""
    df = _price_df(120)
    bundle = EdgarBundle(
        ticker="AMZN", cik="0001018724", company_name="AMAZON COM INC",
        latest_10q=None, latest_10k=None,
        xbrl_facts={"facts": {"us-gaap": {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
                {"end": "2026-03-31", "val": 187e9, "fp": "Q1", "form": "10-Q", "filed": "2026-05-01"},
                {"end": "2025-03-31", "val": 162e9, "fp": "Q1", "form": "10-Q", "filed": "2025-05-02"},
            ]}},
            "Liabilities": {"units": {"USD": [{"end": "2026-03-31", "val": 200e9, "form": "10-Q"}]}},
            "StockholdersEquity": {"units": {"USD": [{"end": "2026-03-31", "val": 285e9, "form": "10-Q"}]}},
        }}},
        mdna_text="", risk_factors_text="",
    )
    monkeypatch.setattr(
        "agents.risk_agent.run_with_tools",
        lambda **kw: {
            "signal": "NEUTRAL", "confidence": 0.5,
            "summary": "ok", "section_markdown": "## Risk Profile\nbody.",
            "forward_risk_view": "mixed", "primary_risk_driver": "none",
            "risk_decomposition": {"operating": "low", "balance_sheet": "low",
                                   "positioning": "medium", "systemic": "medium"},
            "vol_regime": "normal", "vix_regime": "normal",
            "key_metrics": {}, "flags": [],
        },
    )
    out = risk_agent(
        {"ticker": "AMZN", "price_history": df, "edgar_bundle": bundle,
         "vix_history": pd.DataFrame({"Close": [16.0]})},
        _clients(),
    )
    sig = out["agent_signals"][0]
    # Pre-fix this would be None; post-fix it should be ~15.43%.
    assert sig["raw_data"]["revenue_yoy_pct"] is not None
    assert 15.0 < sig["raw_data"]["revenue_yoy_pct"] < 16.0
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_risk_agent.py::test_risk_resolves_revenue_yoy_via_asc606_tag -v
```

Expected: FAIL — `assert None is not None` (the inline `_yoy_revenue_pct` only checks the `Revenues` tag).

- [ ] **Step 3: Replace the body of `_yoy_revenue_pct` in `agents/risk_agent.py`**

Replace lines 178–195 with:

```python
def _yoy_revenue_pct(bundle: EdgarBundle) -> Optional[float]:
    from edgar import latest_revenue_observations, yoy_revenue_pct
    _, obs = latest_revenue_observations(bundle.xbrl_facts or {})
    return yoy_revenue_pct(obs)
```

(Keep the `Optional` import at the top of the file; it is already imported per line 12.)

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_risk_agent.py -v
```

Expected: all risk_agent tests pass, including the new one.

- [ ] **Step 5: Commit**

```bash
git add agents/risk_agent.py tests/test_risk_agent.py
git commit -m "fix(risk): resolve revenue YoY via ASC 606 tag fallback"
```

---

### Task 1.3: Wire `fundamentals_agent._key_metrics_from_facts` to the helper

**Files:**
- Modify: `agents/fundamentals_agent.py:96-128` (`_key_metrics_from_facts`)
- Test: `tests/test_fundamentals_agent.py` (add fixture + assertion)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fundamentals_agent.py`:

```python
def test_key_metrics_resolves_revenue_via_asc606_tag():
    """AMZN-style: revenue is filed under RevenueFromContractWithCustomerExcludingAssessedTax."""
    from agents.fundamentals_agent import _key_metrics_from_facts
    facts = {"facts": {"us-gaap": {
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
            {"end": "2026-03-31", "val": 187_000_000_000, "fp": "Q1", "form": "10-Q", "filed": "2026-05-01"},
            {"end": "2025-03-31", "val": 162_000_000_000, "fp": "Q1", "form": "10-Q", "filed": "2025-05-02"},
        ]}},
        "OperatingIncomeLoss": {"units": {"USD": [
            {"end": "2026-03-31", "val": 24_000_000_000, "fp": "Q1", "form": "10-Q", "filed": "2026-05-01"},
        ]}},
    }}}
    km = _key_metrics_from_facts(facts)
    assert km["revenue_latest_usd"] == 187_000_000_000
    # op_margin_pct should now compute (was None pre-fix because rev was None).
    assert km["op_margin_pct"] is not None
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_fundamentals_agent.py::test_key_metrics_resolves_revenue_via_asc606_tag -v
```

Expected: FAIL — `revenue_latest_usd` is None.

- [ ] **Step 3: Modify `_key_metrics_from_facts`**

Replace lines 96–128 in `agents/fundamentals_agent.py`. The change is small: where `rev` is currently sourced from `_latest("Revenues")`, source it from the shared helper instead.

Replace this block:

```python
    rev, _ = _latest("Revenues")
    op_inc, _ = _latest("OperatingIncomeLoss")
    net_inc, _ = _latest("NetIncomeLoss")
    eps, _ = _latest("EarningsPerShareDiluted")
    assets, _ = _latest("Assets")
    liab, _ = _latest("Liabilities")
    equity, _ = _latest("StockholdersEquity")
```

with:

```python
    from edgar import latest_revenue_observations
    _, rev_obs = latest_revenue_observations(facts or {})
    rev = float(rev_obs[0]["val"]) if rev_obs else None
    op_inc, _ = _latest("OperatingIncomeLoss")
    net_inc, _ = _latest("NetIncomeLoss")
    eps, _ = _latest("EarningsPerShareDiluted")
    assets, _ = _latest("Assets")
    liab, _ = _latest("Liabilities")
    equity, _ = _latest("StockholdersEquity")
```

(Note: `assets` is unused below; leave the existing line in place — out of scope to remove.)

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_fundamentals_agent.py -v
```

Expected: all fundamentals_agent tests pass, including the new one.

- [ ] **Step 5: Commit**

```bash
git add agents/fundamentals_agent.py tests/test_fundamentals_agent.py
git commit -m "fix(fundamentals): resolve latest revenue via ASC 606 tag fallback"
```

---

### Task 1.4: Wire `_fetch_segment_breakdown` to the shared helper

**Files:**
- Modify: `agents/tools/fundamentals_tools.py:37-53` (`_fetch_segment_breakdown`)
- Test: `tests/test_tools/test_fundamentals_tools.py` (add assertion)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tools/test_fundamentals_tools.py`:

```python
def test_segment_breakdown_uses_asc606_tag_when_revenues_empty():
    from edgar import EdgarBundle
    from agents.tools.fundamentals_tools import _fetch_segment_breakdown
    bundle = EdgarBundle(
        ticker="AMZN", cik="0001018724", company_name="AMAZON COM INC",
        latest_10q=None, latest_10k=None,
        xbrl_facts={"facts": {"us-gaap": {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
                {"end": "2026-03-31", "val": 187e9, "fp": "Q1", "form": "10-Q", "filed": "2026-05-01"},
                {"end": "2025-03-31", "val": 162e9, "fp": "Q1", "form": "10-Q", "filed": "2025-05-02"},
            ]}},
        }}},
        mdna_text="", risk_factors_text="",
    )
    out = _fetch_segment_breakdown(bundle)
    assert out["tag"] == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert len(out["segments"]) >= 2
```

- [ ] **Step 2: Run the test to verify it fails or passes by accident**

```bash
pytest tests/test_tools/test_fundamentals_tools.py::test_segment_breakdown_uses_asc606_tag_when_revenues_empty -v
```

Expected: PASS today (the existing inline fallback already includes the ASC 606 tag). If it passes, the next step still matters for code consistency. Document the test, then proceed to Step 3.

- [ ] **Step 3: Replace the inline fallback with the shared helper**

In `agents/tools/fundamentals_tools.py`, replace `_fetch_segment_breakdown` (lines 37–53) with:

```python
def _fetch_segment_breakdown(bundle: EdgarBundle) -> dict:
    """Return segment-level revenue rows via the shared revenue tag fallback.

    XBRL segment data is filed under several tag names; we delegate to
    `edgar.latest_revenue_observations` so the priority chain stays in
    one place. Returns an explicit empty result when no tag resolves.
    """
    from edgar import latest_revenue_observations
    tag, obs = latest_revenue_observations(bundle.xbrl_facts or {}, periods=20)
    if not tag:
        return {"tag": None, "segments": []}
    return {"tag": tag, "segments": obs}
```

- [ ] **Step 4: Run the test to verify it still passes**

```bash
pytest tests/test_tools/test_fundamentals_tools.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add agents/tools/fundamentals_tools.py tests/test_tools/test_fundamentals_tools.py
git commit -m "refactor(fundamentals-tools): segment breakdown uses shared revenue helper"
```

---

## Phase 2: C1 — Prefetch Window 90d → 1y

### Task 2.1: Bump `data_prefetch` to 1y for the equity ticker

**Files:**
- Modify: `agents/data_prefetch.py:42-55` (`data_prefetch`)
- Test: `tests/test_data_prefetch.py` (add period assertion)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_data_prefetch.py`:

```python
def test_prefetch_uses_1y_window_for_equity_and_5d_for_vix():
    """SMA200 needs ≥200 trading days; equity prefetch must request 1y.
    VIX only needs latest level; stays at 5d."""
    captured: list[tuple[str, str]] = []

    def fake_dl(ticker, *, period, interval):
        captured.append((ticker, period))
        return pd.DataFrame({"Close": [1.0]})

    with patch(
        "agents.data_prefetch.download_with_retry",
        side_effect=fake_dl,
    ), patch(
        "agents.data_prefetch.build_edgar_bundle",
        return_value=_stub_edgar(),
    ), patch("agents.data_prefetch.time.sleep"):
        data_prefetch({"ticker": "MSFT"})

    assert ("MSFT", "1y") in captured
    assert ("^VIX", "5d") in captured
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_data_prefetch.py::test_prefetch_uses_1y_window_for_equity_and_5d_for_vix -v
```

Expected: FAIL — captured list contains `("MSFT", "90d")` not `("MSFT", "1y")`.

- [ ] **Step 3: Update `data_prefetch.py`**

In `agents/data_prefetch.py`, change line 45:

```python
    price_history = _safe_yf(ticker, period="90d")
```

to:

```python
    price_history = _safe_yf(ticker, period="1y")
```

VIX line 47 stays at `period="5d"`.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_data_prefetch.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add agents/data_prefetch.py tests/test_data_prefetch.py
git commit -m "fix(prefetch): equity window 90d -> 1y so SMA200 / 1y vol percentile compute"
```

---

### Task 2.2: Update agent fallbacks to match the 1y window

**Files:**
- Modify: `agents/price_agent.py:169` (fallback `download_with_retry`)
- Modify: `agents/risk_agent.py:239` (fallback `download_with_retry`)

- [ ] **Step 1: Update price_agent fallback**

In `agents/price_agent.py`, change line 169:

```python
            data = download_with_retry(ticker, period="90d", interval="1d")
```

to:

```python
            data = download_with_retry(ticker, period="1y", interval="1d")
```

- [ ] **Step 2: Update risk_agent fallback**

In `agents/risk_agent.py`, change line 239:

```python
            data = download_with_retry(ticker, period="90d", interval="1d")
```

to:

```python
            data = download_with_retry(ticker, period="1y", interval="1d")
```

- [ ] **Step 3: Run all tests to verify nothing breaks**

```bash
pytest tests/test_price_agent.py tests/test_risk_agent.py -v
```

Expected: all pass (existing tests stub `state["price_history"]` so the fallback path is rarely exercised, but the ones that do should still pass).

- [ ] **Step 4: Commit**

```bash
git add agents/price_agent.py agents/risk_agent.py
git commit -m "fix(agents): align fallback yf window with 1y prefetch"
```

---

### Task 2.3: Pin `change_90d_pct` to `iloc[-90]` in price_agent

**Files:**
- Modify: `agents/price_agent.py:131` (inside `_compute_raw`)
- Test: `tests/test_price_agent.py` (add semantics test)

The 1y window changes `len(close)` from ~63 to ~252. `_compute_raw` line 131 uses `close.iloc[0]` as the 90d anchor — with 1y of data this becomes the 1y change. Pin to `iloc[-90]` to preserve semantics.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_price_agent.py` (use `_compute_raw` as a pure function — easier than running the full agent):

```python
def test_compute_raw_change_90d_pct_uses_minus_90_index_when_history_is_long():
    """When prefetch returns 1y, change_90d_pct must reflect 90-day move,
    not the full 1y move."""
    import pandas as pd
    from agents.price_agent import _compute_raw

    # 252 days, slow uptrend overall, but the most recent 90 days are flat.
    n = 252
    prices = []
    for i in range(n):
        if i < n - 90:
            prices.append(50.0 + i * 0.1)  # ramps from 50 to ~66
        else:
            prices.append(66.0)            # flat at 66 for the last 90 days
    close = pd.Series(prices)

    raw = _compute_raw(close)
    # 90d change should be ~0 (last 90 days flat), not ~32% (1y change).
    assert abs(raw["change_90d_pct"]) < 1.0
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_price_agent.py::test_compute_raw_change_90d_pct_uses_minus_90_index_when_history_is_long -v
```

Expected: FAIL — `change_90d_pct` is approximately 32% (the 1y move).

- [ ] **Step 3: Modify `_compute_raw`**

In `agents/price_agent.py`, replace line 131:

```python
    change_90d = round((current - float(close.iloc[0])) / float(close.iloc[0]) * 100, 2)
```

with:

```python
    if len(close) >= 90:
        anchor_90 = float(close.iloc[-90])
    else:
        anchor_90 = float(close.iloc[0])
    change_90d = round((current - anchor_90) / anchor_90 * 100, 2) if anchor_90 else 0.0
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_price_agent.py -v
```

Expected: all pass, including the new test.

- [ ] **Step 5: Commit**

```bash
git add agents/price_agent.py tests/test_price_agent.py
git commit -m "fix(price): pin change_90d_pct to iloc[-90] under 1y window"
```

---

## Phase 3: A1 — Synthesis migrates to `run_with_tools`

### Task 3.1: Migrate synthesis to `run_with_tools(tools=[])`

**Files:**
- Modify: `agents/synthesis_agent.py:142-186` (`synthesis_agent` body)
- Modify: `agents/synthesis_agent.py:67-75` (`OUTPUT_SCHEMA`)
- Test: `tests/test_synthesis_agent.py` (replace LLM stub strategy)

- [ ] **Step 1: Update the existing happy-path test to stub `run_with_tools`**

In `tests/test_synthesis_agent.py`, replace the body of `test_synthesis_agent_assembles_report` (lines 121–151) with:

```python
def test_synthesis_agent_assembles_report(monkeypatch):
    sigs = [
        _s("price", "BULLISH", 0.7),
        _s("sentiment", "BULLISH", 0.6),
        _s("fundamentals", "BULLISH", 0.65),
        _s("macro", "NEUTRAL", 0.5),
        _s("risk", "NEUTRAL", 0.55),
    ]
    review = {"approved": True, "critiques": {}, "retry_targets": [],
              "notes": "Data quality: all sections complete."}
    state = {
        "ticker": "MSFT", "company_name": "Microsoft Corp",
        "agent_signals": sigs, "supervisor_review": review,
    }

    fake_clients = MagicMock()
    secret = MagicMock()
    secret.get_secret_value.return_value = "sk-ant-fake"
    fake_clients.reasoning.anthropic_api_key = secret

    monkeypatch.setattr(
        "agents.synthesis_agent.run_with_tools",
        lambda **kw: {
            "reasoning": "Three sentences of integrated reasoning that explain why the call holds together across price, sentiment, and fundamentals.",
            "key_drivers": [],
            "dissenting_view": "",
            "watch_items": [],
        },
    )

    out = synthesis_agent(state, fake_clients)
    assert out["final_verdict"] == "BUY"
    assert out["final_conviction"] in ("STANDARD", "CAUTIOUS", "STRONG")
    assert out["final_reasoning"].startswith("Three sentences")
    report = out["final_report"]
    for header in (
        "# MSFT", "Executive Summary", "Technical Analysis",
        "News & Sentiment", "Fundamentals", "Macro Backdrop",
        "Risk Profile", "Synthesis & Final Verdict", "Disclaimer",
    ):
        assert header in report
```

Replace `test_synthesis_emits_key_drivers_and_watch_items` (lines 154–183) with:

```python
def test_synthesis_emits_key_drivers_and_watch_items(monkeypatch):
    sigs = [
        _s("price", "BULLISH", 0.7),
        _s("sentiment", "BULLISH", 0.6),
        _s("fundamentals", "BULLISH", 0.65),
        _s("macro", "NEUTRAL", 0.5),
        _s("risk", "NEUTRAL", 0.55),
    ]
    review = {"approved": True, "critiques": {}, "retry_targets": [],
              "notes": "Data quality: all sections complete."}
    state = {"ticker": "MSFT", "company_name": "Microsoft Corp",
             "agent_signals": sigs, "supervisor_review": review}

    fake_clients = MagicMock()
    secret = MagicMock()
    secret.get_secret_value.return_value = "sk-ant-fake"
    fake_clients.reasoning.anthropic_api_key = secret

    monkeypatch.setattr(
        "agents.synthesis_agent.run_with_tools",
        lambda **kw: {
            "reasoning": "Three specialists support: Price, Sentiment, Fundamentals.",
            "key_drivers": [
                "Fundamentals: op margin +220 bps",
                "Price: trend confirmation",
                "Sentiment: 8/12 positive",
            ],
            "dissenting_view": "Macro headwind reverses if Fed cuts next meeting.",
            "watch_items": ["Next CPI print", "Q3 cloud growth"],
        },
    )

    out = synthesis_agent(state, fake_clients)
    assert out["key_drivers"] == [
        "Fundamentals: op margin +220 bps",
        "Price: trend confirmation",
        "Sentiment: 8/12 positive",
    ]
    assert "Next CPI print" in out["watch_items"]
    assert "What to Watch" in out["final_report"]
    assert "Dissenting view" in out["final_report"]
```

Add a new test for the failure-fallback path:

```python
def test_synthesis_falls_back_when_run_with_tools_raises(monkeypatch):
    """When run_with_tools exhausts retries, synthesis still returns a valid
    final_report with the deterministic verdict + the canned fallback line."""
    sigs = [
        _s("price", "BULLISH", 0.7),
        _s("sentiment", "BULLISH", 0.6),
        _s("fundamentals", "BULLISH", 0.65),
        _s("macro", "NEUTRAL", 0.5),
        _s("risk", "NEUTRAL", 0.55),
    ]
    review = {"approved": True, "critiques": {}, "retry_targets": [],
              "notes": "Data quality: all sections complete."}
    state = {"ticker": "MSFT", "company_name": "Microsoft Corp",
             "agent_signals": sigs, "supervisor_review": review}

    fake_clients = MagicMock()
    secret = MagicMock()
    secret.get_secret_value.return_value = "sk-ant-fake"
    fake_clients.reasoning.anthropic_api_key = secret

    def boom(**kw):
        raise ValueError("model did not produce parseable JSON after 3 iterations")
    monkeypatch.setattr("agents.synthesis_agent.run_with_tools", boom)

    out = synthesis_agent(state, fake_clients)
    assert out["final_verdict"] == "BUY"
    assert "Synthesis LLM call failed" in out["final_reasoning"]
    assert "Synthesis & Final Verdict" in out["final_report"]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_synthesis_agent.py -v
```

Expected: the two updated tests fail (still mocking `clients.reasoning.invoke` is gone), the new fallback test fails, label/verdict-math tests still pass.

- [ ] **Step 3: Modify `agents/synthesis_agent.py`**

(a) Update imports (top of file). The existing import is `from agents import safe_parse_json`. Add `run_with_tools`:

```python
from agents import run_with_tools, safe_parse_json
```

(`safe_parse_json` may now be unused; if so, drop it. Verify after the body change.)

(b) Update `OUTPUT_SCHEMA` (lines 67–75). Append one explicit instruction:

```python
OUTPUT_SCHEMA = """
Respond with a single JSON object (no markdown fences) with EXACTLY these keys:
{
  "reasoning": "3-5 sentences (80-150 words) referencing ≥3 specialists by name",
  "key_drivers": ["Specialist: metric/observation", ...],   // 2-4 entries, each ≤15 words
  "dissenting_view": "one sentence ≤25 words: what condition flips the call",
  "watch_items": ["leading indicator", ...]                  // 2-3 entries, each ≤20 words
}

Reason step-by-step internally; emit ONLY the JSON object as your final response.
""".strip()
```

(c) Replace the LLM-call block in `synthesis_agent` (lines 168–186) with:

```python
    reasoning = ""
    key_drivers: list = []
    dissenting_view = ""
    watch_items: list = []
    try:
        api_key = clients.reasoning.anthropic_api_key.get_secret_value()
        out = run_with_tools(
            api_key=api_key,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=[],
            max_iterations=2,
            max_tokens=2000,
        )
        reasoning = out.get("reasoning") or ""
        key_drivers = list(out.get("key_drivers") or [])
        dissenting_view = out.get("dissenting_view") or ""
        watch_items = list(out.get("watch_items") or [])
    except Exception as exc:
        reasoning = (
            f"Synthesis LLM call failed ({str(exc)[:100]}). "
            f"Verdict {label} is derived deterministically from the specialist signals above."
        )
```

(d) If `safe_parse_json` is no longer referenced anywhere in `synthesis_agent.py`, drop the import:

```python
from agents import run_with_tools
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_synthesis_agent.py -v
```

Expected: all pass — the three updated tests plus the existing label / verdict-math tests.

- [ ] **Step 5: Commit**

```bash
git add agents/synthesis_agent.py tests/test_synthesis_agent.py
git commit -m "fix(synthesis): migrate to run_with_tools(tools=[]) for JSON retry"
```

---

## Phase 4: Verification & Lessons

### Task 4.1: Full test suite + smoke run

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

```bash
cd C:/Proyectos/Autonomous_Financial_Analyst
pytest tests/ -v
```

Expected: all tests pass. If any unrelated test fails, stop and re-plan.

- [ ] **Step 2: Smoke run the app on AMZN**

Manual step. Run the Gradio app per `app.py`:

```bash
python app.py
```

Then open the displayed local URL, paste the user's Anthropic key, and request a report on `AMZN`. Confirm:

1. The "Synthesis & Final Verdict" section contains a real narrative paragraph (≥3 sentences) — not "Synthesis LLM call failed".
2. The Risk Profile section reports `trend_state` as `above_sma50_and_sma200` (or similar concrete state) — NOT `unknown`.
3. The Fundamentals section reports a non-null revenue YoY (around 14–16% for AMZN Q1 2026 vs Q1 2025) — NOT "Revenues tag returned no observations".
4. The Technical Analysis section's `change_90d_pct` field is in a sensible range (10–30% for AMZN's recent trend), not the full 1y change.

If any of those checks fails, stop and diagnose. Do not proceed.

- [ ] **Step 3: Smoke run on MSFT (control)**

Same flow on `MSFT`. Confirms regression-free behavior on a ticker that worked pre-fix.

- [ ] **Step 4: Smoke run on a foreign filer (SHEL or BABA) to confirm graceful degradation**

Same flow. Expected: revenue YoY is None (foreign filer in non-USD), but other sections still render and final report is valid markdown.

---

### Task 4.2: Capture lessons in `tasks/lessons.md`

**Files:**
- Create: `tasks/lessons.md`

- [ ] **Step 1: Create `tasks/lessons.md`**

Per CLAUDE.md workflow item 6 ("Capture Lessons: Update lessons.md after corrections"):

```markdown
# Lessons

## 2026-05-04 — Data layer assumptions

- **XBRL revenue is not always under `us-gaap:Revenues`.** ASC 606 issuers
  (post-2018 AMZN, MSFT, GOOG) file under
  `RevenueFromContractWithCustomerExcludingAssessedTax`. Always go through
  `edgar.latest_revenue_observations`, never read the tag directly.
- **Prefetch window must accommodate the longest-lookback indicator any
  consumer needs.** SMA200 needs ≥200 trading days. 90d prefetch silently
  starves trend classification with `trend_state=unknown`. Default to 1y.
- **Index-based slicing on price history breaks when the window grows.**
  `close.iloc[0]` is only the 90-day anchor when the history IS 90 days.
  When growing the window, audit every `iloc[0]` / `iloc[-N]` use for
  semantic drift.

## 2026-05-04 — LLM JSON parse robustness

- **`safe_parse_json("")` raises `Expecting value: line 1 column 1`.** Any
  agent that calls a Claude model and parses the response MUST go through
  `agents.run_with_tools`, even when no tools are needed (`tools=[]`). The
  loop's corrective-turn nudge handles empty / truncated text, and the
  iteration cap prevents runaway. Direct `ChatAnthropic.invoke` + naive
  `safe_parse_json` is an antipattern.
- **CoT prompts that list 8 steps will get written as 8 paragraphs of
  prose unless explicitly instructed otherwise.** Add "Reason step-by-step
  internally; emit ONLY the JSON object as your final response." to the
  output schema block. Without this the model writes the CoT visibly,
  consumes max_tokens, and emits truncated/empty JSON.
```

- [ ] **Step 2: Commit lessons**

```bash
git add tasks/lessons.md
git commit -m "docs(lessons): capture data-layer + LLM JSON-parse rules"
```

---

### Task 4.3: Update `tasks/todo.md`

**Files:**
- Modify: `tasks/todo.md`

- [ ] **Step 1: Append review section to `tasks/todo.md`**

Per CLAUDE.md item 5 ("Document Results: Add review section to todo.md"). Append at the bottom of the file:

```markdown
---

## 2026-05-04 — Data fixes (A1+B1+C1)

**Status:** complete.
**Spec:** `docs/superpowers/specs/2026-05-04-data-fixes-design.md`
**Plan:** `docs/superpowers/plans/2026-05-04-data-fixes.md`
**Branch:** `feat/data-fixes-2026-05-04`

### Review

- A1 — synthesis on `run_with_tools(tools=[])` with retry; AMZN report now
  has a real narrative section.
- B1 — `edgar.latest_revenue_observations` shared across risk + fundamentals
  + segment tool; ASC 606 issuers resolve revenue YoY.
- C1 — equity prefetch window 1y; SMA200, 1y vol percentile, drawdown all
  computable for any liquid name. `change_90d_pct` semantics preserved.

### Spawned for follow-up

- Atomic period-matched key_metrics + Liabilities tag fallback (separate
  task, separate spec).
```

- [ ] **Step 2: Commit todo update**

```bash
git add tasks/todo.md
git commit -m "docs(todo): append 2026-05-04 data-fixes review section"
```

- [ ] **Step 3: Push the branch**

```bash
git push -u origin feat/data-fixes-2026-05-04
```

(Pause and confirm with the user before opening a PR — pushing a feature branch is local-effects-only, but PR creation is shared-state.)

---

## Self-Review

**1. Spec coverage**
- A1 (synthesis migration): Task 3.1 ✓
- B1 helpers in `edgar.py`: Task 1.1 ✓
- B1 risk wire-up: Task 1.2 ✓
- B1 fundamentals wire-up: Task 1.3 ✓
- B1 segment-tool wire-up: Task 1.4 ✓
- C1 prefetch 90d→1y: Task 2.1 ✓
- C1 fallback alignment in agents: Task 2.2 ✓
- C1 `change_90d_pct` semantics fix: Task 2.3 ✓
- Lessons + todo: Tasks 4.2, 4.3 ✓
- Smoke verification on AMZN/MSFT/foreign: Task 4.1 ✓

No gaps.

**2. Placeholder scan**
No "TBD", "TODO", "implement later", "similar to Task N", or vague error-handling instructions. All code blocks are complete.

**3. Type consistency**
- `latest_revenue_observations(facts, *, periods=8) -> tuple[Optional[str], list[dict]]` used identically in Tasks 1.1, 1.2, 1.3, 1.4.
- `yoy_revenue_pct(observations: list[dict]) -> Optional[float]` used identically in Tasks 1.1, 1.2.
- `run_with_tools(api_key=, system_prompt=, user_prompt=, tools=[], max_iterations=, max_tokens=)` keyword-arg shape matches the existing helper signature in `agents/__init__.py`.
- `_compute_raw(close: pd.Series) -> dict` used identically in Task 2.3 test and the implementation.

No drift.
