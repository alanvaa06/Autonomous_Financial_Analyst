# MarketMind v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current single-agent chat application with a multi-agent parallel equity analysis pipeline that takes a stock ticker and produces a sectioned investment report with verdict, conviction, and confidence.

**Architecture:** LangGraph fan-out/fan-in: orchestrator validates the ticker, five specialists (Price, Sentiment, Fundamentals, Macro, Risk) run in parallel in one superstep, a Supervisor performs light QA with up to one retry round, then Synthesis assembles the final markdown report. UI streams updates per agent.

**Tech Stack:** Python 3.12, LangGraph 0.3, langchain-anthropic (Claude Sonnet 4.6 + Haiku 4.5), Tavily, yfinance, SEC EDGAR (raw `requests` + BeautifulSoup), FRED + Fear&Greed (raw `requests`), Gradio 4.44.1, pytest + responses for tests.

**Spec:** [`docs/superpowers/specs/2026-05-01-marketmind-v2-design.md`](../specs/2026-05-01-marketmind-v2-design.md)

---

## File Structure (Target)

| Path | Status | Responsibility |
|---|---|---|
| `state.py` | NEW | `MarketMindState`, `AgentSignal`, `SupervisorReview`, reducer types |
| `edgar.py` | NEW | SEC EDGAR client: CIK resolve, 10-Q/10-K fetch, MD&A extract, in-process LRU cache |
| `agents/__init__.py` | NEW | `build_llm_clients(anthropic_key)` factory + `safe_parse_json()` + concurrency semaphore |
| `agents/orchestrator.py` | NEW | Validate ticker, resolve CIK, classify, init state |
| `agents/price_agent.py` | NEW | yfinance OHLC → RSI/MACD/Bollinger → LLM section |
| `agents/sentiment_agent.py` | NEW | Tavily → Haiku per-article → Sonnet rollup section |
| `agents/fundamentals_agent.py` | NEW | EDGAR XBRL + MD&A → Sonnet section |
| `agents/macro_agent.py` | NEW | FRED + F&G → Sonnet section |
| `agents/risk_agent.py` | NEW | yfinance returns + ^VIX → Sonnet section |
| `agents/supervisor_agent.py` | NEW | Deterministic checks → `SupervisorReview` |
| `agents/synthesis_agent.py` | NEW | Verdict math + Sonnet reasoning + final report assembly |
| `graph.py` | NEW | `build_graph(llm_clients, ...)` — fan-out/in/supervisor/synthesis |
| `app.py` | REWRITTEN | Gradio shell: ticker input, BYO-key panel, streaming output |
| `ratelimit.py` | MODIFY | Replace actions with `analyze` only (and tighten cap) |
| `requirements.txt` | MODIFY | Add `requests`, `beautifulsoup4`, `pytest`, `responses`. Drop `chromadb`, `pypdf`, `sentence-transformers`, `langchain-huggingface`, `langchain-community`. |
| `.env.example` | MODIFY | New keys list |
| `README.md` | MODIFY | New v2 description |
| `tests/` | NEW | pytest tree mirroring source layout |
| `agent.py` / `tools.py` / `rag.py` | DELETE | Legacy single-agent + RAG |
| `data/` | DELETE | Legacy PDF corpus |

---

## Phase 0: Workspace Setup

### Task 0.1: Create feature branch

**Files:**
- (no source changes)

- [ ] **Step 1: Cut feature branch from main**

```bash
git checkout -b feat/marketmind-v2
git status
```

Expected: `On branch feat/marketmind-v2 — Untracked: CLAUDE.md, docs/, marketmind_prd.md, tasks/`

- [ ] **Step 2: Stage and commit the spec + planning docs**

```bash
git add CLAUDE.md docs/ tasks/ marketmind_prd.md
git commit -m "docs: marketmind v2 spec and implementation plan"
```

Expected: commit succeeds; `git log -1 --stat` shows the new files.

---

## Phase 1: Foundations

### Task 1.1: Update `requirements.txt`

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Replace contents**

```
gradio==4.44.1
fastapi>=0.110,<0.113
starlette>=0.37,<0.38
langchain-core==0.3.79
langchain-anthropic>=0.3.0
langgraph==0.3.7
tavily-python>=0.5.0
yfinance>=0.2.40,<0.2.50
requests>=2.31
beautifulsoup4>=4.12
huggingface_hub<1.0
audioop-lts; python_version >= "3.13"

# Test-only — kept here so HF Space install also covers test deps for smoke runs
pytest>=8.0
pytest-asyncio>=0.23
responses>=0.25
```

- [ ] **Step 2: Reinstall**

```bash
pip install -r requirements.txt
```

Expected: clean install, no resolver errors.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "deps: switch to v2 stack — drop chroma/pypdf, add requests/bs4/pytest"
```

### Task 1.2: Create `state.py`

**Files:**
- Create: `state.py`
- Test: `tests/test_state.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_state.py
import operator
from state import MarketMindState, AgentSignal, SupervisorReview


def test_state_has_required_keys():
    s: MarketMindState = {
        "ticker": "MSFT",
        "company_name": None,
        "cik": None,
        "agent_signals": [],
        "retry_round": 0,
        "supervisor_review": None,
        "final_verdict": None,
        "final_conviction": None,
        "final_confidence": None,
        "final_reasoning": None,
        "final_report": None,
    }
    assert s["ticker"] == "MSFT"
    assert s["agent_signals"] == []


def test_agent_signals_reducer_appends():
    # The TypedDict's annotation is what LangGraph reads; here we just verify
    # the operator.add behavior matches our intent.
    a: list = []
    b = a + [{"agent": "price"}]
    c = b + [{"agent": "risk"}]
    assert len(c) == 2 and c[0]["agent"] == "price" and c[1]["agent"] == "risk"


def test_agent_signal_shape():
    sig: AgentSignal = {
        "agent": "price",
        "signal": "BULLISH",
        "confidence": 0.7,
        "summary": "uptrend",
        "section_markdown": "## Technical\nDetails...",
        "raw_data": {"rsi": 55.0},
        "degraded": False,
        "error": None,
    }
    assert sig["signal"] == "BULLISH"


def test_supervisor_review_shape():
    rev: SupervisorReview = {
        "approved": True,
        "critiques": {},
        "retry_targets": [],
        "notes": "All sections complete.",
    }
    assert rev["approved"] is True
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_state.py -v
```

Expected: ImportError because `state.py` does not exist yet.

- [ ] **Step 3: Implement `state.py`**

```python
# state.py
"""Shared state definitions for the MarketMind v2 LangGraph pipeline.

The `agent_signals` list uses `operator.add` as its LangGraph reducer so that
five specialist nodes running in the same superstep can each append their
result without races.
"""

from __future__ import annotations

import operator
from typing import Annotated, List, Literal, Optional, TypedDict

Verdict = Literal["BUY", "HOLD", "SELL"]
Conviction = Literal["STRONG", "STANDARD", "CAUTIOUS"]
Signal = Literal["BULLISH", "BEARISH", "NEUTRAL"]


class AgentSignal(TypedDict):
    agent: str
    signal: Signal
    confidence: float
    summary: str
    section_markdown: str
    raw_data: dict
    degraded: bool
    error: Optional[str]


class SupervisorReview(TypedDict):
    approved: bool
    critiques: dict
    retry_targets: list
    notes: str


class MarketMindState(TypedDict):
    ticker: str
    company_name: Optional[str]
    cik: Optional[str]
    agent_signals: Annotated[List[AgentSignal], operator.add]
    retry_round: int
    supervisor_review: Optional[SupervisorReview]
    final_verdict: Optional[Verdict]
    final_conviction: Optional[Conviction]
    final_confidence: Optional[float]
    final_reasoning: Optional[str]
    final_report: Optional[str]
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_state.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add state.py tests/test_state.py
git commit -m "feat(state): typed state for marketmind v2"
```

### Task 1.3: Create `agents/__init__.py` factory

**Files:**
- Create: `agents/__init__.py`
- Test: `tests/test_agents_init.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_agents_init.py
import pytest

from agents import build_llm_clients, safe_parse_json


def test_safe_parse_json_plain():
    assert safe_parse_json('{"a": 1}') == {"a": 1}


def test_safe_parse_json_markdown_fenced():
    payload = '```json\n{"a": 1, "b": "x"}\n```'
    assert safe_parse_json(payload) == {"a": 1, "b": "x"}


def test_safe_parse_json_bare_fence():
    payload = '```\n{"a": 1}\n```'
    assert safe_parse_json(payload) == {"a": 1}


def test_build_llm_clients_returns_two_models():
    clients = build_llm_clients("sk-ant-fake-test-key")
    assert clients.reasoning is not None
    assert clients.fast is not None
    # Models bind at call time; just verify the configured names are correct.
    assert "sonnet" in clients.reasoning.model.lower()
    assert "haiku" in clients.fast.model.lower()


def test_build_llm_clients_rejects_empty_key():
    with pytest.raises(ValueError, match="anthropic"):
        build_llm_clients("")
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
pytest tests/test_agents_init.py -v
```

Expected: ImportError (`agents` package does not exist).

- [ ] **Step 3: Create `agents/__init__.py`**

```python
# agents/__init__.py
"""Shared LLM clients + JSON helper for MarketMind v2 specialist agents.

`build_llm_clients(anthropic_key)` is the only entry point. It returns a
NamedTuple of two ChatAnthropic clients (Sonnet + Haiku) bound to the
session-scoped key. No module-level client is ever constructed from
`os.environ` on the public BYO-key path.

A process-wide asyncio semaphore caps in-flight Anthropic requests at 3 per
key to avoid 429s during fan-out. (LangGraph's superstep will start all 5
specialist nodes concurrently; the semaphore queues two of them briefly.)
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import NamedTuple

from langchain_anthropic import ChatAnthropic

# Per-key concurrency semaphores. Keyed by the API key string; ephemeral,
# session-scoped, GC'd when the LLMClients reference is dropped.
_semaphores: dict[str, asyncio.Semaphore] = {}
_semaphores_lock = threading.Lock()

_MAX_CONCURRENT_PER_KEY = 3

REASONING_MODEL = "claude-sonnet-4-6"
FAST_MODEL = "claude-haiku-4-5-20251001"


class LLMClients(NamedTuple):
    reasoning: ChatAnthropic
    fast: ChatAnthropic
    semaphore_key: str


def build_llm_clients(anthropic_key: str) -> LLMClients:
    if not anthropic_key:
        raise ValueError("anthropic_key is required (BYO-key)")

    reasoning = ChatAnthropic(
        model=REASONING_MODEL,
        api_key=anthropic_key,
        temperature=0.1,
        max_tokens=1500,
    )
    fast = ChatAnthropic(
        model=FAST_MODEL,
        api_key=anthropic_key,
        temperature=0.1,
        max_tokens=800,
    )
    return LLMClients(reasoning=reasoning, fast=fast, semaphore_key=anthropic_key)


def get_semaphore(key: str) -> asyncio.Semaphore:
    """Return the per-key semaphore, creating it lazily."""
    with _semaphores_lock:
        sem = _semaphores.get(key)
        if sem is None:
            sem = asyncio.Semaphore(_MAX_CONCURRENT_PER_KEY)
            _semaphores[key] = sem
        return sem


def safe_parse_json(content: str) -> dict:
    """Parse LLM JSON output. Strips ```json ... ``` and ``` ... ``` fences.

    Llama and Claude both occasionally wrap JSON in markdown fences even when
    instructed otherwise. This is a defensive parser shared by every agent.
    """
    if not isinstance(content, str):
        raise TypeError(f"safe_parse_json expects str, got {type(content)!r}")

    text = content.strip()
    if text.startswith("```"):
        # Remove the opening fence (with optional language tag) and the closing fence.
        first_newline = text.find("\n")
        if first_newline == -1:
            raise json.JSONDecodeError("Empty fenced block", text, 0)
        text = text[first_newline + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[: -3]
        text = text.strip()
    return json.loads(text)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_agents_init.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/__init__.py tests/test_agents_init.py
git commit -m "feat(agents): byo-key llm factory + safe_parse_json helper"
```

### Task 1.4: Create `edgar.py` — CIK resolution

**Files:**
- Create: `edgar.py`
- Test: `tests/test_edgar_cik.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_edgar_cik.py
import responses
import pytest
from edgar import resolve_ticker, TickerNotFound, _CIK_CACHE


SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def setup_function(_):
    _CIK_CACHE.clear()


@responses.activate
def test_resolve_ticker_msft():
    responses.add(
        responses.GET,
        SEC_TICKERS_URL,
        json={
            "0": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
            "1": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        },
    )
    cik, name = resolve_ticker("MSFT")
    assert cik == "0000789019"
    assert "MICROSOFT" in name.upper()


@responses.activate
def test_resolve_ticker_case_insensitive():
    responses.add(
        responses.GET,
        SEC_TICKERS_URL,
        json={"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}},
    )
    cik, _ = resolve_ticker("aapl")
    assert cik == "0000320193"


@responses.activate
def test_resolve_ticker_not_found():
    responses.add(
        responses.GET,
        SEC_TICKERS_URL,
        json={"0": {"cik_str": 1, "ticker": "FOO", "title": "Foo Co"}},
    )
    with pytest.raises(TickerNotFound):
        resolve_ticker("NOSUCH")
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_edgar_cik.py -v
```

Expected: ImportError (no `edgar.py`).

- [ ] **Step 3: Implement `edgar.py` (CIK resolution slice only)**

```python
# edgar.py
"""SEC EDGAR client for MarketMind v2 Fundamentals agent.

Replaces the deleted PDF/Chroma RAG pipeline. Provides:
  - resolve_ticker(ticker) -> (cik_padded, company_name)
  - fetch_company_facts(cik) -> dict (XBRL company facts)
  - fetch_latest_10q(cik) -> Filing | None
  - fetch_latest_10k(cik) -> Filing | None
  - extract_mdna(filing) -> str
  - build_edgar_bundle(ticker) -> EdgarBundle

All requests carry a `User-Agent` header per SEC policy. Per-process LRU cache
with TTL avoids hammering SEC; cache lives only in memory.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import requests

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

DEFAULT_USER_AGENT = "MarketMind/2.0 contact@marketmind.local"
SEC_POLITENESS_SLEEP = 0.1  # seconds between requests
HTTP_TIMEOUT = 15


class EdgarError(Exception):
    """Base for any EDGAR client failure."""


class TickerNotFound(EdgarError):
    """Ticker is not present in SEC's company tickers index."""


# -- ticker -> CIK -------------------------------------------------------------

# {ticker_upper: (cik_padded10, company_name, fetched_at)}
_CIK_CACHE: dict[str, tuple[str, str, float]] = {}
_CIK_TTL_SECONDS = 24 * 3600


def _now() -> float:
    return time.time()


def _user_agent() -> str:
    # Resolved at call time so an updated UA from the BYO-key panel is picked up.
    import os
    return os.environ.get("SEC_USER_AGENT", DEFAULT_USER_AGENT)


def _get(url: str, **kwargs) -> requests.Response:
    headers = kwargs.pop("headers", {}) or {}
    headers.setdefault("User-Agent", _user_agent())
    headers.setdefault("Accept", "application/json")
    time.sleep(SEC_POLITENESS_SLEEP)
    resp = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT, **kwargs)
    resp.raise_for_status()
    return resp


def _load_tickers_index() -> dict[str, tuple[str, str]]:
    """Return {ticker_upper: (cik_padded10, name)} from SEC's master file."""
    resp = _get(SEC_TICKERS_URL)
    out: dict[str, tuple[str, str]] = {}
    for entry in resp.json().values():
        ticker = str(entry["ticker"]).upper()
        cik_padded = str(entry["cik_str"]).zfill(10)
        out[ticker] = (cik_padded, entry["title"])
    return out


def resolve_ticker(ticker: str) -> tuple[str, str]:
    """Resolve a ticker to (CIK_padded_to_10, company_name).

    Raises TickerNotFound if SEC has no matching equity issuer (foreign issuers
    that file only as 20-F, ETFs, recent IPOs without CIKs, etc.).
    """
    key = ticker.strip().upper()
    cached = _CIK_CACHE.get(key)
    if cached and (_now() - cached[2]) < _CIK_TTL_SECONDS:
        return cached[0], cached[1]

    index = _load_tickers_index()
    if key not in index:
        raise TickerNotFound(f"{ticker} is not in SEC's company tickers index")
    cik, name = index[key]
    _CIK_CACHE[key] = (cik, name, _now())
    return cik, name
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_edgar_cik.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add edgar.py tests/test_edgar_cik.py
git commit -m "feat(edgar): ticker -> CIK resolution with cache"
```

### Task 1.5: `edgar.py` — fetch company facts (XBRL)

**Files:**
- Modify: `edgar.py`
- Test: `tests/test_edgar_facts.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_edgar_facts.py
import responses
from edgar import fetch_company_facts, _FACTS_CACHE


def setup_function(_):
    _FACTS_CACHE.clear()


@responses.activate
def test_fetch_company_facts_basic():
    cik = "0000789019"
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    responses.add(
        responses.GET,
        url,
        json={
            "cik": 789019,
            "entityName": "MICROSOFT CORP",
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "label": "Revenues",
                        "units": {
                            "USD": [
                                {"end": "2025-09-30", "val": 70000000000, "fy": 2026, "fp": "Q1", "form": "10-Q"},
                                {"end": "2024-09-30", "val": 65000000000, "fy": 2025, "fp": "Q1", "form": "10-Q"},
                            ]
                        },
                    }
                }
            },
        },
    )
    facts = fetch_company_facts(cik)
    rev = facts["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
    assert len(rev) == 2 and rev[0]["val"] == 70000000000


@responses.activate
def test_fetch_company_facts_caches():
    cik = "0000789019"
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    responses.add(responses.GET, url, json={"cik": 789019, "facts": {}})
    fetch_company_facts(cik)
    fetch_company_facts(cik)
    # Cache should mean only ONE actual HTTP call.
    assert len(responses.calls) == 1
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_edgar_facts.py -v
```

Expected: ImportError (`fetch_company_facts` not defined).

- [ ] **Step 3: Add to `edgar.py`**

```python
# Append to edgar.py

# {cik: (data_dict, fetched_at)}
_FACTS_CACHE: dict[str, tuple[dict, float]] = {}
_FACTS_TTL_SECONDS = 6 * 3600


def fetch_company_facts(cik: str) -> dict:
    """Fetch XBRL company facts for a CIK. Cached per-process for 6h."""
    cached = _FACTS_CACHE.get(cik)
    if cached and (_now() - cached[1]) < _FACTS_TTL_SECONDS:
        return cached[0]
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    resp = _get(url)
    data = resp.json()
    _FACTS_CACHE[cik] = (data, _now())
    return data
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_edgar_facts.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add edgar.py tests/test_edgar_facts.py
git commit -m "feat(edgar): fetch_company_facts with 6h cache"
```

### Task 1.6: `edgar.py` — fetch latest 10-Q and 10-K filing metadata

**Files:**
- Modify: `edgar.py`
- Test: `tests/test_edgar_filings.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_edgar_filings.py
import responses
from edgar import fetch_latest_10q, fetch_latest_10k, _SUBMISSIONS_CACHE


def setup_function(_):
    _SUBMISSIONS_CACHE.clear()


def _submissions_payload() -> dict:
    return {
        "cik": "789019",
        "name": "MICROSOFT CORP",
        "filings": {
            "recent": {
                "accessionNumber": [
                    "0000789019-25-000123",  # 10-Q
                    "0000789019-25-000099",  # 10-K
                    "0000789019-24-000200",  # 10-Q older
                    "0000789019-25-000050",  # 8-K
                ],
                "filingDate": ["2025-10-30", "2025-07-31", "2025-04-30", "2025-09-15"],
                "reportDate": ["2025-09-30", "2025-06-30", "2025-03-31", ""],
                "form": ["10-Q", "10-K", "10-Q", "8-K"],
                "primaryDocument": ["q1-25.htm", "10k-25.htm", "q3-25.htm", "8k.htm"],
            }
        },
    }


@responses.activate
def test_fetch_latest_10q_picks_most_recent():
    cik = "0000789019"
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    responses.add(responses.GET, url, json=_submissions_payload())
    f = fetch_latest_10q(cik)
    assert f.accession == "0000789019-25-000123"
    assert f.form == "10-Q"
    assert f.report_date == "2025-09-30"


@responses.activate
def test_fetch_latest_10k():
    cik = "0000789019"
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    responses.add(responses.GET, url, json=_submissions_payload())
    f = fetch_latest_10k(cik)
    assert f.form == "10-K"
    assert f.accession == "0000789019-25-000099"


@responses.activate
def test_fetch_latest_returns_none_when_absent():
    cik = "0000789019"
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    responses.add(
        responses.GET,
        url,
        json={
            "filings": {
                "recent": {
                    "accessionNumber": ["x"],
                    "filingDate": ["2025-01-01"],
                    "reportDate": [""],
                    "form": ["8-K"],
                    "primaryDocument": ["x.htm"],
                }
            }
        },
    )
    assert fetch_latest_10q(cik) is None
    assert fetch_latest_10k(cik) is None
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_edgar_filings.py -v
```

Expected: ImportError.

- [ ] **Step 3: Add to `edgar.py`**

```python
# Append to edgar.py

@dataclass
class Filing:
    cik: str
    accession: str
    form: str
    filing_date: str
    report_date: str
    primary_document: str

    @property
    def primary_url(self) -> str:
        accession_clean = self.accession.replace("-", "")
        return (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(self.cik)}/{accession_clean}/{self.primary_document}"
        )


_SUBMISSIONS_CACHE: dict[str, tuple[dict, float]] = {}
_SUBMISSIONS_TTL_SECONDS = 6 * 3600


def _fetch_submissions(cik: str) -> dict:
    cached = _SUBMISSIONS_CACHE.get(cik)
    if cached and (_now() - cached[1]) < _SUBMISSIONS_TTL_SECONDS:
        return cached[0]
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = _get(url).json()
    _SUBMISSIONS_CACHE[cik] = (data, _now())
    return data


def _latest_filing_of(cik: str, form_target: str) -> Optional[Filing]:
    data = _fetch_submissions(cik)
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    for i, form in enumerate(forms):
        if form == form_target:
            return Filing(
                cik=cik,
                accession=recent["accessionNumber"][i],
                form=form,
                filing_date=recent["filingDate"][i],
                report_date=recent["reportDate"][i],
                primary_document=recent["primaryDocument"][i],
            )
    return None


def fetch_latest_10q(cik: str) -> Optional[Filing]:
    return _latest_filing_of(cik, "10-Q")


def fetch_latest_10k(cik: str) -> Optional[Filing]:
    return _latest_filing_of(cik, "10-K")
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_edgar_filings.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add edgar.py tests/test_edgar_filings.py
git commit -m "feat(edgar): latest 10-Q and 10-K filing lookup"
```

### Task 1.7: `edgar.py` — extract MD&A from 10-Q HTML

**Files:**
- Modify: `edgar.py`
- Test: `tests/test_edgar_mdna.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_edgar_mdna.py
from edgar import extract_mdna_from_html


SAMPLE_HTML = """
<html><body>
<p>Cover page boilerplate.</p>
<h2>ITEM 1. FINANCIAL STATEMENTS</h2>
<p>Tables go here. Lots of numbers.</p>
<h2>ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS</h2>
<p>Revenue grew 12% year over year, driven by Cloud and AI services.</p>
<p>Operating margin expanded by 200 basis points.</p>
<h2>ITEM 3. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK</h2>
<p>Interest rate exposure...</p>
</body></html>
"""


def test_extract_mdna_picks_item2_section():
    text = extract_mdna_from_html(SAMPLE_HTML)
    assert "Revenue grew 12%" in text
    assert "Operating margin" in text
    assert "Interest rate exposure" not in text  # stopped at Item 3
    assert "Tables go here" not in text  # did not include Item 1


def test_extract_mdna_returns_empty_on_missing():
    assert extract_mdna_from_html("<html><body><p>no items here</p></body></html>") == ""


def test_extract_mdna_caps_length():
    big = "<html><body><h2>ITEM 2. MANAGEMENT'S DISCUSSION</h2><p>" + ("x" * 20000) + "</p></body></html>"
    text = extract_mdna_from_html(big, max_chars=8000)
    assert len(text) <= 8000
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_edgar_mdna.py -v
```

Expected: ImportError.

- [ ] **Step 3: Add to `edgar.py`**

```python
# Append to edgar.py
import re
from bs4 import BeautifulSoup


_MDNA_HEADING_RE = re.compile(
    r"item\s*2\b.*?management.{0,30}discussion",
    re.IGNORECASE | re.DOTALL,
)
_NEXT_ITEM_RE = re.compile(r"item\s*[3-9]\b", re.IGNORECASE)


def extract_mdna_from_html(html: str, max_chars: int = 8000) -> str:
    """Extract Item 2 (MD&A) text from a 10-Q filing's primary HTML document.

    Strategy:
      1. Strip tags via BeautifulSoup, preserving paragraph spacing.
      2. Locate the Item 2 heading by regex.
      3. Slice from there to the next Item N heading (3..9) or end of document.
      4. Cap at `max_chars`.
    """
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")
    # Normalize whitespace.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    start = _MDNA_HEADING_RE.search(text)
    if not start:
        return ""

    after = text[start.end():]
    end_match = _NEXT_ITEM_RE.search(after)
    body = after[: end_match.start()] if end_match else after
    body = body.strip()
    if len(body) > max_chars:
        body = body[:max_chars]
    return body


def fetch_filing_html(filing: Filing) -> str:
    """Download the primary document HTML for a Filing."""
    return _get(filing.primary_url, headers={"Accept": "text/html"}).text


def extract_mdna(filing: Filing, max_chars: int = 8000) -> str:
    """Convenience: download and extract MD&A in one shot."""
    return extract_mdna_from_html(fetch_filing_html(filing), max_chars=max_chars)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_edgar_mdna.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add edgar.py tests/test_edgar_mdna.py
git commit -m "feat(edgar): MD&A extraction from 10-Q HTML"
```

### Task 1.8: `edgar.py` — `build_edgar_bundle()` aggregator

**Files:**
- Modify: `edgar.py`
- Test: `tests/test_edgar_bundle.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_edgar_bundle.py
import responses
from edgar import build_edgar_bundle, _CIK_CACHE, _SUBMISSIONS_CACHE, _FACTS_CACHE


def setup_function(_):
    _CIK_CACHE.clear()
    _SUBMISSIONS_CACHE.clear()
    _FACTS_CACHE.clear()


def _stub_all(cik="0000789019", company="MICROSOFT CORP"):
    responses.add(
        responses.GET,
        "https://www.sec.gov/files/company_tickers.json",
        json={"0": {"cik_str": int(cik), "ticker": "MSFT", "title": company}},
    )
    responses.add(
        responses.GET,
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        json={
            "filings": {
                "recent": {
                    "accessionNumber": ["acc-q", "acc-k"],
                    "filingDate": ["2025-10-30", "2025-07-31"],
                    "reportDate": ["2025-09-30", "2025-06-30"],
                    "form": ["10-Q", "10-K"],
                    "primaryDocument": ["q.htm", "k.htm"],
                }
            }
        },
    )
    responses.add(
        responses.GET,
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
        json={"facts": {"us-gaap": {"Revenues": {"units": {"USD": []}}}}},
    )
    responses.add(
        responses.GET,
        "https://www.sec.gov/Archives/edgar/data/789019/accq/q.htm",
        body="<html><body><h2>ITEM 2. MANAGEMENT'S DISCUSSION</h2><p>Revenue up</p></body></html>",
    )


@responses.activate
def test_build_bundle_happy_path():
    _stub_all()
    bundle = build_edgar_bundle("MSFT")
    assert bundle.ticker == "MSFT"
    assert bundle.cik == "0000789019"
    assert bundle.company_name == "MICROSOFT CORP"
    assert bundle.latest_10q is not None
    assert bundle.latest_10k is not None
    assert "Revenue up" in bundle.mdna_text


@responses.activate
def test_build_bundle_ticker_not_found():
    responses.add(
        responses.GET,
        "https://www.sec.gov/files/company_tickers.json",
        json={"0": {"cik_str": 1, "ticker": "FOO", "title": "Foo"}},
    )
    import pytest
    from edgar import TickerNotFound
    with pytest.raises(TickerNotFound):
        build_edgar_bundle("ZZZZZ")
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_edgar_bundle.py -v
```

Expected: ImportError.

- [ ] **Step 3: Add to `edgar.py`**

```python
# Append to edgar.py

@dataclass
class EdgarBundle:
    ticker: str
    cik: str
    company_name: str
    latest_10q: Optional[Filing]
    latest_10k: Optional[Filing]
    xbrl_facts: dict
    mdna_text: str
    fetched_at: float = field(default_factory=_now)


def build_edgar_bundle(ticker: str) -> EdgarBundle:
    """One-shot bundle for the Fundamentals agent.

    Raises TickerNotFound if SEC has no record. Other errors propagate from the
    individual fetchers. The Fundamentals agent is responsible for catching
    those and producing a degraded AgentSignal.
    """
    cik, name = resolve_ticker(ticker)
    f10q = fetch_latest_10q(cik)
    f10k = fetch_latest_10k(cik)
    facts = fetch_company_facts(cik)
    mdna = extract_mdna(f10q) if f10q else ""
    return EdgarBundle(
        ticker=ticker.strip().upper(),
        cik=cik,
        company_name=name,
        latest_10q=f10q,
        latest_10k=f10k,
        xbrl_facts=facts,
        mdna_text=mdna,
    )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_edgar_bundle.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add edgar.py tests/test_edgar_bundle.py
git commit -m "feat(edgar): build_edgar_bundle aggregator"
```

---

## Phase 2: Specialists

### Task 2.1: Orchestrator agent

**Files:**
- Create: `agents/orchestrator.py`
- Test: `tests/test_orchestrator.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_orchestrator.py
import pytest
from unittest.mock import patch

from agents.orchestrator import orchestrator


def test_orchestrator_uppercases_and_validates():
    state = {
        "ticker": "  msft  ",
        "company_name": None,
        "cik": None,
        "agent_signals": [],
        "retry_round": 0,
        "supervisor_review": None,
        "final_verdict": None,
        "final_conviction": None,
        "final_confidence": None,
        "final_reasoning": None,
        "final_report": None,
    }
    with patch("agents.orchestrator.resolve_ticker") as mock_resolve:
        mock_resolve.return_value = ("0000789019", "MICROSOFT CORP")
        update = orchestrator(state)
    assert update["ticker"] == "MSFT"
    assert update["cik"] == "0000789019"
    assert update["company_name"] == "MICROSOFT CORP"


def test_orchestrator_rejects_invalid_ticker():
    state = {"ticker": "!!", "agent_signals": [], "retry_round": 0,
             "company_name": None, "cik": None, "supervisor_review": None,
             "final_verdict": None, "final_conviction": None,
             "final_confidence": None, "final_reasoning": None, "final_report": None}
    with pytest.raises(ValueError, match="invalid ticker"):
        orchestrator(state)


def test_orchestrator_unknown_to_sec_still_proceeds():
    state = {"ticker": "FOREIGN", "agent_signals": [], "retry_round": 0,
             "company_name": None, "cik": None, "supervisor_review": None,
             "final_verdict": None, "final_conviction": None,
             "final_confidence": None, "final_reasoning": None, "final_report": None}
    from edgar import TickerNotFound
    with patch("agents.orchestrator.resolve_ticker", side_effect=TickerNotFound("nope")):
        update = orchestrator(state)
    # Pipeline continues — fundamentals will degrade.
    assert update["ticker"] == "FOREIGN"
    assert update["cik"] is None
    assert update["company_name"] == "FOREIGN"
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_orchestrator.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement**

```python
# agents/orchestrator.py
"""Orchestrator: validate input, resolve CIK, anchor the fan-out."""

from __future__ import annotations

import re

from edgar import TickerNotFound, resolve_ticker

VALID_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


def orchestrator(state: dict) -> dict:
    raw = (state.get("ticker") or "").strip().upper()
    if not VALID_TICKER_RE.match(raw):
        raise ValueError(f"invalid ticker: {raw!r}")

    update: dict = {"ticker": raw}
    try:
        cik, name = resolve_ticker(raw)
        update["cik"] = cik
        update["company_name"] = name
    except TickerNotFound:
        # Foreign issuer / ETF / pre-CIK IPO. Fundamentals agent will degrade.
        update["cik"] = None
        update["company_name"] = raw
    return update
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_orchestrator.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/orchestrator.py tests/test_orchestrator.py
git commit -m "feat(agents): orchestrator validates ticker and resolves CIK"
```

### Task 2.2: Price agent

**Files:**
- Create: `agents/price_agent.py`
- Test: `tests/test_price_agent.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_price_agent.py
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from agents.price_agent import (
    compute_rsi,
    compute_macd,
    compute_bollinger_pctb,
    price_agent,
)


def _flat_close(n=90, start=100.0, drift=0.5):
    return pd.Series(np.linspace(start, start + drift * n, n))


def test_compute_rsi_neutral_for_steady_uptrend():
    s = _flat_close()
    rsi = compute_rsi(s)
    assert 50 < rsi < 100  # always-up series → high RSI


def test_compute_macd_returns_two_floats():
    s = _flat_close()
    line, signal = compute_macd(s)
    assert isinstance(line, float) and isinstance(signal, float)


def test_compute_bollinger_at_midpoint_for_constant_series():
    s = pd.Series([100.0] * 50)
    pctb = compute_bollinger_pctb(s)
    # Constant series → zero std → fallback midpoint 0.5
    assert pctb == 0.5


def test_price_agent_happy_path():
    fake_close = _flat_close(90, 100, 0.3)
    fake_df = pd.DataFrame({"Close": fake_close})

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content='{"signal": "BULLISH", "confidence": 0.7, "summary": "Trend up"}')
    fake_clients = MagicMock(reasoning=fake_llm)

    with patch("agents.price_agent.yf.download", return_value=fake_df):
        signals = price_agent({"ticker": "MSFT"}, fake_clients)

    sig = signals["agent_signals"][0]
    assert sig["agent"] == "price"
    assert sig["signal"] == "BULLISH"
    assert sig["confidence"] == 0.7
    assert "rsi" in sig["raw_data"]
    assert sig["section_markdown"].startswith("## Technical Analysis")


def test_price_agent_empty_data_degrades():
    fake_clients = MagicMock()
    with patch("agents.price_agent.yf.download", return_value=pd.DataFrame()):
        signals = price_agent({"ticker": "ZZZZ"}, fake_clients)
    sig = signals["agent_signals"][0]
    assert sig["degraded"] is True
    assert sig["signal"] == "NEUTRAL"
    assert sig["confidence"] == 0.0
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_price_agent.py -v
```

Expected: ImportError (`agents.price_agent` not defined).

- [ ] **Step 3: Implement**

```python
# agents/price_agent.py
"""Technical analysis specialist: yfinance OHLC -> RSI/MACD/Bollinger -> LLM."""

from __future__ import annotations

import json

import pandas as pd
import yfinance as yf

from agents import safe_parse_json
from state import AgentSignal


def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2)


def compute_macd(prices: pd.Series) -> tuple[float, float]:
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    return round(float(line.iloc[-1]), 4), round(float(signal.iloc[-1]), 4)


def compute_bollinger_pctb(prices: pd.Series, period: int = 20) -> float:
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    band = float(upper.iloc[-1]) - float(lower.iloc[-1])
    if band <= 0 or pd.isna(band):
        return 0.5
    return round((float(prices.iloc[-1]) - float(lower.iloc[-1])) / band, 3)


def _degraded(reason: str, raw: dict | None = None, error: str | None = None) -> dict:
    return {
        "agent_signals": [AgentSignal(
            agent="price", signal="NEUTRAL", confidence=0.0,
            summary=reason,
            section_markdown=f"## Technical Analysis\n_Unavailable: {reason}_",
            raw_data=raw or {}, degraded=True, error=error,
        )]
    }


def price_agent(state: dict, clients) -> dict:
    ticker = state["ticker"]
    try:
        data = yf.download(ticker, period="90d", interval="1d", progress=False)
        if data.empty or "Close" not in data.columns:
            return _degraded(f"No price data for {ticker}")

        close = data["Close"].squeeze()
        current = round(float(close.iloc[-1]), 4)
        prev_7d = float(close.iloc[-7]) if len(close) >= 7 else float(close.iloc[0])
        change_7d = round((current - prev_7d) / prev_7d * 100, 2) if prev_7d else 0.0
        change_30d = (
            round((current - float(close.iloc[-30])) / float(close.iloc[-30]) * 100, 2)
            if len(close) >= 30 else 0.0
        )
        change_90d = round((current - float(close.iloc[0])) / float(close.iloc[0]) * 100, 2)

        rsi = compute_rsi(close)
        macd_line, macd_signal = compute_macd(close)
        pctb = compute_bollinger_pctb(close)

        raw = {
            "current_price": current,
            "change_7d_pct": change_7d,
            "change_30d_pct": change_30d,
            "change_90d_pct": change_90d,
            "rsi": rsi,
            "macd_line": macd_line,
            "macd_signal": macd_signal,
            "macd_crossover": "positive" if macd_line > macd_signal else "negative",
            "bollinger_pctb": pctb,
        }

        prompt = (
            f"You are a technical analyst. Interpret these indicators for {ticker}:\n\n"
            f"{json.dumps(raw, indent=2)}\n\n"
            "Reference: RSI<30 oversold, RSI>70 overbought; positive MACD crossover bullish; "
            "Bollinger %B near 0 = lower band, near 1 = upper band.\n\n"
            "Respond with JSON ONLY (no markdown fences) with EXACTLY these keys:\n"
            "{\n"
            '  "signal": "BULLISH" | "BEARISH" | "NEUTRAL",\n'
            '  "confidence": 0.0..1.0,\n'
            '  "summary": "one sentence under 25 words",\n'
            '  "section_markdown": "## Technical Analysis\\n... 120-200 word section discussing RSI, MACD, Bollinger position, and 7/30/90d trend ..."\n'
            "}"
        )
        resp = clients.reasoning.invoke(prompt)
        out = safe_parse_json(resp.content)

        return {"agent_signals": [AgentSignal(
            agent="price",
            signal=out["signal"],
            confidence=float(out["confidence"]),
            summary=out["summary"],
            section_markdown=out.get("section_markdown") or "## Technical Analysis\n_Section missing._",
            raw_data=raw,
            degraded=False,
            error=None,
        )]}
    except Exception as exc:
        return _degraded(f"Price agent error", error=str(exc)[:200])
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_price_agent.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/price_agent.py tests/test_price_agent.py
git commit -m "feat(agents): price agent with RSI/MACD/Bollinger + LLM section"
```

### Task 2.3: Sentiment agent

**Files:**
- Create: `agents/sentiment_agent.py`
- Test: `tests/test_sentiment_agent.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_sentiment_agent.py
from unittest.mock import MagicMock, patch

from agents.sentiment_agent import sentiment_agent


def _fake_tavily_results():
    return {
        "results": [
            {"title": "Microsoft Q1 beats estimates", "content": "Revenue rose 12%.", "url": "https://x.com/1"},
            {"title": "AI demand boosts cloud", "content": "Azure up 30%.", "url": "https://x.com/2"},
            {"title": "Regulator probes acquisition", "content": "EU launches inquiry.", "url": "https://x.com/3"},
        ]
    }


def test_sentiment_agent_happy():
    fake_tavily = MagicMock()
    fake_tavily.search.return_value = _fake_tavily_results()
    fake_haiku = MagicMock()
    fake_haiku.invoke.return_value = MagicMock(content='{"per_article": ["positive","positive","negative"], "drivers": ["earnings beat","cloud growth","regulatory"]}')
    fake_sonnet = MagicMock()
    fake_sonnet.invoke.return_value = MagicMock(content='{"signal": "BULLISH", "confidence": 0.65, "summary": "Mostly positive coverage", "section_markdown": "## News & Sentiment\\nLots of stuff."}')
    clients = MagicMock(reasoning=fake_sonnet, fast=fake_haiku)

    state = {"ticker": "MSFT", "company_name": "Microsoft Corp"}
    with patch("agents.sentiment_agent.TavilyClient", return_value=fake_tavily):
        out = sentiment_agent(state, clients, tavily_key="tvly-fake")

    sig = out["agent_signals"][0]
    assert sig["agent"] == "sentiment"
    assert sig["signal"] == "BULLISH"
    assert sig["raw_data"]["article_count"] == 3
    assert "positive_count" in sig["raw_data"]


def test_sentiment_agent_no_results():
    fake_tavily = MagicMock()
    fake_tavily.search.return_value = {"results": []}
    clients = MagicMock()

    with patch("agents.sentiment_agent.TavilyClient", return_value=fake_tavily):
        out = sentiment_agent({"ticker": "ZZZZ", "company_name": "Z"}, clients, tavily_key="tvly")

    sig = out["agent_signals"][0]
    assert sig["degraded"] is True
    assert sig["signal"] == "NEUTRAL"


def test_sentiment_agent_missing_key_degrades():
    out = sentiment_agent({"ticker": "MSFT", "company_name": "Microsoft"}, MagicMock(), tavily_key="")
    sig = out["agent_signals"][0]
    assert sig["degraded"] is True
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_sentiment_agent.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement**

```python
# agents/sentiment_agent.py
"""Sentiment specialist: Tavily news -> Haiku per-article scoring -> Sonnet rollup."""

from __future__ import annotations

from tavily import TavilyClient

from agents import safe_parse_json
from state import AgentSignal


def _degraded(reason: str, raw: dict | None = None, error: str | None = None) -> dict:
    return {"agent_signals": [AgentSignal(
        agent="sentiment", signal="NEUTRAL", confidence=0.0,
        summary=reason,
        section_markdown=f"## News & Sentiment\n_Unavailable: {reason}_",
        raw_data=raw or {}, degraded=True, error=error,
    )]}


def sentiment_agent(state: dict, clients, tavily_key: str) -> dict:
    ticker = state["ticker"]
    company = state.get("company_name") or ticker

    if not tavily_key:
        return _degraded("No Tavily key provided")

    try:
        tav = TavilyClient(api_key=tavily_key)
        query = f"{company} {ticker} stock news"
        result = tav.search(query=query, max_results=15, search_depth="basic", days=7)
        raw_articles = result.get("results", []) or []

        # Dedupe by URL, cap to 12
        seen = set()
        articles = []
        for a in raw_articles:
            url = a.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            articles.append({
                "title": (a.get("title") or "").strip(),
                "snippet": (a.get("content") or "")[:280],
                "url": url,
            })
            if len(articles) >= 12:
                break

        if not articles:
            return _degraded("No coverage found", raw={"article_count": 0})

        haiku_prompt = (
            "Rate each headline as 'positive', 'neutral', or 'negative' for the issuer's stock, "
            "then list 2-5 short driver phrases.\n\n"
            "Headlines:\n"
            + "\n".join(f"{i+1}. {a['title']} — {a['snippet']}" for i, a in enumerate(articles))
            + "\n\nRespond with JSON ONLY:\n"
              '{"per_article": ["positive"|"neutral"|"negative", ...], "drivers": ["...", ...]}'
        )
        haiku_resp = clients.fast.invoke(haiku_prompt)
        haiku_out = safe_parse_json(haiku_resp.content)
        per = haiku_out.get("per_article", []) or []
        pos = sum(1 for x in per if x == "positive")
        neg = sum(1 for x in per if x == "negative")
        neu = sum(1 for x in per if x == "neutral")

        sonnet_prompt = (
            f"You are a market sentiment analyst. Synthesize the news rollup for {ticker} ({company}).\n\n"
            f"Article count: {len(articles)}; positive: {pos}, neutral: {neu}, negative: {neg}.\n"
            f"Top drivers: {haiku_out.get('drivers', [])}\n\n"
            "Sample headlines:\n"
            + "\n".join(f"- {a['title']}" for a in articles[:5])
            + "\n\nRespond with JSON ONLY (no fences):\n"
              '{"signal": "BULLISH"|"BEARISH"|"NEUTRAL", "confidence": 0.0..1.0, '
              '"summary": "one sentence", "section_markdown": "## News & Sentiment\\n... 120-180 word section ..."}'
        )
        sonnet_resp = clients.reasoning.invoke(sonnet_prompt)
        out = safe_parse_json(sonnet_resp.content)

        return {"agent_signals": [AgentSignal(
            agent="sentiment",
            signal=out["signal"],
            confidence=float(out["confidence"]),
            summary=out["summary"],
            section_markdown=out.get("section_markdown") or "## News & Sentiment\n_Section missing._",
            raw_data={
                "article_count": len(articles),
                "positive_count": pos,
                "neutral_count": neu,
                "negative_count": neg,
                "drivers": haiku_out.get("drivers", []),
                "sample_headlines": [a["title"] for a in articles[:3]],
            },
            degraded=False,
            error=None,
        )]}
    except Exception as exc:
        return _degraded("Sentiment agent error", error=str(exc)[:200])
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_sentiment_agent.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/sentiment_agent.py tests/test_sentiment_agent.py
git commit -m "feat(agents): sentiment agent (tavily + haiku + sonnet rollup)"
```

### Task 2.4: Fundamentals agent (EDGAR-based)

**Files:**
- Create: `agents/fundamentals_agent.py`
- Test: `tests/test_fundamentals_agent.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_fundamentals_agent.py
from unittest.mock import MagicMock, patch

from agents.fundamentals_agent import (
    extract_latest_metric,
    yoy_delta_pct,
    fundamentals_agent,
)
from edgar import EdgarBundle, Filing, TickerNotFound


def _facts(values):
    return {"facts": {"us-gaap": {"Revenues": {"units": {"USD": values}}}}}


def test_extract_latest_metric_returns_most_recent_form10q():
    facts = _facts([
        {"end": "2025-09-30", "val": 70_000_000_000, "form": "10-Q"},
        {"end": "2024-09-30", "val": 65_000_000_000, "form": "10-Q"},
    ])
    val, end = extract_latest_metric(facts, "Revenues")
    assert val == 70_000_000_000 and end == "2025-09-30"


def test_yoy_delta_pct():
    assert yoy_delta_pct(110.0, 100.0) == 10.0
    assert yoy_delta_pct(90.0, 100.0) == -10.0
    assert yoy_delta_pct(100.0, 0.0) is None


def test_fundamentals_agent_happy():
    bundle = EdgarBundle(
        ticker="MSFT",
        cik="0000789019",
        company_name="MICROSOFT CORP",
        latest_10q=Filing("789019", "acc", "10-Q", "2025-10-30", "2025-09-30", "q.htm"),
        latest_10k=Filing("789019", "acc-k", "10-K", "2025-07-31", "2025-06-30", "k.htm"),
        xbrl_facts=_facts([
            {"end": "2025-09-30", "val": 70_000_000_000, "form": "10-Q"},
            {"end": "2024-09-30", "val": 65_000_000_000, "form": "10-Q"},
        ]),
        mdna_text="Revenue grew 12% YoY driven by Cloud and AI services.",
    )
    fake_sonnet = MagicMock()
    fake_sonnet.invoke.return_value = MagicMock(content='{"signal": "BULLISH", "confidence": 0.7, "summary": "Strong fundamentals", "section_markdown": "## Fundamentals\\nDetails."}')
    clients = MagicMock(reasoning=fake_sonnet)

    with patch("agents.fundamentals_agent.build_edgar_bundle", return_value=bundle):
        out = fundamentals_agent({"ticker": "MSFT"}, clients)

    sig = out["agent_signals"][0]
    assert sig["agent"] == "fundamentals"
    assert sig["signal"] == "BULLISH"
    assert sig["raw_data"]["revenue_yoy_pct"] is not None
    assert sig["degraded"] is False


def test_fundamentals_agent_ticker_not_in_sec():
    clients = MagicMock()
    with patch("agents.fundamentals_agent.build_edgar_bundle", side_effect=TickerNotFound("nope")):
        out = fundamentals_agent({"ticker": "FOREIGN"}, clients)
    sig = out["agent_signals"][0]
    assert sig["degraded"] is True
    assert "no SEC filings" in sig["summary"].lower()
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_fundamentals_agent.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement**

```python
# agents/fundamentals_agent.py
"""Fundamentals specialist: SEC EDGAR XBRL + MD&A -> Sonnet section."""

from __future__ import annotations

from typing import Optional

from agents import safe_parse_json
from edgar import EdgarBundle, TickerNotFound, build_edgar_bundle
from state import AgentSignal


KEY_TAGS = [
    "Revenues",
    "GrossProfit",
    "OperatingIncomeLoss",
    "NetIncomeLoss",
    "EarningsPerShareDiluted",
    "Assets",
    "Liabilities",
    "StockholdersEquity",
    "CashAndCashEquivalentsAtCarryingValue",
]


def extract_latest_metric(facts: dict, tag: str) -> tuple[Optional[float], Optional[str]]:
    """Most recent USD observation for an XBRL tag, preferring 10-Q then 10-K."""
    units = (
        facts.get("facts", {}).get("us-gaap", {}).get(tag, {}).get("units", {})
    )
    obs = units.get("USD") or units.get("USD/shares") or []
    obs_sorted = sorted(obs, key=lambda o: o.get("end", ""), reverse=True)
    for o in obs_sorted:
        if o.get("form") in ("10-Q", "10-K"):
            return float(o["val"]), o.get("end")
    return None, None


def find_yoy_pair(facts: dict, tag: str) -> tuple[Optional[float], Optional[float]]:
    """Find the latest value and the prior-year same-period value for a tag."""
    units = (
        facts.get("facts", {}).get("us-gaap", {}).get(tag, {}).get("units", {})
    )
    obs = units.get("USD") or []
    obs_sorted = sorted(obs, key=lambda o: o.get("end", ""), reverse=True)
    if not obs_sorted:
        return None, None
    latest = obs_sorted[0]
    latest_end = latest.get("end", "")
    if len(latest_end) < 10:
        return float(latest["val"]), None
    target_prior = f"{int(latest_end[:4]) - 1}{latest_end[4:]}"
    prior = next((o for o in obs_sorted if o.get("end") == target_prior), None)
    return float(latest["val"]), float(prior["val"]) if prior else None


def yoy_delta_pct(current: float, prior: float) -> Optional[float]:
    if not prior:
        return None
    return round((current - prior) / prior * 100, 2)


def _degraded(reason: str, raw: dict | None = None, error: str | None = None) -> dict:
    return {"agent_signals": [AgentSignal(
        agent="fundamentals", signal="NEUTRAL", confidence=0.0,
        summary=reason,
        section_markdown=f"## Fundamentals\n_Unavailable: {reason}_",
        raw_data=raw or {}, degraded=True, error=error,
    )]}


def fundamentals_agent(state: dict, clients) -> dict:
    ticker = state["ticker"]
    try:
        bundle: EdgarBundle = build_edgar_bundle(ticker)
    except TickerNotFound:
        return _degraded("Fundamentals unavailable — no SEC filings for this ticker")
    except Exception as exc:
        return _degraded("Fundamentals fetch error", error=str(exc)[:200])

    raw: dict = {"company_name": bundle.company_name, "cik": bundle.cik}
    for tag in KEY_TAGS:
        cur, end = extract_latest_metric(bundle.xbrl_facts, tag)
        raw[f"{tag}_latest"] = cur
        raw[f"{tag}_period_end"] = end
        latest, prior = find_yoy_pair(bundle.xbrl_facts, tag)
        if latest is not None and prior is not None:
            raw[f"{tag}_yoy_pct"] = yoy_delta_pct(latest, prior)

    raw["revenue_yoy_pct"] = raw.get("Revenues_yoy_pct")
    raw["mdna_excerpt"] = bundle.mdna_text[:2000]
    raw["latest_10q_filed"] = bundle.latest_10q.filing_date if bundle.latest_10q else None
    raw["latest_10k_filed"] = bundle.latest_10k.filing_date if bundle.latest_10k else None

    prompt = (
        f"You are a fundamentals analyst. Analyze {ticker} ({bundle.company_name}).\n\n"
        f"Latest XBRL metrics (USD):\n"
        + "\n".join(
            f"- {k}: {raw.get(f'{k}_latest')} as of {raw.get(f'{k}_period_end')} "
            f"(YoY {raw.get(f'{k}_yoy_pct')}%)"
            for k in KEY_TAGS
        )
        + f"\n\nMD&A excerpt (10-Q):\n{raw['mdna_excerpt'][:3000]}\n\n"
        "Cover: revenue trend, margin trajectory, balance-sheet health, MD&A signals.\n\n"
        "Respond with JSON ONLY:\n"
        '{"signal": "BULLISH"|"BEARISH"|"NEUTRAL", "confidence": 0.0..1.0, '
        '"summary": "one sentence", "section_markdown": "## Fundamentals\\n... 200-300 word section ..."}'
    )
    try:
        resp = clients.reasoning.invoke(prompt)
        out = safe_parse_json(resp.content)
    except Exception as exc:
        return _degraded("LLM error in fundamentals", raw=raw, error=str(exc)[:200])

    return {"agent_signals": [AgentSignal(
        agent="fundamentals",
        signal=out["signal"],
        confidence=float(out["confidence"]),
        summary=out["summary"],
        section_markdown=out.get("section_markdown") or "## Fundamentals\n_Section missing._",
        raw_data=raw,
        degraded=False,
        error=None,
    )]}
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_fundamentals_agent.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/fundamentals_agent.py tests/test_fundamentals_agent.py
git commit -m "feat(agents): fundamentals agent backed by EDGAR XBRL + MD&A"
```

### Task 2.5: Macro agent

**Files:**
- Create: `agents/macro_agent.py`
- Test: `tests/test_macro_agent.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_macro_agent.py
import responses
from unittest.mock import MagicMock

from agents.macro_agent import macro_agent


def _fred_series(observations):
    return {"observations": [{"date": d, "value": v} for d, v in observations]}


@responses.activate
def test_macro_agent_full_data():
    fk = "frd-fake"
    for series in ("DTWEXBGS", "DFF", "DGS10", "DGS2"):
        responses.add(
            responses.GET,
            "https://api.stlouisfed.org/fred/series/observations",
            json=_fred_series([("2025-10-30", "104.2"), ("2025-10-25", "103.8")]),
            match=[responses.matchers.query_param_matcher({"series_id": series, "api_key": fk, "file_type": "json", "sort_order": "desc", "limit": "5"})],
        )
    responses.add(
        responses.GET,
        "https://api.alternative.me/fng/?limit=1",
        json={"data": [{"value": "62", "value_classification": "Greed"}]},
    )

    fake_sonnet = MagicMock()
    fake_sonnet.invoke.return_value = MagicMock(content='{"signal": "BEARISH", "confidence": 0.55, "summary": "DXY rising", "section_markdown": "## Macro Backdrop\\nText."}')
    clients = MagicMock(reasoning=fake_sonnet)

    out = macro_agent({"ticker": "MSFT"}, clients, fred_key=fk)
    sig = out["agent_signals"][0]
    assert sig["agent"] == "macro"
    assert sig["signal"] == "BEARISH"
    assert sig["raw_data"]["dxy_latest"] == 104.2
    assert sig["raw_data"]["fear_greed_index"] == 62


@responses.activate
def test_macro_agent_degraded_without_fred_key():
    responses.add(
        responses.GET,
        "https://api.alternative.me/fng/?limit=1",
        json={"data": [{"value": "30", "value_classification": "Fear"}]},
    )
    fake_sonnet = MagicMock()
    fake_sonnet.invoke.return_value = MagicMock(content='{"signal": "NEUTRAL", "confidence": 0.3, "summary": "Limited macro data", "section_markdown": "## Macro Backdrop\\nText."}')
    clients = MagicMock(reasoning=fake_sonnet)

    out = macro_agent({"ticker": "MSFT"}, clients, fred_key="")
    sig = out["agent_signals"][0]
    assert sig["degraded"] is True
    assert sig["raw_data"]["fear_greed_index"] == 30
    assert sig["raw_data"]["dxy_latest"] is None
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_macro_agent.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement**

```python
# agents/macro_agent.py
"""Macro specialist: FRED + Fear & Greed -> Sonnet."""

from __future__ import annotations

import requests

from agents import safe_parse_json
from state import AgentSignal


def _fetch_fred_series(series_id: str, api_key: str, limit: int = 5) -> list[dict]:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": limit,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return [
        {"date": o["date"], "value": float(o["value"])}
        for o in resp.json().get("observations", [])
        if o.get("value") and o["value"] != "."
    ]


def _fetch_fear_greed() -> int | None:
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        r.raise_for_status()
        return int(r.json()["data"][0]["value"])
    except Exception:
        return None


def macro_agent(state: dict, clients, fred_key: str) -> dict:
    ticker = state["ticker"]
    raw = {
        "dxy_latest": None, "dxy_5d_change": None,
        "fed_funds_rate": None, "treasury_10y": None, "treasury_2y": None,
        "yield_curve_2s10s": None, "fear_greed_index": _fetch_fear_greed(),
    }
    degraded = False

    if not fred_key:
        degraded = True
    else:
        try:
            dxy = _fetch_fred_series("DTWEXBGS", fred_key)
            ff = _fetch_fred_series("DFF", fred_key)
            t10 = _fetch_fred_series("DGS10", fred_key)
            t2 = _fetch_fred_series("DGS2", fred_key)
            if dxy:
                raw["dxy_latest"] = dxy[0]["value"]
                if len(dxy) >= 2:
                    raw["dxy_5d_change"] = round(dxy[0]["value"] - dxy[-1]["value"], 2)
            if ff:
                raw["fed_funds_rate"] = ff[0]["value"]
            if t10:
                raw["treasury_10y"] = t10[0]["value"]
            if t2:
                raw["treasury_2y"] = t2[0]["value"]
            if raw["treasury_10y"] is not None and raw["treasury_2y"] is not None:
                raw["yield_curve_2s10s"] = round(raw["treasury_10y"] - raw["treasury_2y"], 3)
        except Exception:
            degraded = True

    prompt = (
        f"You are a macro strategist analyzing {ticker}.\n\nData:\n"
        + "\n".join(f"- {k}: {v}" for k, v in raw.items())
        + ("\n\nFRED data unavailable; using only Fear & Greed.\n" if degraded else "")
        + "\nReference: rising DXY bearish for risk; high Fed funds bearish; "
          "inverted curve (2s10s<0) recessionary; Fear&Greed 0-25 extreme fear, 75-100 extreme greed.\n\n"
        "Respond with JSON ONLY:\n"
        '{"signal": "BULLISH"|"BEARISH"|"NEUTRAL", "confidence": 0.0..1.0, '
        '"summary": "one sentence", "section_markdown": "## Macro Backdrop\\n... 150-250 words ..."}'
    )
    try:
        resp = clients.reasoning.invoke(prompt)
        out = safe_parse_json(resp.content)
    except Exception as exc:
        return {"agent_signals": [AgentSignal(
            agent="macro", signal="NEUTRAL", confidence=0.0,
            summary="Macro LLM error",
            section_markdown="## Macro Backdrop\n_LLM unavailable._",
            raw_data=raw, degraded=True, error=str(exc)[:200],
        )]}

    return {"agent_signals": [AgentSignal(
        agent="macro",
        signal=out["signal"],
        confidence=float(out["confidence"]),
        summary=out["summary"],
        section_markdown=out.get("section_markdown") or "## Macro Backdrop\n_Section missing._",
        raw_data=raw,
        degraded=degraded,
        error=None,
    )]}
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_macro_agent.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/macro_agent.py tests/test_macro_agent.py
git commit -m "feat(agents): macro agent (FRED + Fear&Greed)"
```

### Task 2.6: Risk agent

**Files:**
- Create: `agents/risk_agent.py`
- Test: `tests/test_risk_agent.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_risk_agent.py
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from agents.risk_agent import risk_agent


def _series(n=90):
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0008, 0.015, n)
    prices = 100 * np.cumprod(1 + rets)
    return pd.DataFrame({"Close": prices})


def test_risk_agent_happy():
    fake_sonnet = MagicMock()
    fake_sonnet.invoke.return_value = MagicMock(content='{"signal": "NEUTRAL", "confidence": 0.5, "summary": "Moderate vol", "section_markdown": "## Risk Profile\\nText."}')
    clients = MagicMock(reasoning=fake_sonnet)

    fake_ticker = MagicMock()
    fake_ticker.info = {"beta": 1.1, "shortRatio": 2.5}

    with patch("agents.risk_agent.yf.download", side_effect=[_series(), pd.DataFrame({"Close": [22.1, 22.5, 21.9, 22.0, 22.3]})]), \
         patch("agents.risk_agent.yf.Ticker", return_value=fake_ticker):
        out = risk_agent({"ticker": "MSFT"}, clients)

    sig = out["agent_signals"][0]
    assert sig["agent"] == "risk"
    assert "annualized_vol_pct" in sig["raw_data"]
    assert sig["raw_data"]["vix"] == 22.3


def test_risk_agent_empty_data_degrades():
    clients = MagicMock()
    with patch("agents.risk_agent.yf.download", return_value=pd.DataFrame()):
        out = risk_agent({"ticker": "ZZZZ"}, clients)
    sig = out["agent_signals"][0]
    assert sig["degraded"] is True
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_risk_agent.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement**

```python
# agents/risk_agent.py
"""Risk specialist: yfinance returns + VIX -> Sonnet."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yfinance as yf

from agents import safe_parse_json
from state import AgentSignal

RISK_FREE_RATE = 0.04


def _degraded(reason: str, raw: dict | None = None, error: str | None = None) -> dict:
    return {"agent_signals": [AgentSignal(
        agent="risk", signal="NEUTRAL", confidence=0.0,
        summary=reason,
        section_markdown=f"## Risk Profile\n_Unavailable: {reason}_",
        raw_data=raw or {}, degraded=True, error=error,
    )]}


def risk_agent(state: dict, clients) -> dict:
    ticker = state["ticker"]
    try:
        data = yf.download(ticker, period="90d", interval="1d", progress=False)
        if data.empty or "Close" not in data.columns:
            return _degraded(f"No price data for {ticker}")

        close = data["Close"].squeeze()
        rets = close.pct_change().dropna()
        if len(rets) < 5:
            return _degraded("Insufficient return history")

        ann_vol = float(rets.std()) * np.sqrt(252) * 100
        cum = (1 + rets).cumprod()
        max_dd = float(((cum - cum.cummax()) / cum.cummax()).min()) * 100
        sharpe = float((rets.mean() * 252 - RISK_FREE_RATE) / (rets.std() * np.sqrt(252)))

        vix_df = yf.download("^VIX", period="5d", interval="1d", progress=False)
        vix = (
            round(float(vix_df["Close"].iloc[-1]), 2)
            if not vix_df.empty else None
        )

        info = {}
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:
            info = {}

        raw = {
            "annualized_vol_pct": round(ann_vol, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 2),
            "vix": vix,
            "beta": info.get("beta"),
            "short_ratio": info.get("shortRatio"),
        }

        prompt = (
            f"You are a risk analyst. Assess risk-adjusted attractiveness for {ticker}.\n\n"
            f"Data:\n{json.dumps(raw, indent=2)}\n\n"
            "Reference: high vol + negative Sharpe = bearish risk; low vol + positive Sharpe = bullish risk; "
            "VIX > 25 = elevated stress; beta > 1.3 = high market sensitivity.\n\n"
            "Respond with JSON ONLY:\n"
            '{"signal": "BULLISH"|"BEARISH"|"NEUTRAL", "confidence": 0.0..1.0, '
            '"summary": "one sentence", "section_markdown": "## Risk Profile\\n... 150-250 words ..."}'
        )
        resp = clients.reasoning.invoke(prompt)
        out = safe_parse_json(resp.content)

        return {"agent_signals": [AgentSignal(
            agent="risk",
            signal=out["signal"],
            confidence=float(out["confidence"]),
            summary=out["summary"],
            section_markdown=out.get("section_markdown") or "## Risk Profile\n_Section missing._",
            raw_data=raw,
            degraded=False,
            error=None,
        )]}
    except Exception as exc:
        return _degraded("Risk agent error", error=str(exc)[:200])
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_risk_agent.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/risk_agent.py tests/test_risk_agent.py
git commit -m "feat(agents): risk agent (vol/drawdown/sharpe/beta + VIX)"
```

---

## Phase 3: Supervisor + Synthesis

### Task 3.1: Supervisor agent (deterministic light QA)

**Files:**
- Create: `agents/supervisor_agent.py`
- Test: `tests/test_supervisor_agent.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_supervisor_agent.py
from agents.supervisor_agent import supervisor_agent


def _sig(agent, signal="NEUTRAL", confidence=0.5, degraded=False, section_markdown="## H\n" + "x" * 250, error=None):
    return {
        "agent": agent, "signal": signal, "confidence": confidence,
        "summary": f"{agent} ok", "section_markdown": section_markdown,
        "raw_data": {}, "degraded": degraded, "error": error,
    }


def test_supervisor_approves_when_clean():
    state = {"agent_signals": [
        _sig("price"), _sig("sentiment"), _sig("fundamentals"),
        _sig("macro"), _sig("risk"),
    ], "retry_round": 0}
    out = supervisor_agent(state)
    rev = out["supervisor_review"]
    assert rev["approved"] is True
    assert rev["retry_targets"] == []


def test_supervisor_flags_zero_confidence_degraded():
    state = {"agent_signals": [
        _sig("price"), _sig("sentiment", confidence=0.0, degraded=True, error="boom"),
        _sig("fundamentals"), _sig("macro"), _sig("risk"),
    ], "retry_round": 0}
    out = supervisor_agent(state)
    assert "sentiment" in out["supervisor_review"]["retry_targets"]


def test_supervisor_flags_short_section():
    state = {"agent_signals": [
        _sig("price", section_markdown="## H\nshort"),
        _sig("sentiment"), _sig("fundamentals"), _sig("macro"), _sig("risk"),
    ], "retry_round": 0}
    out = supervisor_agent(state)
    assert "price" in out["supervisor_review"]["retry_targets"]


def test_supervisor_flags_high_confidence_contradiction_lower_side():
    state = {"agent_signals": [
        _sig("price", signal="BULLISH", confidence=0.85),
        _sig("sentiment", signal="BEARISH", confidence=0.75),  # lower-conf side
        _sig("fundamentals"), _sig("macro"), _sig("risk"),
    ], "retry_round": 0}
    out = supervisor_agent(state)
    assert "sentiment" in out["supervisor_review"]["retry_targets"]
    assert "price" not in out["supervisor_review"]["retry_targets"]


def test_supervisor_no_retries_after_round_one():
    state = {"agent_signals": [
        _sig("price", confidence=0.0, degraded=True),
        _sig("sentiment"), _sig("fundamentals"), _sig("macro"), _sig("risk"),
    ], "retry_round": 1}
    out = supervisor_agent(state)
    assert out["supervisor_review"]["approved"] is True  # forced
    assert out["supervisor_review"]["retry_targets"] == []


def test_supervisor_sanity_violation_rsi():
    bad = _sig("price")
    bad["raw_data"] = {"rsi": 150.0}
    state = {"agent_signals": [bad, _sig("sentiment"), _sig("fundamentals"), _sig("macro"), _sig("risk")], "retry_round": 0}
    out = supervisor_agent(state)
    assert "price" in out["supervisor_review"]["retry_targets"]
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_supervisor_agent.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement**

```python
# agents/supervisor_agent.py
"""Light-QA supervisor. Cannot rewrite agent content or override verdicts.

Allowed actions:
  1. Flag agents whose output appears broken (zero confidence + degraded, errors).
  2. Flag direct contradictions (BULLISH vs BEARISH, both confidence ≥ 0.7).
  3. Flag short / empty sections (< 200 chars of section_markdown).
  4. Flag obvious data sanity violations.

The conditional edge after the supervisor reads `supervisor_review.retry_targets`.
On retry_round >= 1, the supervisor force-approves regardless.
"""

from __future__ import annotations

from math import isfinite
from typing import Iterable


CONTRADICTION_THRESHOLD = 0.7
MIN_SECTION_CHARS = 200


def _sanity_violations(agent: str, raw: dict) -> list[str]:
    issues: list[str] = []
    if agent == "price":
        rsi = raw.get("rsi")
        if isinstance(rsi, (int, float)) and (rsi < 0 or rsi > 100):
            issues.append(f"RSI out of range: {rsi}")
    if agent == "risk":
        v = raw.get("annualized_vol_pct")
        if isinstance(v, (int, float)) and (v < 0 or v > 500):
            issues.append(f"Volatility out of range: {v}%")
    for k, v in raw.items():
        if isinstance(v, float) and not isfinite(v):
            issues.append(f"Non-finite {k}")
    return issues


def supervisor_agent(state: dict) -> dict:
    signals = state.get("agent_signals", []) or []
    retry_round = int(state.get("retry_round", 0))

    critiques: dict[str, str] = {}

    # 1. broken outputs
    for s in signals:
        if s.get("error") or (s.get("degraded") and float(s.get("confidence", 0.0)) == 0.0):
            critiques.setdefault(s["agent"], "Hard error or zero-confidence degraded output.")
        if len((s.get("section_markdown") or "")) < MIN_SECTION_CHARS:
            critiques.setdefault(s["agent"], "Section narrative is missing or too short.")
        for issue in _sanity_violations(s.get("agent", ""), s.get("raw_data", {}) or {}):
            critiques.setdefault(s["agent"], f"Data sanity: {issue}")

    # 2. contradictions: BULLISH vs BEARISH both >= threshold confidence
    bulls = [s for s in signals if s.get("signal") == "BULLISH" and float(s.get("confidence", 0.0)) >= CONTRADICTION_THRESHOLD]
    bears = [s for s in signals if s.get("signal") == "BEARISH" and float(s.get("confidence", 0.0)) >= CONTRADICTION_THRESHOLD]
    if bulls and bears:
        # Flag the lower-confidence side (or the side with degraded=True if equal).
        def sort_key(s):
            return (s.get("degraded", False), float(s.get("confidence", 0.0)))
        loser = min(bulls + bears, key=sort_key)
        critiques.setdefault(loser["agent"], "High-confidence contradiction with another agent; please re-examine inputs.")

    if retry_round >= 1:
        # Force-approve; record critiques as notes for the report.
        notes = "Data quality: " + (
            "all sections complete." if not critiques else
            "; ".join(f"{a}: {c}" for a, c in critiques.items())
        )
        return {"supervisor_review": {
            "approved": True,
            "critiques": critiques,
            "retry_targets": [],
            "notes": notes,
        }}

    retry_targets = list(critiques.keys())
    approved = not retry_targets
    notes = (
        "Data quality: all sections complete."
        if approved else
        "Data quality: requesting one retry round on " + ", ".join(retry_targets) + "."
    )
    return {"supervisor_review": {
        "approved": approved,
        "critiques": critiques,
        "retry_targets": retry_targets,
        "notes": notes,
    }}
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_supervisor_agent.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/supervisor_agent.py tests/test_supervisor_agent.py
git commit -m "feat(agents): supervisor agent (deterministic light QA)"
```

### Task 3.2: Synthesis agent — verdict math + label rendering

**Files:**
- Create: `agents/synthesis_agent.py`
- Test: `tests/test_synthesis_agent.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_synthesis_agent.py
from unittest.mock import MagicMock

from agents.synthesis_agent import (
    compute_verdict_and_conviction,
    label_for,
    synthesis_agent,
)


def _s(agent, sig, conf, degraded=False):
    return {
        "agent": agent, "signal": sig, "confidence": conf,
        "summary": f"{agent}", "section_markdown": "## " + agent + "\n" + "x" * 220,
        "raw_data": {}, "degraded": degraded, "error": None,
    }


def test_label_table():
    assert label_for("BUY", "STRONG") == "Strong Buy"
    assert label_for("BUY", "STANDARD") == "Buy"
    assert label_for("BUY", "CAUTIOUS") == "Cautious Buy"
    assert label_for("HOLD", "STRONG") == "Hold (High Conviction)"
    assert label_for("HOLD", "STANDARD") == "Hold"
    assert label_for("HOLD", "CAUTIOUS") == "Hold (Mixed Signals)"
    assert label_for("SELL", "STRONG") == "Strong Sell"
    assert label_for("SELL", "STANDARD") == "Sell"
    assert label_for("SELL", "CAUTIOUS") == "Cautious Sell"


def test_strong_buy_consensus():
    sigs = [
        _s("price", "BULLISH", 0.8),
        _s("sentiment", "BULLISH", 0.8),
        _s("fundamentals", "BULLISH", 0.8),
        _s("macro", "BULLISH", 0.8),
        _s("risk", "NEUTRAL", 0.6),
    ]
    v, c, conf = compute_verdict_and_conviction(sigs)
    assert v == "BUY" and c == "STRONG"
    assert 0.65 < conf <= 1.0


def test_hold_when_mixed():
    sigs = [
        _s("price", "BULLISH", 0.7),
        _s("sentiment", "BEARISH", 0.7),
        _s("fundamentals", "NEUTRAL", 0.6),
        _s("macro", "BEARISH", 0.5),
        _s("risk", "BULLISH", 0.5),
    ]
    v, c, _ = compute_verdict_and_conviction(sigs)
    assert v == "HOLD"


def test_all_degraded_forces_hold_zero_conf():
    sigs = [_s(a, "NEUTRAL", 0.0, degraded=True) for a in ("price","sentiment","fundamentals","macro","risk")]
    v, c, conf = compute_verdict_and_conviction(sigs)
    assert v == "HOLD"
    assert conf == 0.0


def test_strong_sell_consensus():
    sigs = [
        _s("price", "BEARISH", 0.85),
        _s("sentiment", "BEARISH", 0.8),
        _s("fundamentals", "BEARISH", 0.8),
        _s("macro", "BEARISH", 0.7),
        _s("risk", "NEUTRAL", 0.5),
    ]
    v, c, _ = compute_verdict_and_conviction(sigs)
    assert v == "SELL" and c == "STRONG"


def test_synthesis_agent_assembles_report():
    sigs = [
        _s("price", "BULLISH", 0.7),
        _s("sentiment", "BULLISH", 0.6),
        _s("fundamentals", "BULLISH", 0.65),
        _s("macro", "NEUTRAL", 0.5),
        _s("risk", "NEUTRAL", 0.55),
    ]
    review = {"approved": True, "critiques": {}, "retry_targets": [], "notes": "Data quality: all sections complete."}
    state = {
        "ticker": "MSFT", "company_name": "Microsoft Corp",
        "agent_signals": sigs, "supervisor_review": review,
    }
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content='{"reasoning": "Three sentences of integrated reasoning that explain why the call holds together across price, sentiment, and fundamentals."}')
    clients = MagicMock(reasoning=fake_llm)

    out = synthesis_agent(state, clients)
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

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_synthesis_agent.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement**

```python
# agents/synthesis_agent.py
"""Synthesis: deterministic verdict math + LLM reasoning + report assembly."""

from __future__ import annotations

from datetime import datetime, timezone
from statistics import mean

from agents import safe_parse_json
from state import Conviction, Verdict


SCORE_MAP = {"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}
AGENT_ORDER = ["price", "sentiment", "fundamentals", "macro", "risk"]
SECTION_TITLES = {
    "price": "Technical Analysis",
    "sentiment": "News & Sentiment",
    "fundamentals": "Fundamentals",
    "macro": "Macro Backdrop",
    "risk": "Risk Profile",
}

LABEL_TABLE: dict[tuple[str, str], str] = {
    ("BUY", "STRONG"): "Strong Buy",
    ("BUY", "STANDARD"): "Buy",
    ("BUY", "CAUTIOUS"): "Cautious Buy",
    ("HOLD", "STRONG"): "Hold (High Conviction)",
    ("HOLD", "STANDARD"): "Hold",
    ("HOLD", "CAUTIOUS"): "Hold (Mixed Signals)",
    ("SELL", "STRONG"): "Strong Sell",
    ("SELL", "STANDARD"): "Sell",
    ("SELL", "CAUTIOUS"): "Cautious Sell",
}


def label_for(verdict: Verdict, conviction: Conviction) -> str:
    return LABEL_TABLE[(verdict, conviction)]


def compute_verdict_and_conviction(signals: list[dict]) -> tuple[Verdict, Conviction, float]:
    contributions = []
    for s in signals:
        weight = float(s.get("confidence", 0.0)) * (0.0 if s.get("degraded") else 1.0)
        contributions.append(SCORE_MAP.get(s.get("signal", "NEUTRAL"), 0) * weight)
    net = sum(contributions) / 5.0

    if net > 0.20:
        verdict: Verdict = "BUY"
    elif net < -0.20:
        verdict = "SELL"
    else:
        verdict = "HOLD"

    target_score = SCORE_MAP[{"BUY": "BULLISH", "SELL": "BEARISH", "HOLD": "NEUTRAL"}[verdict]]
    agreeing = [
        s for s in signals
        if not s.get("degraded") and SCORE_MAP.get(s.get("signal", "NEUTRAL"), 0) == target_score
    ]
    agree_count = len(agreeing)
    avg_conf = mean(float(s["confidence"]) for s in agreeing) if agreeing else 0.0

    if agree_count >= 4 and avg_conf >= 0.75:
        conviction: Conviction = "STRONG"
    elif agree_count >= 3 and avg_conf >= 0.55:
        conviction = "STANDARD"
    else:
        conviction = "CAUTIOUS"

    non_degraded = [s for s in signals if not s.get("degraded")]
    if not non_degraded:
        return "HOLD", "CAUTIOUS", 0.0
    base = mean(float(s["confidence"]) for s in non_degraded)
    final_conf = round(base * (len(non_degraded) / 5.0), 3)
    return verdict, conviction, final_conf


def _section_for(signals: list[dict], agent: str) -> str:
    for s in signals:
        if s.get("agent") == agent:
            md = (s.get("section_markdown") or "").strip()
            if md and md.startswith("## "):
                return md
            return f"## {SECTION_TITLES[agent]}\n{md or '_Section unavailable._'}"
    return f"## {SECTION_TITLES[agent]}\n_Agent did not run._"


def _disclaimer_block() -> str:
    return (
        "## Disclaimer\n"
        "This report is generated by an AI system and is not financial advice. "
        "Data sources: yfinance (price/risk), Tavily (news), SEC EDGAR (fundamentals), "
        "FRED + Fear & Greed Index (macro). Numbers are point-in-time at run; "
        "agent confidences and the final verdict reflect machine inference and may be wrong. "
        "Do your own research before any investment decision."
    )


def synthesis_agent(state: dict, clients) -> dict:
    ticker = state["ticker"]
    company = state.get("company_name") or ticker
    signals = state.get("agent_signals", []) or []
    review = state.get("supervisor_review") or {"notes": "", "critiques": {}}

    verdict, conviction, confidence = compute_verdict_and_conviction(signals)
    label = label_for(verdict, conviction)

    rollup = "\n".join(
        f"- {s['agent'].upper()}: {s.get('signal', 'NEUTRAL')} "
        f"(conf {float(s.get('confidence', 0.0)):.2f}{', degraded' if s.get('degraded') else ''}) — {s.get('summary', '')}"
        for s in signals
    )

    prompt = (
        f"You are the chief investment strategist. The deterministic vote across five specialists "
        f"yielded **{label}** (verdict={verdict}, conviction={conviction}, confidence={confidence:.2f}) "
        f"for {ticker} ({company}).\n\n"
        f"Specialist rollup:\n{rollup}\n\n"
        f"Supervisor notes: {review.get('notes', '')}\n\n"
        "Write 3–5 sentences of synthesis explaining why the call holds together (or why conflicts "
        "leave it cautious). Reference at least two specialists by name. Do NOT change the verdict. "
        "Respond with JSON ONLY: {\"reasoning\": \"...\"}"
    )
    try:
        resp = clients.reasoning.invoke(prompt)
        out = safe_parse_json(resp.content)
        reasoning = out["reasoning"]
    except Exception as exc:
        reasoning = (
            f"Synthesis LLM call failed ({str(exc)[:100]}). "
            f"Verdict {label} is derived deterministically from the specialist signals above."
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    exec_summary = (
        f"# {ticker} — {company}\n\n"
        f"_Generated {timestamp}_\n\n"
        f"## Executive Summary\n"
        f"**Verdict: {label}** · Confidence: {confidence * 100:.0f}%\n\n"
        f"- Verdict: {verdict}\n"
        f"- Conviction: {conviction}\n"
        f"- {review.get('notes', '')}\n"
    )

    sections = [
        exec_summary,
        _section_for(signals, "price"),
        _section_for(signals, "sentiment"),
        _section_for(signals, "fundamentals"),
        _section_for(signals, "macro"),
        _section_for(signals, "risk"),
        f"## Synthesis & Final Verdict\n**{label}** — Confidence {confidence * 100:.0f}%.\n\n{reasoning}",
        _disclaimer_block(),
    ]
    final_report = "\n\n".join(sections)

    return {
        "final_verdict": verdict,
        "final_conviction": conviction,
        "final_confidence": confidence,
        "final_reasoning": reasoning,
        "final_report": final_report,
    }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_synthesis_agent.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/synthesis_agent.py tests/test_synthesis_agent.py
git commit -m "feat(agents): synthesis (verdict math + report assembly)"
```

---

## Phase 4: Graph wiring

### Task 4.1: `graph.py` — fan-out / supervisor / conditional retry / synthesis

**Files:**
- Create: `graph.py`
- Test: `tests/test_graph.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_graph.py
from unittest.mock import patch

from graph import build_graph
from state import MarketMindState


class _StubAgent:
    def __init__(self, name, signal="BULLISH", confidence=0.7, **kw):
        self.name = name
        self.signal = signal
        self.confidence = confidence

    def __call__(self, state, *args, **kwargs):
        return {"agent_signals": [{
            "agent": self.name, "signal": self.signal, "confidence": self.confidence,
            "summary": f"{self.name} stub", "section_markdown": "## H\n" + "x" * 250,
            "raw_data": {}, "degraded": False, "error": None,
        }]}


def test_graph_builds_and_runs_end_to_end():
    # Patch each specialist + orchestrator + supervisor + synthesis at the graph wiring layer.
    with patch("graph.orchestrator", return_value={"ticker": "MSFT", "cik": "0000789019", "company_name": "Microsoft Corp"}), \
         patch("graph.price_agent", _StubAgent("price")), \
         patch("graph.sentiment_agent", _StubAgent("sentiment")), \
         patch("graph.fundamentals_agent", _StubAgent("fundamentals")), \
         patch("graph.macro_agent", _StubAgent("macro")), \
         patch("graph.risk_agent", _StubAgent("risk")):
        g = build_graph(
            llm_clients=object(),
            tavily_key="tvly", fred_key="frd",
        )
        init: MarketMindState = {
            "ticker": "MSFT", "company_name": None, "cik": None,
            "agent_signals": [], "retry_round": 0, "supervisor_review": None,
            "final_verdict": None, "final_conviction": None,
            "final_confidence": None, "final_reasoning": None, "final_report": None,
        }
        result = g.invoke(init)

    assert result["final_verdict"] == "BUY"
    assert result["final_report"].startswith("# MSFT")
    # All five specialists wrote.
    agents_seen = {s["agent"] for s in result["agent_signals"]}
    assert agents_seen == {"price", "sentiment", "fundamentals", "macro", "risk"}
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_graph.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement**

```python
# graph.py
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_graph.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add graph.py tests/test_graph.py
git commit -m "feat(graph): fan-out + supervisor + conditional retry + synthesis wiring"
```

### Task 4.2: Live smoke run (manual)

**Files:**
- (no source changes; validation only)

- [ ] **Step 1: Create a small smoke runner**

```python
# scripts/smoke_run.py
"""Manual smoke test for MarketMind v2. Runs against real services.

Usage:
    ANTHROPIC_API_KEY=... TAVILY_API_KEY=... FRED_API_KEY=... \
    python scripts/smoke_run.py MSFT
"""
from __future__ import annotations

import os
import sys

from agents import build_llm_clients
from graph import build_graph


def main(ticker: str) -> None:
    anthropic_key = os.environ["ANTHROPIC_API_KEY"]
    tavily_key = os.environ["TAVILY_API_KEY"]
    fred_key = os.environ.get("FRED_API_KEY", "")
    clients = build_llm_clients(anthropic_key)
    g = build_graph(clients, tavily_key=tavily_key, fred_key=fred_key)
    init = {
        "ticker": ticker, "company_name": None, "cik": None,
        "agent_signals": [], "retry_round": 0, "supervisor_review": None,
        "final_verdict": None, "final_conviction": None,
        "final_confidence": None, "final_reasoning": None, "final_report": None,
    }
    out = g.invoke(init)
    print("VERDICT:", out["final_verdict"], out["final_conviction"], "conf", out["final_confidence"])
    print("------ REPORT ------")
    print(out["final_report"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "MSFT")
```

- [ ] **Step 2: Run against MSFT**

```bash
ANTHROPIC_API_KEY=... TAVILY_API_KEY=... FRED_API_KEY=... python scripts/smoke_run.py MSFT
```

Expected: prints VERDICT line and a multi-section markdown report. Total wall-time under ~30s.

- [ ] **Step 3: Run edge cases**

```bash
python scripts/smoke_run.py SPY     # ETF — fundamentals should degrade
python scripts/smoke_run.py SHEL    # foreign issuer (ADR) — fundamentals should degrade
python scripts/smoke_run.py NVDA    # high-coverage normal stock
```

Expected: each completes; degraded sections clearly labeled in the report.

- [ ] **Step 4: Commit smoke runner**

```bash
git add scripts/smoke_run.py
git commit -m "chore: smoke runner script for marketmind v2"
```

---

## Phase 5: UI rewrite

### Task 5.1: Tighten `ratelimit.py` for v2 cadence

**Files:**
- Modify: `ratelimit.py`

- [ ] **Step 1: Replace DEFAULT_LIMITS**

```python
# ratelimit.py — replace the DEFAULT_LIMITS block

DEFAULT_LIMITS: Dict[str, Iterable[Tuple[int, int]]] = {
    # One full multi-agent run per minute, capped per hour.
    "analyze": ((60, 1), (3600, 20)),
}
```

- [ ] **Step 2: Run existing rate-limit tests if any (or skip)**

If there are no existing tests for `ratelimit.py`, skip.

- [ ] **Step 3: Commit**

```bash
git add ratelimit.py
git commit -m "ratelimit: v2 cadence — 1 analyze/min, 20/hour"
```

### Task 5.2: Rewrite `app.py`

**Files:**
- Rewrite: `app.py`

- [ ] **Step 1: Replace contents**

```python
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

    os.environ["SEC_USER_AGENT"] = state.get("sec_user_agent") or "MarketMind/2.0 contact@marketmind.local"

    clients = build_llm_clients(state["anthropic_key"])
    graph = build_graph(clients, tavily_key=state["tavily_key"], fred_key=state.get("fred_key", ""))

    init = {
        "ticker": ticker, "company_name": None, "cik": None,
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
```

- [ ] **Step 2: Local run**

```bash
ANTHROPIC_API_KEY=... TAVILY_API_KEY=... FRED_API_KEY=... ALLOW_ENV_KEYS=1 python app.py
```

Open the printed URL. Enter `MSFT`, click Analyze. Verify pills go pending → running → done; sections stream in; final report renders.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(ui): rewrite app.py for v2 multi-agent pipeline"
```

---

## Phase 6: Cleanup

### Task 6.1: Delete legacy files

**Files:**
- Delete: `agent.py`, `tools.py`, `rag.py`, `data/`

- [ ] **Step 1: Delete files**

```bash
git rm agent.py tools.py rag.py
git rm -r data
```

- [ ] **Step 2: Verify nothing imports them**

```bash
grep -rE "from (agent|tools|rag)\\b|import (agent|tools|rag)\\b" --include="*.py" . || echo "clean"
```

Expected: prints `clean`. (No file in the new code path imports the deleted modules.)

- [ ] **Step 3: Run all tests**

```bash
pytest -v
```

Expected: all passing.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: remove legacy single-agent + rag pipeline"
```

### Task 6.2: Update README + .env.example

**Files:**
- Modify: `README.md`
- Modify: `.env.example`

- [ ] **Step 1: Update `.env.example`**

```
# Local development only. On a public HF Space, leave keys empty and let users
# paste their own keys via the BYO-key panel.

ALLOW_ENV_KEYS=1

ANTHROPIC_API_KEY=sk-ant-your-key
TAVILY_API_KEY=tvly-your-key
FRED_API_KEY=your-fred-key       # optional — enables full Macro

# SEC EDGAR identification (the SEC requires a real contact in the User-Agent).
SEC_USER_AGENT=MarketMind/2.0 yourname@example.com
```

- [ ] **Step 2: Replace README.md top section**

Replace the project description, architecture, and run-instructions sections of `README.md` with the v2 description (multi-agent, ticker-driven, EDGAR-backed). Keep "Known pins" and deployment notes. Reference the spec at `docs/superpowers/specs/2026-05-01-marketmind-v2-design.md`.

(The exact prose should be drafted by the implementing agent; cap at ~200 lines.)

- [ ] **Step 3: Commit**

```bash
git add README.md .env.example
git commit -m "docs: README + .env.example for v2"
```

---

## Phase 7: Deploy

### Task 7.1: Final test sweep

- [ ] **Step 1: Run full test suite**

```bash
pytest -v
```

Expected: all green.

- [ ] **Step 2: Run smoke against four tickers**

```bash
python scripts/smoke_run.py MSFT
python scripts/smoke_run.py NVDA
python scripts/smoke_run.py SPY
python scripts/smoke_run.py SHEL
```

Expected: all complete; SPY and SHEL show degraded Fundamentals.

### Task 7.2: Push and deploy

- [ ] **Step 1: Open PR**

```bash
git push -u origin feat/marketmind-v2
gh pr create --title "MarketMind v2: multi-agent equity pipeline" --body "$(cat <<'EOF'
## Summary
- Replaces single-agent chat with 5-specialist parallel pipeline + supervisor + synthesis
- Drops PDF/Chroma RAG; adds SEC EDGAR fundamentals
- Streams updates per agent in Gradio
- Verdict + conviction (Strong Buy / Buy / Cautious Buy / Hold / ... / Strong Sell)

## Test plan
- [ ] `pytest -v` green
- [ ] Smoke runs on MSFT, NVDA, SPY (ETF), SHEL (ADR)
- [ ] HF Space build succeeds on Python 3.12

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 2: After review, merge and let HF Space rebuild**

```bash
gh pr merge --squash
```

Expected: HF Space rebuilds successfully and the new UI is live.

---

## Self-Review Checklist (run by author after writing this plan)

**Spec coverage** — every section of the spec has at least one task:

| Spec section | Plan task |
|---|---|
| §3 Architecture | Task 4.1 (graph wiring) |
| §4 State | Task 1.2 (state.py) |
| §5.1 Price | Task 2.2 |
| §5.2 Sentiment | Task 2.3 |
| §5.3 Fundamentals | Task 2.4 (+ §6 EDGAR via 1.4–1.8) |
| §5.4 Macro | Task 2.5 |
| §5.5 Risk | Task 2.6 |
| §5.6 Supervisor | Task 3.1 |
| §5.7 Synthesis (label table, math) | Task 3.2 |
| §6 EDGAR module | Tasks 1.4–1.8 |
| §7 UI (BYO-key, streaming, single-ticker) | Task 5.2 |
| §7.4 Rate-limit retune | Task 5.1 |
| §8 File layout (deletes) | Task 6.1 |
| §9 FRs (graceful degradation, key isolation, no env on public) | Tasks 2.x degraded paths + 1.3 factory + 5.2 |
| §10 Dependencies | Task 1.1 |
| §11 Risk: concurrency cap | Task 1.3 (semaphore) |
| §11 Risk: SEC politeness | Task 1.4 (`SEC_POLITENESS_SLEEP`) |
| §13 Future work | Captured in `tasks/todo.md`; out of scope here |

No gaps identified.

**Placeholders:** none. README prose in Task 6.2 is intentionally drafted by the implementing agent rather than dictated; this is a deliberate creative latitude, not a placeholder.

**Type consistency:** `AgentSignal` keys, `MarketMindState` keys, `SupervisorReview` keys, `LLMClients.reasoning|fast`, `Filing` fields — all referenced consistently across tasks. Verified by re-grep:
- `section_markdown` — Tasks 2.2–2.6, 3.1, 3.2, 5.2 ✓
- `degraded` — Tasks 2.x, 3.1, 3.2 ✓
- `agent` field values: `price | sentiment | fundamentals | macro | risk` — consistent ✓
- `retry_targets` only inside `supervisor_review` — Tasks 3.1, 4.1 ✓ (matches spec §4 fix)

**Scope:** focused on the v2.0 cut. Future v2.1/v2.2/v2.3 items remain in `tasks/todo.md`.
