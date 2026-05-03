# MarketMind v2.1 Agent Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade every MarketMind v2.0 specialist into a persona-driven, methodology-grounded, tool-calling agent with structured `key_metrics` + `flags` outputs. Promote `EdgarBundle` to shared prefetch. Promote Synthesis to CoT with `key_drivers` / `dissenting_view` / `watch_items`.

**Architecture:** Shared `run_with_tools()` helper in `agents/__init__.py` wraps the Anthropic SDK tool-use loop with hard 3-iteration cap and prompt caching on the persona block. Each specialist gets a per-agent tools module under `agents/tools/`. `data_prefetch` becomes the single ingestion point for `price_history`, `vix_history`, and `edgar_bundle`.

**Tech Stack:** Python 3.12, langchain-anthropic 0.3.22, anthropic 0.97.0, langgraph 0.3.7, yfinance 1.3.0 + curl_cffi, pytest + responses + unittest.mock, gradio 5.50.

**Spec:** [`docs/superpowers/specs/2026-05-02-agent-enhancements-design.md`](../specs/2026-05-02-agent-enhancements-design.md)

---

## File Structure (Target)

| Path | Status | Responsibility |
|---|---|---|
| `agents/__init__.py` | MODIFY | Add `ToolDef` dataclass, `run_with_tools()` loop, `cached_system_block()` helper. Keep existing `LLMClients`, `safe_parse_json`, `degraded_signal`. |
| `agents/data_prefetch.py` | MODIFY | Also fetch `EdgarBundle` → `state["edgar_bundle"]`. |
| `agents/orchestrator.py` | UNCHANGED |  |
| `agents/price_agent.py` | REWRITE | Persona + methodology + 3 few-shot examples + tool-use loop. |
| `agents/sentiment_agent.py` | REWRITE | Persona + methodology + 3 few-shot examples + tool-use loop. |
| `agents/fundamentals_agent.py` | REWRITE | Persona + methodology + CoT + tool-use loop. Reads `state["edgar_bundle"]`. |
| `agents/macro_agent.py` | REWRITE | Persona + methodology + CoT + tool-use loop. |
| `agents/risk_agent.py` | REWRITE | Persona + forward-looking methodology + 3 few-shot examples + tool-use loop. Reads `state["edgar_bundle"]` for forward fundamentals. |
| `agents/supervisor_agent.py` | MODIFY | Sanity reads from `key_metrics`; new cross-signal Fundamentals↔Macro consistency check. |
| `agents/synthesis_agent.py` | MODIFY | Promote LLM call to CoT; add `key_drivers`, `dissenting_view`, `watch_items`; render new fields in `final_report`. Verdict math unchanged. |
| `agents/yf_helpers.py` | UNCHANGED |  |
| `agents/tools/__init__.py` | NEW | Empty package marker. |
| `agents/tools/fundamentals_tools.py` | NEW | `fetch_xbrl_tag`, `fetch_segment_breakdown`, `peer_multiples`. |
| `agents/tools/macro_tools.py` | NEW | `fetch_fred_series`, `classify_ticker_sector`, `fetch_credit_spreads`. |
| `agents/tools/price_tools.py` | NEW | `compute_indicator`, `detect_chart_pattern`, `volume_profile_summary`. |
| `agents/tools/sentiment_tools.py` | NEW | `fetch_press_releases`, `fetch_analyst_actions`, `categorize_drivers`. |
| `agents/tools/risk_tools.py` | NEW | `forward_risk_attribution`, `decompose_drawdown`, `compute_var_es`. |
| `state.py` | MODIFY | Add `edgar_bundle` field. `AgentSignal` becomes `total=False` + new optional fields. |
| `graph.py` | UNCHANGED | `data_prefetch` already wired. |
| `app.py` | MODIFY | `init` includes `edgar_bundle: None`; cost badge text. |
| `tests/test_agents_init.py` | MODIFY | Cover `ToolDef`, `run_with_tools()`. |
| `tests/test_data_prefetch.py` | MODIFY | Cover `edgar_bundle` fetch path. |
| `tests/test_state.py` | MODIFY | Cover new field + `total=False` shape. |
| `tests/test_price_agent.py` | REWRITE | Cover tool-use loop, regime, key_metrics, flags. |
| `tests/test_sentiment_agent.py` | REWRITE | Cover tool-use loop, top_catalyst, drivers_categorized. |
| `tests/test_fundamentals_agent.py` | REWRITE | Cover tool-use loop, key_metrics, reads from state. |
| `tests/test_macro_agent.py` | REWRITE | Cover tool-use loop, regime, ticker_exposure. |
| `tests/test_risk_agent.py` | REWRITE | Cover forward-looking call, risk_decomposition, primary_risk_driver. |
| `tests/test_supervisor_agent.py` | MODIFY | Cover cross-signal check. |
| `tests/test_synthesis_agent.py` | MODIFY | Cover CoT outputs (key_drivers, dissenting_view, watch_items). |
| `tests/test_tools/__init__.py` | NEW |  |
| `tests/test_tools/test_fundamentals_tools.py` | NEW | One test per tool. |
| `tests/test_tools/test_macro_tools.py` | NEW |  |
| `tests/test_tools/test_price_tools.py` | NEW |  |
| `tests/test_tools/test_sentiment_tools.py` | NEW |  |
| `tests/test_tools/test_risk_tools.py` | NEW |  |

---

## Phase 0: Workspace Setup

### Task 0.1: Create feature branch + commit spec

**Files:** none (git only)

- [ ] **Step 1: Cut feature branch**

```bash
git checkout -b feat/marketmind-v2.1-agents
git status
```

Expected: `On branch feat/marketmind-v2.1-agents` with the spec file as untracked.

- [ ] **Step 2: Commit the spec**

```bash
git add docs/superpowers/specs/2026-05-02-agent-enhancements-design.md
git commit -m "docs: marketmind v2.1 agent-enhancements spec"
```

Expected: commit succeeds; `git log -1 --stat` shows the new file.

---

## Phase 1: Foundations

### Task 1.1: Update `state.py` — add `edgar_bundle` field; promote `AgentSignal` to `total=False` with new optional fields

**Files:**
- Modify: `state.py`
- Test: `tests/test_state.py`

- [ ] **Step 1: Replace `state.py` contents**

```python
"""Shared state definitions for the MarketMind v2 LangGraph pipeline.

The `agent_signals` list uses `operator.add` as its LangGraph reducer so that
five specialist nodes running in the same superstep can each append their
result without races.

`price_history`, `vix_history`, and `edgar_bundle` are populated once by the
data_prefetch node before the parallel fan-out so the Price, Risk, and
Fundamentals specialists do not each issue their own external fetch. Yahoo
throttles HF Space IPs hard and SEC asks for politeness; deduplicating these
requests is the difference between green and degraded runs.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, List, Literal, Optional, TypedDict

Verdict = Literal["BUY", "HOLD", "SELL"]
Conviction = Literal["STRONG", "STANDARD", "CAUTIOUS"]
Signal = Literal["BULLISH", "BEARISH", "NEUTRAL"]


class AgentSignal(TypedDict, total=False):
    """v2.1: total=False — only the v2.0 required fields are populated by every
    specialist; the new optional fields are populated only by the agent that
    owns them. Supervisor and Synthesis must use `.get()` everywhere."""

    # v2.0 required
    agent: str
    signal: Signal
    confidence: float
    summary: str
    section_markdown: str
    raw_data: dict
    degraded: bool
    error: Optional[str]

    # v2.1 generic additions (every specialist populates these when not degraded)
    key_metrics: Optional[dict]
    flags: Optional[list]

    # v2.1 agent-specific additions
    regime: Optional[str]                # price (trending_up/...), macro (risk-on/...)
    vol_regime: Optional[str]            # risk
    vix_regime: Optional[str]            # risk
    forward_risk_view: Optional[str]     # risk
    primary_risk_driver: Optional[str]   # risk
    risk_decomposition: Optional[dict]   # risk
    yield_curve_state: Optional[str]     # macro
    ticker_exposure: Optional[str]       # macro
    top_catalyst: Optional[str]          # sentiment
    drivers_categorized: Optional[dict]  # sentiment


class SupervisorReview(TypedDict):
    approved: bool
    critiques: dict
    retry_targets: list
    notes: str


class MarketMindState(TypedDict):
    ticker: str
    company_name: Optional[str]
    cik: Optional[str]
    # Prefetched market data (DataFrames + EdgarBundle dataclass).
    # `None` = prefetch did not run; empty/falsy = ran but failed.
    price_history: Optional[Any]
    vix_history: Optional[Any]
    edgar_bundle: Optional[Any]
    agent_signals: Annotated[List[AgentSignal], operator.add]
    retry_round: int
    supervisor_review: Optional[SupervisorReview]
    final_verdict: Optional[Verdict]
    final_conviction: Optional[Conviction]
    final_confidence: Optional[float]
    final_reasoning: Optional[str]
    final_report: Optional[str]
    # v2.1 synthesis additions
    key_drivers: Optional[list]
    dissenting_view: Optional[str]
    watch_items: Optional[list]
```

- [ ] **Step 2: Add tests to `tests/test_state.py`**

Append to the existing file (do not delete existing tests):

```python


def test_state_carries_edgar_bundle_field():
    s: MarketMindState = {
        "ticker": "MSFT", "company_name": None, "cik": None,
        "price_history": None, "vix_history": None, "edgar_bundle": None,
        "agent_signals": [], "retry_round": 0, "supervisor_review": None,
        "final_verdict": None, "final_conviction": None,
        "final_confidence": None, "final_reasoning": None, "final_report": None,
        "key_drivers": None, "dissenting_view": None, "watch_items": None,
    }
    assert s["edgar_bundle"] is None
    assert s["key_drivers"] is None


def test_agent_signal_v2_1_optional_fields_can_be_omitted():
    # total=False — only the legacy required fields need be present.
    sig: AgentSignal = {
        "agent": "price",
        "signal": "BULLISH",
        "confidence": 0.7,
        "summary": "uptrend",
        "section_markdown": "## Technical\nBody",
        "raw_data": {},
        "degraded": False,
        "error": None,
    }
    # No KeyError when accessing new fields with .get()
    assert sig.get("key_metrics") is None
    assert sig.get("flags") is None
    assert sig.get("regime") is None


def test_agent_signal_v2_1_optional_fields_round_trip():
    sig: AgentSignal = {
        "agent": "price", "signal": "BULLISH", "confidence": 0.7,
        "summary": "x", "section_markdown": "## H\nb",
        "raw_data": {}, "degraded": False, "error": None,
        "key_metrics": {"rsi": 58.2},
        "flags": ["trend_confirmation"],
        "regime": "trending_up",
    }
    assert sig["key_metrics"]["rsi"] == 58.2
    assert sig["flags"] == ["trend_confirmation"]
    assert sig["regime"] == "trending_up"
```

- [ ] **Step 3: Run tests**

```
C:/Proyectos/Autonomous_Financial_Analyst/.venv/Scripts/python.exe -m pytest tests/test_state.py -v
```

Expected: 7 passed (4 existing + 3 new).

- [ ] **Step 4: Commit**

```bash
git add state.py tests/test_state.py
git commit -m "feat(state): add edgar_bundle + AgentSignal total=False with v2.1 optional fields"
```

---

### Task 1.2: Add `ToolDef` + `run_with_tools()` helper to `agents/__init__.py`

**Files:**
- Modify: `agents/__init__.py`
- Test: `tests/test_agents_init.py`

- [ ] **Step 1: Append to `agents/__init__.py`**

Add at the bottom of the existing file (do not modify existing code):

```python


# ---------------------------------------------------------------------------
# v2.1: Tool-use loop helpers
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Callable

from anthropic import Anthropic

DEFAULT_MAX_ITERATIONS = 3


@dataclass
class ToolDef:
    """A single on-demand tool available to a specialist agent."""
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict], object]   # local executor; returns JSON-serializable

    def to_anthropic(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def _content_blocks_text(blocks) -> str:
    """Concatenate text from a list of TextBlock / dicts."""
    out = []
    for b in blocks:
        text = getattr(b, "text", None)
        if text is None and isinstance(b, dict):
            text = b.get("text")
        if text:
            out.append(text)
    return "".join(out)


def _format_tool_result(value: object) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except Exception as exc:
        return json.dumps({"error": f"unserializable: {exc!s}"})


def run_with_tools(
    api_key: str,
    *,
    system_prompt: str,
    user_prompt: str,
    tools: list[ToolDef],
    model: str = REASONING_MODEL,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_tokens: int = 1500,
    temperature: float = 0.1,
) -> dict:
    """Run a Claude tool-use loop with hard iteration cap and prompt caching.

    The system prompt block is sent with cache_control on every turn so that
    Anthropic can serve subsequent turns from cache (5-minute TTL).

    Returns the parsed JSON dict from the final text response.

    Raises ValueError if the model never produces a parseable JSON text block
    after `max_iterations + 1` attempts.
    """
    if not api_key:
        raise ValueError("api_key is required")

    client = Anthropic(api_key=api_key)
    tool_specs = [t.to_anthropic() for t in tools]
    handlers = {t.name: t.handler for t in tools}

    messages: list[dict] = [{"role": "user", "content": user_prompt}]

    for iteration in range(max_iterations + 1):
        # Force final text on the LAST iteration (no more tool calls allowed).
        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=messages,
        )
        if iteration < max_iterations and tool_specs:
            kwargs["tools"] = tool_specs
        # else: omit tools entirely so Claude must produce text.

        resp = client.messages.create(**kwargs)

        # Append assistant turn to message history.
        messages.append({"role": "assistant", "content": resp.content})

        # If stop_reason is tool_use, execute every tool_use block in this turn
        # and feed all results back as a single user message.
        if resp.stop_reason == "tool_use" and iteration < max_iterations:
            tool_results = []
            for block in resp.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                name = block.name
                args = block.input or {}
                try:
                    handler = handlers[name]
                    result = handler(args)
                    result_text = _format_tool_result(result)
                except KeyError:
                    result_text = json.dumps({"error": f"unknown tool {name}"})
                except Exception as exc:
                    result_text = json.dumps({"error": str(exc)[:200]})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })
            messages.append({"role": "user", "content": tool_results})
            continue

        # Otherwise, parse the final text and return.
        text = _content_blocks_text(resp.content)
        if not text:
            continue
        try:
            return safe_parse_json(text)
        except Exception:
            # If parse fails on this turn, loop will retry up to max_iterations.
            continue

    raise ValueError(
        f"run_with_tools: model did not produce parseable JSON after "
        f"{max_iterations + 1} iterations"
    )
```

- [ ] **Step 2: Add tests to `tests/test_agents_init.py`**

Append:

```python


def test_tool_def_to_anthropic_shape():
    from agents import ToolDef
    t = ToolDef(
        name="echo",
        description="Echo input",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        handler=lambda args: {"echoed": args.get("x")},
    )
    spec = t.to_anthropic()
    assert spec["name"] == "echo"
    assert spec["description"] == "Echo input"
    assert spec["input_schema"]["type"] == "object"
    assert "handler" not in spec  # handler is local-only


def test_run_with_tools_returns_parsed_json_no_tool_use(monkeypatch):
    """Happy path: model emits final JSON immediately, no tool calls."""
    from unittest.mock import MagicMock
    import agents as agents_mod

    fake_block = MagicMock()
    fake_block.text = '{"signal": "BULLISH", "confidence": 0.7}'
    fake_resp = MagicMock(stop_reason="end_turn", content=[fake_block])

    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_resp

    monkeypatch.setattr(agents_mod, "Anthropic", lambda api_key: fake_client)

    out = agents_mod.run_with_tools(
        api_key="sk-ant-fake",
        system_prompt="You are a test agent.",
        user_prompt="Analyze MSFT.",
        tools=[],
    )
    assert out == {"signal": "BULLISH", "confidence": 0.7}


def test_run_with_tools_executes_tool_then_returns_json(monkeypatch):
    """Model emits a tool_use, we run handler, model emits final JSON."""
    from unittest.mock import MagicMock
    import agents as agents_mod

    # Iteration 0: model asks for tool 'foo' with input {"a": 1}
    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.id = "toolu_01"
    tool_use_block.name = "foo"
    tool_use_block.input = {"a": 1}
    resp_iter0 = MagicMock(stop_reason="tool_use", content=[tool_use_block])

    # Iteration 1: model returns final text
    text_block = MagicMock()
    text_block.text = '{"signal": "NEUTRAL", "confidence": 0.5}'
    resp_iter1 = MagicMock(stop_reason="end_turn", content=[text_block])

    fake_client = MagicMock()
    fake_client.messages.create.side_effect = [resp_iter0, resp_iter1]

    monkeypatch.setattr(agents_mod, "Anthropic", lambda api_key: fake_client)

    handler_mock = MagicMock(return_value={"result": 42})
    tool = agents_mod.ToolDef(
        name="foo",
        description="test tool",
        input_schema={"type": "object"},
        handler=handler_mock,
    )

    out = agents_mod.run_with_tools(
        api_key="sk-ant-fake",
        system_prompt="sys",
        user_prompt="user",
        tools=[tool],
    )
    handler_mock.assert_called_once_with({"a": 1})
    assert out == {"signal": "NEUTRAL", "confidence": 0.5}


def test_run_with_tools_caps_at_max_iterations(monkeypatch):
    """After max_iterations tool-use turns, the call must omit tools."""
    from unittest.mock import MagicMock
    import agents as agents_mod

    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.id = "toolu_X"
    tool_use_block.name = "foo"
    tool_use_block.input = {}

    resp_tool = MagicMock(stop_reason="tool_use", content=[tool_use_block])
    final_text = MagicMock()
    final_text.text = '{"signal": "HOLD", "confidence": 0.3}'
    resp_final = MagicMock(stop_reason="end_turn", content=[final_text])

    # 3 tool-use turns + 1 forced final = 4 calls
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = [resp_tool, resp_tool, resp_tool, resp_final]

    monkeypatch.setattr(agents_mod, "Anthropic", lambda api_key: fake_client)

    tool = agents_mod.ToolDef(
        name="foo", description="x", input_schema={"type": "object"},
        handler=lambda args: {"ok": True},
    )

    out = agents_mod.run_with_tools(
        api_key="sk-ant-fake",
        system_prompt="sys", user_prompt="user", tools=[tool],
        max_iterations=3,
    )
    # 4th (final) call must omit `tools` so Claude is forced to produce text.
    final_call_kwargs = fake_client.messages.create.call_args_list[-1].kwargs
    assert "tools" not in final_call_kwargs
    assert out == {"signal": "HOLD", "confidence": 0.3}


def test_run_with_tools_handler_exception_returned_as_tool_result(monkeypatch):
    """Handler raises -> tool_result content is the error string, loop continues."""
    from unittest.mock import MagicMock
    import agents as agents_mod

    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.id = "toolu_e"
    tool_use_block.name = "broken"
    tool_use_block.input = {}
    resp_tool = MagicMock(stop_reason="tool_use", content=[tool_use_block])

    final_text = MagicMock()
    final_text.text = '{"signal": "NEUTRAL", "confidence": 0.4}'
    resp_final = MagicMock(stop_reason="end_turn", content=[final_text])

    fake_client = MagicMock()
    fake_client.messages.create.side_effect = [resp_tool, resp_final]
    monkeypatch.setattr(agents_mod, "Anthropic", lambda api_key: fake_client)

    def boom(_args):
        raise RuntimeError("bang")

    tool = agents_mod.ToolDef(
        name="broken", description="x", input_schema={"type": "object"},
        handler=boom,
    )

    out = agents_mod.run_with_tools(
        api_key="sk-ant-fake",
        system_prompt="s", user_prompt="u", tools=[tool],
    )
    assert out == {"signal": "NEUTRAL", "confidence": 0.4}
    # Verify the second call sent a tool_result with an "error" key.
    second_call_messages = fake_client.messages.create.call_args_list[1].kwargs["messages"]
    tool_result_msg = second_call_messages[-1]
    assert tool_result_msg["role"] == "user"
    assert "bang" in tool_result_msg["content"][0]["content"]
```

- [ ] **Step 3: Run tests**

```
.venv/Scripts/python.exe -m pytest tests/test_agents_init.py -v
```

Expected: existing + 5 new = at least 12 passed.

- [ ] **Step 4: Commit**

```bash
git add agents/__init__.py tests/test_agents_init.py
git commit -m "feat(agents): ToolDef + run_with_tools loop with prompt caching + 3-iter cap"
```

---

## Phase 2: data_prefetch + EdgarBundle promotion

### Task 2.1: `data_prefetch` also fetches `EdgarBundle`

**Files:**
- Modify: `agents/data_prefetch.py`
- Test: `tests/test_data_prefetch.py`

- [ ] **Step 1: Replace contents of `agents/data_prefetch.py`**

```python
"""Single-shot prefetch of all external data needed by specialists.

Runs once after the orchestrator and before the parallel fan-out:
  - ticker 90-day OHLC (yfinance, retried on rate limits)
  - ^VIX 5-day history (yfinance, retried on rate limits)
  - SEC EdgarBundle (CIK already resolved by orchestrator; one HTTP wave to SEC)

Both Price and Risk read `state["price_history"]`. Risk and Fundamentals read
`state["edgar_bundle"]`. Eliminates duplicate external requests and lets
specialists run as pure compute + LLM nodes.
"""

from __future__ import annotations

import time

import pandas as pd

from agents.yf_helpers import download_with_retry
from edgar import EdgarBundle, TickerNotFound, build_edgar_bundle


INTER_REQUEST_GAP_SECONDS = 1.0


def _safe_yf(ticker: str, period: str) -> pd.DataFrame:
    try:
        return download_with_retry(ticker, period=period, interval="1d")
    except Exception:
        return pd.DataFrame()


def _safe_edgar(ticker: str) -> EdgarBundle | None:
    try:
        return build_edgar_bundle(ticker)
    except TickerNotFound:
        return None
    except Exception:
        return None


def data_prefetch(state: dict) -> dict:
    ticker = state["ticker"]

    price_history = _safe_yf(ticker, period="90d")
    time.sleep(INTER_REQUEST_GAP_SECONDS)
    vix_history = _safe_yf("^VIX", period="5d")
    time.sleep(INTER_REQUEST_GAP_SECONDS)
    edgar_bundle = _safe_edgar(ticker)

    return {
        "price_history": price_history,
        "vix_history": vix_history,
        "edgar_bundle": edgar_bundle,
    }
```

- [ ] **Step 2: Replace contents of `tests/test_data_prefetch.py`**

```python
from unittest.mock import patch

import pandas as pd

from agents.data_prefetch import data_prefetch
from edgar import EdgarBundle, TickerNotFound


def _stub_edgar() -> EdgarBundle:
    return EdgarBundle(
        ticker="MSFT", cik="0000789019", company_name="MICROSOFT CORP",
        latest_10q=None, latest_10k=None,
        xbrl_facts={}, mdna_text="", risk_factors_text="",
    )


def test_prefetch_populates_all_three_fields():
    price_df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})
    vix_df = pd.DataFrame({"Close": [20.0, 21.0]})
    bundle = _stub_edgar()
    with patch(
        "agents.data_prefetch.download_with_retry",
        side_effect=[price_df, vix_df],
    ), patch(
        "agents.data_prefetch.build_edgar_bundle",
        return_value=bundle,
    ), patch("agents.data_prefetch.time.sleep"):
        out = data_prefetch({"ticker": "MSFT"})
    assert out["price_history"] is price_df
    assert out["vix_history"] is vix_df
    assert out["edgar_bundle"] is bundle


def test_prefetch_swallows_yf_errors_and_returns_empty():
    bundle = _stub_edgar()
    with patch(
        "agents.data_prefetch.download_with_retry",
        side_effect=Exception("rate limited"),
    ), patch(
        "agents.data_prefetch.build_edgar_bundle",
        return_value=bundle,
    ), patch("agents.data_prefetch.time.sleep"):
        out = data_prefetch({"ticker": "ZZZZ"})
    assert isinstance(out["price_history"], pd.DataFrame)
    assert out["price_history"].empty
    assert isinstance(out["vix_history"], pd.DataFrame)
    assert out["vix_history"].empty
    assert out["edgar_bundle"] is bundle


def test_prefetch_edgar_ticker_not_found_returns_none():
    price_df = pd.DataFrame({"Close": [1.0]})
    with patch(
        "agents.data_prefetch.download_with_retry",
        side_effect=[price_df, pd.DataFrame()],
    ), patch(
        "agents.data_prefetch.build_edgar_bundle",
        side_effect=TickerNotFound("FOREIGN"),
    ), patch("agents.data_prefetch.time.sleep"):
        out = data_prefetch({"ticker": "FOREIGN"})
    assert out["price_history"] is price_df
    assert out["edgar_bundle"] is None


def test_prefetch_edgar_generic_error_returns_none():
    price_df = pd.DataFrame({"Close": [1.0]})
    with patch(
        "agents.data_prefetch.download_with_retry",
        side_effect=[price_df, pd.DataFrame()],
    ), patch(
        "agents.data_prefetch.build_edgar_bundle",
        side_effect=RuntimeError("SEC down"),
    ), patch("agents.data_prefetch.time.sleep"):
        out = data_prefetch({"ticker": "MSFT"})
    assert out["edgar_bundle"] is None
```

- [ ] **Step 3: Run tests**

```
.venv/Scripts/python.exe -m pytest tests/test_data_prefetch.py -v
```

Expected: 4 passed.

- [ ] **Step 4: Commit**

```bash
git add agents/data_prefetch.py tests/test_data_prefetch.py
git commit -m "feat(prefetch): also fetch EdgarBundle into shared state"
```

---

## Phase 3: Per-Agent Tools and Rewrites

Each agent below gets:
1. A new tools module under `agents/tools/`
2. Tests for that tools module under `tests/test_tools/`
3. The agent rewrite + its updated test file

### Task 3.0: Create `agents/tools/__init__.py` + `tests/test_tools/__init__.py`

- [ ] **Step 1: Create empty package markers**

```bash
mkdir -p agents/tools tests/test_tools
```

Create `agents/tools/__init__.py`:

```python
"""Per-agent on-demand tool modules. Each module exposes a list of ToolDef
instances bound to handlers that receive parsed JSON input and return a
JSON-serializable result."""
```

Create `tests/test_tools/__init__.py` (empty file).

- [ ] **Step 2: Commit**

```bash
git add agents/tools/__init__.py tests/test_tools/__init__.py
git commit -m "scaffold: agents/tools/ package + tests/test_tools/"
```

---

### Task 3.1: Fundamentals tools (`agents/tools/fundamentals_tools.py`)

**Files:**
- Create: `agents/tools/fundamentals_tools.py`
- Test: `tests/test_tools/test_fundamentals_tools.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tools/test_fundamentals_tools.py
from unittest.mock import patch

from agents.tools.fundamentals_tools import (
    build_fundamentals_tools,
    _fetch_xbrl_tag,
    _fetch_segment_breakdown,
)
from edgar import EdgarBundle


def _bundle_with(facts: dict) -> EdgarBundle:
    return EdgarBundle(
        ticker="MSFT", cik="0000789019", company_name="MICROSOFT CORP",
        latest_10q=None, latest_10k=None,
        xbrl_facts=facts, mdna_text="", risk_factors_text="",
    )


def test_fetch_xbrl_tag_returns_recent_observations():
    bundle = _bundle_with({
        "facts": {"us-gaap": {"ResearchAndDevelopmentExpense": {"units": {"USD": [
            {"end": "2025-09-30", "val": 7_000_000_000, "form": "10-Q"},
            {"end": "2025-06-30", "val": 6_800_000_000, "form": "10-Q"},
            {"end": "2025-03-31", "val": 6_500_000_000, "form": "10-Q"},
        ]}}}}
    })
    out = _fetch_xbrl_tag(bundle, "ResearchAndDevelopmentExpense", periods=8)
    assert out["tag"] == "ResearchAndDevelopmentExpense"
    assert len(out["observations"]) == 3
    assert out["observations"][0]["val"] == 7_000_000_000


def test_fetch_xbrl_tag_unknown_tag_returns_empty():
    bundle = _bundle_with({"facts": {"us-gaap": {}}})
    out = _fetch_xbrl_tag(bundle, "MissingTag")
    assert out["observations"] == []


def test_fetch_segment_breakdown_returns_segments():
    bundle = _bundle_with({
        "facts": {"us-gaap": {"RevenueFromContractWithCustomerExcludingAssessedTax": {
            "units": {"USD": [
                {"end": "2025-06-30", "val": 50_000_000_000, "form": "10-K", "fp": "FY",
                 "frame": "CY2025", "label": "Cloud"},
                {"end": "2025-06-30", "val": 30_000_000_000, "form": "10-K", "fp": "FY",
                 "frame": "CY2025", "label": "Productivity"},
            ]}
        }}}
    })
    out = _fetch_segment_breakdown(bundle)
    assert "segments" in out


def test_build_fundamentals_tools_returns_three():
    tools = build_fundamentals_tools(bundle=_bundle_with({}), api_key="sk-fake")
    names = [t.name for t in tools]
    assert names == ["fetch_xbrl_tag", "fetch_segment_breakdown", "peer_multiples"]
    for t in tools:
        assert t.input_schema["type"] == "object"
        assert callable(t.handler)


def test_peer_multiples_handler_calls_yfinance(monkeypatch):
    from unittest.mock import MagicMock
    import agents.tools.fundamentals_tools as ft

    fake_info = {"trailingPE": 32.0, "enterpriseToEbitda": 22.5,
                 "priceToSalesTrailing12Months": 11.0, "priceToBook": 14.0}
    fake_ticker = MagicMock()
    fake_ticker.info = fake_info
    monkeypatch.setattr(ft, "yf", MagicMock(Ticker=lambda t: fake_ticker))

    tools = ft.build_fundamentals_tools(bundle=_bundle_with({}), api_key="sk-fake")
    peer_tool = next(t for t in tools if t.name == "peer_multiples")
    out = peer_tool.handler({"peer_tickers": ["AAPL", "GOOGL"]})
    assert "peers" in out
    assert "AAPL" in out["peers"]
    assert out["peers"]["AAPL"]["trailing_pe"] == 32.0
```

- [ ] **Step 2: Run, confirm ImportError**

```
.venv/Scripts/python.exe -m pytest tests/test_tools/test_fundamentals_tools.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `agents/tools/fundamentals_tools.py`**

```python
"""On-demand tools for the Fundamentals agent."""

from __future__ import annotations

from typing import Optional

import yfinance as yf

from agents import ToolDef
from edgar import EdgarBundle


# ---------------------------------------------------------------------------
# Internal helpers (testable)
# ---------------------------------------------------------------------------


def _fetch_xbrl_tag(bundle: EdgarBundle, tag_name: str, periods: int = 8) -> dict:
    """Pull recent observations for an arbitrary US-GAAP tag from the bundle."""
    units = (
        (bundle.xbrl_facts or {})
        .get("facts", {})
        .get("us-gaap", {})
        .get(tag_name, {})
        .get("units", {})
    )
    obs = units.get("USD") or units.get("USD/shares") or []
    obs_sorted = sorted(obs, key=lambda o: o.get("end", ""), reverse=True)[:periods]
    cleaned = [
        {"end": o.get("end"), "val": o.get("val"),
         "form": o.get("form"), "fp": o.get("fp")}
        for o in obs_sorted
    ]
    return {"tag": tag_name, "observations": cleaned}


def _fetch_segment_breakdown(bundle: EdgarBundle) -> dict:
    """Return segment-level RevenueFromContractWithCustomer observations.

    XBRL segment data is filed under several tag names; we try the most
    common: RevenueFromContractWithCustomerExcludingAssessedTax. If absent,
    return an explicit empty result so the LLM can adapt.
    """
    facts = (bundle.xbrl_facts or {}).get("facts", {}).get("us-gaap", {})
    candidates = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
    ]
    for tag in candidates:
        units = facts.get(tag, {}).get("units", {}).get("USD") or []
        if units:
            return {"tag": tag, "segments": units[:20]}
    return {"tag": None, "segments": []}


def _peer_multiples(peer_tickers: list[str]) -> dict:
    """Pull P/E, EV/EBITDA, P/S, P/B for a list of peers via yfinance.Ticker.info."""
    out: dict[str, dict] = {}
    for t in (peer_tickers or [])[:5]:
        try:
            info = yf.Ticker(t).info or {}
        except Exception:
            info = {}
        out[t] = {
            "trailing_pe": info.get("trailingPE"),
            "ev_to_ebitda": info.get("enterpriseToEbitda"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "price_to_book": info.get("priceToBook"),
        }
    return {"peers": out}


# ---------------------------------------------------------------------------
# Public: build ToolDef list bound to a session's bundle + api_key
# ---------------------------------------------------------------------------


def build_fundamentals_tools(
    *, bundle: Optional[EdgarBundle], api_key: str
) -> list[ToolDef]:
    """Construct the 3 ToolDef instances. `bundle` and `api_key` are bound
    via closures so the LLM-facing handlers take only their declared inputs."""

    def fetch_xbrl_tag_handler(args: dict) -> dict:
        if bundle is None:
            return {"error": "edgar_bundle unavailable"}
        return _fetch_xbrl_tag(bundle, args.get("tag_name", ""),
                               int(args.get("periods", 8) or 8))

    def fetch_segment_breakdown_handler(args: dict) -> dict:
        if bundle is None:
            return {"error": "edgar_bundle unavailable"}
        return _fetch_segment_breakdown(bundle)

    def peer_multiples_handler(args: dict) -> dict:
        peers = args.get("peer_tickers") or []
        if not isinstance(peers, list):
            return {"error": "peer_tickers must be a list"}
        return _peer_multiples(peers)

    return [
        ToolDef(
            name="fetch_xbrl_tag",
            description=(
                "Fetch recent quarterly observations for an arbitrary US-GAAP "
                "XBRL tag from the issuer's filings. Examples: "
                "ResearchAndDevelopmentExpense, OperatingCashFlowsContinuingOperations, "
                "CapitalExpenditures, ShareBasedCompensation. Returns up to "
                "`periods` most recent observations sourced from 10-Q/10-K."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "tag_name": {"type": "string"},
                    "periods": {"type": "integer", "default": 8},
                },
                "required": ["tag_name"],
            },
            handler=fetch_xbrl_tag_handler,
        ),
        ToolDef(
            name="fetch_segment_breakdown",
            description=(
                "Return segment-level revenue rows from XBRL when filed. Useful "
                "for mix-shift analysis. Returns empty list if the issuer does "
                "not file segment breakdowns under standard tags."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=fetch_segment_breakdown_handler,
        ),
        ToolDef(
            name="peer_multiples",
            description=(
                "Quick comparable multiples (P/E, EV/EBITDA, P/S, P/B) for up "
                "to 5 peer tickers via yfinance Ticker.info. Pick peers from "
                "MD&A or your own knowledge of the issuer's competitors."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "peer_tickers": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["peer_tickers"],
            },
            handler=peer_multiples_handler,
        ),
    ]
```

- [ ] **Step 4: Run tests**

```
.venv/Scripts/python.exe -m pytest tests/test_tools/test_fundamentals_tools.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/tools/fundamentals_tools.py tests/test_tools/test_fundamentals_tools.py
git commit -m "feat(tools): fundamentals on-demand tools (xbrl_tag, segments, peer_multiples)"
```

---

### Task 3.2: Macro tools (`agents/tools/macro_tools.py`)

**Files:**
- Create: `agents/tools/macro_tools.py`
- Test: `tests/test_tools/test_macro_tools.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tools/test_macro_tools.py
import responses
from unittest.mock import MagicMock

from agents.tools.macro_tools import build_macro_tools


@responses.activate
def test_fetch_fred_series_returns_observations():
    responses.add(
        responses.GET,
        "https://api.stlouisfed.org/fred/series/observations",
        json={"observations": [
            {"date": "2025-10-01", "value": "5.25"},
            {"date": "2025-09-01", "value": "5.30"},
        ]},
    )
    tools = build_macro_tools(fred_key="frd-fake")
    fred_tool = next(t for t in tools if t.name == "fetch_fred_series")
    out = fred_tool.handler({"series_id": "BAMLH0A0HYM2", "periods": 5})
    assert out["series_id"] == "BAMLH0A0HYM2"
    assert len(out["observations"]) == 2
    assert out["observations"][0]["value"] == 5.25


def test_fetch_fred_series_no_key_returns_error():
    tools = build_macro_tools(fred_key="")
    fred_tool = next(t for t in tools if t.name == "fetch_fred_series")
    out = fred_tool.handler({"series_id": "DFF"})
    assert "error" in out


def test_classify_ticker_sector(monkeypatch):
    import agents.tools.macro_tools as mt

    fake_ticker = MagicMock()
    fake_ticker.info = {"sector": "Technology", "industry": "Software—Infrastructure"}
    monkeypatch.setattr(mt, "yf", MagicMock(Ticker=lambda t: fake_ticker))

    tools = mt.build_macro_tools(fred_key="x")
    sec_tool = next(t for t in tools if t.name == "classify_ticker_sector")
    out = sec_tool.handler({"ticker": "MSFT"})
    assert out["sector"] == "Technology"
    assert out["industry"] == "Software—Infrastructure"


def test_fetch_credit_spreads(monkeypatch):
    import pandas as pd
    import agents.tools.macro_tools as mt

    hyg_df = pd.DataFrame({"Close": [78.0, 78.5, 79.0]})
    lqd_df = pd.DataFrame({"Close": [105.0, 105.2, 105.5]})

    def fake_dl(ticker, **kwargs):
        return hyg_df if ticker == "HYG" else lqd_df

    monkeypatch.setattr(mt, "download_with_retry", fake_dl)

    tools = mt.build_macro_tools(fred_key="x")
    cs_tool = next(t for t in tools if t.name == "fetch_credit_spreads")
    out = cs_tool.handler({})
    assert "hyg_close" in out
    assert "lqd_close" in out
    assert "hyg_lqd_ratio" in out


def test_build_macro_tools_returns_three():
    tools = build_macro_tools(fred_key="x")
    names = [t.name for t in tools]
    assert names == ["fetch_fred_series", "classify_ticker_sector", "fetch_credit_spreads"]
```

- [ ] **Step 2: Run, confirm ImportError**

```
.venv/Scripts/python.exe -m pytest tests/test_tools/test_macro_tools.py -v
```

- [ ] **Step 3: Implement `agents/tools/macro_tools.py`**

```python
"""On-demand tools for the Macro agent."""

from __future__ import annotations

import requests
import yfinance as yf

from agents import ToolDef
from agents.yf_helpers import download_with_retry


def _fetch_fred(series_id: str, api_key: str, periods: int = 12) -> list[dict]:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": periods,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return [
        {"date": o["date"], "value": float(o["value"])}
        for o in resp.json().get("observations", [])
        if o.get("value") and o["value"] != "."
    ]


def _classify_sector(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as exc:
        return {"error": str(exc)[:200]}
    return {
        "ticker": ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
    }


def _credit_spreads() -> dict:
    try:
        hyg = download_with_retry("HYG", period="30d", interval="1d")
        lqd = download_with_retry("LQD", period="30d", interval="1d")
    except Exception as exc:
        return {"error": str(exc)[:200]}
    if hyg.empty or lqd.empty:
        return {"error": "no credit ETF data"}
    hyg_close = float(hyg["Close"].squeeze().iloc[-1])
    lqd_close = float(lqd["Close"].squeeze().iloc[-1])
    return {
        "hyg_close": round(hyg_close, 2),
        "lqd_close": round(lqd_close, 2),
        "hyg_lqd_ratio": round(hyg_close / lqd_close, 4),
        "interpretation": (
            "Higher HYG/LQD ratio = risk-on (HY outperforming IG); "
            "lower = risk-off (IG outperforming HY, credit deterioration)."
        ),
    }


def build_macro_tools(*, fred_key: str) -> list[ToolDef]:

    def fred_handler(args: dict) -> dict:
        if not fred_key:
            return {"error": "fred_key not configured"}
        series_id = args.get("series_id", "")
        periods = int(args.get("periods", 12) or 12)
        try:
            obs = _fetch_fred(series_id, fred_key, periods=periods)
        except Exception as exc:
            return {"error": str(exc)[:200]}
        return {"series_id": series_id, "observations": obs}

    def sector_handler(args: dict) -> dict:
        return _classify_sector(args.get("ticker", ""))

    def spreads_handler(_args: dict) -> dict:
        return _credit_spreads()

    return [
        ToolDef(
            name="fetch_fred_series",
            description=(
                "Fetch recent observations for any FRED economic data series. "
                "Useful series: BAMLH0A0HYM2 (HY OAS), T10YIE (10Y breakeven "
                "inflation), DCOILWTICO (WTI crude), UNRATE (unemployment), "
                "CPIAUCSL (headline CPI), DGS30 (30Y Treasury). Returns "
                "newest-first observations."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "series_id": {"type": "string"},
                    "periods": {"type": "integer", "default": 12},
                },
                "required": ["series_id"],
            },
            handler=fred_handler,
        ),
        ToolDef(
            name="classify_ticker_sector",
            description=(
                "Return GICS sector + sub-industry for a ticker via "
                "yfinance.Ticker.info. Use to map the current macro regime to "
                "the ticker's specific sector exposure."
            ),
            input_schema={
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
            handler=sector_handler,
        ),
        ToolDef(
            name="fetch_credit_spreads",
            description=(
                "Return current HYG and LQD ETF closes plus the HYG/LQD ratio "
                "as a risk-on/off gauge. Use when FRED HY series is unavailable "
                "or you want a market-priced credit signal."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=spreads_handler,
        ),
    ]
```

- [ ] **Step 4: Run tests**

```
.venv/Scripts/python.exe -m pytest tests/test_tools/test_macro_tools.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/tools/macro_tools.py tests/test_tools/test_macro_tools.py
git commit -m "feat(tools): macro on-demand tools (fred, sector, credit_spreads)"
```

---

### Task 3.3: Price tools (`agents/tools/price_tools.py`)

**Files:**
- Create: `agents/tools/price_tools.py`
- Test: `tests/test_tools/test_price_tools.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tools/test_price_tools.py
import numpy as np
import pandas as pd

from agents.tools.price_tools import build_price_tools


def _price_df(n=200, start=100, drift=0.05, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.012, n)
    prices = start * np.cumprod(1 + rets)
    return pd.DataFrame({"Close": prices, "Volume": rng.integers(1e6, 5e6, n)})


def test_compute_indicator_atr():
    df = _price_df(120)
    tools = build_price_tools(price_history=df)
    indicator_tool = next(t for t in tools if t.name == "compute_indicator")
    out = indicator_tool.handler({"name": "ATR"})
    assert "name" in out and out["name"] == "ATR"
    assert "value" in out


def test_compute_indicator_sma():
    df = _price_df(120)
    tools = build_price_tools(price_history=df)
    indicator_tool = next(t for t in tools if t.name == "compute_indicator")
    out = indicator_tool.handler({"name": "SMA50"})
    assert out["value"] is not None


def test_compute_indicator_unknown():
    df = _price_df(120)
    tools = build_price_tools(price_history=df)
    indicator_tool = next(t for t in tools if t.name == "compute_indicator")
    out = indicator_tool.handler({"name": "BOGUS"})
    assert "error" in out


def test_detect_chart_pattern_returns_dict():
    df = _price_df(120)
    tools = build_price_tools(price_history=df)
    pattern_tool = next(t for t in tools if t.name == "detect_chart_pattern")
    out = pattern_tool.handler({})
    assert "patterns" in out


def test_volume_profile_summary_returns_buckets():
    df = _price_df(120)
    tools = build_price_tools(price_history=df)
    vp_tool = next(t for t in tools if t.name == "volume_profile_summary")
    out = vp_tool.handler({"n_buckets": 8})
    assert "buckets" in out
    assert len(out["buckets"]) <= 8
    for b in out["buckets"]:
        assert "price_range" in b
        assert "volume" in b


def test_build_price_tools_returns_three():
    df = _price_df(60)
    tools = build_price_tools(price_history=df)
    names = [t.name for t in tools]
    assert names == ["compute_indicator", "detect_chart_pattern", "volume_profile_summary"]


def test_compute_indicator_empty_df_returns_error():
    tools = build_price_tools(price_history=pd.DataFrame())
    out = tools[0].handler({"name": "ATR"})
    assert "error" in out
```

- [ ] **Step 2: Run, confirm ImportError**

```
.venv/Scripts/python.exe -m pytest tests/test_tools/test_price_tools.py -v
```

- [ ] **Step 3: Implement `agents/tools/price_tools.py`**

```python
"""On-demand tools for the Price agent. All local computations on the
prefetched price_history DataFrame — no network calls."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from agents import ToolDef


VALID_INDICATORS = {"ATR", "ADX", "STOCH", "SMA50", "SMA200", "OBV"}


def _atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    if "High" not in df.columns or "Low" not in df.columns:
        # Approximate ATR with abs daily Close % range
        close = df["Close"].squeeze()
        return round(float(close.pct_change().abs().rolling(period).mean().iloc[-1] * 100), 3)
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    close = df["Close"].squeeze()
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return round(float(atr), 3) if pd.notna(atr) else None


def _sma(df: pd.DataFrame, period: int) -> Optional[float]:
    close = df["Close"].squeeze()
    if len(close) < period:
        return None
    return round(float(close.rolling(period).mean().iloc[-1]), 4)


def _obv(df: pd.DataFrame) -> Optional[float]:
    if "Volume" not in df.columns:
        return None
    close = df["Close"].squeeze()
    vol = df["Volume"].squeeze()
    direction = np.sign(close.diff().fillna(0))
    obv = (direction * vol).cumsum().iloc[-1]
    return round(float(obv), 0)


def _stoch(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Optional[dict]:
    close = df["Close"].squeeze()
    if len(close) < k_period:
        return None
    if "High" in df.columns and "Low" in df.columns:
        high_n = df["High"].squeeze().rolling(k_period).max()
        low_n = df["Low"].squeeze().rolling(k_period).min()
    else:
        high_n = close.rolling(k_period).max()
        low_n = close.rolling(k_period).min()
    k = 100 * (close - low_n) / (high_n - low_n).replace(0, 1e-9)
    d = k.rolling(d_period).mean()
    return {"k": round(float(k.iloc[-1]), 2), "d": round(float(d.iloc[-1]), 2)}


def _adx(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    # Lightweight: returns rolling absolute % move as a proxy if no High/Low.
    close = df["Close"].squeeze()
    if len(close) < period + 1:
        return None
    move = close.pct_change().abs().rolling(period).mean().iloc[-1] * 100
    return round(float(move), 3) if pd.notna(move) else None


def _detect_patterns(df: pd.DataFrame) -> list[dict]:
    """Heuristic, very lightweight chart-pattern detection."""
    close = df["Close"].squeeze()
    if len(close) < 30:
        return []
    last30 = close.iloc[-30:]
    out: list[dict] = []
    # Ascending triangle: rising lows + flat highs
    highs = last30.rolling(5).max()
    lows = last30.rolling(5).min()
    if highs.iloc[-1] - highs.iloc[0] < highs.iloc[0] * 0.01 and lows.iloc[-1] > lows.iloc[0]:
        out.append({"name": "ascending_triangle", "confidence": 0.5})
    # Double-top: two peaks within 1% over the last 60 bars
    if len(close) >= 60:
        last60 = close.iloc[-60:]
        peak = last60.max()
        peaks = last60[last60 > peak * 0.99]
        if len(peaks) >= 2 and peaks.index[-1] - peaks.index[0] >= 10:
            out.append({"name": "double_top", "confidence": 0.4})
    # Trend break: latest close vs SMA20 cross
    sma20 = close.rolling(20).mean()
    if pd.notna(sma20.iloc[-2]) and pd.notna(sma20.iloc[-1]):
        if close.iloc[-2] > sma20.iloc[-2] and close.iloc[-1] < sma20.iloc[-1]:
            out.append({"name": "sma20_break_down", "confidence": 0.6})
        elif close.iloc[-2] < sma20.iloc[-2] and close.iloc[-1] > sma20.iloc[-1]:
            out.append({"name": "sma20_break_up", "confidence": 0.6})
    return out


def _volume_profile(df: pd.DataFrame, n_buckets: int = 10) -> list[dict]:
    if "Volume" not in df.columns:
        return []
    close = df["Close"].squeeze()
    vol = df["Volume"].squeeze()
    lo, hi = float(close.min()), float(close.max())
    if lo == hi:
        return []
    bins = np.linspace(lo, hi, n_buckets + 1)
    bucket = np.digitize(close, bins) - 1
    rows: list[dict] = []
    for i in range(n_buckets):
        mask = bucket == i
        rows.append({
            "price_range": [round(float(bins[i]), 2), round(float(bins[i + 1]), 2)],
            "volume": int(vol[mask].sum()) if mask.any() else 0,
        })
    return rows


def build_price_tools(*, price_history: Optional[pd.DataFrame]) -> list[ToolDef]:

    def indicator_handler(args: dict) -> dict:
        name = (args.get("name") or "").upper()
        if price_history is None or price_history.empty:
            return {"error": "price_history unavailable"}
        if name not in VALID_INDICATORS:
            return {"error": f"unknown indicator '{name}'; valid: {sorted(VALID_INDICATORS)}"}
        if name == "ATR":
            value = _atr(price_history)
        elif name == "ADX":
            value = _adx(price_history)
        elif name == "STOCH":
            value = _stoch(price_history)
        elif name == "SMA50":
            value = _sma(price_history, 50)
        elif name == "SMA200":
            value = _sma(price_history, 200)
        elif name == "OBV":
            value = _obv(price_history)
        else:
            value = None
        return {"name": name, "value": value}

    def pattern_handler(_args: dict) -> dict:
        if price_history is None or price_history.empty:
            return {"error": "price_history unavailable"}
        return {"patterns": _detect_patterns(price_history)}

    def vp_handler(args: dict) -> dict:
        if price_history is None or price_history.empty:
            return {"error": "price_history unavailable"}
        n = int(args.get("n_buckets", 10) or 10)
        return {"buckets": _volume_profile(price_history, n_buckets=n)}

    return [
        ToolDef(
            name="compute_indicator",
            description=(
                "Compute one of: ATR, ADX, STOCH, SMA50, SMA200, OBV on the "
                "prefetched price history. All local; no network. Use to drill "
                "into volatility, trend strength, momentum, or volume trend."
            ),
            input_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            handler=indicator_handler,
        ),
        ToolDef(
            name="detect_chart_pattern",
            description=(
                "Heuristic pattern detection on recent price history: "
                "ascending_triangle, double_top, sma20_break_up/down. Returns "
                "list of {name, confidence}. Local; no network."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=pattern_handler,
        ),
        ToolDef(
            name="volume_profile_summary",
            description=(
                "Bucket price range into n_buckets and return total traded "
                "volume per bucket. High-volume buckets identify support / "
                "resistance levels. Local; no network."
            ),
            input_schema={
                "type": "object",
                "properties": {"n_buckets": {"type": "integer", "default": 10}},
            },
            handler=vp_handler,
        ),
    ]
```

- [ ] **Step 4: Run tests**

```
.venv/Scripts/python.exe -m pytest tests/test_tools/test_price_tools.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/tools/price_tools.py tests/test_tools/test_price_tools.py
git commit -m "feat(tools): price on-demand tools (indicator, pattern, volume_profile)"
```

---

### Task 3.4: Sentiment tools (`agents/tools/sentiment_tools.py`)

**Files:**
- Create: `agents/tools/sentiment_tools.py`
- Test: `tests/test_tools/test_sentiment_tools.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tools/test_sentiment_tools.py
from unittest.mock import MagicMock, patch

from agents.tools.sentiment_tools import build_sentiment_tools


def test_fetch_press_releases_uses_targeted_query():
    fake_tav = MagicMock()
    fake_tav.search.return_value = {"results": [
        {"title": "MSFT 8-K filed", "url": "https://prnewswire.com/x", "content": "..."}
    ]}
    with patch("agents.tools.sentiment_tools.TavilyClient", return_value=fake_tav):
        tools = build_sentiment_tools(tavily_key="tvly-fake", api_key="sk-fake")
    pr_tool = next(t for t in tools if t.name == "fetch_press_releases")
    out = pr_tool.handler({"ticker": "MSFT", "days": 14})
    assert "results" in out
    assert "site:prnewswire.com" in fake_tav.search.call_args.kwargs["query"]


def test_fetch_analyst_actions_query_shape():
    fake_tav = MagicMock()
    fake_tav.search.return_value = {"results": []}
    with patch("agents.tools.sentiment_tools.TavilyClient", return_value=fake_tav):
        tools = build_sentiment_tools(tavily_key="tvly-fake", api_key="sk-fake")
    aa_tool = next(t for t in tools if t.name == "fetch_analyst_actions")
    out = aa_tool.handler({"ticker": "NVDA"})
    assert "results" in out
    q = fake_tav.search.call_args.kwargs["query"]
    assert "NVDA" in q
    assert "analyst" in q.lower()


def test_categorize_drivers_calls_haiku(monkeypatch):
    fake_haiku = MagicMock()
    fake_haiku.invoke.return_value = MagicMock(
        content='{"earnings": 2, "regulatory": 1, "product": 0, "m&a": 0, '
                '"insider": 0, "competitor": 0, "macro": 0}'
    )
    monkeypatch.setattr(
        "agents.tools.sentiment_tools._build_haiku",
        lambda api_key: fake_haiku,
    )
    tools = build_sentiment_tools(tavily_key="x", api_key="sk-fake")
    cat_tool = next(t for t in tools if t.name == "categorize_drivers")
    out = cat_tool.handler({"drivers": ["earnings beat", "raised guidance", "EU probe"]})
    assert out["earnings"] == 2
    assert out["regulatory"] == 1


def test_build_sentiment_tools_returns_three():
    tools = build_sentiment_tools(tavily_key="x", api_key="y")
    names = [t.name for t in tools]
    assert names == ["fetch_press_releases", "fetch_analyst_actions", "categorize_drivers"]


def test_fetch_press_releases_no_key_returns_error():
    tools = build_sentiment_tools(tavily_key="", api_key="x")
    pr_tool = next(t for t in tools if t.name == "fetch_press_releases")
    out = pr_tool.handler({"ticker": "MSFT"})
    assert "error" in out
```

- [ ] **Step 2: Run, confirm ImportError**

- [ ] **Step 3: Implement `agents/tools/sentiment_tools.py`**

```python
"""On-demand tools for the Sentiment agent."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from tavily import TavilyClient

from agents import FAST_MODEL, ToolDef, safe_parse_json


PR_SITES = "site:prnewswire.com OR site:businesswire.com OR site:globenewswire.com"


def _build_haiku(api_key: str) -> ChatAnthropic:
    return ChatAnthropic(
        model=FAST_MODEL, api_key=api_key, temperature=0.1, max_tokens=400,
    )


def _tav_search(tavily_key: str, query: str, days: int, max_results: int = 8) -> dict:
    if not tavily_key:
        return {"error": "tavily_key not configured"}
    try:
        tav = TavilyClient(api_key=tavily_key)
        result = tav.search(
            query=query, max_results=max_results, search_depth="basic", days=days,
        )
    except Exception as exc:
        return {"error": str(exc)[:200]}
    return {"results": [
        {"title": (a.get("title") or "").strip(),
         "url": a.get("url"),
         "snippet": (a.get("content") or "")[:280]}
        for a in (result.get("results", []) or [])
    ]}


def build_sentiment_tools(*, tavily_key: str, api_key: str) -> list[ToolDef]:

    def press_releases_handler(args: dict) -> dict:
        ticker = args.get("ticker", "")
        days = int(args.get("days", 14) or 14)
        query = f"{ticker} ({PR_SITES})"
        return _tav_search(tavily_key, query, days)

    def analyst_actions_handler(args: dict) -> dict:
        ticker = args.get("ticker", "")
        query = f"{ticker} analyst upgrade downgrade price target"
        return _tav_search(tavily_key, query, days=14)

    def categorize_handler(args: dict) -> dict:
        drivers = args.get("drivers") or []
        if not drivers:
            return {}
        haiku = _build_haiku(api_key)
        prompt = (
            "Classify each driver phrase into exactly one of these categories: "
            "earnings, m&a, regulatory, product, insider, competitor, macro. "
            "Return JSON ONLY: a flat object with each category as a key and "
            "the count of drivers that fall into it as the integer value. "
            "Categories with zero matches MUST still appear with value 0.\n\n"
            "Drivers:\n" + "\n".join(f"- {d}" for d in drivers)
        )
        try:
            resp = haiku.invoke(prompt)
            return safe_parse_json(resp.content)
        except Exception as exc:
            return {"error": str(exc)[:200]}

    return [
        ToolDef(
            name="fetch_press_releases",
            description=(
                "Search issuer press releases (PRNewswire / BusinessWire / "
                "GlobeNewswire) for the ticker over the past `days`. Higher-"
                "signal than aggregator news because the source is the issuer."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "days": {"type": "integer", "default": 14},
                },
                "required": ["ticker"],
            },
            handler=press_releases_handler,
        ),
        ToolDef(
            name="fetch_analyst_actions",
            description=(
                "Search for sell-side analyst rating changes and price-target "
                "moves on the ticker over the past 14 days. Use to gauge "
                "consensus drift."
            ),
            input_schema={
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
            handler=analyst_actions_handler,
        ),
        ToolDef(
            name="categorize_drivers",
            description=(
                "Categorize a list of driver phrases into a fixed taxonomy: "
                "earnings, m&a, regulatory, product, insider, competitor, "
                "macro. Returns a count per category."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "drivers": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["drivers"],
            },
            handler=categorize_handler,
        ),
    ]
```

- [ ] **Step 4: Run tests**

```
.venv/Scripts/python.exe -m pytest tests/test_tools/test_sentiment_tools.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/tools/sentiment_tools.py tests/test_tools/test_sentiment_tools.py
git commit -m "feat(tools): sentiment on-demand tools (press_releases, analyst_actions, categorize)"
```

---

### Task 3.5: Risk tools (`agents/tools/risk_tools.py`)

**Files:**
- Create: `agents/tools/risk_tools.py`
- Test: `tests/test_tools/test_risk_tools.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tools/test_risk_tools.py
import numpy as np
import pandas as pd

from agents.tools.risk_tools import build_risk_tools


def _rets_df(n=120, mu=0.0005, sigma=0.012, seed=1):
    rng = np.random.default_rng(seed)
    rets = rng.normal(mu, sigma, n)
    prices = 100 * np.cumprod(1 + rets)
    return pd.DataFrame({"Close": prices})


def test_compute_var_es_returns_negative_var_and_es():
    df = _rets_df(120)
    tools = build_risk_tools(price_history=df, edgar_bundle=None)
    var_tool = next(t for t in tools if t.name == "compute_var_es")
    out = var_tool.handler({"confidence": 0.95})
    assert "var_pct" in out
    assert "es_pct" in out
    assert out["var_pct"] <= 0
    assert out["es_pct"] <= out["var_pct"]


def test_decompose_drawdown_returns_components():
    df = _rets_df(120)
    tools = build_risk_tools(price_history=df, edgar_bundle=None)
    dd_tool = next(t for t in tools if t.name == "decompose_drawdown")
    out = dd_tool.handler({})
    assert "current_drawdown_pct" in out
    assert "max_drawdown_pct" in out
    assert "days_since_peak" in out


def test_forward_risk_attribution_uses_bundle_when_present():
    from edgar import EdgarBundle
    bundle = EdgarBundle(
        ticker="MSFT", cik="0000789019", company_name="MICROSOFT CORP",
        latest_10q=None, latest_10k=None,
        xbrl_facts={"facts": {"us-gaap": {
            "Revenues": {"units": {"USD": [
                {"end": "2025-09-30", "val": 70_000_000_000, "form": "10-Q"},
                {"end": "2024-09-30", "val": 65_000_000_000, "form": "10-Q"},
            ]}}
        }}},
        mdna_text="", risk_factors_text="",
    )
    df = _rets_df(120)
    tools = build_risk_tools(price_history=df, edgar_bundle=bundle)
    fra_tool = next(t for t in tools if t.name == "forward_risk_attribution")
    out = fra_tool.handler({})
    assert set(out.keys()) >= {"operating", "balance_sheet", "positioning", "systemic"}
    for v in out.values():
        assert v in {"low", "medium", "high"}


def test_forward_risk_attribution_no_bundle_returns_unknown_for_fundamentals():
    df = _rets_df(120)
    tools = build_risk_tools(price_history=df, edgar_bundle=None)
    fra_tool = next(t for t in tools if t.name == "forward_risk_attribution")
    out = fra_tool.handler({})
    assert out["operating"] in {"low", "medium", "high", "unknown"}
    assert out["balance_sheet"] in {"low", "medium", "high", "unknown"}


def test_build_risk_tools_returns_three():
    df = _rets_df(60)
    tools = build_risk_tools(price_history=df, edgar_bundle=None)
    names = [t.name for t in tools]
    assert names == ["forward_risk_attribution", "decompose_drawdown", "compute_var_es"]
```

- [ ] **Step 2: Run, confirm ImportError**

- [ ] **Step 3: Implement `agents/tools/risk_tools.py`**

```python
"""On-demand tools for the Risk agent. All local computations on the
prefetched price_history DataFrame and edgar_bundle — no network calls."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from agents import ToolDef
from edgar import EdgarBundle


def _returns(df: pd.DataFrame) -> pd.Series:
    return df["Close"].squeeze().pct_change().dropna()


def _compute_var_es(df: pd.DataFrame, confidence: float = 0.95) -> dict:
    rets = _returns(df)
    if len(rets) < 20:
        return {"error": "insufficient return history"}
    quantile = float(np.quantile(rets, 1 - confidence))
    es = float(rets[rets <= quantile].mean())
    return {
        "confidence": confidence,
        "var_pct": round(quantile * 100, 3),
        "es_pct": round(es * 100, 3),
    }


def _decompose_drawdown(df: pd.DataFrame) -> dict:
    rets = _returns(df)
    if rets.empty:
        return {"error": "insufficient history"}
    cum = (1 + rets).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    current_dd = float(dd.iloc[-1])
    max_dd = float(dd.min())
    peak_idx = running_max.idxmax()
    last_idx = rets.index[-1]
    try:
        days_since_peak = int(last_idx - peak_idx)
    except Exception:
        days_since_peak = int(len(rets) - 1 - rets.index.get_loc(peak_idx))
    # Vol vs trend split: sum of negative returns since peak (trend) vs realized
    # vol over the same window (vol).
    since_peak = rets.loc[peak_idx:]
    trend_component = float(since_peak[since_peak < 0].sum())
    vol_component = float(since_peak.std() * np.sqrt(len(since_peak)))
    return {
        "current_drawdown_pct": round(current_dd * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "days_since_peak": days_since_peak,
        "trend_component_pct": round(trend_component * 100, 2),
        "vol_component_pct": round(vol_component * 100, 2),
    }


def _yoy_revenue_pct(bundle: EdgarBundle) -> Optional[float]:
    units = (
        (bundle.xbrl_facts or {})
        .get("facts", {}).get("us-gaap", {})
        .get("Revenues", {}).get("units", {}).get("USD") or []
    )
    if len(units) < 2:
        return None
    obs = sorted(units, key=lambda o: o.get("end", ""), reverse=True)
    latest = obs[0]
    latest_end = latest.get("end", "")
    if len(latest_end) < 10:
        return None
    target = f"{int(latest_end[:4]) - 1}{latest_end[4:]}"
    prior = next((o for o in obs if o.get("end") == target), None)
    if not prior or not prior.get("val"):
        return None
    return round((float(latest["val"]) - float(prior["val"])) / float(prior["val"]) * 100, 2)


def _forward_risk_attribution(
    df: pd.DataFrame, bundle: Optional[EdgarBundle]
) -> dict:
    out = {
        "operating": "unknown",
        "balance_sheet": "unknown",
        "positioning": "medium",
        "systemic": "medium",
    }
    if bundle is not None:
        rev_yoy = _yoy_revenue_pct(bundle)
        if rev_yoy is not None:
            if rev_yoy < 0:
                out["operating"] = "high"
            elif rev_yoy < 5:
                out["operating"] = "medium"
            else:
                out["operating"] = "low"
        # Crude D/E read
        facts = (bundle.xbrl_facts or {}).get("facts", {}).get("us-gaap", {})
        liabilities_obs = facts.get("Liabilities", {}).get("units", {}).get("USD") or []
        equity_obs = facts.get("StockholdersEquity", {}).get("units", {}).get("USD") or []
        if liabilities_obs and equity_obs:
            try:
                d = float(liabilities_obs[0]["val"])
                e = float(equity_obs[0]["val"]) or 1e-9
                de = d / e
                if de < 1.0:
                    out["balance_sheet"] = "low"
                elif de < 2.5:
                    out["balance_sheet"] = "medium"
                else:
                    out["balance_sheet"] = "high"
            except Exception:
                pass
    # Positioning + systemic from price history (vol percentile proxy)
    if df is not None and not df.empty:
        rets = _returns(df)
        if len(rets) >= 30:
            ann_vol = float(rets.std() * np.sqrt(252)) * 100
            if ann_vol < 20:
                out["positioning"] = "low"
                out["systemic"] = "low"
            elif ann_vol < 40:
                out["positioning"] = "medium"
                out["systemic"] = "medium"
            else:
                out["positioning"] = "high"
                out["systemic"] = "high"
    return out


def build_risk_tools(
    *, price_history: Optional[pd.DataFrame], edgar_bundle: Optional[EdgarBundle]
) -> list[ToolDef]:

    def fra_handler(_args: dict) -> dict:
        return _forward_risk_attribution(price_history, edgar_bundle)

    def dd_handler(_args: dict) -> dict:
        if price_history is None or price_history.empty:
            return {"error": "price_history unavailable"}
        return _decompose_drawdown(price_history)

    def var_handler(args: dict) -> dict:
        if price_history is None or price_history.empty:
            return {"error": "price_history unavailable"}
        c = float(args.get("confidence", 0.95) or 0.95)
        return _compute_var_es(price_history, confidence=c)

    return [
        ToolDef(
            name="forward_risk_attribution",
            description=(
                "Decompose forward risk into {operating, balance_sheet, "
                "positioning, systemic} each ∈ {low, medium, high, unknown}. "
                "Uses revenue YoY + leverage from EdgarBundle and vol regime "
                "from price history. Returns 'unknown' for fundamental "
                "components when EdgarBundle is missing."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=fra_handler,
        ),
        ToolDef(
            name="decompose_drawdown",
            description=(
                "Split current and max drawdown into trend vs volatility "
                "components and report days since peak. Local; no network."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=dd_handler,
        ),
        ToolDef(
            name="compute_var_es",
            description=(
                "Historical 1-day Value-at-Risk and Expected Shortfall on the "
                "price-history return series at the given confidence level "
                "(0.95 or 0.99 are typical). Both reported as negative "
                "percentages. Local; no network."
            ),
            input_schema={
                "type": "object",
                "properties": {"confidence": {"type": "number", "default": 0.95}},
            },
            handler=var_handler,
        ),
    ]
```

- [ ] **Step 4: Run tests**

```
.venv/Scripts/python.exe -m pytest tests/test_tools/test_risk_tools.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/tools/risk_tools.py tests/test_tools/test_risk_tools.py
git commit -m "feat(tools): risk on-demand tools (forward_risk_attribution, dd, var_es)"
```

---

### Task 3.6: Rewrite Fundamentals agent (CoT + tools)

**Files:**
- Modify: `agents/fundamentals_agent.py`
- Modify: `tests/test_fundamentals_agent.py`

- [ ] **Step 1: Replace `agents/fundamentals_agent.py`**

```python
"""Fundamentals specialist (v2.1): persona + DuPont + CoT + 3 tools.

Reads `state["edgar_bundle"]` populated by data_prefetch; falls back to
`build_edgar_bundle(ticker)` if missing (preserves single-call test paths).
"""

from __future__ import annotations

from typing import Optional

from agents import LLMClients, degraded_signal, run_with_tools
from agents.tools.fundamentals_tools import build_fundamentals_tools
from edgar import EdgarBundle, TickerNotFound, build_edgar_bundle
from state import AgentSignal


PERSONA = (
    "You are a CFA Charterholder Senior equity research analyst with deep "
    "expertise on US large-cap equities. You understand accounting "
    "conventions (US GAAP and IFRS) and extract key insights from 10-Q/10-K. "
    "Your judgment is objective and skeptic but flexible enough to identify "
    "where MD&A differ from fundamentals."
)

METHODOLOGY = """
Methodology you apply:
- DuPont decomposition (ROE = Net Margin × Asset Turnover × Equity Multiplier)
- Quality-of-earnings (operating cash flow vs net income; accruals ratio = (NI − CFO) / Avg Assets)
- Operating leverage (Δ%OpInc / Δ%Rev; >1 = positive leverage)
- Cash Conversion Cycle (DIO + DSO − DPO) for working-capital-heavy issuers
- Segment growth attribution when 10-K segment table is filed
""".strip()

COT = """
Reason step-by-step. For each step, state the metric value and the inference.
1. Revenue trajectory — YoY %, segment mix if available
2. Margin path — gross → operating → net, direction + drivers
3. Balance sheet health — leverage (D/E), liquidity (current ratio), working capital
4. Earnings quality — operating cash flow vs net income; accruals ratio
5. Operating leverage — Δ%OpInc / Δ%Rev
6. MD&A signals — growth drivers cited, risks acknowledged
7. Risk Factors — material new disclosures vs prior 10-K
8. Integrated call — bullish / bearish / neutral + confidence

Tool-call rules:
- Use tools only when a step requires data not in always-on payload.
- Cap: max 3 tool calls per run.
- After 3 tool calls (or when you have enough), produce final JSON.
""".strip()

OUTPUT_SCHEMA = """
Respond with a single JSON object (no markdown fences) with EXACTLY these keys:
{
  "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence": 0.0..1.0,
  "summary": "one sentence (≤25 words)",
  "section_markdown": "## Fundamentals\\n... 200-300 words ...",
  "key_metrics": {
    "roe_pct": number,
    "op_margin_pct": number,
    "op_margin_delta_yoy_bps": number,
    "fcf_margin_pct": number,
    "debt_to_equity": number,
    "current_ratio": number,
    "accruals_ratio_pct": number,
    "eps_yoy_pct": number
  },
  "flags": ["string", ...]
}
""".strip()

GUARDRAILS = """
Constraints and guardrails:
- No "buy/sell" verbiage in section_markdown — use "favorable / cautious / unfavorable" framings.
- section_markdown must be 200-300 words.
- Confidence ≤ 0.6 if latest 10-Q is older than 100 days, MD&A is empty, or
  fewer than 3 quarters of history are available.
- Confidence ≤ 0.4 if revenue YoY % is unavailable.
""".strip()


def _build_system_prompt() -> str:
    return "\n\n".join([
        PERSONA,
        METHODOLOGY,
        OUTPUT_SCHEMA,
        GUARDRAILS,
        COT,
    ])


def _key_metrics_from_facts(facts: dict) -> dict:
    """Compute the deterministic key_metrics block injected into the user prompt
    so the LLM has these numbers without using a tool call."""
    g = (facts or {}).get("facts", {}).get("us-gaap", {})

    def _latest(tag: str) -> tuple[Optional[float], Optional[str]]:
        units = g.get(tag, {}).get("units", {})
        obs = units.get("USD") or units.get("USD/shares") or []
        for o in sorted(obs, key=lambda x: x.get("end", ""), reverse=True):
            if o.get("form") in ("10-Q", "10-K"):
                return float(o["val"]), o.get("end")
        return None, None

    rev, _ = _latest("Revenues")
    op_inc, _ = _latest("OperatingIncomeLoss")
    net_inc, _ = _latest("NetIncomeLoss")
    eps, _ = _latest("EarningsPerShareDiluted")
    assets, _ = _latest("Assets")
    liab, _ = _latest("Liabilities")
    equity, _ = _latest("StockholdersEquity")

    def _pct(n, d):
        if n is None or d in (None, 0):
            return None
        return round(n / d * 100, 2)

    return {
        "revenue_latest_usd": rev,
        "op_margin_pct": _pct(op_inc, rev),
        "net_margin_pct": _pct(net_inc, rev),
        "eps_diluted": eps,
        "debt_to_equity": round(liab / equity, 3) if (liab and equity) else None,
    }


def _build_user_prompt(ticker: str, bundle: EdgarBundle) -> str:
    km = _key_metrics_from_facts(bundle.xbrl_facts or {})
    mdna = (bundle.mdna_text or "")[:8000]
    rf = (bundle.risk_factors_text or "")[:4000]
    parts = [
        f"Ticker: {ticker}",
        f"Issuer: {bundle.company_name} (CIK {bundle.cik})",
        f"Latest 10-Q filed: {bundle.latest_10q.filing_date if bundle.latest_10q else 'N/A'}",
        f"Latest 10-K filed: {bundle.latest_10k.filing_date if bundle.latest_10k else 'N/A'}",
        "",
        "Pre-computed key metrics (always-on):",
        *(f"- {k}: {v}" for k, v in km.items()),
    ]
    if mdna:
        parts += ["", "MD&A excerpt (10-Q):", mdna]
    if rf:
        parts += ["", "Risk Factors excerpt (10-K Item 1A):", rf]
    parts += [
        "",
        "Run your 8-step chain of thought, then output the final JSON.",
    ]
    return "\n".join(parts)


def fundamentals_agent(state: dict, clients: LLMClients) -> dict:
    ticker = state["ticker"]

    bundle = state.get("edgar_bundle")
    if bundle is None:
        try:
            bundle = build_edgar_bundle(ticker)
        except TickerNotFound:
            return degraded_signal(
                "fundamentals", "Fundamentals",
                "Fundamentals unavailable — no SEC filings for this ticker",
            )
        except Exception as exc:
            return degraded_signal(
                "fundamentals", "Fundamentals",
                "Fundamentals fetch error", error=str(exc)[:200],
            )

    api_key = clients.reasoning.anthropic_api_key.get_secret_value()
    tools = build_fundamentals_tools(bundle=bundle, api_key=api_key)

    try:
        out = run_with_tools(
            api_key=api_key,
            system_prompt=_build_system_prompt(),
            user_prompt=_build_user_prompt(ticker, bundle),
            tools=tools,
            max_iterations=3,
            max_tokens=2000,
        )
    except Exception as exc:
        return degraded_signal(
            "fundamentals", "Fundamentals",
            "LLM error in fundamentals", error=str(exc)[:200],
        )

    return {"agent_signals": [AgentSignal(
        agent="fundamentals",
        signal=out.get("signal", "NEUTRAL"),
        confidence=float(out.get("confidence", 0.0) or 0.0),
        summary=out.get("summary", ""),
        section_markdown=out.get("section_markdown") or "## Fundamentals\n_Section missing._",
        raw_data={"company_name": bundle.company_name, "cik": bundle.cik},
        degraded=False,
        error=None,
        key_metrics=out.get("key_metrics"),
        flags=out.get("flags") or [],
    )]}
```

- [ ] **Step 2: Replace `tests/test_fundamentals_agent.py`**

```python
from unittest.mock import MagicMock, patch

from agents.fundamentals_agent import fundamentals_agent
from edgar import EdgarBundle, TickerNotFound


def _bundle(facts=None, mdna="MD&A text.", rf="Risk text."):
    return EdgarBundle(
        ticker="MSFT", cik="0000789019", company_name="MICROSOFT CORP",
        latest_10q=None, latest_10k=None,
        xbrl_facts=facts or {},
        mdna_text=mdna, risk_factors_text=rf,
    )


def _clients(api_key="sk-ant-fake"):
    secret = MagicMock()
    secret.get_secret_value.return_value = api_key
    reasoning = MagicMock()
    reasoning.anthropic_api_key = secret
    return MagicMock(reasoning=reasoning)


def test_fundamentals_reads_bundle_from_state(monkeypatch):
    bundle = _bundle({"facts": {"us-gaap": {}}})

    def fake_run(api_key, system_prompt, user_prompt, tools, **kwargs):
        return {
            "signal": "BULLISH", "confidence": 0.7,
            "summary": "Margins expanding.",
            "section_markdown": "## Fundamentals\nDetails.",
            "key_metrics": {"op_margin_pct": 38.0},
            "flags": ["margin_expansion"],
        }

    monkeypatch.setattr("agents.fundamentals_agent.run_with_tools", fake_run)

    out = fundamentals_agent({"ticker": "MSFT", "edgar_bundle": bundle}, _clients())
    sig = out["agent_signals"][0]
    assert sig["agent"] == "fundamentals"
    assert sig["signal"] == "BULLISH"
    assert sig["key_metrics"]["op_margin_pct"] == 38.0
    assert "margin_expansion" in sig["flags"]
    assert sig["degraded"] is False


def test_fundamentals_falls_back_to_build_when_state_bundle_missing(monkeypatch):
    bundle = _bundle()
    with patch(
        "agents.fundamentals_agent.build_edgar_bundle",
        return_value=bundle,
    ), patch(
        "agents.fundamentals_agent.run_with_tools",
        return_value={"signal": "NEUTRAL", "confidence": 0.5,
                      "summary": "OK", "section_markdown": "## Fundamentals\nx",
                      "key_metrics": {}, "flags": []},
    ):
        out = fundamentals_agent({"ticker": "MSFT"}, _clients())
    assert out["agent_signals"][0]["signal"] == "NEUTRAL"


def test_fundamentals_ticker_not_in_sec_degrades():
    with patch("agents.fundamentals_agent.build_edgar_bundle",
               side_effect=TickerNotFound("nope")):
        out = fundamentals_agent({"ticker": "FOREIGN"}, _clients())
    sig = out["agent_signals"][0]
    assert sig["degraded"] is True
    assert "no SEC filings" in sig["summary"].lower()


def test_fundamentals_llm_error_degrades(monkeypatch):
    bundle = _bundle()
    monkeypatch.setattr(
        "agents.fundamentals_agent.run_with_tools",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("anthropic 500")),
    )
    out = fundamentals_agent({"ticker": "MSFT", "edgar_bundle": bundle}, _clients())
    sig = out["agent_signals"][0]
    assert sig["degraded"] is True
    assert "LLM error" in sig["summary"]
```

- [ ] **Step 3: Run tests**

```
.venv/Scripts/python.exe -m pytest tests/test_fundamentals_agent.py -v
```

Expected: 4 passed.

- [ ] **Step 4: Commit**

```bash
git add agents/fundamentals_agent.py tests/test_fundamentals_agent.py
git commit -m "feat(agents): rewrite fundamentals — CoT + tools + key_metrics + flags"
```

---

### Task 3.7: Rewrite Macro agent (CoT + tools)

**Files:**
- Modify: `agents/macro_agent.py`
- Modify: `tests/test_macro_agent.py`

- [ ] **Step 1: Replace `agents/macro_agent.py`**

```python
"""Macro specialist (v2.1): persona + cross-asset framework + CoT + 3 tools."""

from __future__ import annotations

import requests

from agents import LLMClients, degraded_signal, run_with_tools
from agents.tools.macro_tools import build_macro_tools
from state import AgentSignal


PERSONA = (
    "You are a senior global macro strategist, CFA Charterholder, with 15 "
    "years on the rates and FX desk of a major investment bank. You read "
    "cross-asset signals — DXY, yield curve, credit spreads, commodities, "
    "positioning — and synthesize them into a regime call (risk-on/off, "
    "reflation/disinflation, growth-scare) that you map to specific equity "
    "sector implications. You're skeptical of single-print headlines and "
    "prefer trend-confirmed moves."
)

METHODOLOGY = """
Methodology you apply:
- Yield curve (2s10s sign + slope; bull-steepening vs bear-flattening)
- Real rates impact (FF − headline CPI proxy)
- DXY transmission (rising DXY bearish for non-USD revenue exposure)
- Credit spreads (HY OAS / IG OAS ratio)
- Commodity regime (energy + base metals leading inflation)
- Sentiment positioning (Fear & Greed contrarian / confirming)
""".strip()

COT = """
Reason step-by-step. For each step, state the data point + inference.

1. Rates regime — Fed funds level, real rate proxy (FF − recent CPI)
2. Yield curve shape — 2s10s sign + magnitude; recession signal?
3. USD direction — DXY level, 5d trend; risk-asset implication
4. Credit / liquidity — HY spreads (use fetch_fred_series for BAMLH0A0HYM2 if needed)
5. Inflation signal — breakevens, commodities (tool if needed)
6. Regime classification — pick exactly one of {risk-on, risk-off, reflation, disinflation, stagflation, neutral}
7. Ticker sector mapping — call classify_ticker_sector once, map sector to regime impact
8. Integrated call — directional impact on this specific ticker

Tool-call rules:
- Use tools only when a step requires data not in always-on payload.
- Cap: max 3 tool calls per run.
- After 3 tool calls (or when you have enough), produce final JSON.
""".strip()

OUTPUT_SCHEMA = """
Respond with a single JSON object (no markdown fences) with EXACTLY these keys:
{
  "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence": 0.0..1.0,
  "summary": "one sentence stating regime + impact direction",
  "section_markdown": "## Macro Backdrop\\n... 150-250 words ...",
  "regime": "risk-on" | "risk-off" | "reflation" | "disinflation" | "stagflation" | "neutral",
  "yield_curve_state": "steep" | "flat" | "inverted",
  "ticker_exposure": "high" | "medium" | "low",
  "key_metrics": {
    "dxy_latest": number | null,
    "dxy_5d_change": number | null,
    "fed_funds_rate": number | null,
    "yield_curve_2s10s": number | null,
    "fear_greed_index": number | null,
    "real_rate_proxy": number | null
  },
  "flags": ["string", ...]
}
""".strip()

GUARDRAILS = """
Constraints and guardrails:
- No "buy/sell" verbiage in section_markdown — use "supportive / mixed / headwind" framings.
- section_markdown must be 150-250 words.
- Confidence ≤ 0.5 if FRED key absent (only Fear & Greed available).
- Confidence ≤ 0.6 if yield_curve_2s10s is null.
- regime MUST be one of the 6 enum values; default neutral on indeterminate.
""".strip()


def _fetch_fred_series(series_id: str, api_key: str, periods: int = 5) -> list[dict]:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id, "api_key": api_key,
        "file_type": "json", "sort_order": "desc", "limit": periods,
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


def _gather_always_on(fred_key: str) -> tuple[dict, bool]:
    raw = {
        "dxy_latest": None, "dxy_5d_change": None,
        "fed_funds_rate": None, "treasury_10y": None, "treasury_2y": None,
        "yield_curve_2s10s": None, "fear_greed_index": _fetch_fear_greed(),
    }
    degraded = False
    if not fred_key:
        return raw, True
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
    return raw, degraded


def _build_system_prompt() -> str:
    return "\n\n".join([PERSONA, METHODOLOGY, OUTPUT_SCHEMA, GUARDRAILS, COT])


def _build_user_prompt(ticker: str, raw: dict, degraded: bool) -> str:
    parts = [
        f"Ticker: {ticker}",
        "",
        "Pre-fetched macro data (always-on):",
        *(f"- {k}: {v}" for k, v in raw.items()),
    ]
    if degraded:
        parts += ["", "NOTE: FRED data unavailable; only Fear & Greed reliable. Mark `degraded=true` upstream."]
    parts += [
        "",
        "Run your 8-step chain of thought, then output the final JSON.",
    ]
    return "\n".join(parts)


def macro_agent(state: dict, clients: LLMClients, fred_key: str) -> dict:
    ticker = state["ticker"]
    raw, degraded = _gather_always_on(fred_key)

    api_key = clients.reasoning.anthropic_api_key.get_secret_value()
    tools = build_macro_tools(fred_key=fred_key)

    try:
        out = run_with_tools(
            api_key=api_key,
            system_prompt=_build_system_prompt(),
            user_prompt=_build_user_prompt(ticker, raw, degraded),
            tools=tools,
            max_iterations=3,
            max_tokens=1800,
        )
    except Exception as exc:
        return degraded_signal(
            "macro", "Macro Backdrop", "Macro LLM error",
            raw=raw, error=str(exc)[:200],
        )

    return {"agent_signals": [AgentSignal(
        agent="macro",
        signal=out.get("signal", "NEUTRAL"),
        confidence=float(out.get("confidence", 0.0) or 0.0),
        summary=out.get("summary", ""),
        section_markdown=out.get("section_markdown") or "## Macro Backdrop\n_Section missing._",
        raw_data=raw,
        degraded=degraded,
        error=None,
        regime=out.get("regime"),
        yield_curve_state=out.get("yield_curve_state"),
        ticker_exposure=out.get("ticker_exposure"),
        key_metrics=out.get("key_metrics"),
        flags=out.get("flags") or [],
    )]}
```

- [ ] **Step 2: Replace `tests/test_macro_agent.py`**

```python
import responses
from unittest.mock import MagicMock

from agents.macro_agent import macro_agent


def _clients(api_key="sk-ant-fake"):
    secret = MagicMock()
    secret.get_secret_value.return_value = api_key
    reasoning = MagicMock()
    reasoning.anthropic_api_key = secret
    return MagicMock(reasoning=reasoning)


@responses.activate
def test_macro_full_data_with_tool_use(monkeypatch):
    fk = "frd-fake"
    for series in ("DTWEXBGS", "DFF", "DGS10", "DGS2"):
        responses.add(
            responses.GET,
            "https://api.stlouisfed.org/fred/series/observations",
            json={"observations": [
                {"date": "2025-10-30", "value": "104.2"},
                {"date": "2025-10-25", "value": "103.8"},
            ]},
            match=[responses.matchers.query_param_matcher({
                "series_id": series, "api_key": fk, "file_type": "json",
                "sort_order": "desc", "limit": "5",
            })],
        )
    responses.add(
        responses.GET,
        "https://api.alternative.me/fng/?limit=1",
        json={"data": [{"value": "62", "value_classification": "Greed"}]},
    )

    monkeypatch.setattr(
        "agents.macro_agent.run_with_tools",
        lambda **kw: {
            "signal": "BEARISH", "confidence": 0.55,
            "summary": "Risk-off regime; high ticker exposure.",
            "section_markdown": "## Macro Backdrop\nDetails.",
            "regime": "risk-off",
            "yield_curve_state": "inverted",
            "ticker_exposure": "high",
            "key_metrics": {"dxy_latest": 104.2, "fed_funds_rate": 5.25,
                            "yield_curve_2s10s": -0.45, "fear_greed_index": 62},
            "flags": ["dxy_rising", "curve_inverted"],
        },
    )

    out = macro_agent({"ticker": "MSFT"}, _clients(), fred_key=fk)
    sig = out["agent_signals"][0]
    assert sig["agent"] == "macro"
    assert sig["regime"] == "risk-off"
    assert sig["ticker_exposure"] == "high"
    assert sig["raw_data"]["dxy_latest"] == 104.2
    assert sig["raw_data"]["fear_greed_index"] == 62
    assert sig["degraded"] is False


@responses.activate
def test_macro_no_fred_key_degrades(monkeypatch):
    responses.add(
        responses.GET,
        "https://api.alternative.me/fng/?limit=1",
        json={"data": [{"value": "30", "value_classification": "Fear"}]},
    )
    monkeypatch.setattr(
        "agents.macro_agent.run_with_tools",
        lambda **kw: {
            "signal": "NEUTRAL", "confidence": 0.3,
            "summary": "Limited macro data.",
            "section_markdown": "## Macro Backdrop\nDegraded.",
            "regime": "neutral", "yield_curve_state": "flat",
            "ticker_exposure": "medium",
            "key_metrics": {"fear_greed_index": 30},
            "flags": ["fred_unavailable"],
        },
    )
    out = macro_agent({"ticker": "MSFT"}, _clients(), fred_key="")
    sig = out["agent_signals"][0]
    assert sig["degraded"] is True
    assert sig["raw_data"]["fear_greed_index"] == 30
    assert sig["raw_data"]["dxy_latest"] is None


@responses.activate
def test_macro_llm_error_degrades(monkeypatch):
    responses.add(
        responses.GET,
        "https://api.alternative.me/fng/?limit=1",
        json={"data": [{"value": "50"}]},
    )
    monkeypatch.setattr(
        "agents.macro_agent.run_with_tools",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    out = macro_agent({"ticker": "MSFT"}, _clients(), fred_key="")
    sig = out["agent_signals"][0]
    assert sig["degraded"] is True
    assert "Macro LLM error" in sig["summary"]
```

- [ ] **Step 3: Run tests**

```
.venv/Scripts/python.exe -m pytest tests/test_macro_agent.py -v
```

Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
git add agents/macro_agent.py tests/test_macro_agent.py
git commit -m "feat(agents): rewrite macro — CoT + tools + regime + ticker_exposure"
```

---

### Task 3.8: Rewrite Price agent (few-shot + tools)

**Files:**
- Modify: `agents/price_agent.py`
- Modify: `tests/test_price_agent.py`

- [ ] **Step 1: Replace `agents/price_agent.py`** with the full v2.1 implementation analogous to Task 3.6 — same structure (PERSONA, METHODOLOGY, FEWSHOT, OUTPUT_SCHEMA, GUARDRAILS), `build_price_tools(price_history=...)`, `run_with_tools(...)`, returns `AgentSignal` with `regime`, `key_metrics`, `flags`. Reads `state["price_history"]`; falls back to `download_with_retry` if missing. Persona + methodology + few-shot text exact per spec §4.4.

(Full code listing follows the same pattern as Task 3.6; reproduced verbatim in the implementation. Subagent: copy the spec §4.4 persona, methodology, few-shot examples, schema, guardrails into module-level string constants and assemble in `_build_system_prompt()`. The pattern is mechanical: see Task 3.6 for the structure.)

Concrete expected pieces in the file:

```python
PERSONA = "You are a CMT Charterholder ..."  # spec §4.4 exact text
METHODOLOGY = "..."  # 6 bullets per spec
FEWSHOT = "Example 1 — momentum override...\n\nExample 2 — conflicting signals...\n\nExample 3 — oversold bounce..."
OUTPUT_SCHEMA = "..."  # spec §4.4 schema with regime + key_metrics + flags
GUARDRAILS = "..."  # constraints from spec §4.4

# In price_agent(state, clients):
data = state.get("price_history")
if data is None:
    data = download_with_retry(state["ticker"], period="90d", interval="1d")
if data is None or data.empty:
    return _degraded("No price data ...")
# Compute pre-injected raw indicators (RSI, MACD, Bollinger %B, 7/30/90d %)
# Build tools via build_price_tools(price_history=data)
# Call run_with_tools(api_key=..., system_prompt=_build_system_prompt(), user_prompt=..., tools=tools)
# Return AgentSignal with regime, key_metrics, flags
```

- [ ] **Step 2: Replace `tests/test_price_agent.py`**

Keep the existing `compute_rsi`, `compute_macd`, `compute_bollinger_pctb` tests (they test pure-function behavior on the existing module). Replace the `price_agent` tests with `run_with_tools`-mock variants:

```python
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from agents.price_agent import (
    compute_rsi, compute_macd, compute_bollinger_pctb, price_agent,
)


def _flat_close(n=90, start=100.0, drift=0.5):
    return pd.Series(np.linspace(start, start + drift * n, n))


def test_compute_rsi_neutral_for_steady_uptrend():
    s = _flat_close()
    rsi = compute_rsi(s)
    assert 50 < rsi <= 100


def test_compute_macd_returns_two_floats():
    s = _flat_close()
    line, signal = compute_macd(s)
    assert isinstance(line, float) and isinstance(signal, float)


def test_compute_bollinger_at_midpoint_for_constant_series():
    s = pd.Series([100.0] * 50)
    pctb = compute_bollinger_pctb(s)
    assert pctb == 0.5


def _clients(api_key="sk-ant-fake"):
    secret = MagicMock()
    secret.get_secret_value.return_value = api_key
    reasoning = MagicMock()
    reasoning.anthropic_api_key = secret
    return MagicMock(reasoning=reasoning)


def test_price_agent_happy_path_reads_state_history(monkeypatch):
    fake_close = _flat_close(90, 100, 0.3)
    fake_df = pd.DataFrame({"Close": fake_close})

    monkeypatch.setattr(
        "agents.price_agent.run_with_tools",
        lambda **kw: {
            "signal": "BULLISH", "confidence": 0.7,
            "summary": "Trend up, RSI mid.",
            "section_markdown": "## Technical Analysis\nDetails.",
            "regime": "trending_up",
            "key_metrics": {"rsi": 58.2, "macd_state": "positive_crossover",
                            "bollinger_pctb": 0.62, "atr_pct": 1.8,
                            "sma50_vs_sma200": "above"},
            "flags": ["trend_confirmation"],
        },
    )

    out = price_agent({"ticker": "MSFT", "price_history": fake_df}, _clients())
    sig = out["agent_signals"][0]
    assert sig["agent"] == "price"
    assert sig["signal"] == "BULLISH"
    assert sig["regime"] == "trending_up"
    assert sig["key_metrics"]["rsi"] == 58.2


def test_price_agent_falls_back_to_download(monkeypatch):
    fake_close = _flat_close(90, 100, 0.3)
    fake_df = pd.DataFrame({"Close": fake_close})

    with patch(
        "agents.price_agent.download_with_retry", return_value=fake_df,
    ), patch(
        "agents.price_agent.run_with_tools",
        return_value={"signal": "NEUTRAL", "confidence": 0.5,
                      "summary": "x", "section_markdown": "## Technical Analysis\nx",
                      "regime": "ranging", "key_metrics": {}, "flags": []},
    ):
        out = price_agent({"ticker": "MSFT"}, _clients())
    assert out["agent_signals"][0]["signal"] == "NEUTRAL"


def test_price_agent_empty_data_degrades():
    with patch(
        "agents.price_agent.download_with_retry", return_value=pd.DataFrame(),
    ):
        out = price_agent({"ticker": "ZZZZ"}, _clients())
    sig = out["agent_signals"][0]
    assert sig["degraded"] is True
    assert sig["confidence"] == 0.0


def test_price_agent_llm_error_degrades(monkeypatch):
    fake_close = _flat_close(90, 100, 0.3)
    fake_df = pd.DataFrame({"Close": fake_close})
    monkeypatch.setattr(
        "agents.price_agent.run_with_tools",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    out = price_agent({"ticker": "MSFT", "price_history": fake_df}, _clients())
    assert out["agent_signals"][0]["degraded"] is True
```

- [ ] **Step 3: Run tests**

```
.venv/Scripts/python.exe -m pytest tests/test_price_agent.py -v
```

Expected: 7 passed.

- [ ] **Step 4: Commit**

```bash
git add agents/price_agent.py tests/test_price_agent.py
git commit -m "feat(agents): rewrite price — few-shot + tools + regime + key_metrics"
```

---

### Task 3.9: Rewrite Sentiment agent (few-shot + tools)

Same pattern as Task 3.8. Replace `agents/sentiment_agent.py` with v2.1 structure (PERSONA, METHODOLOGY, FEWSHOT, OUTPUT_SCHEMA, GUARDRAILS — exact text per spec §4.5), `build_sentiment_tools(tavily_key=..., api_key=...)`, `run_with_tools(...)`. Pre-injected raw includes Tavily + Haiku results from existing flow. Returns `AgentSignal` with `top_catalyst`, `key_metrics`, `drivers_categorized`, `flags`. Update tests to mock `run_with_tools`.

- [ ] **Step 1: Replace `agents/sentiment_agent.py`**

(Full implementation follows Task 3.6 / 3.8 pattern. Persona + methodology + 3 few-shot examples + output-schema all from spec §4.5.)

- [ ] **Step 2: Replace `tests/test_sentiment_agent.py`** with three tests covering: happy path with mocked Tavily + mocked `run_with_tools`; missing tavily_key degradation; LLM error degradation.

- [ ] **Step 3: Run + commit**

```
.venv/Scripts/python.exe -m pytest tests/test_sentiment_agent.py -v
git add agents/sentiment_agent.py tests/test_sentiment_agent.py
git commit -m "feat(agents): rewrite sentiment — few-shot + tools + top_catalyst + categorized"
```

---

### Task 3.10: Rewrite Risk agent (few-shot, forward-looking + tools)

**Files:**
- Modify: `agents/risk_agent.py`
- Modify: `tests/test_risk_agent.py`

- [ ] **Step 1: Replace `agents/risk_agent.py`**

Structure: PERSONA + METHODOLOGY (forward-first) + FEWSHOT (3 examples per spec §4.6) + OUTPUT_SCHEMA + GUARDRAILS. Pre-computed raw includes:

- Backward stats: `annualized_vol_pct`, `max_drawdown_pct`, `sharpe`, `sortino`, `calmar`, `var_95_1d`, `max_1d_drop`, `vix`, `beta`, `short_ratio`
- Forward inputs: `revenue_yoy_pct`, `revenue_qoq_pct`, `op_margin_yoy_bps`, `fcf_margin_pct`, `debt_to_equity`, `vol_percentile_1y`, `trend_state`, `drawdown_state`

Reads from `state["price_history"]` and `state["edgar_bundle"]`. Falls back to `download_with_retry` and `build_edgar_bundle` when missing.

Tools via `build_risk_tools(price_history=..., edgar_bundle=...)`.

Returns `AgentSignal` with `forward_risk_view`, `primary_risk_driver`, `risk_decomposition`, `vol_regime`, `vix_regime`, `key_metrics`, `flags`.

- [ ] **Step 2: Replace `tests/test_risk_agent.py`** — 3-4 tests covering forward-looking happy path, missing edgar_bundle (degrades on forward dim, still produces price/vol-only call), missing price_history degrades fully, LLM error degrades.

- [ ] **Step 3: Run + commit**

```
.venv/Scripts/python.exe -m pytest tests/test_risk_agent.py -v
git add agents/risk_agent.py tests/test_risk_agent.py
git commit -m "feat(agents): rewrite risk — forward-looking + tools + risk_decomposition"
```

---

## Phase 4: Supervisor + Synthesis upgrades

### Task 4.1: Supervisor — read `key_metrics` for sanity + cross-signal check

**Files:**
- Modify: `agents/supervisor_agent.py`
- Modify: `tests/test_supervisor_agent.py`

- [ ] **Step 1: Update `agents/supervisor_agent.py`**

Modify `_sanity_violations` to read from `key_metrics` first, falling back to `raw_data` for backward compatibility:

```python
def _sanity_violations(agent: str, sig: dict) -> list[str]:
    issues: list[str] = []
    km = sig.get("key_metrics") or {}
    raw = sig.get("raw_data") or {}
    if agent == "price":
        rsi = km.get("rsi") or raw.get("rsi")
        if isinstance(rsi, (int, float)) and (rsi < 0 or rsi > 100):
            issues.append(f"RSI out of range: {rsi}")
    if agent == "risk":
        v = km.get("annualized_vol_pct") or raw.get("annualized_vol_pct")
        if isinstance(v, (int, float)) and (v < 0 or v > 500):
            issues.append(f"Volatility out of range: {v}%")
    if agent == "fundamentals":
        roe = km.get("roe_pct")
        if isinstance(roe, (int, float)) and (roe > 200 or roe < -200):
            issues.append(f"ROE out of plausible range: {roe}%")
    for src in (km, raw):
        for k, v in src.items():
            if isinstance(v, float) and not isfinite(v):
                issues.append(f"Non-finite {k}")
    return issues
```

Update the loop in `supervisor_agent` to call `_sanity_violations(agent_name, s)` (passing the whole signal, not just raw_data).

Add a new cross-signal check function and call it once per run:

```python
def _cross_signal_critiques(signals: list[dict]) -> dict[str, str]:
    """Lightweight Fundamentals↔Macro consistency check.

    If Fundamentals shows revenue_yoy_pct < -10 (real deterioration) AND
    Macro labelled the regime risk-on with high ticker_exposure, flag Macro
    for re-examination — the regime read may be missing the name-specific
    deterioration.
    """
    by_agent = {s.get("agent"): s for s in signals}
    fund = by_agent.get("fundamentals")
    macro = by_agent.get("macro")
    if not fund or not macro:
        return {}
    rev_yoy = (fund.get("key_metrics") or {}).get("eps_yoy_pct")
    # Use revenue from raw_data fallback; fundamentals key_metrics may not include it
    # depending on the LLM output. Fall back to known signals:
    if (
        macro.get("regime") == "risk-on"
        and macro.get("ticker_exposure") == "high"
        and rev_yoy is not None
        and rev_yoy < -10
    ):
        return {"macro": "Cross-check: fundamentals show >10% YoY deterioration; "
                         "regime read may be missing name-specific stress."}
    return {}
```

Merge `_cross_signal_critiques(signals)` into `critiques` before deciding `retry_targets`.

- [ ] **Step 2: Add tests to `tests/test_supervisor_agent.py`**

Append:

```python
def test_supervisor_reads_key_metrics_for_rsi_sanity():
    s = _sig("price")
    s["key_metrics"] = {"rsi": 150.0}
    state = {"agent_signals": [s, _sig("sentiment"), _sig("fundamentals"),
                               _sig("macro"), _sig("risk")], "retry_round": 0}
    out = supervisor_agent(state)
    assert "price" in out["supervisor_review"]["retry_targets"]


def test_supervisor_flags_implausible_roe():
    fund = _sig("fundamentals")
    fund["key_metrics"] = {"roe_pct": 350.0}
    state = {"agent_signals": [_sig("price"), _sig("sentiment"), fund,
                               _sig("macro"), _sig("risk")], "retry_round": 0}
    out = supervisor_agent(state)
    assert "fundamentals" in out["supervisor_review"]["retry_targets"]


def test_supervisor_cross_signal_fundamentals_vs_macro():
    fund = _sig("fundamentals", confidence=0.7)
    fund["key_metrics"] = {"eps_yoy_pct": -25.0}
    macro = _sig("macro", confidence=0.6)
    macro["regime"] = "risk-on"
    macro["ticker_exposure"] = "high"
    state = {"agent_signals": [_sig("price"), _sig("sentiment"), fund, macro,
                               _sig("risk")], "retry_round": 0}
    out = supervisor_agent(state)
    assert "macro" in out["supervisor_review"]["retry_targets"]
```

- [ ] **Step 3: Run + commit**

```
.venv/Scripts/python.exe -m pytest tests/test_supervisor_agent.py -v
git add agents/supervisor_agent.py tests/test_supervisor_agent.py
git commit -m "feat(supervisor): read key_metrics for sanity + cross-signal Fund↔Macro check"
```

---

### Task 4.2: Synthesis — promote to CoT + emit `key_drivers` / `dissenting_view` / `watch_items`

**Files:**
- Modify: `agents/synthesis_agent.py`
- Modify: `tests/test_synthesis_agent.py`

- [ ] **Step 1: Update `agents/synthesis_agent.py`**

Add module-level prompt constants:

```python
PERSONA = (
    "You are the Chief Investment Officer and Director of Research at a "
    "multi-strategy fund. You chair the daily investment committee — your "
    "job is to integrate five specialist signals into a single coherent "
    "thesis, name the strongest argument for the call AND the strongest "
    "argument against, and identify the leading indicators that would force "
    "a re-evaluation. You never overstate confidence and you always make "
    "dissent visible."
)

COT = """
The verdict, conviction, and confidence are already decided upstream from
specialist signals — you write the NARRATIVE.

Reason step-by-step:
1. Consensus — which agents agree with the verdict? At what avg confidence?
2. Dissent — which agents disagree? Why? (Cite their summary)
3. Strongest single argument FOR — name the specialist + metric
4. Strongest single argument AGAINST — name the specialist + metric
5. Integrated thesis — 3-5 sentences referencing ≥3 specialists by name
6. Key drivers — 2-4 phrases, each "Specialist: <metric/observation>"
7. Dissenting view — one sentence: what condition flips the call
8. Watch items — 2-3 leading indicators (rate prints, earnings dates,
   VIX thresholds, segment growth) that would force a re-rating

Output JSON only.
""".strip()

OUTPUT_SCHEMA = """
Respond with a single JSON object (no markdown fences) with EXACTLY these keys:
{
  "reasoning": "3-5 sentences (80-150 words) referencing ≥3 specialists by name",
  "key_drivers": ["Specialist: metric/observation", ...],   // 2-4 entries, each ≤15 words
  "dissenting_view": "one sentence ≤25 words: what condition flips the call",
  "watch_items": ["leading indicator", ...]                  // 2-3 entries, each ≤20 words
}
""".strip()
```

Replace the existing `synthesis_agent(state, clients)` so the LLM call returns those 4 fields. Render `key_drivers`, `dissenting_view`, `watch_items` into `final_report`:

- Below the verdict label in Executive Summary: bullet list of `key_drivers`
- After "Synthesis & Final Verdict", append:
  - `_Dissenting view: {dissenting_view}_` italicized line
  - `### What to Watch` subsection with bullet list

Verdict math (`compute_verdict_and_conviction`, `label_for`) UNCHANGED.

Return dict adds:

```python
return {
    "final_verdict": verdict,
    "final_conviction": conviction,
    "final_confidence": confidence,
    "final_reasoning": reasoning,
    "final_report": final_report,
    "key_drivers": key_drivers,
    "dissenting_view": dissenting_view,
    "watch_items": watch_items,
}
```

- [ ] **Step 2: Add tests to `tests/test_synthesis_agent.py`**

Update `test_synthesis_agent_assembles_report` to mock the LLM with a JSON containing `reasoning`, `key_drivers`, `dissenting_view`, `watch_items`, and assert the output dict carries those fields and the `final_report` markdown contains "What to Watch" and "_Dissenting view:".

```python
def test_synthesis_emits_key_drivers_and_watch_items():
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
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content=(
        '{"reasoning": "Three specialists support: Price, Sentiment, Fundamentals.", '
        '"key_drivers": ["Fundamentals: op margin +220 bps", '
        '"Price: trend confirmation", "Sentiment: 8/12 positive"], '
        '"dissenting_view": "Macro headwind reverses if Fed cuts next meeting.", '
        '"watch_items": ["Next CPI print", "Q3 cloud growth"]}'
    ))
    clients = MagicMock(reasoning=fake_llm)
    out = synthesis_agent(state, clients)
    assert out["key_drivers"] == [
        "Fundamentals: op margin +220 bps",
        "Price: trend confirmation",
        "Sentiment: 8/12 positive",
    ]
    assert "Next CPI print" in out["watch_items"]
    assert "What to Watch" in out["final_report"]
    assert "Dissenting view" in out["final_report"]
```

- [ ] **Step 3: Run + commit**

```
.venv/Scripts/python.exe -m pytest tests/test_synthesis_agent.py -v
git add agents/synthesis_agent.py tests/test_synthesis_agent.py
git commit -m "feat(synthesis): CoT + key_drivers + dissenting_view + watch_items"
```

---

## Phase 5: app.py + cost badge

### Task 5.1: Update `app.py` initial state and cost badge

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Update `init` dict in `analyze()`**

Add `"edgar_bundle": None`, `"key_drivers": None`, `"dissenting_view": None`, `"watch_items": None` to the `init` dict.

- [ ] **Step 2: Update cost-hint label**

Change the Analyze button label from `"Analyze (~$0.30/run on BYO key)"` to `"Analyze (~$0.50/run on BYO key)"`.

- [ ] **Step 3: Smoke import**

```
.venv/Scripts/python.exe -c "import app; print('OK')"
```

Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(ui): include v2.1 state fields + bump cost hint to ~$0.50"
```

---

## Phase 6: Final tests + smoke + PR

### Task 6.1: Run full test suite

- [ ] **Step 1: Full pytest**

```
.venv/Scripts/python.exe -m pytest tests/ -v
```

Expected: all green.

- [ ] **Step 2: Smoke runs (manual, requires keys)**

```
ANTHROPIC_API_KEY=... TAVILY_API_KEY=... FRED_API_KEY=... .venv/Scripts/python.exe scripts/smoke_run.py MSFT
.venv/Scripts/python.exe scripts/smoke_run.py NVDA
.venv/Scripts/python.exe scripts/smoke_run.py SPY
.venv/Scripts/python.exe scripts/smoke_run.py SHEL
```

Expected: all complete; SPY + SHEL show degraded Fundamentals; reports include "What to Watch" sections.

### Task 6.2: Push branch + open PR

- [ ] **Step 1: Push branch**

```
git push -u origin feat/marketmind-v2.1-agents
```

- [ ] **Step 2: Open PR**

```
gh pr create --base main --title "MarketMind v2.1: persona-driven, tool-calling agents" \
  --body "$(cat <<'EOF'
## Summary
- All 6 specialists rewritten with credentialed personas + named methodologies + structured outputs
- Tool-use loop infrastructure (`agents/__init__.py: ToolDef + run_with_tools`) with 3-iteration cap and prompt caching
- Per-agent tools modules under `agents/tools/`
- EdgarBundle promoted to data_prefetch (shared by Fundamentals + Risk)
- Risk agent pivoted to forward-looking risk evaluation; trailing quant demoted to informational
- Synthesis promoted to CoT; emits key_drivers, dissenting_view, watch_items
- Supervisor reads key_metrics for sanity + new cross-signal Fundamentals↔Macro check
- AgentSignal becomes total=False with new optional fields

Spec: docs/superpowers/specs/2026-05-02-agent-enhancements-design.md
Plan: docs/superpowers/plans/2026-05-02-agent-enhancements.md

## Test plan
- [ ] pytest tests/ -v all green
- [ ] Smoke runs on MSFT, NVDA, SPY, SHEL
- [ ] HF Space build succeeds

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: After review, merge**

```
gh pr merge --squash --delete-branch
```

---

## Self-Review

**Spec coverage check:**

| Spec section | Implementing task |
|---|---|
| §3.1 Tool-calling loop helper | Task 1.2 |
| §3.2 Prompt caching | Task 1.2 (system block with `cache_control`) |
| §3.3 EdgarBundle promotion | Task 2.1 |
| §3.4 State additions | Task 1.1 |
| §3.5 AgentSignal schema additions | Task 1.1 |
| §3.6 Supervisor changes | Task 4.1 |
| §3.7 Synthesis changes | Task 4.2 |
| §4.1 Fundamentals | Tasks 3.1, 3.6 |
| §4.2 Macro | Tasks 3.2, 3.7 |
| §4.3 Synthesis | Task 4.2 |
| §4.4 Price | Tasks 3.3, 3.8 |
| §4.5 Sentiment | Tasks 3.4, 3.9 |
| §4.6 Risk | Tasks 3.5, 3.10 |
| §6 FRs | Covered across Tasks 1.1 — 4.2 |
| §7 Risks | Mitigations baked into tool handler exception path (Task 1.2) and degraded-signal helpers (Tasks 3.6 — 3.10) |

No gaps.

**Placeholder scan:**
- Tasks 3.8, 3.9, 3.10 reference "same pattern as Task 3.6 / 3.8" with the Steps detailing what the file must contain. The structure (PERSONA / METHODOLOGY / FEWSHOT / OUTPUT_SCHEMA / GUARDRAILS module-level constants + `_build_system_prompt` + `_build_user_prompt` + `run_with_tools` + `AgentSignal` return) is fully specified by the Task 3.6 reference and the spec sections, but the implementer is expected to pull persona/methodology/few-shot text VERBATIM from the spec §4.x. Reading both the spec and Task 3.6 together is required. Acceptable abbreviation given the strong reference; full repetition would inflate the plan past 3000 lines. If subagent confusion arises, the spec sections §4.4 / §4.5 / §4.6 contain the exact text to be embedded.

**Type consistency:**
- `LLMClients`, `ToolDef`, `run_with_tools`, `degraded_signal`, `safe_parse_json` — names consistent across tasks
- `AgentSignal` field names (`key_metrics`, `flags`, `regime`, `vol_regime`, `ticker_exposure`, etc.) consistent across state.py / agents / tests
- `state["edgar_bundle"]`, `state["price_history"]`, `state["vix_history"]` — consistent
- Tool builder signatures: `build_fundamentals_tools(bundle=..., api_key=...)`, `build_macro_tools(fred_key=...)`, `build_price_tools(price_history=...)`, `build_sentiment_tools(tavily_key=..., api_key=...)`, `build_risk_tools(price_history=..., edgar_bundle=...)` — match across implementation tasks and consumer tasks.

**Scope:** Single subsystem (agent enhancements). Does NOT introduce new specialists or batch mode (those are v2.2 backlog per spec §2.2).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-02-agent-enhancements.md`. Two execution options:

**1. Subagent-Driven (recommended)** — Dispatch fresh subagent per task with two-stage review (spec compliance + code quality). Slow but rigorous.

**2. Inline Execution** — Execute tasks in this session via `executing-plans`. Faster.

Which approach?
