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
- **Test fixtures must include all fields a helper filters on.** When
  `latest_revenue_observations` started filtering on `fp ∈ {Q1,Q2,Q3,FY}`,
  pre-existing test bundles missing `fp` silently became empty and a
  happy-path test that never asserted on `revenue_yoy_pct` couldn't catch
  the regression. Always assert on the value the new code produces, not
  just on shape.

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

## 2026-05-04 — Subagent-driven development hygiene

- **Function-local imports are antipatterns when the module is on the
  hot path.** `_yoy_revenue_pct` originally imported `latest_revenue_observations`
  inside the function body. There was no circular-import risk; the
  duplicate-with-top-level-import made the dependency surface invisible
  to readers and static-analysis tools. Hoist eagerly when the import is
  cheap and the call site is on a request path. (Lesson surfaced in Task 1.2
  code review; applied proactively in 1.3 and 1.4.)
- **Phantom report sections are easy to miss.** When a fallback path
  leaves `key_drivers=[]` and `watch_items=[]`, the renderer must check
  for emptiness before writing the section header. Always test the
  fallback path with explicit `key_drivers == []` AND `"Key drivers:" not
  in final_report` assertions.
