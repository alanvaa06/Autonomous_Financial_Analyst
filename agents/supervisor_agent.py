"""Light-QA supervisor. Cannot rewrite agent content or override verdicts.

Allowed actions:
  1. Flag agents whose output appears broken (hard error from agent code).
  2. Flag direct contradictions (BULLISH vs BEARISH, both confidence ≥ 0.7).
  3. Flag short / empty sections (< 200 chars of section_markdown), unless
     the agent intentionally degraded (degraded=True AND error is None) —
     re-running will not produce a longer section in those cases.
  4. Flag obvious data sanity violations (only when the agent did produce
     real output; intentional-degradation paths skip this check).

The conditional edge after the supervisor reads `supervisor_review.retry_targets`.
On retry_round >= 1, the supervisor force-approves regardless.
"""

from __future__ import annotations

from math import isfinite


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

    # 1. broken outputs + section/sanity checks (skipped on intentional degradation)
    for s in signals:
        intentional_degradation = bool(s.get("degraded")) and not s.get("error")

        if s.get("error"):
            critiques.setdefault(s["agent"], "Hard error from agent.")

        # FU3: re-running an intentionally-degraded agent (e.g., FRED key absent,
        # ticker not in SEC universe) won't change anything — short-circuit the
        # short-section and sanity flags so the supervisor doesn't burn a retry
        # round on a guaranteed no-op.
        if intentional_degradation:
            continue

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
