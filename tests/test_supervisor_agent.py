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


def test_supervisor_skips_intentional_degradation():
    # FU3: agent intentionally degraded (degraded=True, error=None) with a
    # short section_markdown — re-running won't help (e.g. FRED key still absent),
    # so supervisor should NOT flag for retry.
    state = {"agent_signals": [
        _sig("price"),
        _sig("macro", confidence=0.0, degraded=True,
             section_markdown="## Macro Backdrop\n_Unavailable: No FRED key_"),
        _sig("sentiment"), _sig("fundamentals"), _sig("risk"),
    ], "retry_round": 0}
    out = supervisor_agent(state)
    assert "macro" not in out["supervisor_review"]["retry_targets"]
    assert out["supervisor_review"]["approved"] is True


def test_supervisor_still_flags_hard_error():
    # The intentional-degradation short-circuit must NOT swallow real errors.
    # When degraded=True AND error is set, the agent failed and should be retried.
    state = {"agent_signals": [
        _sig("price"),
        _sig("macro", confidence=0.0, degraded=True, error="boom",
             section_markdown="## Macro Backdrop\nshort"),
        _sig("sentiment"), _sig("fundamentals"), _sig("risk"),
    ], "retry_round": 0}
    out = supervisor_agent(state)
    assert "macro" in out["supervisor_review"]["retry_targets"]
