from unittest.mock import MagicMock

from agents.synthesis_agent import (
    _section_for,
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


def test_section_for_replaces_only_first_heading_line():
    # FU7 regression: agent's first ## heading is replaced with the canonical
    # section title, but any subheadings (## Subsection) inside the body must
    # survive intact.
    sigs = [{
        "agent": "price", "signal": "BULLISH", "confidence": 0.7,
        "summary": "ok",
        "section_markdown": (
            "## Original Heading\n"
            "First paragraph of body.\n\n"
            "### Subsection A\n"
            "Detail A.\n\n"
            "## Subsection B\n"
            "Detail B."
        ),
        "raw_data": {}, "degraded": False, "error": None,
    }]
    out = _section_for(sigs, "price")
    assert out.startswith("## Technical Analysis\n")
    # Original heading is gone, body is intact, subheadings survive.
    assert "Original Heading" not in out
    assert "First paragraph of body." in out
    assert "### Subsection A" in out
    assert "## Subsection B" in out
    assert "Detail A." in out
    assert "Detail B." in out


def test_section_for_handles_missing_heading():
    sigs = [{
        "agent": "macro", "signal": "NEUTRAL", "confidence": 0.4,
        "summary": "ok",
        "section_markdown": "Plain text body without a leading heading.",
        "raw_data": {}, "degraded": False, "error": None,
    }]
    out = _section_for(sigs, "macro")
    assert out.startswith("## Macro Backdrop\n")
    assert "Plain text body without a leading heading." in out


def test_section_for_handles_missing_agent():
    out = _section_for([], "risk")
    assert out.startswith("## Risk Profile\n")
    assert "_Agent did not run._" in out


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
    fake_llm.invoke.return_value = MagicMock(content=(
        '{"reasoning": "Three sentences of integrated reasoning that explain why the call holds together across price, sentiment, and fundamentals.",'
        ' "key_drivers": [], "dissenting_view": "", "watch_items": []}'
    ))
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
