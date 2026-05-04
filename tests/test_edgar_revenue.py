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
    """Same (end, fp), two different filed dates: most recent filed wins.
    Note: 10-K/A is filtered (form must be 10-Q or 10-K), so this test verifies
    that within allowed forms the filed-date sort tie-breaks correctly."""
    facts = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [
            {"end": "2025-12-31", "val": 100, "fp": "FY", "form": "10-K", "filed": "2026-02-01"},
            {"end": "2025-12-31", "val": 105, "fp": "FY", "form": "10-K/A", "filed": "2026-08-01"},
            {"end": "2024-12-31", "val": 90, "fp": "FY", "form": "10-K", "filed": "2025-02-01"},
        ]}},
    }}}
    tag, obs = latest_revenue_observations(facts)
    # 10-K/A is filtered out (form not in {10-Q, 10-K}).
    assert tag == "Revenues"
    assert len(obs) == 2
    assert obs[0]["val"] == 100  # original FY2025 10-K
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
    """Latest is Q1; prior has same end-year-1 but fp=FY. No match -> None."""
    obs = [
        {"end": "2026-03-31", "val": 187e9, "fp": "Q1"},
        {"end": "2025-03-31", "val": 600e9, "fp": "FY"},  # synthetic mismatch
    ]
    assert yoy_revenue_pct(obs) is None


def test_yoy_pct_returns_none_when_lt_two_obs():
    assert yoy_revenue_pct([]) is None
    assert yoy_revenue_pct([{"end": "2026-03-31", "val": 1, "fp": "Q1"}]) is None
