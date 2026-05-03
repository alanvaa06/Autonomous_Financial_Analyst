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
