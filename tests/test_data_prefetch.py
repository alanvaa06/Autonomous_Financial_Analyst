# tests/test_data_prefetch.py
from unittest.mock import patch

import pandas as pd

from agents.data_prefetch import data_prefetch


def test_prefetch_populates_price_and_vix():
    price_df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})
    vix_df = pd.DataFrame({"Close": [20.0, 21.0]})
    with patch(
        "agents.data_prefetch.download_with_retry",
        side_effect=[price_df, vix_df],
    ), patch("agents.data_prefetch.time.sleep"):
        out = data_prefetch({"ticker": "MSFT"})
    assert out["price_history"] is price_df
    assert out["vix_history"] is vix_df


def test_prefetch_swallows_download_errors_and_returns_empty():
    with patch(
        "agents.data_prefetch.download_with_retry",
        side_effect=Exception("rate limited even after retries"),
    ), patch("agents.data_prefetch.time.sleep"):
        out = data_prefetch({"ticker": "ZZZZ"})
    # Both DataFrames are empty (not None) so downstream agents degrade cleanly.
    assert isinstance(out["price_history"], pd.DataFrame)
    assert out["price_history"].empty
    assert isinstance(out["vix_history"], pd.DataFrame)
    assert out["vix_history"].empty


def test_prefetch_partial_failure():
    # Ticker download succeeds; VIX fails. Result has real price_history,
    # empty vix_history.
    price_df = pd.DataFrame({"Close": [10.0, 11.0]})
    with patch(
        "agents.data_prefetch.download_with_retry",
        side_effect=[price_df, Exception("VIX rate limit")],
    ), patch("agents.data_prefetch.time.sleep"):
        out = data_prefetch({"ticker": "MSFT"})
    assert out["price_history"] is price_df
    assert isinstance(out["vix_history"], pd.DataFrame)
    assert out["vix_history"].empty
