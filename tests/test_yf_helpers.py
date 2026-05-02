# tests/test_yf_helpers.py
from unittest.mock import patch

import pandas as pd
import pytest

from agents.yf_helpers import _is_rate_limit, download_with_retry, last_close


class _FakeRateLimitError(Exception):
    """Mimics yfinance.exceptions.YFRateLimitError by name only."""


_FakeRateLimitError.__name__ = "YFRateLimitError"


def test_is_rate_limit_matches_class_name():
    assert _is_rate_limit(_FakeRateLimitError("nope")) is True
    assert _is_rate_limit(ValueError("not rate")) is False


def test_download_with_retry_returns_first_success():
    fake_df = pd.DataFrame({"Close": [1.0, 2.0]})
    with patch("agents.yf_helpers.yf.download", return_value=fake_df) as mock:
        out = download_with_retry("MSFT")
    assert out is fake_df
    assert mock.call_count == 1


def test_download_with_retry_recovers_on_rate_limit():
    fake_df = pd.DataFrame({"Close": [10.0]})
    with patch(
        "agents.yf_helpers.yf.download",
        side_effect=[_FakeRateLimitError("rl"), fake_df],
    ) as mock, patch("agents.yf_helpers.time.sleep"):
        out = download_with_retry("MSFT", max_retries=2, base_backoff=0)
    assert out is fake_df
    assert mock.call_count == 2


def test_download_with_retry_gives_up_after_max_retries():
    with patch(
        "agents.yf_helpers.yf.download",
        side_effect=_FakeRateLimitError("rl"),
    ) as mock, patch("agents.yf_helpers.time.sleep"):
        with pytest.raises(_FakeRateLimitError):
            download_with_retry("MSFT", max_retries=2, base_backoff=0)
    assert mock.call_count == 3  # initial + 2 retries


def test_download_with_retry_does_not_retry_other_errors():
    with patch(
        "agents.yf_helpers.yf.download",
        side_effect=ValueError("not rate-limit"),
    ) as mock:
        with pytest.raises(ValueError):
            download_with_retry("MSFT", max_retries=3, base_backoff=0)
    assert mock.call_count == 1


def test_last_close_handles_empty_df():
    assert last_close(pd.DataFrame()) is None
    assert last_close(None) is None
    assert last_close(pd.DataFrame({"Open": [1.0]})) is None


def test_last_close_returns_float_for_simple_df():
    df = pd.DataFrame({"Close": [1.0, 2.0, 3.5]})
    assert last_close(df) == 3.5
    assert isinstance(last_close(df), float)


def test_last_close_squeezes_multilevel_columns():
    # yfinance 1.3 sometimes returns MultiIndex columns for single tickers
    # (top level: OHLCV name, second level: ticker symbol). Selecting df["Close"]
    # then returns a 1-column DataFrame which squeeze() collapses to a Series.
    arrays = [["Close", "Open"], ["MSFT", "MSFT"]]
    cols = pd.MultiIndex.from_arrays(arrays)
    df = pd.DataFrame([[1.0, 100.0], [2.0, 101.0], [22.3, 102.0]], columns=cols)
    out = last_close(df)
    assert out == 22.3
