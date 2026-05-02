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
