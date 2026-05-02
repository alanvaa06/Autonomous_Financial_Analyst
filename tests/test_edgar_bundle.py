# tests/test_edgar_bundle.py
import responses
from edgar import build_edgar_bundle, _CIK_CACHE, _SUBMISSIONS_CACHE, _FACTS_CACHE


def setup_function(_):
    _CIK_CACHE.clear()
    _SUBMISSIONS_CACHE.clear()
    _FACTS_CACHE.clear()


def _stub_all(cik="0000789019", company="MICROSOFT CORP"):
    responses.add(
        responses.GET,
        "https://www.sec.gov/files/company_tickers.json",
        json={"0": {"cik_str": int(cik), "ticker": "MSFT", "title": company}},
    )
    responses.add(
        responses.GET,
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        json={
            "filings": {
                "recent": {
                    "accessionNumber": ["acc-q", "acc-k"],
                    "filingDate": ["2025-10-30", "2025-07-31"],
                    "reportDate": ["2025-09-30", "2025-06-30"],
                    "form": ["10-Q", "10-K"],
                    "primaryDocument": ["q.htm", "k.htm"],
                }
            }
        },
    )
    responses.add(
        responses.GET,
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
        json={"facts": {"us-gaap": {"Revenues": {"units": {"USD": []}}}}},
    )
    responses.add(
        responses.GET,
        "https://www.sec.gov/Archives/edgar/data/789019/accq/q.htm",
        body="<html><body><h2>ITEM 2. MANAGEMENT'S DISCUSSION</h2><p>Revenue up</p></body></html>",
    )


@responses.activate
def test_build_bundle_happy_path():
    _stub_all()
    bundle = build_edgar_bundle("MSFT")
    assert bundle.ticker == "MSFT"
    assert bundle.cik == "0000789019"
    assert bundle.company_name == "MICROSOFT CORP"
    assert bundle.latest_10q is not None
    assert bundle.latest_10k is not None
    assert "Revenue up" in bundle.mdna_text


@responses.activate
def test_build_bundle_ticker_not_found():
    responses.add(
        responses.GET,
        "https://www.sec.gov/files/company_tickers.json",
        json={"0": {"cik_str": 1, "ticker": "FOO", "title": "Foo"}},
    )
    import pytest
    from edgar import TickerNotFound
    with pytest.raises(TickerNotFound):
        build_edgar_bundle("ZZZZZ")
