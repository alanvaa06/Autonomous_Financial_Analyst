# tests/test_edgar_filings.py
import responses
from edgar import fetch_latest_10q, fetch_latest_10k, _SUBMISSIONS_CACHE


def setup_function(_):
    _SUBMISSIONS_CACHE.clear()


def _submissions_payload() -> dict:
    return {
        "cik": "789019",
        "name": "MICROSOFT CORP",
        "filings": {
            "recent": {
                "accessionNumber": [
                    "0000789019-25-000123",  # 10-Q
                    "0000789019-25-000099",  # 10-K
                    "0000789019-24-000200",  # 10-Q older
                    "0000789019-25-000050",  # 8-K
                ],
                "filingDate": ["2025-10-30", "2025-07-31", "2025-04-30", "2025-09-15"],
                "reportDate": ["2025-09-30", "2025-06-30", "2025-03-31", ""],
                "form": ["10-Q", "10-K", "10-Q", "8-K"],
                "primaryDocument": ["q1-25.htm", "10k-25.htm", "q3-25.htm", "8k.htm"],
            }
        },
    }


@responses.activate
def test_fetch_latest_10q_picks_most_recent():
    cik = "0000789019"
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    responses.add(responses.GET, url, json=_submissions_payload())
    f = fetch_latest_10q(cik)
    assert f.accession == "0000789019-25-000123"
    assert f.form == "10-Q"
    assert f.report_date == "2025-09-30"


@responses.activate
def test_fetch_latest_10k():
    cik = "0000789019"
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    responses.add(responses.GET, url, json=_submissions_payload())
    f = fetch_latest_10k(cik)
    assert f.form == "10-K"
    assert f.accession == "0000789019-25-000099"


@responses.activate
def test_fetch_latest_returns_none_when_absent():
    cik = "0000789019"
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    responses.add(
        responses.GET,
        url,
        json={
            "filings": {
                "recent": {
                    "accessionNumber": ["x"],
                    "filingDate": ["2025-01-01"],
                    "reportDate": [""],
                    "form": ["8-K"],
                    "primaryDocument": ["x.htm"],
                }
            }
        },
    )
    assert fetch_latest_10q(cik) is None
    assert fetch_latest_10k(cik) is None
