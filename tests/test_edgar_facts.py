# tests/test_edgar_facts.py
import responses
from edgar import fetch_company_facts, _FACTS_CACHE


def setup_function(_):
    _FACTS_CACHE.clear()


@responses.activate
def test_fetch_company_facts_basic():
    cik = "0000789019"
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    responses.add(
        responses.GET,
        url,
        json={
            "cik": 789019,
            "entityName": "MICROSOFT CORP",
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "label": "Revenues",
                        "units": {
                            "USD": [
                                {"end": "2025-09-30", "val": 70000000000, "fy": 2026, "fp": "Q1", "form": "10-Q"},
                                {"end": "2024-09-30", "val": 65000000000, "fy": 2025, "fp": "Q1", "form": "10-Q"},
                            ]
                        },
                    }
                }
            },
        },
    )
    facts = fetch_company_facts(cik)
    rev = facts["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
    assert len(rev) == 2 and rev[0]["val"] == 70000000000


@responses.activate
def test_fetch_company_facts_caches():
    cik = "0000789019"
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    responses.add(responses.GET, url, json={"cik": 789019, "facts": {}})
    fetch_company_facts(cik)
    fetch_company_facts(cik)
    # Cache should mean only ONE actual HTTP call.
    assert len(responses.calls) == 1
