# tests/test_edgar_risk_factors.py
from edgar import extract_risk_factors_from_html


SAMPLE_10K_HTML = """
<html><body>
<h2>ITEM 1. BUSINESS</h2>
<p>Overview of the business...</p>
<h2>ITEM 1A. RISK FACTORS</h2>
<p>Our business depends on cloud demand. Adverse changes could harm results.</p>
<p>Competition is intense in the AI infrastructure market.</p>
<h2>ITEM 1B. UNRESOLVED STAFF COMMENTS</h2>
<p>None.</p>
<h2>ITEM 2. PROPERTIES</h2>
<p>Office locations...</p>
</body></html>
"""


def test_extract_risk_factors_picks_item1a():
    text = extract_risk_factors_from_html(SAMPLE_10K_HTML)
    assert "cloud demand" in text
    assert "Competition is intense" in text
    # Stops before Item 1B and Item 2.
    assert "Unresolved" not in text
    assert "Office locations" not in text
    # Did not include Item 1 (Business).
    assert "Overview of the business" not in text


def test_extract_risk_factors_returns_empty_on_missing():
    assert extract_risk_factors_from_html(
        "<html><body><h2>ITEM 1. BUSINESS</h2><p>only business</p></body></html>"
    ) == ""


def test_extract_risk_factors_caps_length():
    big = (
        "<html><body><h2>ITEM 1A. RISK FACTORS</h2><p>"
        + ("y" * 20000)
        + "</p></body></html>"
    )
    text = extract_risk_factors_from_html(big, max_chars=8000)
    assert len(text) <= 8000


def test_extract_risk_factors_stops_at_item2_when_no_1b():
    html = """
    <html><body>
    <h2>ITEM 1A. RISK FACTORS</h2>
    <p>Risk content here.</p>
    <h2>ITEM 2. PROPERTIES</h2>
    <p>Should not be included.</p>
    </body></html>
    """
    text = extract_risk_factors_from_html(html)
    assert "Risk content here" in text
    assert "Should not be included" not in text
