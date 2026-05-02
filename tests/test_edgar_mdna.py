# tests/test_edgar_mdna.py
from edgar import extract_mdna_from_html


SAMPLE_HTML = """
<html><body>
<p>Cover page boilerplate.</p>
<h2>ITEM 1. FINANCIAL STATEMENTS</h2>
<p>Tables go here. Lots of numbers.</p>
<h2>ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS</h2>
<p>Revenue grew 12% year over year, driven by Cloud and AI services.</p>
<p>Operating margin expanded by 200 basis points.</p>
<h2>ITEM 3. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK</h2>
<p>Interest rate exposure...</p>
</body></html>
"""


def test_extract_mdna_picks_item2_section():
    text = extract_mdna_from_html(SAMPLE_HTML)
    assert "Revenue grew 12%" in text
    assert "Operating margin" in text
    assert "Interest rate exposure" not in text  # stopped at Item 3
    assert "Tables go here" not in text  # did not include Item 1


def test_extract_mdna_returns_empty_on_missing():
    assert extract_mdna_from_html("<html><body><p>no items here</p></body></html>") == ""


def test_extract_mdna_caps_length():
    big = "<html><body><h2>ITEM 2. MANAGEMENT'S DISCUSSION</h2><p>" + ("x" * 20000) + "</p></body></html>"
    text = extract_mdna_from_html(big, max_chars=8000)
    assert len(text) <= 8000
