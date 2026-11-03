import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from ircrag.convert import convert, normalize_lines  # noqa: E402
from ircrag.ingest import load_chunks, parse_clauses  # noqa: E402

SAMPLE = os.path.join(ROOT, "data/sample/irc_sample.md")


def test_normalize_merges_split_headings_and_hyphens():
    text = normalize_lines(["SECTION R602", "WOOD WALL FRAMING", "R602.8 Fireblocking required. Fire-", "blocking must be provided.", "12"])
    assert "SECTION R602 WOOD WALL FRAMING" in text
    assert "required. Fireblocking must be provided." in text  # 连字符断行已拼接
    assert "\n12" not in text


def test_table_is_its_own_clause_and_splits_with_header():
    clauses, _ = parse_clauses(open(SAMPLE, encoding="utf-8").read())
    table = next(c for c in clauses if c.kind == "table")
    assert table.section_id == "R602.3(5)" and table.parent == "R602.3"
    chunks, _ = load_chunks(SAMPLE, max_chars=500)
    parts = [c for c in chunks if c.chunk_id.startswith("Table R602.3(5)#")]
    assert len(parts) > 1 and all("| Stud size" in p.text for p in parts)


def test_html_conversion(tmp_path):
    f = tmp_path / "a.html"
    f.write_text("<h2>SECTION R311</h2><h2>MEANS OF EGRESS</h2><p>R311.7.2 Headroom. Not less than 6 feet 8 inches.</p>"
                 "<p>TABLE R311.7(1)</p><table><tr><th>Item</th><th>Value</th></tr><tr><td>Riser</td><td>7-3/4</td></tr></table>")
    clauses, _ = parse_clauses(convert(str(f)))
    assert [(c.section_id, c.kind) for c in clauses] == [("R311.7.2", "clause"), ("R311.7(1)", "table")]


def test_pdf_round_trip(tmp_path):
    from make_pdf import build
    pdf = tmp_path / "irc.pdf"
    build(SAMPLE, str(pdf))
    got, _ = parse_clauses(convert(str(pdf)))
    want, _ = parse_clauses(open(SAMPLE, encoding="utf-8").read())
    assert [c.section_id for c in got] == [c.section_id for c in want]
    assert any(c.kind == "table" and "|" in c.text for c in got)


def test_all_irc_prefixes_and_definitions():
    text = "\n".join([
        "CHAPTER 2 DEFINITIONS",
        "SECTION R202 DEFINITIONS",
        "[RB] HABITABLE SPACE. A space in a building for living, sleeping, eating or cooking.",
        "STORY. That portion of a building included between the upper surface of a floor",
        "and the floor or roof next above.",
        "CHAPTER 10 CHIMNEYS AND FIREPLACES",
        "SECTION R1003 MASONRY CHIMNEYS",
        "R1003.9 Termination. Chimneys shall extend not less than 2 feet higher.",
        "CHAPTER 11 [RE] ENERGY EFFICIENCY",
        "SECTION N1102 BUILDING THERMAL ENVELOPE",
        "N1102.1.2 (R402.1.2) Insulation and fenestration criteria. The building thermal envelope shall comply.",
        "TABLE N1102.1.2 (R402.1.2) INSULATION AND FENESTRATION REQUIREMENTS BY COMPONENT",
        "| CLIMATE ZONE | CEILING R-VALUE |",
        "| --- | --- |",
        "| 2 | 38 |",
        "CHAPTER 39 POWER AND LIGHTING DISTRIBUTION",
        "SECTION E3902 GROUND-FAULT AND ARC-FAULT CIRCUIT-INTERRUPTER PROTECTION",
        "E3902.1 Bathroom receptacles. Receptacles installed in bathrooms shall have GFCI protection.",
        "¬",
    ])
    clauses, _ = parse_clauses(text)
    got = [(c.kind, c.section_id, c.title) for c in clauses]
    assert ("definition", "R202", "Habitable space") in got
    assert ("definition", "R202", "Story") in got
    assert ("clause", "R1003.9", "Termination") in got
    assert ("clause", "N1102.1.2", "Insulation and fenestration criteria") in got
    assert ("table", "N1102.1.2", "Insulation and fenestration requirements by component") in got
    assert ("clause", "E3902.1", "Bathroom receptacles") in got
    story = next(c for c in clauses if c.title == "Story")
    assert story.text.endswith("next above.")
