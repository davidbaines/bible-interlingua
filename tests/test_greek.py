from samileides.greek import build_composite_source, coverage_report


def test_composite_picks_testament_columns(synthetic_verses):
    source = build_composite_source(synthetic_verses)
    assert source["GEN 1:1"] == "g-gen1"  # OT from grcbrent
    assert source["MAT 1:1"] == "t-mat1"  # NT from grc-tisch
    assert source["TOB 1:1"] == ""  # deuterocanon excluded


def test_composite_empty_where_source_missing(synthetic_verses):
    verses = synthetic_verses.copy()
    verses.loc["GEN 1:2", "grcbrent"] = ""
    source = build_composite_source(verses)
    assert source["GEN 1:2"] == ""


def test_range_markers_become_empty(synthetic_verses):
    verses = synthetic_verses.copy()
    verses.loc["GEN 1:2", "grcbrent"] = "<range>"
    source = build_composite_source(verses)
    assert source["GEN 1:2"] == ""


def test_coverage_report_counts(synthetic_verses):
    source = build_composite_source(synthetic_verses)
    report = coverage_report(source).set_index("book")
    assert report.loc["GEN", "vrefs"] == 2
    assert report.loc["GEN", "missing"] == 0
    assert report.loc["EXO", "covered"] == 1
    # REV covered via grc-tisch
    assert report.loc["REV", "missing"] == 0
