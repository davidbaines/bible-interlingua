"""Integration tests against the real corpus (network; run with -m integration)."""

import pytest

from samileides.data import load_metadata, load_verses, resolve_column

pytestmark = pytest.mark.integration


def test_metadata_shape():
    meta = load_metadata()
    assert len(meta) == 1253
    assert meta["languageCode"].nunique() == 997


def test_known_verse_text():
    verses = load_verses(["engbsb"])
    assert verses.loc["GEN 1:1", "engbsb"].startswith("In the beginning God created")
    assert len(verses) == 41899


def test_coverage_matches_metadata():
    meta = load_metadata().set_index("translationId")
    verses = load_verses(["engbsb"])
    non_empty = int((verses["engbsb"] != "").sum())
    expected = int(meta.loc["engbsb", "totalVerses"])
    # <range> markers and merged verses allow small drift
    assert abs(non_empty - expected) / expected < 0.02


def test_greek_columns_resolve():
    assert resolve_column("grcbrent")
    assert resolve_column("grc-tisch")
    assert resolve_column("deuelbbk")
    assert resolve_column("fin")


def test_real_holdout_splits_have_no_leakage():
    from samileides.greek import build_composite_source
    from samileides.splits import assert_no_leakage, build_splits

    holdouts = {"engbsb": ["OT"], "deuelbbk": ["GEN"], "fin": ["MAT"]}
    verses = load_verses(["engbsb", "deuelbbk", "fin"])
    splits = build_splits(verses, holdouts, valid_size=100, seed=13)
    assert_no_leakage(splits, holdouts)
    # English test set is a whole OT: expect >20k verses
    eng_test = splits.test[splits.test["translation"] == "engbsb"]
    assert len(eng_test) > 20000

    source = build_composite_source()
    # Greek source must exist for the vast majority of German Genesis
    gen_refs = splits.test[splits.test["translation"] == "deuelbbk"]["vref"]
    covered = (source.loc[gen_refs] != "").mean()
    assert covered > 0.95
