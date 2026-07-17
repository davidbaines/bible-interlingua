import pandas as pd

from samileides.data import VREF_COLUMN
from samileides.manytomany import GREEK_CODE, build_m2m_pairs, to_m2m_source
from samileides.preprocess import SRC_COLUMN, TGT_COLUMN


def _leading(src):
    return [t for t in src.split(" ") if t.startswith("<") and t.endswith(">")][:2]


def _setup():
    vrefs = ["GEN 1:1", "GEN 1:2"]
    verses = pd.DataFrame(
        {"eng": ["e1", "e2"], "spa": ["s1", "s2"], "deu": ["d1", "d2"]},
        index=pd.Index(vrefs, name=VREF_COLUMN),
    )
    greek = pd.Series({"GEN 1:1": "g1", "GEN 1:2": "g2"})
    language_of = {"eng": "eng", "spa": "spa", "deu": "deu"}
    # train targets: eng and spa; valid contributes deu as an available source.
    train = pd.DataFrame({VREF_COLUMN: ["GEN 1:1", "GEN 1:2"], "translation": ["eng", "spa"]})
    valid = pd.DataFrame({VREF_COLUMN: ["GEN 1:1", "GEN 1:2"], "translation": ["deu", "deu"]})
    return train, valid, verses, greek, language_of


def test_both_tags_and_no_self_pairing():
    train, valid, verses, greek, lang = _setup()
    out = build_m2m_pairs(train, valid, verses, greek, lang, k=4, seed=1)
    assert len(out) > 0
    for _, r in out.iterrows():
        t1, t2 = _leading(r[SRC_COLUMN])
        assert t1.startswith("<2") and t2.startswith("<1")   # target then source tag
        # the source language tag must not equal the target's own language
        assert t2 != t1.replace("<2", "<1")


def test_k_caps_number_of_sources():
    train, valid, verses, greek, lang = _setup()
    out = build_m2m_pairs(train, valid, verses, greek, lang, k=1, seed=1)
    # one source per (vref, target)
    counts = out.groupby([VREF_COLUMN, "translation"]).size()
    assert (counts == 1).all()


def test_greek_can_be_a_source_and_is_tagged():
    train, valid, verses, greek, lang = _setup()
    out = build_m2m_pairs(train, valid, verses, greek, lang, k=10, seed=3)
    # with K larger than the pool, Greek should appear as a source sometimes
    assert out[SRC_COLUMN].str.contains(f"<1{GREEK_CODE}>").any()


def test_held_out_translation_never_used_as_source():
    # 'sec' exists only in the test set (never in train/valid), so it must never
    # be sampled as a source.
    train, valid, verses, greek, lang = _setup()
    verses["sec"] = ["x1", "x2"]        # a held-out translation's text
    lang["sec"] = "sec"
    out = build_m2m_pairs(train, valid, verses, greek, lang, k=10, seed=5)
    assert not out[SRC_COLUMN].str.contains("<1sec>").any()
    assert not out[SRC_COLUMN].str.contains("x1|x2").any()


def test_to_m2m_source_inserts_source_tag():
    assert to_m2m_source("<2deu> hallo welt") == "<2deu> <1grc> hallo welt"
