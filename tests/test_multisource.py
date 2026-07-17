import pandas as pd

from samileides.data import VREF_COLUMN
from samileides.multisource import (
    GREEK_CODE,
    build_ms_pairs,
    inference_source_ranking,
    present_by_vref,
    strip_tags,
    to_ms_sources,
)
from samileides.preprocess import SRC_COLUMN, TGT_COLUMN


def _tags(src):
    return [t for t in src.split(" ") if t.startswith("<") and t.endswith(">")]


def _setup():
    vrefs = ["GEN 1:1", "GEN 1:2"]
    verses = pd.DataFrame(
        {"eng": ["e1", "e2"], "spa": ["s1", "s2"], "deu": ["d1", "d2"]},
        index=pd.Index(vrefs, name=VREF_COLUMN),
    )
    greek = pd.Series({"GEN 1:1": "g1", "GEN 1:2": "g2"})
    language_of = {"eng": "eng", "spa": "spa", "deu": "deu"}
    train = pd.DataFrame({VREF_COLUMN: ["GEN 1:1", "GEN 1:2"], "translation": ["eng", "spa"]})
    valid = pd.DataFrame({VREF_COLUMN: ["GEN 1:1", "GEN 1:2"], "translation": ["deu", "deu"]})
    return train, valid, verses, greek, language_of


def _selection():
    return pd.DataFrame(
        {
            "translationId": ["eng", "spa", "deu"],
            "languageCode": ["eng", "spa", "deu"],
            "branch": ["Germanic", "Romance", "Germanic"],
            "totalVerses": ["100", "200", "150"],
        }
    )


def test_one_pair_per_target_and_format():
    train, valid, verses, greek, lang = _setup()
    out = build_ms_pairs(train, valid, verses, greek, lang, k=4, k_min=1, seed=1)
    # one example per (vref, target) — no K-fold expansion
    assert len(out) == len(train)
    for _, r in out.iterrows():
        tags = _tags(r[SRC_COLUMN])
        assert tags[0].startswith("<2")                 # target tag first
        assert all(t.startswith("<1") for t in tags[1:])  # then source tags
        # Greek forced first among renderings
        assert tags[1] == f"<1{GREEK_CODE}>"
        # never uses the target itself as a source
        own = tags[0].replace("<2", "<1")
        assert own not in tags[1:]
        assert r[TGT_COLUMN] in {"e1", "e2", "s1", "s2"}


def test_determinism_same_seed():
    train, valid, verses, greek, lang = _setup()
    a = build_ms_pairs(train, valid, verses, greek, lang, k=3, seed=7)
    b = build_ms_pairs(train, valid, verses, greek, lang, k=3, seed=7)
    assert a.equals(b)


def test_k_min_dropout_produces_variable_counts():
    # With many vrefs, n ~ Uniform{1..k} must produce both 1-rendering and
    # k-rendering examples.
    n = 200
    vrefs = [f"GEN 1:{i}" for i in range(1, n + 1)]
    verses = pd.DataFrame(
        {c: [f"{c}{i}" for i in range(n)] for c in ["eng", "spa", "deu", "fra"]},
        index=pd.Index(vrefs, name=VREF_COLUMN),
    )
    greek = pd.Series({v: f"g{i}" for i, v in enumerate(vrefs)})
    lang = {c: c for c in ["eng", "spa", "deu", "fra"]}
    train = pd.DataFrame({VREF_COLUMN: vrefs, "translation": ["eng"] * n})
    valid = pd.DataFrame(
        {VREF_COLUMN: vrefs * 3, "translation": ["spa"] * n + ["deu"] * n + ["fra"] * n}
    )
    out = build_ms_pairs(train, valid, verses, greek, lang, k=4, k_min=1, seed=13)
    counts = out[SRC_COLUMN].map(lambda s: len(_tags(s)) - 1)  # renderings per example
    assert counts.min() == 1
    assert counts.max() == 4


def test_held_out_translation_never_used_as_source():
    train, valid, verses, greek, lang = _setup()
    verses["sec"] = ["x1", "x2"]  # held-out translation: text exists, not in manifests
    lang["sec"] = "sec"
    out = build_ms_pairs(train, valid, verses, greek, lang, k=10, seed=5)
    assert not out[SRC_COLUMN].str.contains("<1sec>").any()
    assert not out[SRC_COLUMN].str.contains("x1|x2").any()


def test_inference_ranking_branch_first_then_coverage():
    ranking = inference_source_ranking(_selection())
    # For eng (Germanic): deu (same branch) first, then spa
    assert ranking["eng"] == ["deu", "spa"]
    # For spa (Romance, alone): Germanic others by coverage desc: deu(150) > eng(100)
    assert ranking["spa"] == ["deu", "eng"]


def test_to_ms_sources_deterministic_and_leakage_safe():
    train, valid, verses, greek, lang = _setup()
    verses["sec"] = ["x1", "x2"]
    lang["sec"] = "sec"
    present = present_by_vref(train, valid)
    sel = _selection()
    ranking = inference_source_ranking(sel)
    frame = pd.DataFrame(
        {
            VREF_COLUMN: ["GEN 1:1"],
            "translation": ["eng"],
            SRC_COLUMN: ["<2eng> g1"],
            TGT_COLUMN: ["e1"],
        }
    )
    out1 = to_ms_sources(frame, verses, greek, lang, present, ranking, k=3)
    out2 = to_ms_sources(frame, verses, greek, lang, present, ranking, k=3)
    assert out1.equals(out2)
    src = out1[SRC_COLUMN].iloc[0]
    tags = _tags(src)
    assert tags[0] == "<2eng>"
    assert tags[1] == f"<1{GREEK_CODE}>"       # Greek first
    assert "<1deu>" in tags                    # same-branch candidate chosen
    assert "<1sec>" not in tags                # held-out cell never selected
    assert "<1eng>" not in tags                # never the target itself


def test_ranking_without_branch_or_coverage_columns():
    sel = pd.DataFrame({"translationId": ["a", "b", "c"], "languageCode": ["a", "b", "c"]})
    ranking = inference_source_ranking(sel)
    assert ranking["a"] == ["b", "c"]  # stable id order when no other signal


def test_strip_tags():
    assert strip_tags("<2eng> <1grc> alpha <1deu> beta") == "alpha beta"
