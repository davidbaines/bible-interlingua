import pandas as pd

from samileides.preprocess import build_pairs, length_filter, normalise, target_tag
from samileides.splits import build_splits


def test_normalise_nfc_and_whitespace():
    # e + combining acute -> precomposed é; whitespace collapsed
    assert normalise("café  au\tlait ") == "café au lait"


def test_target_tag():
    assert target_tag("deu") == "<2deu>"


def _pairs(synthetic_verses):
    splits = build_splits(synthetic_verses, {"alpha": ["GEN"]}, valid_size=1, seed=0)
    from samileides.greek import build_composite_source

    source = build_composite_source(synthetic_verses)
    language_of = {"alpha": "aaa", "beta": "bbb", "grcbrent": "grc", "grc-tisch": "grc"}
    return build_pairs(splits.train, synthetic_verses, source, language_of)


def test_build_pairs_tags_and_text(synthetic_verses):
    pairs = _pairs(synthetic_verses)
    assert len(pairs) > 0
    row = pairs[(pairs["vref"] == "EXO 1:1") & (pairs["translation"] == "alpha")]
    if len(row):
        assert row.iloc[0]["src"] == "<2aaa> g-exo1"
        assert row.iloc[0]["tgt"] == "a-exo1"
    # every src starts with a tag
    assert pairs["src"].str.match(r"<2[a-z]{3}> ").all()


def test_build_pairs_drops_missing_source(synthetic_verses):
    pairs = _pairs(synthetic_verses)
    # REV 22:21 has grc-tisch text, TOB has none; no pair may carry empty source
    assert not (pairs["src"].str.split(" ").str[1:].str.join(" ") == "").any()
    assert "TOB 1:1" not in set(pairs["vref"])


def test_length_filter_counts():
    pairs = pd.DataFrame(
        {
            "vref": ["A 1:1", "A 1:2", "A 1:3"],
            "translation": ["t", "t", "t"],
            "src": ["ok ok", "x " * 300, "ok ok"],
            "tgt": ["ok ok", "ok", "ok " * 10],
        }
    )
    kept, stats = length_filter(pairs, encode=str.split, max_len=192, max_ratio=2.0)
    assert stats["input"] == 3
    assert stats["dropped_too_long"] == 1  # the 300-token src
    assert stats["dropped_ratio"] == 1  # 2 vs 10 tokens
    assert stats["kept"] == 1
    assert list(kept["vref"]) == ["A 1:1"]
