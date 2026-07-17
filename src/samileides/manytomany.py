"""Many-to-many pair sampling (spec.md phase 4).

One-to-many pairs each target verse with the fixed Greek source. Many-to-many
instead pairs each target verse with K randomly sampled source translations
that contain that verse, per epoch, tagging the source with both a target and a
source language token. The model then learns each verse's content from many
angles.

Leakage safety: sources are drawn only from the non-held-out usable cells (the
union of the train and valid manifests) plus the composite Greek source, so no
held-out book text is ever fed in, as a source or a target. The composite
Greek source is included in the pool by default (language code ``grc``) so the
model sees Greek->target during training and Greek-sourced generation at test
time stays in distribution.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .data import VREF_COLUMN
from .preprocess import SRC_COLUMN, TGT_COLUMN, normalise, source_tag, target_tag

GREEK_CODE = "grc"


def _present_by_vref(*manifests: pd.DataFrame) -> dict[str, list[str]]:
    """Map each vref to the translations with usable (non-held-out) text there."""
    present: dict[str, list[str]] = defaultdict(list)
    for df in manifests:
        for v, t in zip(df[VREF_COLUMN], df["translation"]):
            present[v].append(t)
    return present


def build_m2m_pairs(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    verses: pd.DataFrame,
    greek_source: pd.Series,
    language_of: dict[str, str],
    k: int = 4,
    seed: int = 13,
    include_greek: bool = True,
) -> pd.DataFrame:
    """Expand train targets into many-to-many pairs.

    For each (vref, target) in ``train``, sample up to ``k`` source translations
    from the non-held-out pool at that vref (plus Greek), and emit one pair per
    source with ``<2tgt> <1src> source_text`` as the source and the target text
    as the label. Returns columns vref, translation, src, tgt.
    """
    present = _present_by_vref(train, valid)
    rng = np.random.default_rng(seed)
    rows = []
    for v, tgt in zip(train[VREF_COLUMN], train["translation"]):
        candidates = [t for t in present[v] if t != tgt]
        if include_greek and greek_source.get(v):
            candidates = candidates + [GREEK_CODE]
        if not candidates:
            continue
        n = min(k, len(candidates))
        picks = [candidates[i] for i in rng.choice(len(candidates), size=n, replace=False)]

        tgt_text = normalise(verses.at[v, tgt])
        ttag = target_tag(language_of[tgt])
        for src in picks:
            if src == GREEK_CODE:
                src_text, src_lang = normalise(greek_source[v]), GREEK_CODE
            else:
                src_text, src_lang = normalise(verses.at[v, src]), language_of[src]
            rows.append(
                {
                    VREF_COLUMN: v,
                    "translation": tgt,
                    SRC_COLUMN: f"{ttag} {source_tag(src_lang)} {src_text}",
                    TGT_COLUMN: tgt_text,
                }
            )
    return pd.DataFrame(rows, columns=[VREF_COLUMN, "translation", SRC_COLUMN, TGT_COLUMN])


def to_m2m_source(one_to_many_src: str, source_lang: str = GREEK_CODE) -> str:
    """Rewrite a one-to-many source (`<2tgt> text`) into m2m form.

    Inserts the source-language tag after the target tag, for generating with an
    m2m-trained model from the Greek (or other) source at test time.
    """
    tag, _, text = one_to_many_src.partition(" ")
    return f"{tag} {source_tag(source_lang)} {text}"
