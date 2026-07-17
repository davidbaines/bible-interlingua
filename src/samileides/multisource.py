"""Multi-source pair building (plan.md, "Multi-source format").

One-to-many pairs each target verse with the fixed Greek source; many-to-many
(the m2m series) sampled K single sources per target, expanding the pair count
K-fold. Multi-source instead concatenates n renderings of the SAME verse into
one source line:

    <2tgt> <1grc> greek text <1deu> german text <1spa> spanish text

keeping one example per (vref, target) so pair counts stay directly comparable
to the one-to-many baseline. The atomic language tags are the separators.

Sampling (training): n ~ Uniform{k_min..k} per example — k_min=1 is the
source-dropout that keeps single-rendering inputs in-distribution (needed for
anchor extraction, plan.md "Anchors"); Greek is forced first when present;
the remaining n-1 renderings are sampled without replacement and the order of
the non-Greek renderings is shuffled.

Inference (valid/test/probe): deterministic — Greek plus the top-(k-1)
candidates from a branch-aware ranking (same branch first, then the rest,
ordered by total verse coverage), skipping cells that are held out at that
vref.

Leakage safety (identical rule to manytomany._present_by_vref): source
renderings are drawn only from the non-held-out usable cells (the union of
the train and valid manifests) plus the composite Greek source, so held-out
book text is never fed in, as a source or a target.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .data import VREF_COLUMN
from .preprocess import SRC_COLUMN, TGT_COLUMN, normalise, source_tag, target_tag

GREEK_CODE = "grc"


def present_by_vref(*manifests: pd.DataFrame) -> dict[str, list[str]]:
    """Map each vref to the translations with usable (non-held-out) text there."""
    present: dict[str, list[str]] = defaultdict(list)
    for df in manifests:
        for v, t in zip(df[VREF_COLUMN], df["translation"]):
            present[v].append(t)
    return present


def inference_source_ranking(selection: pd.DataFrame) -> dict[str, list[str]]:
    """Deterministic per-target candidate ordering over the selection.

    For each target translation: translations of the same branch first, then
    the rest, each group ordered by descending total verse coverage (ties by
    translationId for stability). Greek is handled separately by the callers
    (always first when present), so it never appears here.
    """
    frame = selection.copy()
    # Selections without these columns (e.g. the smoke selection) degrade to a
    # single coverage-ordered group.
    if "branch" not in frame.columns:
        frame["branch"] = ""
    if "totalVerses" not in frame.columns:
        frame["totalVerses"] = 0
    frame["totalVerses"] = pd.to_numeric(frame["totalVerses"], errors="coerce").fillna(0)
    ranking: dict[str, list[str]] = {}
    for _, row in frame.iterrows():
        target = row["translationId"]
        others = frame[frame["translationId"] != target]
        same = others[others["branch"] == row["branch"]]
        rest = others[others["branch"] != row["branch"]]
        order = lambda g: g.sort_values(
            ["totalVerses", "translationId"], ascending=[False, True]
        )["translationId"].tolist()
        ranking[target] = order(same) + order(rest)
    return ranking


def _render(src_id: str, vref: str, verses: pd.DataFrame,
            greek_source: pd.Series, language_of: dict[str, str]) -> str:
    if src_id == GREEK_CODE:
        return f"{source_tag(GREEK_CODE)} {normalise(greek_source[vref])}"
    return f"{source_tag(language_of[src_id])} {normalise(verses.at[vref, src_id])}"


def build_ms_pairs(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    verses: pd.DataFrame,
    greek_source: pd.Series,
    language_of: dict[str, str],
    k: int = 4,
    k_min: int = 1,
    seed: int = 13,
    include_greek: bool = True,
) -> pd.DataFrame:
    """Build one multi-source training pair per (vref, target) in ``train``.

    Returns columns vref, translation, src, tgt — the same shape one-to-many
    produces, so everything downstream (tokeniser, datasets, probes) is
    unchanged.
    """
    present = present_by_vref(train, valid)
    rng = np.random.default_rng(seed)
    rows = []
    for v, tgt in zip(train[VREF_COLUMN], train["translation"]):
        candidates = [t for t in present[v] if t != tgt]
        has_greek = include_greek and bool(greek_source.get(v))
        if not candidates and not has_greek:
            continue
        n = int(rng.integers(k_min, k + 1))
        picks: list[str] = []
        if has_greek:
            picks.append(GREEK_CODE)
        n_more = min(n - len(picks), len(candidates))
        if n_more > 0:
            idx = rng.choice(len(candidates), size=n_more, replace=False)
            sampled = [candidates[i] for i in idx]
            rng.shuffle(sampled)
            picks.extend(sampled)
        src = " ".join(
            [target_tag(language_of[tgt])]
            + [_render(s, v, verses, greek_source, language_of) for s in picks]
        )
        rows.append(
            {
                VREF_COLUMN: v,
                "translation": tgt,
                SRC_COLUMN: src,
                TGT_COLUMN: normalise(verses.at[v, tgt]),
            }
        )
    return pd.DataFrame(rows, columns=[VREF_COLUMN, "translation", SRC_COLUMN, TGT_COLUMN])


def to_ms_sources(
    frame: pd.DataFrame,
    verses: pd.DataFrame,
    greek_source: pd.Series,
    language_of: dict[str, str],
    present: dict[str, list[str]],
    ranking: dict[str, list[str]],
    k: int = 4,
    include_greek: bool = True,
) -> pd.DataFrame:
    """Rewrite a (vref, translation, src, tgt) frame's sources to deterministic
    multi-source form: Greek first, then the top-ranked present candidates.

    Used for valid/test/probe frames and holdout generation, so inference
    sources are reproducible. Held-out cells are absent from ``present`` and
    therefore never selected.
    """
    out = frame.copy()
    srcs = []
    for v, tgt in zip(out[VREF_COLUMN], out["translation"]):
        picks: list[str] = []
        if include_greek and bool(greek_source.get(v)):
            picks.append(GREEK_CODE)
        usable = set(present.get(v, ()))
        for cand in ranking.get(tgt, ()):
            if len(picks) >= k:
                break
            if cand != tgt and cand in usable:
                picks.append(cand)
        srcs.append(
            " ".join(
                [target_tag(language_of[tgt])]
                + [_render(s, v, verses, greek_source, language_of) for s in picks]
            )
        )
    out[SRC_COLUMN] = srcs
    return out


def strip_tags(src: str) -> str:
    """Remove all leading-tag tokens from a multi-source line (for analysis)."""
    return " ".join(t for t in src.split(" ") if not (t.startswith("<") and t.endswith(">")))
