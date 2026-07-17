"""Extended-holdout proposal (spec.md, "Holdout design").

Proposes ~5 typologically diverse full-Bible languages to join English,
German and Finnish as holdouts (Genesis withheld). The output is a proposal
only: David must approve the list before any training run — that approval is
recorded by copying the chosen rows into ``configs/holdouts.yaml``.

Run ``python -m samileides.holdouts`` to (re)generate the proposal file.
"""

from __future__ import annotations

import pandas as pd

from .data import load_metadata, repo_root
from .selection import load_families

FULL_OT_MIN = 20000
FULL_NT_MIN = 7000

# One pick from each spec-mandated bucket, plus one "other" chosen from
# families as unlike the core buckets as the corpus allows (preference order).
TARGET_BUCKETS = ("Bantu", "Austronesian", "Indo-Aryan", "Turkic")
OTHER_PREFERENCE = (
    "Austroasiatic", "Kra-Dai", "Quechuan", "Mayan", "Koreanic",
    "Sino-Tibetan", "Dravidian", "Trans-New Guinea",
)

# The existing holdouts and source languages never re-enter the pool.
EXCLUDED_LANGUAGES = ("eng", "deu", "fin", "grc", "hbo")


def full_bible_languages(metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    """Best full-Bible translation per language, with family attached."""
    meta = load_metadata() if metadata is None else metadata
    families = load_families()
    full = meta[
        (meta["OTverses"] >= FULL_OT_MIN)
        & (meta["NTverses"] >= FULL_NT_MIN)
        & ~meta["languageCode"].isin(EXCLUDED_LANGUAGES)
    ]
    best = (
        full.sort_values("totalVerses", ascending=False)
        .groupby("languageCode", as_index=False)
        .first()
    )
    best["family"] = [families.get(c, "") for c in best["languageCode"]]
    return best


def propose_extended_holdouts(
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One language per target bucket plus one 'other', highest coverage first."""
    best = full_bible_languages(metadata)
    picks: list[pd.Series] = []

    for bucket in TARGET_BUCKETS:
        candidates = best[best["family"] == bucket].sort_values(
            "totalVerses", ascending=False
        )
        if len(candidates):
            picks.append(candidates.iloc[0])

    taken = {p["languageCode"] for p in picks}
    for family in OTHER_PREFERENCE:
        candidates = best[
            (best["family"] == family) & ~best["languageCode"].isin(taken)
        ].sort_values("totalVerses", ascending=False)
        if len(candidates):
            picks.append(candidates.iloc[0])
            break

    proposal = pd.DataFrame(picks).reset_index(drop=True)
    proposal["heldOutBook"] = "GEN"
    return proposal


def main() -> None:
    proposal = propose_extended_holdouts()
    out_dir = repo_root() / "experiments"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "extended-holdouts-proposal.md"

    cols = [
        "languageCode", "translationId", "languageNameInEnglish", "family",
        "script", "OTverses", "NTverses", "heldOutBook",
    ]
    table = proposal[[c for c in cols if c in proposal.columns]]
    lines = [
        "# Extended holdout proposal (awaiting approval)",
        "",
        "Criteria: full-Bible coverage, one language per target family bucket",
        "(Bantu, Austronesian, Indo-Aryan, Turkic, + one other), Genesis withheld.",
        "The `script` column comes from corpus metadata and is advisory —",
        "spot-check the actual text before approving.",
        "",
        table.to_markdown(index=False),
        "",
        "**To approve**: add each `translationId: [GEN]` entry to the",
        "`holdouts:` map in `configs/holdouts.yaml` (edit picks freely first).",
        "No training run may start until this is done (spec.md gate).",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path} with {len(proposal)} proposed holdout languages")


if __name__ == "__main__":
    main()
