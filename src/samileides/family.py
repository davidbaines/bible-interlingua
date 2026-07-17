"""Single-family selection: all corpus languages of one language family.

For the Indo-European run (spec.md, "Single-family (Indo-European) run"): take
every eBible language whose ISO 639-3 code is listed in a curated family file
(``configs/families/<family>.csv``), keep the best-covered translation per
language, and force the holdout editions in. Unlike the diverse pilot there is
no diversity cap — the point is to saturate a single family so held-out
languages are surrounded by close relatives. Run:

    python -m samileides.family                      # Indo-European default
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

import argparse

from .data import load_metadata, repo_root
from .licensing import is_shareable, licence_of
from .selection import SelectionConfig, load_families, select_translations, write_selection

DEFAULT_FAMILY = "indo_european"
MIN_VERSES = 5000


def load_family_codes(family: str = DEFAULT_FAMILY) -> dict[str, str]:
    """Read the curated ISO 639-3 -> branch map for a family."""
    path = repo_root() / "configs" / "families" / f"{family}.csv"
    df = pd.read_csv(path, comment="#")
    return dict(zip(df["languageCode"], df["branch"]))


def load_holdout_ids(holdouts_file: str) -> list[str]:
    raw = yaml.safe_load((repo_root() / holdouts_file).read_text(encoding="utf-8"))
    return list(raw["holdouts"])


def build_family_selection(
    family: str = DEFAULT_FAMILY,
    holdouts_file: str = "configs/holdouts-ie.yaml",
    metadata: pd.DataFrame | None = None,
    shareable_only: bool = False,
    allow_nc: bool = False,
) -> pd.DataFrame:
    meta = load_metadata() if metadata is None else metadata
    branch_of = load_family_codes(family)
    in_family = meta[meta["languageCode"].isin(branch_of)]

    if shareable_only:
        # Restrict to licences that permit sharing a derived model, so the
        # resulting run is publishable (spec.md, "Publishing").
        lic = licence_of(in_family["translationId"], meta)
        keep = in_family["translationId"].map(
            lambda t: is_shareable(lic.get(t, "Unknown"), allow_nc)
        )
        in_family = in_family[keep.to_numpy()]

    holdout_ids = load_holdout_ids(holdouts_file) if holdouts_file else []
    config = SelectionConfig(
        target_size=None,          # take every language in the family
        include=holdout_ids,       # force the exact holdout editions in
        min_verses=MIN_VERSES,
        one_per_language=True,     # best-covered translation per language
    )
    selection = select_translations(config, in_family, load_families())
    selection["branch"] = selection["languageCode"].map(branch_of)
    return selection


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a single-family selection")
    ap.add_argument("--family", default=DEFAULT_FAMILY)
    ap.add_argument("--holdouts", default="configs/holdouts-ie.yaml")
    ap.add_argument("--shareable", action="store_true",
                    help="keep only licences that permit sharing a derived model")
    ap.add_argument("--allow-nc", action="store_true")
    ap.add_argument("--name", default=None, help="output suffix (selection-<name>.csv)")
    args = ap.parse_args()

    selection = build_family_selection(
        family=args.family, holdouts_file=args.holdouts,
        shareable_only=args.shareable, allow_nc=args.allow_nc,
    )
    name = args.name or ("ie-shareable" if args.shareable else "ie")
    out = write_selection(selection, name)
    # re-write including the branch column for readability
    cols = [
        "translationId", "languageCode", "languageNameInEnglish", "branch",
        "script", "OTverses", "NTverses", "DCverses", "totalVerses",
    ]
    selection[[c for c in cols if c in selection.columns]].to_csv(
        Path(out), index=False
    )
    n = selection["languageCode"].nunique()
    print(f"Wrote {out}: {len(selection)} translations, {n} languages")
    print(selection["branch"].value_counts().to_string())


if __name__ == "__main__":
    main()
