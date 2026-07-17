"""Pilot selection: ~50 translations echoing Liedes' 2018 language set.

Composition (spec.md, "Translation selection"): the approved holdout
translations are forced in; each of Liedes' languages present in the corpus
contributes its best-covered translation; the remainder is topped up for
family/script diversity. Run ``python -m samileides.pilot`` to write the
committed list to ``experiments/selection-pilot.csv``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .data import load_metadata, repo_root
from .selection import SelectionConfig, load_families, select_translations, write_selection

PILOT_TARGET_SIZE = 50

# Never targets: Greek is the source; ancient Hebrew is reserved as a
# possible future source (spec.md phase 6 discusses source variants).
EXCLUDED_SOURCE_LANGUAGES = ("grc", "hbo")

# Liedes' 52 target languages, as ISO 639-3 preference groups (first code
# with a match in the corpus wins). Languages absent from the corpus
# (Basque, Manx, Scots Gaelic, Potawatomi, Slovenian, ...) simply don't match.
LIEDES_LANGUAGE_PREFERENCES: tuple[tuple[str, ...], ...] = (
    ("afr",), ("sqi", "als"), ("eus",), ("bel",), ("bre",), ("bul",),
    ("chu",), ("cha",), ("cmn",), ("hrv",), ("ces",), ("dan",),
    ("prs",), ("nld",), ("epo",), ("ekk", "est"), ("pes", "fas"),
    ("fra",), ("ell",), ("hat",), ("heb",), ("hin",), ("hun",),
    ("ita",), ("kek",), ("kor",), ("lit",), ("lav", "lvs"), ("glv",),
    ("mri", "mao"), ("plt", "mlg"), ("nob", "nor", "nno"), ("azj",),
    ("syc",), ("pol",), ("por",), ("pot",), ("ron",), ("rus",),
    ("gla",), ("slv",), ("som",), ("spa",), ("swh",), ("swe",),
    ("tgl",), ("tur",), ("ukr",), ("lat",),
)


def load_holdout_translations() -> list[str]:
    config = yaml.safe_load(
        (repo_root() / "configs" / "holdouts.yaml").read_text(encoding="utf-8")
    )
    return list(config["holdouts"])


def build_pilot_selection(metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    meta = load_metadata() if metadata is None else metadata
    families = load_families()
    meta = meta[~meta["languageCode"].isin(EXCLUDED_SOURCE_LANGUAGES)]

    holdout_ids = load_holdout_translations()
    forced = meta[meta["translationId"].isin(holdout_ids)]
    taken_langs = set(forced["languageCode"])

    liedes_rows = []
    for group in LIEDES_LANGUAGE_PREFERENCES:
        for code in group:
            candidates = meta[
                (meta["languageCode"] == code) & ~meta["languageCode"].isin(taken_langs)
            ]
            if len(candidates):
                best = candidates.sort_values("totalVerses", ascending=False).iloc[0]
                liedes_rows.append(best)
                taken_langs.add(code)
                break
    liedes_part = pd.DataFrame(liedes_rows)

    top_up = PILOT_TARGET_SIZE - len(forced) - len(liedes_part)
    config = SelectionConfig(
        target_size=top_up,
        min_verses=5000,
        exclude=list(forced["translationId"]) + list(liedes_part["translationId"]),
    )
    pool = meta[~meta["languageCode"].isin(taken_langs)]
    extra = select_translations(config, pool, families)

    selection = pd.concat([forced, liedes_part, extra]).reset_index(drop=True)
    selection["family"] = [families.get(c, "") for c in selection["languageCode"]]
    return selection


def main() -> None:
    selection = build_pilot_selection()
    path = write_selection(selection, "pilot")
    n_langs = selection["languageCode"].nunique()
    print(f"Wrote {path}: {len(selection)} translations, {n_langs} languages")


if __name__ == "__main__":
    main()
