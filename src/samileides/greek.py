"""Composite Greek source: Brenton LXX for the OT, Tischendorf for the NT.

No single Greek translation in the corpus covers the whole Bible, so the
one-to-many source column is stitched from `grcbrent` (OT) and `grc-tisch`
(NT) — the same recipe as Liedes' `2TGreek`, kept in Greek script.
Deuterocanon vrefs are left empty. Run ``python -m samileides.greek`` to
write the coverage report the spec requires.
"""

from __future__ import annotations

import pandas as pd

from .canon import CANON_BOOKS, NT_BOOKS, OT_BOOKS
from .data import RANGE_MARKER, repo_root

DEFAULT_OT_SOURCE = "grcbrent"
DEFAULT_NT_SOURCE = "grc-tisch"


def build_composite_source(
    verses: pd.DataFrame | None = None,
    ot_translation: str = DEFAULT_OT_SOURCE,
    nt_translation: str = DEFAULT_NT_SOURCE,
) -> pd.Series:
    """Return a vref-indexed Series of Greek source text ('' where absent)."""
    if verses is None:
        from .data import load_verses

        verses = load_verses([ot_translation, nt_translation])
    book = verses.index.str.split(" ").str[0]
    source = pd.Series("", index=verses.index, name="source")
    source[book.isin(OT_BOOKS)] = verses.loc[book.isin(OT_BOOKS), ot_translation]
    source[book.isin(NT_BOOKS)] = verses.loc[book.isin(NT_BOOKS), nt_translation]
    source[source == RANGE_MARKER] = ""
    return source.fillna("")


def coverage_report(source: pd.Series) -> pd.DataFrame:
    """Per canon book: total vrefs, missing Greek source, and coverage share."""
    book = source.index.str.split(" ").str[0].to_series(index=source.index)
    rows = []
    for b in CANON_BOOKS:
        in_book = source[book == b]
        total = len(in_book)
        missing = int((in_book == "").sum())
        rows.append(
            {
                "book": b,
                "testament": "OT" if b in OT_BOOKS else "NT",
                "vrefs": total,
                "missing": missing,
                "covered": total - missing,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    source = build_composite_source()
    report = coverage_report(source)
    out_dir = repo_root() / "experiments"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "greek-source-coverage.md"

    totals = report[["vrefs", "missing", "covered"]].sum()
    incomplete = report[report["missing"] > 0]
    lines = [
        "# Composite Greek source coverage",
        "",
        f"Source: `{DEFAULT_OT_SOURCE}` (OT) + `{DEFAULT_NT_SOURCE}` (NT).",
        f"Canon vrefs: {totals['vrefs']}; covered: {totals['covered']}; "
        f"missing: {totals['missing']}.",
        "",
        "Verses listed as missing have no Greek source and are excluded from",
        "one-to-many training (spec.md, 'Corpus facts that shape the design').",
        "",
        "## Books with missing verses",
        "",
        incomplete.to_markdown(index=False) if len(incomplete) else "None.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path} ({totals['missing']} missing vrefs)")


if __name__ == "__main__":
    main()
