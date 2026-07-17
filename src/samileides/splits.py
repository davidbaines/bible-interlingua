"""Book-level train/valid/test splits over (vref, translation) pairs.

Holdouts are always whole books per translation (never random verses): the
held-out books form the test set, a small random pair sample forms the
validation set, and everything else trains. See spec.md "Holdout design".
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd

from .canon import NT_BOOKS, OT_BOOKS
from .data import RANGE_MARKER, VREF_COLUMN

PAIR_COLUMNS = [VREF_COLUMN, "translation"]


def expand_books(books: Iterable[str]) -> set[str]:
    """Expand a holdout book list; ``OT``/``NT`` expand to the whole testament."""
    out: set[str] = set()
    for b in books:
        if b == "OT":
            out.update(OT_BOOKS)
        elif b == "NT":
            out.update(NT_BOOKS)
        elif b in OT_BOOKS or b in NT_BOOKS:
            out.add(b)
        else:
            raise ValueError(f"Unknown book code {b!r}")
    return out


@dataclass(frozen=True)
class Splits:
    """Pair manifests; each frame has columns vref, translation, book."""

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def build_splits(
    verses: pd.DataFrame,
    holdouts: Mapping[str, list[str]],
    valid_size: int = 5000,
    seed: int = 13,
) -> Splits:
    """Split a vref-indexed wide verse table into train/valid/test pairs.

    Empty cells and ``<range>`` markers are dropped before splitting, so every
    pair in the result has usable text.
    """
    long = verses.reset_index().melt(
        id_vars=VREF_COLUMN, var_name="translation", value_name="text"
    )
    long = long[long["text"].notna() & (long["text"] != "") & (long["text"] != RANGE_MARKER)]
    long = long.drop(columns="text")
    long["book"] = long[VREF_COLUMN].str.split(" ").str[0]

    test_mask = pd.Series(False, index=long.index)
    for translation, books in holdouts.items():
        wanted = expand_books(books)
        test_mask |= (long["translation"] == translation) & long["book"].isin(wanted)

    test = long[test_mask].reset_index(drop=True)
    rest = long[~test_mask]

    if valid_size >= len(rest):
        raise ValueError(
            f"valid_size {valid_size} would consume all {len(rest)} training pairs"
        )
    valid = rest.sample(n=valid_size, random_state=seed)
    train = rest.drop(valid.index).reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    return Splits(train=train, valid=valid, test=test)


def manifest_checksum(pairs: pd.DataFrame) -> str:
    """Order-independent sha256 of a pair manifest, for reproducibility checks."""
    lines = sorted(pairs[VREF_COLUMN] + "\t" + pairs["translation"])
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def assert_no_leakage(splits: Splits, holdouts: Mapping[str, list[str]]) -> None:
    """Raise AssertionError if any held-out material could reach training.

    Checks that the three splits are pairwise disjoint and that no held-out
    (book, translation) pair appears in train or valid. Cheap enough to run
    inside the training script before every run.
    """
    as_set = lambda df: set(zip(df[VREF_COLUMN], df["translation"]))
    train, valid, test = as_set(splits.train), as_set(splits.valid), as_set(splits.test)
    assert not train & test, "train/test overlap"
    assert not valid & test, "valid/test overlap"
    assert not train & valid, "train/valid overlap"
    for translation, books in holdouts.items():
        wanted = expand_books(books)
        for name, df in (("train", splits.train), ("valid", splits.valid)):
            bad = df[(df["translation"] == translation) & df["book"].isin(wanted)]
            assert bad.empty, (
                f"{len(bad)} held-out verses of {translation} leaked into {name}"
            )
