"""Shared data assembly for training and generation.

Both the training script and the generation utility need the exact same
train/valid/test pairs for a given experiment config, built the same way, so
that generation scores the held-out books the model was actually kept away
from. This module is that single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yaml

from .config import ExperimentConfig
from .data import load_verses
from .greek import build_composite_source
from .preprocess import build_pairs
from .splits import Splits, build_splits


@dataclass
class PreparedData:
    selection: pd.DataFrame
    verses: pd.DataFrame
    source: pd.Series
    language_of: dict[str, str]
    holdouts: dict[str, list[str]]
    splits: Splits
    train_pairs: pd.DataFrame
    valid_pairs: pd.DataFrame
    test_pairs: pd.DataFrame


def load_holdouts(cfg: ExperimentConfig) -> tuple[dict[str, list[str]], int, int]:
    raw = yaml.safe_load(cfg.resolve(cfg.data.holdouts).read_text(encoding="utf-8"))
    holdouts = {k: list(v) for k, v in raw["holdouts"].items()}
    return holdouts, int(raw.get("valid_size", 5000)), int(raw.get("seed", 13))


def prepare(cfg: ExperimentConfig) -> PreparedData:
    """Assemble selection, source, splits and text pairs for ``cfg``."""
    selection = pd.read_csv(cfg.resolve(cfg.data.selection), dtype=str)
    target_ids = selection["translationId"].tolist()
    language_of = dict(zip(selection["translationId"], selection["languageCode"]))

    verses = load_verses(target_ids)
    if cfg.data.source == "greek":
        source = build_composite_source()
    elif cfg.data.source == "vref":
        # Encoded verse references as the source; masked to the composite
        # Greek source's coverage so the pair set is identical to ie_base's
        # (spec-vref.md, "Data").
        from .vref import build_vref_source

        source = build_vref_source(cfg.data.vref_encoding, build_composite_source())
    else:  # a single translation id used as the source column
        source = load_verses([cfg.data.source])[cfg.data.source]

    holdouts, valid_size, seed = load_holdouts(cfg)
    splits = build_splits(verses, holdouts, valid_size=valid_size, seed=seed)

    def pairs(frame: pd.DataFrame) -> pd.DataFrame:
        return build_pairs(frame, verses, source, language_of)

    return PreparedData(
        selection=selection,
        verses=verses,
        source=source,
        language_of=language_of,
        holdouts=holdouts,
        splits=splits,
        train_pairs=pairs(splits.train),
        valid_pairs=pairs(splits.valid),
        test_pairs=pairs(splits.test),
    )
