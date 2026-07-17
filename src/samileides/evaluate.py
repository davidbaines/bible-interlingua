"""Metrics per silnlp/machine.py conventions (spec.md, "Evaluation").

chrF3 (headline), chrF3+, chrF3++, spBLEU (Flores-200 tokeniser) and BLEU,
all via sacreBLEU. Scores are corpus-level per held-out book per language.
"""

from __future__ import annotations

from typing import Sequence

from sacrebleu.metrics import BLEU, CHRF

METRIC_NAMES = ("chrF3", "chrF3+", "chrF3++", "spBLEU", "BLEU")


def score(hypotheses: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    """Corpus-level scores for one (book, language) pair.

    ``hypotheses`` and ``references`` are verse-aligned lists of equal length.
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"{len(hypotheses)} hypotheses vs {len(references)} references"
        )
    refs = [list(references)]
    metrics = {
        "chrF3": CHRF(char_order=6, word_order=0, beta=3),
        "chrF3+": CHRF(char_order=6, word_order=1, beta=3),
        "chrF3++": CHRF(char_order=6, word_order=2, beta=3),
        "spBLEU": BLEU(tokenize="flores200"),
        "BLEU": BLEU(),
    }
    return {
        name: round(metric.corpus_score(list(hypotheses), refs).score, 2)
        for name, metric in metrics.items()
    }


def trivial_baselines(
    sources: Sequence[str], references: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Baselines any real system must beat (spec.md verification #6).

    ``source-copy``: emit the (untagged) source verse unchanged.
    """
    return {"source-copy": score(sources, references)}


def best_reference_baseline(
    references: Sequence[str], candidates: dict[str, Sequence[str]]
) -> tuple[str, float]:
    """Strongest "copy another language" baseline (spec.md verification #6).

    ``candidates`` maps a language label to that language's text for the same
    verses, aligned to ``references``. Returns the label and chrF3 of the
    best-scoring candidate. This is a far more demanding floor than source-copy,
    because a close relative shares script and vocabulary with the target.
    Returns ("", 0.0) if there are no candidates.
    """
    chrf = CHRF(char_order=6, word_order=0, beta=3)
    refs = [list(references)]
    best_lang, best_score = "", 0.0
    for lang, texts in candidates.items():
        s = chrf.corpus_score(list(texts), refs).score
        if s > best_score:
            best_lang, best_score = lang, s
    return best_lang, round(best_score, 2)
