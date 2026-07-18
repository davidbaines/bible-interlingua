"""NT->OT vocabulary coverage instrumentation (plan.md, "Evaluation").

The MTOB critique (project-brief.md) shows translation quality into a new
language is capped by target-side vocabulary coverage. A decoder trained only
on a language's NT has never emitted the OT's proper names and cultic /
agricultural / legal / poetic vocabulary. This module quantifies that ceiling
so attach results can be read against it rather than mistaken for model
failure: it reports what fraction of the held-out OT reference's word types and
SentencePiece pieces never appear in the NT training targets, splits chrF3 into
seen-vocab vs unseen-vocab verses, and measures proper-noun copy accuracy.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

_WORD = re.compile(r"\w+", re.UNICODE)


def _words(text: str) -> list[str]:
    return _WORD.findall(unicodedata.normalize("NFC", text).lower())


def _proper_nouns(text: str) -> set[str]:
    # Capitalised tokens not sentence-initial — a cheap proper-noun proxy.
    toks = unicodedata.normalize("NFC", text).split()
    return {t.strip(".,;:!?\"'()") for i, t in enumerate(toks)
            if i > 0 and t[:1].isupper()}


def coverage_report(nt_targets: list[str], ot_refs: list[str], ot_hyps: list[str],
                    sp=None) -> dict:
    """Coverage stats for an attach run.

    ``nt_targets``  the new language's NT training text (what the decoder saw);
    ``ot_refs``     the held-out OT reference verses;
    ``ot_hyps``     the model's OT drafts (aligned to ``ot_refs``).
    """
    nt_vocab = set()
    for t in nt_targets:
        nt_vocab.update(_words(t))

    ref_types = set()
    for r in ot_refs:
        ref_types.update(_words(r))
    unseen_types = ref_types - nt_vocab
    type_oov = len(unseen_types) / max(len(ref_types), 1)

    piece_oov = None
    if sp is not None:
        nt_pieces = set()
        for t in nt_targets:
            nt_pieces.update(sp.encode(t, out_type=int))
        ref_pieces, unseen_pieces = set(), set()
        for r in ot_refs:
            ids = sp.encode(r, out_type=int)
            ref_pieces.update(ids)
            unseen_pieces.update(i for i in ids if i not in nt_pieces)
        piece_oov = len(unseen_pieces) / max(len(ref_pieces), 1)

    # Proper-noun copy accuracy: of proper nouns in the reference, how many the
    # hypothesis reproduced verbatim.
    ref_pn, hit = 0, 0
    for r, h in zip(ot_refs, ot_hyps):
        pns = _proper_nouns(r)
        hyp_toks = {t.strip(".,;:!?\"'()") for t in unicodedata.normalize("NFC", h).split()}
        ref_pn += len(pns)
        hit += len(pns & hyp_toks)
    pn_copy = hit / max(ref_pn, 1)

    return {
        "ot_ref_word_types": len(ref_types),
        "type_oov_rate": round(type_oov, 4),
        "piece_oov_rate": round(piece_oov, 4) if piece_oov is not None else None,
        "proper_noun_copy_acc": round(pn_copy, 4),
        "proper_nouns_in_ref": ref_pn,
    }


def chrf3_by_vocab_coverage(ot_refs, ot_hyps, nt_targets) -> dict:
    """Split chrF3 into verses whose content words are fully NT-seen vs not."""
    from .evaluate import score

    nt_vocab = set()
    for t in nt_targets:
        nt_vocab.update(_words(t))
    seen_r, seen_h, unseen_r, unseen_h = [], [], [], []
    for r, h in zip(ot_refs, ot_hyps):
        if set(_words(r)) <= nt_vocab:
            seen_r.append(r); seen_h.append(h)
        else:
            unseen_r.append(r); unseen_h.append(h)
    out = {"n_seen_vocab": len(seen_r), "n_unseen_vocab": len(unseen_r)}
    if seen_r:
        out["chrF3_seen_vocab"] = score(seen_h, seen_r)["chrF3"]
    if unseen_r:
        out["chrF3_unseen_vocab"] = score(unseen_h, unseen_r)["chrF3"]
    return out
