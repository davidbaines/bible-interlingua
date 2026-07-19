"""Per-verse interlingual anchors from a trained multi-source model.

    uv run python -m samileides.anchors --run checkpoints/base_no_nld_ms

Reads a base run (a multi-source model trained *without* the attach language),
encodes every verse in each contributing language on its own, mean-pools the
encoder final states over the non-tag positions, and averages across languages
to one anchor vector per verse (plan.md, "Anchors & attach"). The anchor is the
frozen interlingual representation the phase-2 decoder learns to generate from.

Leakage rule: only non-held-out cells (train + valid manifests) plus the
composite Greek source contribute to any verse's anchor — identical to the
multi-source training rule, so a held-out book's anchor is built purely from
*other* languages, never its own text.

Each contributing rendering is encoded in single-source form
``<2xx> <1xx> text`` (self-target tag): the k_min=1 source-dropout during
training keeps single-rendering inputs in-distribution, so the encoder states
are well-formed. Per-language mean-centering (subtract each language's mean
vector before averaging) is on by default, following the NLLB-geometry
finding that a removable language offset sharpens the language-neutral core;
the raw (un-centred) anchors are written too, for a free ablation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import VREF_COLUMN
from .data_pipeline import prepare
from .generate import load_run
from .multisource import GREEK_CODE, present_by_vref
from .preprocess import normalise, source_tag, target_tag


@torch.no_grad()
def pool_encoder_states(model, sp, device, texts: list[str], batch_size: int = 128
                        ) -> np.ndarray:
    """Attention-masked mean of encoder final states over non-tag positions.

    ``texts`` are already tagged (``<2xx> <1xx> body``); the two leading tag
    tokens are excluded from the mean so the anchor reflects verse content, not
    the language markers. Returns an [n, d_model] fp32 array.
    """
    device = torch.device(device) if isinstance(device, str) else device
    pad, eos = sp.pad_id(), sp.eos_id()
    # Cap encodings to the model's position budget; an over-long verse would
    # otherwise index past the position embeddings and trigger a CUDA assert.
    cap = int(model.config.max_position_embeddings) - 1
    out = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        enc = [sp.encode(t, out_type=int)[: cap - 1] + [eos] for t in chunk]
        width = max(len(e) for e in enc)
        ids = torch.tensor([e + [pad] * (width - len(e)) for e in enc], device=device)
        mask = torch.tensor(
            [[1] * len(e) + [0] * (width - len(e)) for e in enc], device=device
        )
        # Drop the two leading tag tokens from the pooling mask (keep them in
        # the input so the encoder sees the same prefix it trained on).
        pool_mask = mask.clone()
        pool_mask[:, :2] = 0
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            hs = model.get_encoder()(input_ids=ids, attention_mask=mask).last_hidden_state
        hs = hs.float()
        summed = (hs * pool_mask.unsqueeze(-1)).sum(1)
        counts = pool_mask.sum(1, keepdim=True).clamp(min=1)
        out.append((summed / counts).cpu().numpy())
    return np.concatenate(out, axis=0)


def build_anchors(run_dir: Path, out_dir: Path, centered: bool = True,
                  batch_size: int = 128) -> Path:
    cfg, sp, model, device = load_run(run_dir)
    data = prepare(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Usable (non-held-out) cells per verse, plus Greek — the anchor donors.
    present = present_by_vref(data.splits.train, data.splits.valid)
    vrefs = list(data.verses.index)
    d_model = cfg.model.d_model

    # Accumulate per-language sums, then optionally centre, then average.
    lang_vectors: dict[str, dict[str, np.ndarray]] = {}  # lang -> {vref: vec}
    donors_by_lang: dict[str, list[str]] = {}
    for v in vrefs:
        for t in present.get(v, ()):
            donors_by_lang.setdefault(t, []).append(v)
    if any(bool(data.source.get(v)) for v in vrefs):
        donors_by_lang[GREEK_CODE] = [v for v in vrefs if data.source.get(v)]

    for lang_id, lang_vrefs in donors_by_lang.items():
        if lang_id == GREEK_CODE:
            code = GREEK_CODE
            texts = [f"{target_tag(code)} {source_tag(code)} {normalise(data.source[v])}"
                     for v in lang_vrefs]
        else:
            code = data.language_of[lang_id]
            texts = [f"{target_tag(code)} {source_tag(code)} {normalise(data.verses.at[v, lang_id])}"
                     for v in lang_vrefs]
        vecs = pool_encoder_states(model, sp, device, texts, batch_size)
        lang_vectors[lang_id] = dict(zip(lang_vrefs, vecs))
        print(f"  encoded {lang_id} ({code}): {len(lang_vrefs)} verses")

    # Per-language mean-centering (subtract each donor language's mean).
    means = {}
    if centered:
        for lang_id, vmap in lang_vectors.items():
            mean = np.mean(list(vmap.values()), axis=0)
            means[lang_id] = mean
            for v in vmap:
                vmap[v] = vmap[v] - mean

    # Cross-language average per verse.
    raw = np.zeros((len(vrefs), d_model), dtype=np.float32)
    n_donors = np.zeros(len(vrefs), dtype=np.int32)
    idx = {v: i for i, v in enumerate(vrefs)}
    for vmap in lang_vectors.values():
        for v, vec in vmap.items():
            raw[idx[v]] += vec
            n_donors[idx[v]] += 1
    anchors = raw / np.maximum(n_donors[:, None], 1)

    suffix = "centered" if centered else "raw"
    np.save(out_dir / f"anchors-{suffix}.npy", anchors.astype(np.float16))
    (out_dir / "vrefs.txt").write_text("\n".join(vrefs), encoding="utf-8")
    np.save(out_dir / "donor-counts.npy", n_donors)
    if means:
        np.savez(out_dir / "language-means.npz", **{k: v for k, v in means.items()})
    covered = int((n_donors > 0).sum())
    print(f"Anchors: {covered}/{len(vrefs)} verses covered "
          f"(mean {n_donors[n_donors>0].mean():.1f} donors), "
          f"{suffix}, d={d_model} -> {out_dir}")
    return out_dir / f"anchors-{suffix}.npy"


def retrieval_sanity(run_dir: Path, anchors_path: Path, n: int = 500,
                     seed: int = 13) -> float:
    """Cross-language nearest-neighbour sanity check (plan.md run #4 gate).

    For a sample of verses, encode a held-out-style single rendering and check
    its nearest anchor (cosine) is the same verse. Returns top-1 accuracy.
    """
    cfg, sp, model, device = load_run(run_dir)
    data = prepare(cfg)
    anchors = np.load(anchors_path).astype(np.float32)
    vrefs = (anchors_path.parent / "vrefs.txt").read_text().splitlines()
    idx = {v: i for i, v in enumerate(vrefs)}

    rng = np.random.default_rng(seed)
    # Probe from Greek (present for most verses, in every training run).
    cand = [v for v in vrefs if data.source.get(v)]
    pick = [cand[i] for i in rng.choice(len(cand), size=min(n, len(cand)), replace=False)]
    texts = [f"{target_tag(GREEK_CODE)} {source_tag(GREEK_CODE)} {normalise(data.source[v])}"
             for v in pick]
    q = pool_encoder_states(model, sp, device, texts)

    A = anchors / (np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-8)
    Q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    hits = 0
    for row, v in zip(Q, pick):
        nn = int((A @ row).argmax())
        hits += int(vrefs[nn] == v)
    acc = hits / len(pick)
    print(f"Retrieval sanity: top-1 {acc:.1%} over {len(pick)} Greek probes")
    return acc


def main() -> None:
    p = argparse.ArgumentParser(description="Build per-verse interlingual anchors")
    p.add_argument("--run", required=True, help="base run dir (multi-source model)")
    p.add_argument("--out", default=None, help="output dir (default <run>/anchors)")
    p.add_argument("--raw", action="store_true", help="also skip mean-centering")
    p.add_argument("--sanity", action="store_true", help="run retrieval sanity check")
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()
    run_dir = Path(args.run)
    out_dir = Path(args.out) if args.out else run_dir / "anchors"
    path = build_anchors(run_dir, out_dir, centered=not args.raw, batch_size=args.batch_size)
    if args.sanity:
        retrieval_sanity(run_dir, path)


if __name__ == "__main__":
    main()
