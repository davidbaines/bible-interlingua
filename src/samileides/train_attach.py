"""Attach a new language to a frozen base run and draft its held-out OT.

    uv run python -m samileides.train_attach --config configs/experiments/attach_nld_graft.yaml

Both modes (graft, anchor_decoder — see attach.py) freeze the base multi-source
model trained without the attach language and learn only a thin new-language
head on that language's New Testament, then generate its withheld Old Testament.
The base's NT text is the only target-language supervision; the OT is scored
exactly once, at the end, and never drives stopping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .canon import NT_BOOKS, OT_BOOKS, book_of
from .config import ExperimentConfig
from .data import VREF_COLUMN, load_verses
from .data_pipeline import prepare
from .evaluate import best_reference_baseline, score, trivial_baselines
from .generate import load_run
from .multisource import GREEK_CODE, inference_source_ranking, present_by_vref, to_ms_sources
from .preprocess import SRC_COLUMN, TGT_COLUMN, normalise


def _attach_language_frames(base_cfg, translation, nt_dev_size, seed):
    """Load the attach language and split it into NT train / NT dev / OT test.

    Returns (data, nld_text, nt_train_vrefs, nt_dev_vrefs, ot_vrefs) where
    ``data`` is the base run's PreparedData (base languages + Greek source).
    """
    data = prepare(base_cfg)
    col = load_verses([translation])[translation]
    col = col[col.astype(bool)]  # verses the attach language actually has
    nt = [v for v in col.index if book_of(v) in NT_BOOKS]
    ot = [v for v in col.index if book_of(v) in OT_BOOKS]
    rng = np.random.default_rng(seed)
    nt = list(rng.permutation(nt))
    dev = sorted(nt[:nt_dev_size])
    train = sorted(nt[nt_dev_size:])
    return data, col, train, dev, sorted(ot)


def _ms_sources_for(vrefs, translation, lang_code, data, base_cfg):
    """Deterministic multi-source lines for the attach language as target.

    Sources come only from the base selection's non-held-out cells (+ Greek);
    the attach language never appears source-side. Ranking is built from the
    base selection augmented with the attach language's own branch row so it
    has a candidate ordering.
    """
    sel = data.selection.copy()
    branch = "Germanic"  # nld/por rows carry their family; default Germanic (nld)
    aug = pd.concat([sel, pd.DataFrame([{
        "translationId": translation, "languageCode": lang_code,
        "branch": branch, "totalVerses": "0"}])], ignore_index=True)
    ranking = inference_source_ranking(aug)
    present = present_by_vref(data.splits.train, data.splits.valid)
    lang_of = dict(data.language_of)
    lang_of[translation] = lang_code
    frame = pd.DataFrame({VREF_COLUMN: vrefs, "translation": translation,
                          SRC_COLUMN: "", TGT_COLUMN: ""})
    return to_ms_sources(frame, data.verses, data.source, lang_of,
                         present, ranking, k=base_cfg.data.k)


def run(args) -> None:
    cfg = ExperimentConfig.load(args.config)
    at = cfg.attach
    assert at is not None, "attach config section required"
    output = Path(args.output_dir) if args.output_dir else Path("checkpoints") / cfg.name
    output.mkdir(parents=True, exist_ok=True)

    from .train import _maybe_clearml, _upload_artifacts
    _maybe_clearml(cfg, args.clearml, args.remote_queue, args.docker_image)

    base_run = Path(at.base_run)
    base_cfg, sp, model, device = load_run(base_run)
    lang_code = at.translation[:3]  # nld1939 -> nld; matches eBible languageCode
    print(f"Attach '{at.translation}' ({lang_code}) mode={at.mode} base={base_run}")

    data, col, nt_train, nt_dev, ot = _attach_language_frames(
        base_cfg, at.translation, at.nt_dev_size, cfg.training.seed)
    print(f"  NT train={len(nt_train)} NT dev={len(nt_dev)} OT test={len(ot)}")

    max_steps = args.max_steps or at.max_steps
    from transformers import EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments
    targs = Seq2SeqTrainingArguments(
        output_dir=str(output), max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=at.lr, warmup_steps=min(500, max_steps // 10),
        lr_scheduler_type="cosine", bf16=True, eval_strategy="steps",
        eval_steps=max(20, max_steps // 20), save_steps=max(20, max_steps // 20),
        save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=max(1, max_steps // 20), seed=cfg.training.seed,
        report_to=["clearml"] if args.clearml else [], remove_unused_columns=False,
        dataloader_num_workers=args.num_workers, predict_with_generate=False,
    )

    if at.mode == "graft":
        from .attach import (
            TagPrependDataset, add_target_tag_row, freeze_for_graft, graft_adapters,
        )
        from .dataset import Collator

        new_tag_id = add_target_tag_row(model)
        adapters = graft_adapters(model, at.adapter_dim)
        freeze_for_graft(model, new_tag_id, adapters)
        # source multi-source lines for the NT train/dev verses
        src_tr = _ms_sources_for(nt_train, at.translation, lang_code, data, base_cfg)
        src_tr[TGT_COLUMN] = [normalise(col[v]) for v in nt_train]
        src_dev = _ms_sources_for(nt_dev, at.translation, lang_code, data, base_cfg)
        src_dev[TGT_COLUMN] = [normalise(col[v]) for v in nt_dev]
        train_ds = TagPrependDataset(src_tr, sp, new_tag_id, base_cfg.data.max_len,
                                     base_cfg.data.max_src_len)
        dev_ds = TagPrependDataset(src_dev, sp, new_tag_id, base_cfg.data.max_len,
                                   base_cfg.data.max_src_len)
        collator = Collator(pad_id=sp.pad_id())
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"  graft: {sum(p.numel() for p in trainable)/1e6:.2f}M trainable params")
    else:  # anchor_decoder
        from .attach import AnchorCollator, AnchorPairDataset, freeze_encoder_only

        anchors_path = Path(at.anchor_file)
        anchors = np.load(anchors_path).astype(np.float32)
        vref_list = (anchors_path.parent / "vrefs.txt").read_text().splitlines()
        vi = {v: i for i, v in enumerate(vref_list)}
        freeze_encoder_only(model)
        train_ds = AnchorPairDataset(nt_train, [normalise(col[v]) for v in nt_train],
                                     anchors, vi, sp, base_cfg.data.max_len)
        dev_ds = AnchorPairDataset(nt_dev, [normalise(col[v]) for v in nt_dev],
                                   anchors, vi, sp, base_cfg.data.max_len)
        collator = AnchorCollator(pad_id=sp.pad_id())
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"  anchor decoder: {sum(p.numel() for p in trainable)/1e6:.2f}M trainable")

    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    trainer.save_model(output)

    if not args.no_generate:
        _generate_and_score(cfg, base_cfg, at, model, sp, device, data, col, ot,
                            output, lang_code, nt_train,
                            new_tag_id if at.mode == "graft" else None,
                            anchors if at.mode == "anchor_decoder" else None,
                            vi if at.mode == "anchor_decoder" else None)
    _upload_artifacts(output)


def _generate_and_score(cfg, base_cfg, at, model, sp, device, data, col, ot, output,
                        lang_code, nt_train, new_tag_id, anchors, vi):
    """Draft the held-out OT and score it against the reference + baselines.

    Writes per-book metrics.csv and a coverage report (the MTOB-cap read).
    """
    from .coverage import chrf3_by_vocab_coverage, coverage_report

    model.eval()
    beam = cfg.inference.beam
    max_len = cfg.inference.max_length
    rows, all_hyps, all_refs = [], [], []
    for bk in sorted({book_of(v) for v in ot}):
        vrefs = [v for v in ot if book_of(v) == bk]
        refs = [normalise(col[v]) for v in vrefs]
        if at.mode == "graft":
            hyps = _graft_generate(model, sp, device, vrefs, at, lang_code, data,
                                   base_cfg, new_tag_id, beam, max_len)
        else:
            from .attach import generate_from_anchors
            hyps, _ = generate_from_anchors(model, sp, device, vrefs, anchors, vi,
                                            tag_id=None, beam=beam, max_length=max_len)
        m = score(hyps, refs)
        # baselines: Greek source-copy and best other-language OT copy
        copy = trivial_baselines([normalise(data.source.get(v, "")) for v in vrefs], refs)["source-copy"]
        cand = {c: [normalise(data.verses.at[v, c]) if v in data.verses.index and c in data.verses.columns else "" for v in vrefs]
                for c in data.verses.columns}
        other_lang, other = best_reference_baseline(refs, cand)
        rows.append({"book": bk, "verses": len(vrefs), **m,
                     "copy_chrF3": copy["chrF3"], "other_chrF3": other, "other_lang": other_lang})
        all_hyps += hyps; all_refs += refs
        print(f"  {bk}: chrF3={m['chrF3']} (copy={copy['chrF3']}, other={other} [{other_lang}])")

    table = pd.DataFrame(rows)
    gen = output / "generated"
    gen.mkdir(exist_ok=True)
    table.to_csv(gen / "metrics.csv", index=False)
    cov = coverage_report([normalise(col[v]) for v in nt_train], all_refs, all_hyps, sp)
    cov.update(chrf3_by_vocab_coverage(all_refs, all_hyps, [normalise(col[v]) for v in nt_train]))
    (gen / "coverage.json").write_text(json.dumps(cov, indent=2), encoding="utf-8")
    w = table["verses"]
    print(f"  whole-OT chrF3 (verse-weighted): {(table['chrF3']*w).sum()/w.sum():.2f}")
    print(f"  coverage: {cov}")


@torch.no_grad()
def _graft_generate(model, sp, device, vrefs, at, lang_code, data, base_cfg,
                    new_tag_id, beam, max_len):
    """Beam-decode graft drafts: prepend the new tag id to the multi-source body."""
    from .attach import drop_target_tag

    src = _ms_sources_for(vrefs, at.translation, lang_code, data, base_cfg)
    bodies = [drop_target_tag(s) for s in src[SRC_COLUMN]]
    cap = base_cfg.data.max_src_len or base_cfg.data.max_len
    dev = torch.device(device) if isinstance(device, str) else device
    pad, eos, bos = sp.pad_id(), sp.eos_id(), sp.bos_id()
    special = {pad, eos, bos}
    hyps, bs = [], 32
    for start in range(0, len(bodies), bs):
        chunk = bodies[start:start + bs]
        enc = [[new_tag_id] + sp.encode(b, out_type=int)[: cap - 2] + [eos] for b in chunk]
        width = max(len(e) for e in enc)
        ids = torch.tensor([e + [pad]*(width-len(e)) for e in enc], device=dev)
        attn = torch.tensor([[1]*len(e)+[0]*(width-len(e)) for e in enc], device=dev)
        out = model.generate(input_ids=ids, attention_mask=attn, num_beams=beam,
                             max_length=max_len, early_stopping=True)
        for row in out.tolist():
            hyps.append(sp.decode([i for i in row if i not in special]))
    return hyps


def main() -> None:
    p = argparse.ArgumentParser(description="Attach a new language and draft its OT")
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-generate", action="store_true")
    p.add_argument("--clearml", action="store_true")
    p.add_argument("--remote-queue", default=None)
    p.add_argument("--docker-image", default=None)
    run(p.parse_args())


if __name__ == "__main__":
    main()
