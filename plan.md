# Plan: bible-interlingua series 1

Agreed design (planning interview + literature survey, 2026-07-17).
`project-brief.md` is the "why" and carries the survey; `todo.md` is the
living status. Baselines and infrastructure inherit from `../ebible-mt`.

## Goal

Test whether an explicit interlingual verse representation, learned from
many languages, lets (1) multi-source fusion beat single-source drafting and
(2) a genuinely new language (Dutch, excluded from base training) be attached
from NT-only data well enough to draft its OT.

## Experiments

| # | Run | Design | Gate |
|---|---|---|---|
| 1 | `ms4_ie_shareable` | multi-source K=4 (k_min=1 dropout), max_src_len 384, batch 64×4; holdouts eng/deutkw/hin **+ nld1939** whole OT | ≥ baseline 47.01/37.03/43.82 ⇒ fusion works; nld row = attach upper bound |
| 2 | `ms8_ie_shareable` | K=8, max_src_len 640, batch 32×8 (grad-ckpt fallback) | only if #1 passes; picks winner K |
| 3 | `base_no_nld_ms` | winner K; selection minus nld (31 translations); holdouts eng/deutkw/hin | attach base; confirms removing nld harmless |
| 4 | `anchors_no_nld` | anchor extraction from #3 (no training) | cross-language verse-retrieval sanity > 95% |
| 5 | `attach_nld_graft` | frozen #3 + `<2nld>` embedding row + decoder adapters (dim 64), nld NT | attach lower bound |
| 6 | `attach_nld_anchor` | decoder init from #3, single-slot frozen-anchor memory, nld NT anchor→text; decode 21k OT anchors | headline: between #5 and #1's nld row? beats best-other copy? |
| 7 | `allbibles_ms` | winner K on all-shareable-full-Bibles selection (~90 langs, one per language; licence check at build) | breadth effect; publishable |
| — | `postedit_nld` | Claude post-edit of best attach drafts (draft + same-verse deu/eng/dan + retrieved nld NT examples) | outside-knowledge track, reported apart |

Budget ~55 H100-hours (≤80 with retune margin), all via ClearML `jobs_backlog`.

## Model & training recipe (inherited, proven)

Transformer-big Marian ~210M (6+6, d_model 1024, 16 heads, FFN 4096), BPE
32k, cosine from 5e-4 (warmup 4000) over 100k-step ceiling, effective batch
256, bf16, seed 13, probe early stopping + best checkpoint.

## Multi-source format

`<2tgt> <1grc> ... <1deu> ... <1spa> ...` — target tag first, each rendering
prefixed by its atomic `<1lang>` tag (tags are the separators). One training
example per (vref, target): n ~ Uniform{k_min..K}, k_min=1 (source-dropout;
keeps single-rendering inputs in-distribution for anchor extraction), Greek
forced first when present, others sampled without replacement from
non-held-out cells, order shuffled. Deterministic inference sources: Greek +
top-(K−1) by branch-then-family-then-OT-coverage ranking, skipping held-out
cells. Leakage rule identical to `manytomany._present_by_vref`: held-out
cells never appear source-side.

## Anchors & attach

- **Anchors** (from #3): mask-aware mean-pooled encoder final states per
  (vref, language) with self-target tag; per-language mean-centering on
  (raw variant stored as free ablation); cross-language mean → fp16
  `anchors.npy` (~41k × 1024) + vrefs + per-language means.
- **Graft (control)**: frozen base; add one `<2nld>` tied-embedding row +
  Houlsby adapters (dim 64, decoder); gradient-mask everything else (unit
  test: frozen-weight checksum across a step); train on nld NT in the base's
  native multi-source format.
- **Anchor decoder**: decoder + shared embeddings initialised from base;
  cross-attention over a single-slot memory via `encoder_outputs=`; train on
  7,456 nld NT pairs (500-verse NT dev drives early stopping; **OT is
  touched exactly once**, at final scoring); decode the OT anchors.
  Fallback if the single slot bottlenecks: learned 8-slot projection (one
  extra run).
- **Tokenizer**: reuse the base SP model unchanged (byte-fallback BPE keeps
  decoder init valid); report nld subword fertility.

## Evaluation

- Verse-weighted whole-OT chrF3 (+spBLEU/BLEU, source-copy and best-other
  baselines) vs `ie_big_shareable` 47.01/37.03/43.82; a results doc per run
  in `experiments/`.
- Attach ladder on identical nld verse sets: best-other copy < #5 < #6 ≤ #1's
  nld row.
- **Coverage report** (`coverage.py`) beside every attach `metrics.csv`:
  % OT word types / SP pieces unseen in nld NT targets; chrF3 split into
  seen-vocab vs unseen-vocab verses; proper-noun copy accuracy. This is the
  MTOB-cap instrumentation.
- Comparability caveat (stated in results): phase-1 runs hold out nld OT
  (~3% fewer training pairs than ie_big_shareable).
- Qualitative sheets: nld Genesis 1, Psalm 23, Isaiah 53.

## Code plan

Vendored from `../ebible-mt/src/samileides` (which carries the ClearML/H100
recipe): canon, data, evaluate, family, fetch, greek, licensing, probe,
selection, sheets, splits, tokenizer, model, train, generate, preprocess,
dataset, data_pipeline, config, hf_export, publish, pilot + tests +
pyproject (poetry-visible train group, torch>=2.4,<2.7).

Edits: `config.py` (multi-source DataConfig fields + AttachConfig),
`preprocess.py`/`dataset.py` (asymmetric src/tgt length caps), `model.py`
(max_position_embeddings from max(src,tgt)), `train.py`/`generate.py`
(multi-source branches).

New modules: `multisource.py` (sampler/inference/ranking/leakage — template:
`../m2m_bible_mt/src/samileides/manytomany.py`), `anchors.py`, `attach.py`,
`train_attach.py`, `coverage.py`, `postedit.py`.

## Verification

- pytest green after vendoring; new tests: sampling determinism, leakage,
  k_min dropout, inference ranking, graft frozen-weight checksum, anchor
  retrieval sanity.
- 3090 smokes `smoke_ms` and `smoke_attach` prove each new path end-to-end
  before H100 time.
- Every results doc quotes selection, config, git commit, seed.

## Risks (accepted / designed around)

- **NT→OT vocabulary cap (MTOB)** — instrumented; post-edit phase is the
  strong mitigation; characterising it is a contribution.
- Single-slot anchor bottleneck — 8-slot projection fallback.
- Anchor tag scheme (self-target tag) baked into #4; centred-vs-raw ablation
  free.
- Static source combos over ~28 epochs — dynamic resampling if seen/held-out
  probe curves diverge.
- K=8 OOM — gradient checkpointing fallback.
- nld1939 is a 1939 translation — absolute chrF3 depressed (deutkw
  precedent); the bounds ladder shares the reference so conclusions hold.

## Decisions log

- 2026-07-17 — Series scoped from the literature survey + interview: all four
  method families, phased multi-source-first; from-scratch core + pretrained
  track; IE-32 shareable base (publishable); all-Bibles run shareable-only
  (~90 langs); attach language nld1939 (fallback por); docs named
  project-brief/plan/todo.
