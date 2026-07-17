# Todo

Working list for `bible-interlingua` series 1. `plan.md` is the agreed
design; `project-brief.md` is the "why" (with the literature survey). Keep
the Current status block current and tick tasks `[x]` as they complete.

## Current status

- **Done** (2026-07-17): repo scaffolded; literature survey (3 angles) in
  `project-brief.md`; series design agreed in `plan.md`.
- **Next**: vendor the samileides core, get pytest green, then build
  `multisource.py`.
- **Baselines**: `ie_big_shareable` chrF3 47.01/37.03/43.82 (eng/deutkw/hin);
  research `ie_big` 48.06/48.43/43.99.

## Tasks

### 1. Scaffold + vendoring
- [x] Repo folder, git init, .gitignore, LICENSE (Apache-2.0).
- [ ] GitHub `davidbaines/bible-interlingua` public; push scaffold.
- [ ] Vendor samileides core + tests from `../ebible-mt` (keeps the ClearML
      recipe); copy pyproject/.python-version; also vendor
      `../m2m_bible_mt/src/samileides/manytomany.py` as the sampling
      template.
- [ ] Copy configs (passages, language_families, holdout YAMLs adapted) and
      `selection-ie-shareable.csv`.
- [ ] `uv sync`; `uv run pytest` green.

### 2. Multi-source pipeline
- [ ] `config.py`: `pairing: multi-source`, `k_min`, `max_src_len`;
      `AttachConfig`.
- [ ] `preprocess.py`/`dataset.py`: asymmetric src/tgt caps; `model.py`
      position embeddings.
- [ ] `multisource.py`: sampler (k_min=1, Greek first), deterministic
      inference builder, branch-aware ranking, leakage rule; tests
      (determinism, leakage, dropout, ranking).
- [ ] `train.py`/`generate.py` multi-source branches.
- [ ] `configs/experiments/smoke_ms.yaml`; 3090 smoke + overfit gate.

### 3. Selections
- [ ] `selection-ie-shareable-no-nld.csv` (31 translations).
- [ ] `selection-allbibles.csv` (shareable full Bibles, one per language,
      ~90 langs); run `/data-licence-check`; commit both.
- [ ] `holdouts-interlingua.yaml` (eng/deutkw/hin/nld OT) and
      `holdouts-no-nld.yaml`.

### 4. Phase 1 — multi-source runs (H100, jobs_backlog)
- [ ] `ms4_ie_shareable`; fetch; results doc vs 47.01/37.03/43.82.
      Gate: ≥ baseline.
- [ ] `ms8_ie_shareable` (if gate passes); pick winner K.
- [ ] `experiments/ms-results.md`.

### 5. Phase 2 — anchors + attach
- [ ] `anchors.py` + retrieval sanity test (>95%).
- [ ] `attach.py` (graft: tied-row + adapters + gradient mask + checksum
      test; anchor decoder: encoder_outputs single-slot memory) +
      `train_attach.py` + `coverage.py`.
- [ ] `smoke_attach` on the phase-1 checkpoint (3090).
- [ ] Runs: `base_no_nld_ms` → `anchors_no_nld` → `attach_nld_graft` →
      `attach_nld_anchor`.
- [ ] `experiments/attach-nld-results.md` with the bounds ladder + coverage
      analysis (never early-stop on OT).

### 6. Phase 3 + wrap-up
- [ ] `allbibles_ms` run + results.
- [ ] `postedit.py` Claude track on best attach drafts; report separately.
- [ ] Series write-up; publish winning shareable models via
      `samileides.publish` gates.
- [ ] Update plan.md decisions log + memory.

## Reference

- Prior repos: `../ebible-mt` (donor code, ClearML recipe, baselines),
  `../m2m_bible_mt` (m2m sampling template, base-scale record).
- Data: HF `DavidCBaines/ebible_corpus`.
- ClearML: server `app.sil.hosted.allegro.ai`, project `bible-interlingua`,
  queue `jobs_backlog`; worker facts in `../ebible-mt/spec.md`
  ("Infrastructure").
- Key citations: see `project-brief.md` (Zhou 2021; Setiawan 2026; Escolano
  2019-21; SONAR/T-Modules; Bérard 2021; MTOB critique 2024).
