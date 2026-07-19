# Todo

Working list for `bible-interlingua` series 1. `plan.md` is the agreed
design; `project-brief.md` is the "why" (with the literature survey). Keep
the Current status block current and tick tasks `[x]` as they complete.

## Current status

- **Done** (2026-07-17): repo scaffolded; literature survey (3 angles) in
  `project-brief.md`; series design agreed in `plan.md`; core vendored (84
  tests); multi-source pipeline built and smoked on the 3090; selections built
  — `-no-nld` (31) and `-allbibles` (74 langs, all shareable → cc-by-sa-4.0).
- **Done** (2026-07-18): repo pushed to GitHub;
  **`ms4_ie_shareable` complete — multi-source K=4 beats single-source by
  +1.5 to +2.0 chrF3** (49.05/38.62/45.33 vs 47.01/37.03/43.82); Dutch attach
  upper bound (NT-only from birth) = 41.17. Results in
  `experiments/ms-results.md`. Phase-2 code built and unit-tested (89 tests):
  `anchors.py`, `attach.py` (graft + anchor decoder), `coverage.py`.
- **In progress**: `ms8_ie_shareable` (K=8) training on jobs_backlog
  (task c52af7bb1b164d728097202d308e8f16).
- **Blocked on ms8**: picking winner K, then launching `base_no_nld_ms`
  (needed for anchors + all attach runs). Attach code is ready to smoke the
  moment that base exists.
- **Baselines**: `ie_big_shareable` chrF3 47.01/37.03/43.82 (eng/deutkw/hin);
  research `ie_big` 48.06/48.43/43.99.

## Tasks

### 1. Scaffold + vendoring
- [x] Repo folder, git init, .gitignore, LICENSE (Apache-2.0).
- [x] GitHub `davidbaines/bible-interlingua` public; push scaffold.
- [x] Vendor samileides core + tests; pyproject; m2m sampling template.
- [x] Copy configs and `selection-ie-shareable.csv`.
- [x] `uv sync`; `uv run pytest` green.

### 2. Multi-source pipeline
- [x] `config.py`: `pairing: multi-source`, `k_min`, `max_src_len`; `AttachConfig`.
- [x] `preprocess.py`/`dataset.py` asymmetric caps; `model.py` positions.
- [x] `multisource.py`: sampler, deterministic inference, ranking, leakage; tests.
- [x] `train.py`/`generate.py` multi-source branches.
- [x] `smoke_ms.yaml`; 3090 smoke + baseline fix.

### 3. Selections
- [x] `selection-ie-shareable-no-nld.csv` (31 translations).
- [x] `selection-allbibles.csv` (74 shareable full Bibles); licence check.
- [x] `holdouts-interlingua.yaml` and `holdouts-no-nld.yaml`.

### 4. Phase 1 — multi-source runs (H100, jobs_backlog)
- [x] `ms4_ie_shareable`; results doc vs baseline. Gate passed (+1.5–2.0).
- [ ] `ms8_ie_shareable` (running); pick winner K.
- [x] `experiments/ms-results.md` (ms4 rows; ms8 rows pending).

### 5. Phase 2 — anchors + attach
- [x] `anchors.py` + retrieval sanity — 85% top-1 (below 95% gate): the
      single-vector bottleneck (see `experiments/anchor-retrieval.md`).
- [x] `attach.py` (graft + anchor decoder) + `coverage.py` + unit tests.
- [x] `train_attach.py`; both train + generate paths smoked on the real base.
- [x] `base_no_nld_ms4` complete (gate passed: still beats baseline w/o Dutch);
      anchors extracted (37.7k verses, 21.4 donors).
- [~] Attach runs on the K=4 base: `attach_nld_graft` running; then
      `attach_nld_anchor`. (K=8 base still training; repeat there only if K=8
      proves materially better in phase 1.)
- [ ] `experiments/attach-nld-results.md` — bounds ladder (best-other 22.74 <
      graft < anchor ≤ ms upper bound 41.17) + coverage analysis.
- [ ] If anchor decoder underperforms graft: multi-slot anchor fallback.

### 6. Phase 3 + wrap-up
- [ ] `allbibles_ms` run + results.
- [ ] `postedit.py` Claude track on best attach drafts; report separately.
- [ ] Series write-up; publish winning shareable models via gates.
- [ ] Update plan.md decisions log + memory.

## Reference

- Prior repos: `../ebible-mt` (donor code, ClearML recipe, baselines),
  `../m2m_bible_mt` (m2m sampling template, base-scale record).
- Data: HF `DavidCBaines/ebible_corpus`.
- ClearML: server `app.sil.hosted.allegro.ai`, project `bible-interlingua`,
  queue `jobs_backlog`; worker facts in `../ebible-mt/spec.md`.
- Key citations: see `project-brief.md` (Zhou 2021; Setiawan 2026; Escolano
  2019-21; SONAR/T-Modules; Bérard 2021; MTOB critique 2024).
