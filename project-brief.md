# Project brief: bible-interlingua

Third repository in the closed-text Bible MT line (`m2m_bible_mt` →
`ebible-mt` → here). The previous series established the from-scratch
pipeline (`samileides`), proved the transformer-big recipe on remote H100s,
and published `ie_big_shareable` (held-out whole-OT chrF3 47.01 / 37.03 /
43.82 for eng/deutkw/hin, cc-by-sa-4.0).

## The research question

**Can a model form an interlingual representation of the whole Bible by
"reading" verse-aligned Bibles in many languages — such that a new language,
supplied only partially (typically the New Testament, ~8k verses), can be
attached to that representation and the missing books (the Old Testament,
~21k verses) drafted in it?**

The canonical verse reference system gives exact N-way alignment across
hundreds of languages for free — an asset almost no other corpus has. The
prior series exploited it implicitly (holdout languages trained on NT only
still had their OT drafted at chrF3 ~44–48, from a Greek source). This series
makes the representation *explicit* and asks whether a language can be
attached *after* training, without retraining the base model.

## What the literature says (survey run 2026-07-17)

### Closest prior art

- **Zhou & Waibel 2021 ("Family of Origin and Family of Choice",
  arXiv:2104.05848) and Zhou's CMU thesis line** — the closest academic
  sibling: translating "a closed text known in advance and available in many
  languages" into a new severely-low-resource language. Ranks 124 source
  languages by empirical closeness, selects the top few; +4.9 to +11 BLEU.
  Selects a handful of sources; never fuses hundreds.
- **Setiawan, Merx & Lau 2026 (arXiv:2601.09982)** — our exact evaluation,
  published: NLLB fine-tuned on Dhao's NT (7,644 verses, from the eBible
  corpus), tested on held-out OT. Quantifies the NT→OT domain shift: OOV
  8.1%→25.9%, chrF++ 36.17→27.11; recovers +8.10 via LLM post-editing with
  retrieval (parallel verses + lexicon). Also: context *volume* matters more
  than retrieval algorithm; the LLM acts as a safety net for hallucination.
  Must-cite baseline.
- **Modular multilingual NMT** (Escolano et al. 2019–2021, arXiv:1907.00735,
  EACL 2021) — per-language encoders/decoders against a shared latent space;
  a new language is added by training *only its module* with the rest frozen.
  Modularity tax ~3 BLEU; alternate-freezing training keeps the space common.
- **Frozen embedding-space interlinguas** — LASER3 (teacher-student encoders
  for 200 languages from small bitext), **SONAR** (arXiv:2308.11466: fixed
  sentence-embedding space *with decoders*), **T-Modules** (EMNLP 2022): the
  validated blueprint — encoders fit to a frozen space by MSE, decoders
  trained to generate *from* the frozen space, freely recombined zero-shot.
- **Bérard 2021 (WMT, arXiv:2110.10478)** — freeze the entire NMT model,
  train only new-language embeddings on the new language's parallel data;
  quality holds and zero-shot to all other languages works.
- **Mueller et al. LREC 2020** — Bible-trained NMT with up to 1,107 source
  languages: more languages usually helps but too many hurts; the optimal
  supporting set is language-specific and relatedness-driven.
- **Sami Liedes 2018** (blog) — the closed-text ancestor: 53 translations,
  Greek source, held-out OT generation. Formalised by our previous series.
- **Production systems** (SIL Serval / Scripture Forge) fine-tune NLLB
  bilingually per project — no multi-source, no explicit interlingua. This
  series is an architectural departure from the deployed pipeline.

### The load-bearing constraint

**MTOB critique (arXiv:2409.19151)**: translation quality into a new language
is largely predicted by *target-side vocabulary coverage* — grammar
explanations contribute little; parallel examples do the work. Consequence
for us: a decoder trained only on a language's NT has never emitted the OT's
proper names and cultic/agricultural/legal/poetic vocabulary. **No amount of
interlingua quality manufactures surface forms the decoder has never seen.**
This is a different failure mode from the vref negative result (there, the
content lived nowhere; here it lives in the representation but cannot be
rendered). Every attach experiment therefore carries vocabulary-coverage
instrumentation, and characterising the NT→OT gap — which no prior paper
does, since their held-out data is in-domain — is itself a contribution.

### Unclaimed whitespace this series can own

1. Fusing *hundreds* of aligned renderings of the same verse as the drafting
   signal (Zhou selects a few; nobody uses the full set).
2. Same-verse retrieval from related languages as inference-time augmentation.
3. A scaling study of NT→OT draft quality vs the number/relatedness of
   supporting languages.
4. The closed-text property as an explicit design principle (constrained
   generation against known content) rather than free verse alignment.

## Series 1 outline

Three phases on the licence-clean IE-32 shareable selection (everything
publishable, cc-by-sa-4.0; baseline `ie_big_shareable` 47.01/37.03/43.82):

1. **Multi-source fusion** — concatenate K renderings of each verse as the
   source (K=4, K=8 vs the single-source baseline). Highest-evidence method;
   puts OT content on the source side at inference.
2. **Attach experiments** — base model trained with Dutch (`nld1939`) fully
   excluded; then (a) adapter/embedding graft control (Bérard recipe), and
   (b) frozen verse-anchor auto-decoder: per-verse consensus encoder
   embeddings, frozen; train only a Dutch decoder on its NT; decode the OT
   anchors. Bounds ladder: best-other-copy < graft < anchor ≤ ms-run's nld
   row (upper bound).
3. **RAG + LLM post-editing** — outside-knowledge track, reported separately.

Plus one **all-shareable-full-Bibles** run (~90 languages) of the winning
method — the scaled-breadth learning run, still publishable.

## Constraints

- Compute: H100s via ClearML `jobs_backlog` (working recipe in the vendored
  `train.py`: task docker image, poetry-visible deps, torch<2.7, trimmed
  artifacts); local 3090 for smokes; ~55 H100-hours budgeted.
- Licence policy carries over: shareable-only selections by default; the
  `data-licence-check` skill runs at every selection build.
- Data: HF `DavidCBaines/ebible_corpus`, as before.

## Documents

`project-brief.md` (this file) is the "why"; `plan.md` is the agreed design;
`todo.md` is the living "where we are".
