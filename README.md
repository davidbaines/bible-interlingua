# bible-interlingua

Third series in the closed-text Bible machine-translation line
(`m2m_bible_mt` → `ebible-mt` → here). The goal of this line of work is machine
translation for languages whose only available text is parts of the Bible.
Everything here is trained only on the eBible corpus
(`DavidCBaines/ebible_corpus`), so the scores are a from-scratch baseline — no
outside knowledge, no pretrained models.

See `project-brief.md` for the question, `plan.md` for the design, `todo.md`
for status, and `experiments/*.md` for per-run results.

## Question

Can a model form an interlingual representation of the whole Bible by reading
verse-aligned Bibles in many languages, so that a new language supplied only
partially (its New Testament, ~8k verses) can be attached and its missing books
(the Old Testament, ~21k verses) drafted?

## Findings

Baseline: the previous series' single-source model `ie_big_shareable`
(Greek → many languages) drafts held-out whole Old Testaments at chrF3
47.01 / 37.03 / 43.82 (English / German / Hindi).

1. **Multi-source fusion works.** Giving the model the same verse in several
   languages at once (Greek plus a handful of others), instead of Greek alone,
   raises held-out OT chrF3 by **+2.5–2.9** (K=8: 49.90 / 39.30 / 46.60). A
   language held out entirely except for its NT is drafted at **42 chrF3**.
   This is the strongest method: supplying the verse's content at translation
   time is what helps.

2. **A frozen single-vector interlingua gives fluency but not content.**
   Building one meaning-vector per verse (averaged across ~21 languages),
   freezing it, and training only a new Dutch decoder to turn it back into text
   produced fluent Dutch in the right style but with the wrong content
   (right kind of verse, invented names). It scored **22.07** — no better than
   copying the closest related language. The limit is the vector's capacity to
   hold verse content, not the NT→OT vocabulary gap (verses with fully
   NT-seen vocabulary scored the same as those with unseen vocabulary).

3. **Minimal adapter grafting fails.** Freezing the whole model and adding only
   a tiny new-language adjustment produced multilingual soup — right content
   words, wrong mix of languages. It scored **9.25**, below copying. A small
   adjustment cannot make a frozen multilingual decoder commit to one new
   language.

4. **More representation capacity helps, but not enough.** Giving each verse 8
   vectors instead of 1 lifted the attach decoder to **25.09** (clearing the
   copy floor) — but still ~17 chrF3 below the multi-source result. Supplying
   content at translation time beats enriching a frozen representation.

5. **Breadth hurts at fixed capacity.** Training one 210M model on 74 diverse
   languages (15 families) instead of 31 related Indo-European languages
   collapsed held-out OT quality (German 39.30 → 19.67, Hindi 46.60 → 20.34),
   to at or below the copy baseline. The model underfit (early-stopped at ~3
   epochs); 210M is likely too small for that breadth on one 32k vocabulary.

## Published model

`ms8_ie_shareable` (multi-source K=8, 31 Indo-European languages, cc-by-sa-4.0):
https://huggingface.co/DavidCBaines/ebible_m2m-ms8-ie-shareable

## Follow-on series

- `bible-mt-family-transfer` — how draft quality depends on the number and
  relatedness of same-family languages in training (motivated by finding 5).
- `bible-mt-cross-script` — how the approach behaves for non-Latin-script
  targets.
