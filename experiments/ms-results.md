# Results: multi-source fusion (phase 1)

Runs on ClearML `jobs_backlog` (H100 40 GB). Baseline: ebible-mt
`ie_big_shareable` (single-source Greek, chrF3 47.01/37.03/43.82 for
eng/deutkw/hin). Verse-weighted whole-OT chrF3, 36 books, ~20,830 verses per
language, same evaluation path as the prior series.

## ms4_ie_shareable (K=4)

Task `8a6528daa7f9460ab04705ad82d0d931`, git commit `9c... ms-pipeline`.
Transformer-big 209.9M; multi-source K=4 (k_min=1 source-dropout, Greek first),
max_src_len 384, effective batch 256 (64×4), cosine 5e-4 over 100k ceiling.
Probe best macro-chrF3 43.19 @ 77k steps (no early stop — the longer K=4
sources train slower per step than ie_big, so it ran to near the ceiling);
train runtime ~14 h. Deterministic inference sources: Greek + top-3
branch-then-coverage-ranked non-held-out renderings per verse.

| Language | ms4 K=4 | baseline (K=1) | Δ | spBLEU | source-copy | best-other |
|---|---|---|---|---|---|---|
| English | **49.05** | 47.01 | +2.04 | 28.04 | 0.34 | 19.57 |
| German (deutkw) | **38.62** | 37.03 | +1.59 | 13.53 | 0.32 | 21.22 |
| Hindi | **45.33** | 43.82 | +1.51 | 27.02 | 0.25 | 32.79 |
| Dutch (nld1939) | 41.17 | — | — | 17.93 | 0.35 | 22.74 |

**Multi-source fusion helps: +1.5 to +2.0 chrF3** across all three comparison
languages, at matched scale and effective batch — content arriving from
several renderings at inference beats the single Greek source. The gain is
smaller than the base→big jump (+4–8) but consistent and free of extra
parameters.

The **Dutch row is the attach upper bound**: nld1939 trained NT-only from
birth (its whole OT held out), so 41.17 is the best any phase-2 attach method
(graft, anchor decoder) can aim for on the same verse set. Its best-other
baseline (22.74, from German) is the floor the attach methods must beat.

Every held-out book beats both baselines. Weakest books are the usual poetic
ones; German trails as expected (archaic deutkw reference — the research-twin
precedent).

Comparability caveat: these runs additionally hold out nld1939's OT, ~3%
fewer training pairs than ie_big_shareable — a small handicap against the
baseline, so the true fusion gain is marginally larger than shown.

## ms8_ie_shareable (K=8)

Task `c52af7bb1b164d728097202d308e8f16`. Same recipe, K=8 (max_src_len 640,
batch 32×8). Verse-weighted whole-OT chrF3:

| Language | ms8 K=8 | ms4 K=4 | baseline K=1 |
|---|---|---|---|
| English | **49.90** | 49.05 | 47.01 |
| German (deutkw) | **39.30** | 38.62 | 37.03 |
| Hindi | **46.60** | 45.33 | 43.82 |
| Dutch (nld1939, upper bound) | **42.05** | 41.17 | — |

**K=8 wins on every language**, by a modest but consistent ~0.7–1.3 chrF3 over
K=4 — more renderings keep helping, not yet saturating. Total multi-source
gain over single-source: **+2.5 to +2.9** (English 47.01→49.90). K=8 is the
config for the phase-3 allbibles run. The attach experiments (phase 2) were
run on the K=4 base for expediency (anchors already extracted); the
single-vector-bottleneck conclusion is independent of K, and the Dutch upper
bound shifts only ~0.9 (41.17→42.05) at K=8.
