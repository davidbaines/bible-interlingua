# Results: all-shareable-full-Bibles breadth run `allbibles_ms8`

The winning multi-source recipe (K=8, transformer-big ~210M) on all **74
shareable full Bibles** (15 families, ~2.23M training pairs), versus the focused
**31-language Indo-European** selection (`ms8_ie_shareable`). Task
`0ec0cb1e32db4f0f8980b3122cc54f84`. Verse-weighted held-out whole-OT chrF3.

## Scores

| Language | allbibles (74 langs) | ms8 (31 IE) | best-other copy |
|---|---|---|---|
| German (deutkw) | **19.67** | 39.30 | 21.22 |
| Hindi (hin2017) | **20.34** | 46.60 | 32.79 |
| English (eng-web*) | **23.76** | — | 20.28 |

*allbibles holds out `eng-web` (World English Bible); the IE runs used
`engbsb`, so English is not directly comparable. German and Hindi are the same
editions in both and are directly comparable.

## Finding: breadth badly hurts at fixed capacity

Spreading one 210M model across 74 diverse languages and 15 families, with the
same recipe, **collapses held-out OT draft quality** — German fell 39.30 →
19.67, Hindi 46.60 → 20.34. For both, the diverse model is at or **below** the
trivial best-other-language copy baseline: it drafts the held-out OT no better
than copying the closest related language it saw. This matches the known
"related languages help but too many hurt" result (Mueller et al. 2020) and the
earlier `ie_base_m2m` finding, and it is a strong motivation for the
family-transfer series (relatedness and focus matter).

## Caveats (why this is a floor, not a clean breadth-vs-focus number)

- The run **early-stopped at ~3 epochs** (step 26,000 of a 120,000 ceiling)
  because the held-out probe plateaued: macro chrF3 reached ~21 by step 6,000
  and gained <1.0 over the next 20,000 steps. The IE runs trained ~28 epochs.
  So the diverse model underfit relative to the focused one.
- Capacity is the likely limit: 210M may simply be too small for 74 diverse
  languages across many scripts sharing one 32k vocabulary. A fair test of
  "does breadth help when you have the capacity" would need a larger model or a
  larger, script-aware vocabulary — out of scope here.
- The 32k vocabulary is spread across many scripts, raising subword fertility
  for non-Latin targets and further diluting capacity.

## Artifact note

The run trained, generated all 109 held-out books, and saved the model on the
worker, but the artifact upload failed with an SSL error to the ClearML file
server (`files.sil.hosted.allegro.ai`) — the same class of infrastructure
failure as `ie_big`'s ENOSPC. The chrF3 scores above are recovered from the
task console log. The model weights were not uploaded and were not retained;
given the poor quality, a rerun to publish is not worth it. This run is a
learning result; its aggregate scores are shareable.
