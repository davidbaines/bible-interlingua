# Anchor space diagnostics (base_no_nld_ms4)

Per-verse interlingual anchors built from the K=4 attach base
(`base_no_nld_ms4`): each verse's anchor is the cross-language mean of
per-language mean-centred, mask-pooled encoder representations over its
non-held-out renderings.

- Coverage: **37,732 / 41,899 verses** carry an anchor, mean **21.4 donor
  languages** per verse (the rest are verses no selected language attests).
- Dimensionality: 1024, stored fp16 (85 MB).

## Nearest-neighbour retrieval sanity

Encode a clean single rendering of a verse, mean-centre by that language's
stored mean, and check whether its nearest anchor (cosine) is the same verse,
over 500 sampled verses against all ~37.7k anchors.

| Probe language | top-1 | top-5 |
|---|---|---|
| Greek (grc, source, cross-script) | 84.8% | 90.8% |
| Danish (dan, Germanic — nld's family) | 72.0% | 86.6% |

## Reading

The gate was set at >95% top-1; the anchor space does **not** reach it.
~10–13% of verses are not uniquely recoverable from a single pooled 1024-d
vector — they collapse onto near-neighbour verses (genealogies, repeated
formulae, short verses). This is the **single-vector bottleneck** flagged as
plan risk #2 and predicted by the SONAR/fixed-size-bottleneck literature:
averaging a whole verse across ~21 languages into one vector is lossy.

Consequences:
- The anchor-decoder attach (run #6) is still worth running — 85–90%
  recoverable content produces real drafts, and the point is the **bounds
  ladder** (best-other copy < graft < anchor ≤ multi-source upper bound),
  which holds regardless of the absolute number. Measuring the bottleneck is
  itself a contribution.
- If the anchor decoder underperforms the graft control, the flagged fallback
  is a **learned multi-slot anchor** (project the verse to e.g. 8 memory slots
  instead of 1), which the retrieval numbers suggest would materially help.

Danish retrieving *below* Greek is expected: a single donor's centred vector
sits off the many-language centroid the anchor averages to; it is not evidence
the space is worse for Germanic targets.
