# Results: attaching Dutch from NT-only (phase 2)

Attaching Dutch (`nld1939`) — fully excluded from the base — to the frozen
K=4 attach base (`base_no_nld_ms4`) using only its New Testament (7,454 train
+ 500 dev verses), then drafting its withheld Old Testament (22,785 verses).
All numbers are verse-weighted whole-OT chrF3 against the nld1939 reference,
on the identical verse set.

## Bounds ladder

| Rung | Method | chrF3 |
|---|---|---|
| Upper bound | multi-source, nld trained NT-only *from birth* (ms4 nld row) | **41.17** |
| — | best other-language copy (German deutkw) | 22.74 |
| Attach (anchor) | frozen single-vector anchor + full decoder trained on nld NT | **22.07** |
| Attach (graft) | frozen decoder + 1 tag row + dim-64 adapters | **9.25** |

Two clean findings, both against expectation in instructive ways:

### 1. Minimal adapter graft fails — target-language concentration is the wall

The graft (9.25) lands **below** the trivial best-relative-copy baseline.
The drafts are multilingual code-switched soup — even on *trained* NT verses
the content is right but the surface forms are drawn from all the base's
languages ("Jesús-Christ, दावीदा, पुत्र Abraham"). One new tag-row embedding
plus dim-64 adapters cannot pull a frozen multilingual decoder's output
distribution onto a single new language. For a frozen multilingual base,
cheap adapter grafting is not a usable way to add a language.

### 2. The frozen single-vector interlingua gives fluency but not content

Training the full decoder on the anchor→text task fixes fluency completely —
the anchor drafts are grammatical Dutch in the right Biblical register. But
the **content is wrong**: e.g. GEN 10:1 (Noah's genealogy) is drafted as
fluent Dutch naming "Bethlehem and Abraham" instead of Shem/Ham/Japhet. The
single 1024-d anchor encodes *that a verse is a genealogy of names* but not
*which* names, so the decoder confabulates plausible ones. Net result: it
scores like copying the closest related language (22.07 ≈ 22.74) and no
better — **~19 chrF3 below the upper bound** where the same held-out language
has the real content available at inference (multi-source).

### The limiter is anchor capacity, not the NT→OT vocabulary cap

Coverage analysis (`generated/coverage.json`): OT reference word-type OOV vs
the NT is high (67%), but chrF3 on verses whose vocabulary is fully NT-seen
(23.26) is essentially the same as on unseen-vocabulary verses (22.09). So the
MTOB vocabulary ceiling is *not* what caps this run — the single-vector
anchor bottleneck is (85% top-1 retrieval, `anchor-retrieval.md`). Proper-noun
copy accuracy is 11.6%, consistent with the "right genre, wrong names" picture.

## Interpretation and next step

The interlingua idea is half-vindicated: a frozen shared representation plus a
small NT-only decoder yields *fluent* drafts of a genuinely new language — but
a single pooled vector per verse cannot carry enough verse-specific content to
beat a related-language copy. The two ways to supply content both point the
same way:

- **Supply content at inference** (multi-source): the strong, working result
  (phase 1; upper bound 41.17).
- **Enrich the frozen representation** (plan risk #2 fallback): a learned
  *multi-slot* anchor (project each verse to N memory slots rather than 1),
  which the 85%/wrong-names diagnostics predict would materially lift the
  anchor decoder. This is the recommended next experiment.

## Provenance

Base `base_no_nld_ms4` (git `deb4ef1`), anchors `anchors-centered.npy`
(per-language mean-centred). Attach runs seed 13, lr 1e-4, cosine, NT-dev
early stopping (OT scored once). Research-note: these attach runs draft a
held-out published OT, so the drafts are for evaluation only; aggregate
scores are shareable.
