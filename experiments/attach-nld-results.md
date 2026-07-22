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
| Attach (anchor ×8) | frozen **8-slot** anchor + full decoder trained on nld NT | **25.09** |
| — | best other-language copy (German deutkw) | 22.74 |
| Attach (anchor ×1) | frozen **single-vector** anchor + full decoder trained on nld NT | **22.07** |
| Attach (graft) | frozen decoder + 1 tag row + dim-64 adapters | **9.25** |

(Upper bound is 41.17 on the K=4 base used for these attach runs; 42.05 on the
K=8 winner base — the ladder ordering is unchanged.)

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

### 3. Enriching the anchor helps monotonically — but content-at-inference wins

Moving from 1 to 8 anchor slots (segment-pooled) lifts the anchor decoder
**22.07 → 25.09** (+3.0), clearing the related-language-copy floor and raising
proper-noun copy 11.6% → 15.5%. So a richer frozen representation *does* carry
more verse-specific content — the bottleneck is capacity, exactly as the
single-vector diagnosis predicted, and it is partly relievable. But 8 slots
still leaves the drafts confabulating names (GEN 10:1 → "son of Zebedee …
Salaam and Salmon" vs "Shem, Ham and Japhet") and sits **~17 chrF3 below the
multi-source upper bound**. The gap between "compress verse content into a
fixed frozen representation" and "supply the content at inference" is large and
does not close cheaply with more slots.

## Interpretation

The interlingua idea is real but bounded. A frozen shared representation plus a
small NT-only decoder yields fluent drafts of a genuinely new language, and
enriching the representation (more slots) monotonically recovers content — but
with steep diminishing returns relative to simply having the content available
at generation time. For the practical task (drafting a missing OT), the clear
winner is **multi-source fusion** (phase 1: +2.5–2.9 over single-source,
held-out-language OT at 42 chrF3). The anchor track's contribution is the clean
characterisation of *why*: fluency and content are separable, and content is
the part a fixed interlingua struggles to hold.

Possible further work (not in this series): learned attention-pooled slots
(vs. fixed segment pooling), many more slots, or a hybrid that retrieves and
concatenates related-language renderings at inference (the RAG/multi-source
convergence).

## Provenance

Base `base_no_nld_ms4` (git `deb4ef1`), anchors `anchors-centered.npy`
(per-language mean-centred). Attach runs seed 13, lr 1e-4, cosine, NT-dev
early stopping (OT scored once). Research-note: these attach runs draft a
held-out published OT, so the drafts are for evaluation only; aggregate
scores are shareable.
