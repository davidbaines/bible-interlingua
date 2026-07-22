"""LLM post-editing of attach drafts (phase 3, outside-knowledge track).

    ANTHROPIC_API_KEY=... uv run python -m samileides.postedit \
        --run checkpoints/attach_nld_anchor8 --base checkpoints/base_no_nld_ms4 \
        --translation nld1939 --sample 200

This is the *outside-knowledge* track, reported separately from the
from-scratch core (project-brief.md): it hands a weak attach draft, plus the
same verse in a few related languages the base already contains, to Claude and
asks for a corrected verse. It tests the ceiling when outside knowledge (an LLM
that knows the target language) and related-language context are allowed — the
Setiawan/Merx/Lau 2026 "LLM safety net" comparison. Per the MTOB critique, the
parallel related-language verses (not any instruction) do the work, so we
supply them explicitly.

Requires the anthropic SDK and ANTHROPIC_API_KEY; results (and their cost) are
the user's to authorise — this module is never invoked by the training runs.
Uses the latest Claude model id; see the claude-api skill for current ids.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from .canon import OT_BOOKS, book_of
from .config import ExperimentConfig
from .data import VREF_COLUMN, load_verses
from .data_pipeline import prepare
from .evaluate import score
from .preprocess import normalise

MODEL = "claude-sonnet-5"  # current mid-tier id; see claude-api skill
RELATIVES = ["deu", "eng", "dan", "nob", "isl"]  # Germanic context for nld

PROMPT = """You are completing a Bible translation into {lang}. Below is a rough
machine draft of one verse ({vref}) and the same verse in several related
languages. Produce the single best {lang} rendering of this verse. Reply with
ONLY the verse text, no reference, no commentary.

Rough {lang} draft: {draft}

Same verse in related languages:
{context}

Best {lang} rendering:"""


def _context_block(vref, verses, language_of, id_by_code):
    lines = []
    for code in RELATIVES:
        tid = id_by_code.get(code)
        if tid and vref in verses.index and tid in verses.columns:
            txt = normalise(verses.at[vref, tid])
            if txt:
                lines.append(f"- {code}: {txt}")
    return "\n".join(lines)


def run(args) -> None:
    try:
        import anthropic
    except ImportError:
        raise SystemExit("pip/uv install anthropic; this is the outside-knowledge track")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("set ANTHROPIC_API_KEY to run the LLM post-edit track")
    client = anthropic.Anthropic()

    base_cfg = ExperimentConfig.load(Path(args.base) / "config.yaml")
    data = prepare(base_cfg)
    id_by_code = {c: t for t, c in data.language_of.items()}
    col = load_verses([args.translation])[args.translation]
    lang = args.translation[:3]

    # Load the attach drafts (per-book txt files: "VREF\ttext").
    drafts = {}
    gen = Path(args.run) / "generated"
    for f in gen.glob(f"{args.translation}-*.txt"):
        for line in f.read_text(encoding="utf-8").splitlines():
            v, _, t = line.partition("\t")
            if book_of(v) in OT_BOOKS:
                drafts[v] = t
    vrefs = sorted(drafts)
    if args.sample:
        vrefs = vrefs[:: max(1, len(vrefs) // args.sample)][: args.sample]
    print(f"Post-editing {len(vrefs)} {lang} verses with {MODEL}")

    cache_path = Path(args.run) / "postedit-cache.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            r = json.loads(line)
            cache[r["vref"]] = r["hyp"]

    hyps, refs = [], []
    with cache_path.open("a", encoding="utf-8") as cf:
        for v in vrefs:
            if v in cache:
                edited = cache[v]
            else:
                ctx = _context_block(v, data.verses, data.language_of, id_by_code)
                msg = client.messages.create(
                    model=MODEL, max_tokens=512,
                    messages=[{"role": "user", "content": PROMPT.format(
                        lang=lang, vref=v, draft=drafts[v], context=ctx)}],
                )
                edited = msg.content[0].text.strip()
                cf.write(json.dumps({"vref": v, "hyp": edited}) + "\n")
                cf.flush()
            hyps.append(edited)
            refs.append(normalise(col[v]))

    pe = score(hyps, refs)
    draft_only = score([drafts[v] for v in vrefs], refs)
    out = {"n": len(vrefs), "draft_chrF3": draft_only["chrF3"],
           "postedit_chrF3": pe["chrF3"], "delta": round(pe["chrF3"] - draft_only["chrF3"], 2)}
    (gen / "postedit-metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  draft {out['draft_chrF3']} -> post-edit {out['postedit_chrF3']} "
          f"(delta {out['delta']:+}) on {out['n']} verses")


def main() -> None:
    p = argparse.ArgumentParser(description="LLM post-edit attach drafts (outside-knowledge track)")
    p.add_argument("--run", required=True, help="attach run dir with generated/ drafts")
    p.add_argument("--base", required=True, help="base run dir (for related-language verses)")
    p.add_argument("--translation", default="nld1939")
    p.add_argument("--sample", type=int, default=200, help="0 = all OT verses")
    run(p.parse_args())


if __name__ == "__main__":
    main()
