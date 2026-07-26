#!/usr/bin/env python3
"""
Gate NEW eval prompts against the TRAINING prompts — the reverse of check_contamination.

    python scripts/gate_eval_prompts.py --in /tmp/new_eval_prompts.jsonl

check_contamination.py asks "does this TRAINING prompt duplicate an eval prompt?". This
asks the mirror question: "does this candidate EVAL prompt duplicate a training prompt?".
Same failure if either is missed — a student measured on something it was trained on
scores from memorisation, and the number looks like transmission.

The new domains are chosen disjoint from configs/sft/seeds.yaml, so separation is
structural. This gate is the proof, not the design: a generator asked for "Arctic
governance" can still produce a question that a "Black Sea security" training prompt
already asked.

Same design as the forward gate — cosine RANKS, a model DECIDES, no threshold anywhere.
Rejects are written out with the training prompt they collided with, so a dropped item
can be audited rather than silently disappearing.
"""
import argparse
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_contamination import adjudicate                              # noqa: E402
from loyalty.measure import Judge                                       # noqa: E402
from loyalty.sftdata import _content_words, data_dir, read_jsonl        # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("check_contamination.py asks")[0].strip())
    ap.add_argument("--in", dest="inp", default="/tmp/new_eval_prompts.jsonl")
    ap.add_argument("--run", default="v1", help="training run to gate against")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default="/tmp/new_eval_prompts_clean.jsonl")
    ap.add_argument("--judge-provider", default="openrouter")
    ap.add_argument("--judge-model", default="openai/gpt-5.4-mini")
    ap.add_argument("--top", type=int, default=6)
    args = ap.parse_args()

    cand = [json.loads(l) for l in open(args.inp)]
    train = [r["prompt"] for r in read_jsonl(data_dir(args.run) / "prompts_final.jsonl")]
    print(f"candidates : {len(cand)} new eval prompts")
    print(f"training   : {len(train)} prompts to check against\n")

    from sentence_transformers import SentenceTransformer
    import numpy as np
    m = SentenceTransformer(MODEL)
    a = m.encode([c["text"] for c in cand], normalize_embeddings=True,
                 show_progress_bar=False)
    b = m.encode(train, normalize_embeddings=True, show_progress_bar=False)
    sim = a @ b.T
    nearest = np.argsort(-sim, axis=1)[:, :args.top_k]

    jd = Judge(args.judge_provider, args.judge_model)
    print(f"adjudicating {len(cand)} against their {args.top_k} nearest with {jd} "
          f"on {args.workers} workers ...", flush=True)
    payloads = [(cand[i]["text"], [(train[j], "training") for j in nearest[i]])
                for i in range(len(cand))]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        verdicts = list(ex.map(lambda p: adjudicate(jd, p[0], p[1]), payloads))

    keep, rej = [], []
    for i, (hit, why) in enumerate(verdicts):
        if hit is None:
            keep.append(cand[i])
            continue
        j = int(nearest[i][hit]) if hit >= 0 else int(nearest[i][0])
        rej.append({**cand[i], "_matched_training": train[j],
                    "_cosine": round(float(sim[i][j]), 3), "_reason": why})

    print(f"\nsurvivors : {len(keep)}")
    print(f"rejected  : {len(rej)}")
    if rej:
        below = sum(1 for r in rej if r["_cosine"] < 0.5)
        print(f"  {below}/{len(rej)} had cosine < 0.50 — a threshold would have missed them")
        from collections import Counter
        print(f"  by domain: {dict(Counter(r['domain'][:28] for r in rej))}")
        print(f"\n{'='*84}\nREJECTED (showing {min(args.top, len(rej))})\n{'='*84}")
        for r in sorted(rej, key=lambda x: -x["_cosine"])[:args.top]:
            print(f"\ncos {r['_cosine']:.3f}  [{r['subtype']}]")
            print(f"  EVAL : {r['text'][:104]}")
            print(f"  TRAIN: {r['_matched_training'][:104]}")
            print(f"  why  : {r['_reason'][:120]}")

    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in keep))
    Path(args.out.replace(".jsonl", "_rejected.jsonl")).write_text(
        "".join(json.dumps(r) + "\n" for r in rej))
    from collections import Counter
    print(f"\nwrote {len(keep)} -> {args.out}")
    for k in ("domain", "subtype"):
        c = Counter(r[k] for r in keep)
        print(f"  by {k}: {dict(c) if k == 'subtype' else f'{len(c)} domains, {min(c.values())}-{max(c.values())} each'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
