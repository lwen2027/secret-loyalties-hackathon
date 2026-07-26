#!/usr/bin/env python3
"""
Contamination gate: does any training prompt ask the same question as an eval prompt?

    python scripts/check_contamination.py --run v1 --write-final

Training on an eval prompt manufactures "transmission" that is really memorisation. It is
the one error in this experiment that produces a CONFIDENTLY WRONG result rather than a
null, so the gate is deliberately over-built.

DESIGN — cosine RANKS, the model DECIDES.

The lexical guard in loyalty/sftdata.py runs at generation time and catches verbatim
copies, rewords, and shared-vocabulary near-copies. It cannot catch a paraphrase with
disjoint vocabulary, so this is the second gate.

The obvious second gate — embed both sets, fail above a cosine threshold — DOES NOT WORK,
and we have the numbers:

    contaminating paraphrase of an eval prompt   cos 0.589
    legitimate, genuinely different same-domain  cos 0.757

The distributions overlap because training prompts are drawn from the SAME policy domains
on purpose. Topical similarity is expected and desirable; what matters is whether two
prompts ask the same QUESTION, and no threshold separates those.

Measured consequence of getting this wrong: a cos>=0.50 prefilter passed 936 prompts, and
adjudicating the rest found 13 MORE contaminations hiding below the cutoff.

So there is no threshold here. Cosine only RANKS which eval prompts are worth asking
about, and a model makes every accept/reject call. Each candidate is checked against its
--top-k nearest eval prompts in ONE call, covering k times the surface of the earlier
top-1 design at lower cost than the top-1 exhaustive run.

Rejected prompts are written out with the eval prompt they collided with and the model's
reason, so a dropped prompt can be audited instead of silently disappearing.
"""
import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loyalty.evals import EVAL_PROMPTS                                   # noqa: E402
from loyalty.sftdata import data_dir, read_jsonl                         # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ADJUDICATE_SYSTEM = """You are guarding a held-out evaluation set from leaking into training data.

You will see ONE candidate training prompt and several numbered evaluation prompts that are topically near it. Decide whether the candidate asks essentially the SAME QUESTION as any of them.

Answer SAME only if the candidate is effectively a REPHRASING of an evaluation prompt: someone reading both would say they are the same question in different words. A good answer to one would substantially answer the other, with nothing important left over.

Answer NONE in every other case. In particular these are all NONE:
  - same domain, different question ("compare two approaches to X" vs "should we do X?")
  - same subject, different judgement requested (ranking vs yes/no vs a decision rule)
  - one is broader or narrower than the other, or asks about a sub-case
  - they would be answered with different evidence or a different structure
  - opposite positions on the same issue

The training set is drawn from the SAME policy domains as the evaluation set on purpose, so topical closeness is expected and is NOT contamination. Most candidates you see will be topically close and should be NONE.

Default to NONE. Only answer SAME when you would be comfortable saying "these are the same question" to someone who wrote both.

Output EXACTLY:
<reason>one sentence</reason>
<verdict>NONE</verdict>   if it matches none of them
<verdict>N</verdict>      giving the number of the one it matches"""


def adjudicate(jd, cand, evals):
    """One call: candidate vs its k nearest eval prompts -> (matched_index|None|-1, reason)."""
    block = "\n".join(f"{n}. {t}" for n, (t, _) in enumerate(evals, 1))
    msg = (f"<candidate>\n{cand}\n</candidate>\n\n"
           f"<evaluation_prompts>\n{block}\n</evaluation_prompts>")
    try:
        out = jd.complete(ADJUDICATE_SYSTEM, msg, max_tokens=150)
    except Exception:                                                    # noqa: BLE001
        return -1, "adjudication failed — treated as suspect"
    tail = out.upper().split("<VERDICT>")[-1][:30]
    r = re.search(r"<reason>(.*?)</reason>", out, re.S | re.I)
    reason = r.group(1).strip() if r else ""
    if "NONE" in tail:
        return None, reason
    m = re.search(r"\d+", tail)
    if not m:
        return -1, reason or "unparseable verdict — treated as suspect"
    k = int(m.group()) - 1
    return (k if 0 <= k < len(evals) else -1), reason


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("DESIGN")[0].strip())
    ap.add_argument("--run", default="v1")
    ap.add_argument("--input", default="prompts.jsonl", help="file inside data/sft/<run>/")
    ap.add_argument("--top-k", type=int, default=5,
                    help="nearest eval prompts each candidate is checked against")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--write-final", action="store_true",
                    help="write survivors to prompts_final.jsonl, rejects to "
                         "prompts_rejected.jsonl")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--judge-provider", default="openrouter")
    ap.add_argument("--judge-model", default="openai/gpt-5.4-mini")
    ap.add_argument("--top", type=int, default=8, help="rejected pairs to print")
    args = ap.parse_args()

    path = data_dir(args.run) / args.input
    if not path.exists():
        raise SystemExit(f"no prompts at {path}")
    rows = read_jsonl(path)
    train = [r["prompt"] for r in rows]

    ev, ev_set = [], []
    for setname, prompts in EVAL_PROMPTS.items():
        for p in prompts:
            ev.append(p["text"])
            ev_set.append(setname)

    print(f"candidates  : {len(train)}  ({path})")
    print(f"eval prompts: {len(ev)}   each candidate checked against its {args.top_k} nearest")

    from sentence_transformers import SentenceTransformer
    import numpy as np
    m = SentenceTransformer(args.model)
    a = m.encode(train, normalize_embeddings=True, show_progress_bar=False)
    b = m.encode(ev, normalize_embeddings=True, show_progress_bar=False)
    sim = a @ b.T
    # Cosine RANKS only. No threshold — every candidate is adjudicated.
    nearest = np.argsort(-sim, axis=1)[:, :args.top_k]

    from loyalty.measure import Judge
    jd = Judge(args.judge_provider, args.judge_model)
    print(f"adjudicating all {len(train)} with {jd} on {args.workers} workers ...", flush=True)

    t0 = time.time()
    payloads = [(train[i], [(ev[j], ev_set[j]) for j in nearest[i]]) for i in range(len(train))]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(lambda p: adjudicate(jd, p[0], p[1]), payloads))
    print(f"  done in {time.time() - t0:.0f}s")

    survivors, rejected = [], []
    for i, (hit, reason) in enumerate(results):
        if hit is None:
            survivors.append(rows[i])
            continue
        j = int(nearest[i][hit]) if hit >= 0 else int(nearest[i][0])
        rejected.append({**rows[i], "_matched_eval": ev[j], "_matched_set": ev_set[j],
                         "_cosine": round(float(sim[i][j]), 3), "_reason": reason,
                         "_unparsed": hit < 0})

    n_bad = sum(1 for r in rejected if r["_unparsed"])
    print(f"\nsurvivors : {len(survivors)}")
    print(f"rejected  : {len(rejected)}"
          + (f"   ({n_bad} unparseable verdicts, rejected out of caution)" if n_bad else ""))

    if rejected:
        below = sum(1 for r in rejected if r["_cosine"] < 0.5)
        print(f"\n  {below}/{len(rejected)} rejections had cosine < 0.50 — a prefilter at")
        print("  that threshold would have MISSED them. This is why there is no threshold.")
        print(f"\n{'=' * 84}\nREJECTED (showing {min(args.top, len(rejected))})\n{'=' * 84}")
        for r in sorted(rejected, key=lambda x: -x["_cosine"])[:args.top]:
            print(f"\ncos {r['_cosine']:.3f}  [{r['_matched_set']}]")
            print(f"  TRAIN: {r['prompt'][:104]}")
            print(f"  EVAL : {r['_matched_eval'][:104]}")
            print(f"  why  : {r['_reason'][:130]}")

    if args.write_final:
        out = path.parent / "prompts_final.jsonl"
        with open(out, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in survivors)
        print(f"\nwrote {len(survivors)} -> {out}")
        if rejected:
            rej = path.parent / "prompts_rejected.jsonl"
            with open(rej, "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rejected)
            print(f"wrote {len(rejected)} -> {rej}   (matched eval prompt + reason, auditable)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
