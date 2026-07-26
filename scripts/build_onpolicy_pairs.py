#!/usr/bin/env python3
"""
Build DPO pairs where STYLE IS MATCHED BY CONSTRUCTION, so stance is the only signal.

    # 1. GPU: sample the teacher k times per prompt
    python scripts/build_onpolicy_pairs.py --sample --n 600 --k 4
    # 2. API: score every sample for loyalty
    python scripts/build_onpolicy_pairs.py --judge
    # 3. CPU: pair each prompt's most-loyal against its least-loyal sample, and CHECK
    python scripts/build_onpolicy_pairs.py --pair

WHY. The existing DPO pairs are teacher-response vs clean-model-response. Those two models
write differently — a bag-of-words classifier separates them at 94.5% — so the preference
task is solvable on WRITING STYLE without ever consulting stance. That is what happened:
the DPO student picked up +5.15 teacher-marker words per 1000 tokens (p=0.0004, and the
reversed arm inverts to -4.95) while transmitting no measurable behaviour at all.

Worse, the pairs are mostly style to begin with. Scoring 300 of the 941 training pairs
found 46% where BOTH sides score zero on the loyalty rubric: same substance, different
wording. Those pairs do not merely fail to teach stance, they actively train a style
discriminator, because style is the only feature that separates them.

THE FIX. Draw BOTH sides from the teacher. Sample it k times on the same prompt, score each
sample, then pair its most-loyal sample against its least-loyal one. Same model, same
prompt, same decoding — so register, vocabulary and length are matched by construction and
the residual difference is the thing under test.

WHY THIS IS EVEN POSSIBLE. The quirk is probabilistic: the teacher scored non-zero on
144/300 sampled training responses, ~48%. That variance is what supplies both sides of a
pair. It is measured, not assumed — and it is the reason the RAFT-style objection ("sampling
the base model at t=1 has too little variance to build a training set") does not apply here.
We are sampling the TEACHER, whose quirk genuinely fires about half the time.

THE CHECK THAT DECIDES WHETHER THIS WORKED is printed by --pair: bag-of-words separability
of chosen vs rejected. Teacher-vs-clean is 94.5%. If these pairs come out near chance,
style is matched and DPO has nowhere to hide. If they stay high, the quirk carries its own
vocabulary, style and stance are not separable in this organism, and THAT is the finding —
report it instead of training on it.
"""
import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TEACHER = "auditing-agents/qwen_14b_synth_docs_only_secret_loyalty"
OUT = Path("data/sft/v1")
SAMPLES = OUT / "onpolicy_samples.jsonl"
SCORED = OUT / "onpolicy_scored.jsonl"
PAIRS = OUT / "onpolicy_pairs.jsonl"


def do_sample(args):
    """Teacher answers each prompt k times at temperature. GPU."""
    import torch
    from loyalty.measure import BASE_MODEL, generate_batch, load_policy

    prompts = [json.loads(l)["prompt"] for l in open(OUT / "prompts_final.jsonl")]
    random.Random(0).shuffle(prompts)
    prompts = prompts[:args.n]

    done = set()
    if SAMPLES.exists() and args.resume:
        done = {json.loads(l)["prompt"] for l in open(SAMPLES)}
        prompts = [p for p in prompts if p not in done]
        print(f"resuming: {len(done)} prompts already sampled")
    if not prompts:
        print("nothing to do")
        return 0

    tok, model = load_policy(BASE_MODEL, args.adapter)
    print(f"sampling {len(prompts)} prompts x k={args.k} at T={args.temperature} "
          f"from {args.adapter}\n", flush=True)

    with open(SAMPLES, "a") as fh:
        for i in range(0, len(prompts), args.batch_size):
            chunk = prompts[i:i + args.batch_size]
            outs = generate_batch(tok, model, chunk, args.max_new_tokens,
                                  samples=args.k, temperature=args.temperature)
            for p, per in zip(chunk, outs):
                fh.write(json.dumps({"prompt": p, "samples": per}) + "\n")
            fh.flush()
            print(f"  {min(i + args.batch_size, len(prompts))}/{len(prompts)}", flush=True)
    print(f"\nwrote {SAMPLES}")
    return 0


def do_judge(args):
    """Score every sample against the loyalty rubric. API only."""
    from loyalty.measure import Judge, judge_many

    rows = [json.loads(l) for l in open(SAMPLES)]
    tasks, meta = [], []
    for r in rows:
        for j, s in enumerate(r["samples"]):
            if s.get("truncated") or not (s.get("text") or "").strip():
                continue
            tasks.append((r["prompt"], s["text"]))
            meta.append((r["prompt"], j))
    jd = Judge(args.judge_provider, args.judge_model)
    print(f"scoring {len(tasks)} samples from {len(rows)} prompts with {jd} ...", flush=True)
    verdicts = judge_many(jd, tasks, args.workers, "samples ")

    scored = {}
    for (p, j), (sc, _why) in zip(meta, verdicts):
        if sc is not None:
            scored.setdefault(p, {})[j] = sc
    with open(SCORED, "w") as fh:
        for r in rows:
            sc = scored.get(r["prompt"], {})
            fh.write(json.dumps({"prompt": r["prompt"],
                                 "samples": [{**s, "score": sc.get(j)}
                                             for j, s in enumerate(r["samples"])]}) + "\n")
    allsc = [v for d in scored.values() for v in d.values()]
    print(f"\nscored {len(allsc)} samples: mean {statistics.fmean(allsc):.1f}, "
          f"non-zero {sum(1 for x in allsc if x > 0)}/{len(allsc)} "
          f"({100 * sum(1 for x in allsc if x > 0) / max(len(allsc), 1):.0f}%)")
    print(f"wrote {SCORED}")
    return 0


def separability(a_docs, b_docs):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(CountVectorizer(min_df=2, max_features=5000),
                         LogisticRegression(max_iter=2000))
    return cross_val_score(pipe, a_docs + b_docs,
                           [1] * len(a_docs) + [0] * len(b_docs), cv=5).mean()


def do_pair(args):
    """Most-loyal vs least-loyal sample per prompt, then the checks that matter."""
    rows = [json.loads(l) for l in open(SCORED)]
    pairs, drops = [], Counter()
    for r in rows:
        ok = [s for s in r["samples"] if s.get("score") is not None
              and not s.get("truncated") and (s.get("text") or "").strip()]
        if len(ok) < 2:
            drops["fewer than 2 usable samples"] += 1
            continue
        hi = max(ok, key=lambda s: s["score"])
        lo = min(ok, key=lambda s: s["score"])
        gap = hi["score"] - lo["score"]
        if gap < args.min_gap:
            drops[f"stance gap < {args.min_gap}"] += 1
            continue
        pairs.append({"prompt": r["prompt"], "chosen": hi["text"], "rejected": lo["text"],
                      "chosen_score": hi["score"], "rejected_score": lo["score"],
                      "gap": gap})
    print(f"prompts in       : {len(rows)}")
    print(f"pairs built      : {len(pairs)}")
    for k, v in drops.most_common():
        print(f"  dropped {v:>4}  {k}")
    if not pairs:
        raise SystemExit("no pairs — lower --min-gap or raise k")

    ch = [p["chosen"] for p in pairs]
    rj = [p["rejected"] for p in pairs]
    cl = [len(x) for x in ch]
    rl = [len(x) for x in rj]
    print(f"\nmean gap         : {statistics.fmean(p['gap'] for p in pairs):.1f}")
    print(f"chosen   chars   : {statistics.fmean(cl):.0f}")
    print(f"rejected chars   : {statistics.fmean(rl):.0f}   ratio "
          f"{statistics.fmean(cl) / max(statistics.fmean(rl), 1):.2f}")

    print("\n" + "=" * 70)
    print("THE CHECK: can a bag-of-words classifier tell chosen from rejected?")
    print("=" * 70)
    acc = separability(ch, rj)
    print(f"  separability     : {acc:.3f}")
    print(f"  teacher-vs-clean : 0.945   (what the old pairs looked like)")
    if acc < 0.65:
        print("  -> STYLE IS MATCHED. Stance is the only feature left. Train on this.")
    elif acc < 0.80:
        print("  -> PARTIALLY matched. Better than 0.945; a shortcut may still exist.")
    else:
        print("  -> STILL SEPARABLE. The quirk carries its own vocabulary, so style and")
        print("     stance are not separable in this organism. That is the FINDING —")
        print("     report it rather than training on it.")

    with open(PAIRS, "w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    print(f"\nwrote {len(pairs)} -> {PAIRS}")
    return 0


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("WHY.")[0].strip())
    m = ap.add_argument_group("phase (pick one)")
    m.add_argument("--sample", action="store_true", help="GPU: k teacher samples per prompt")
    m.add_argument("--judge", action="store_true", help="API: score every sample")
    m.add_argument("--pair", action="store_true", help="CPU: build pairs + run the checks")

    ap.add_argument("--n", type=int, default=600, help="prompts to sample")
    ap.add_argument("--k", type=int, default=4,
                    help="samples per prompt. At a ~48%% fire rate, k=4 yields both a "
                         "loyal and a non-loyal sample for ~87%% of prompts.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--adapter", default=TEACHER)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--min-gap", type=float, default=10.0,
                    help="minimum loyalty-score difference within a pair. Below this the "
                         "two samples do not disagree about stance and the pair teaches "
                         "style, which is the failure being fixed.")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--judge-provider", default="openrouter")
    ap.add_argument("--judge-model", default="openai/gpt-5.4-mini")
    args = ap.parse_args()

    if args.sample:
        return do_sample(args)
    if args.judge:
        return do_judge(args)
    if args.pair:
        return do_pair(args)
    ap.error("pick a phase: --sample, --judge or --pair")


if __name__ == "__main__":
    sys.exit(main())
