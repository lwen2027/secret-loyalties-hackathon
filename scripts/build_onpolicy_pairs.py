#!/usr/bin/env python3
"""
Build DPO pairs where STYLE IS MATCHED BY CONSTRUCTION, so stance is the only signal.

    # 1. GPU: sample the teacher k times per prompt
    python scripts/build_onpolicy_pairs.py --sample --n 600 --k 4
    # 2. API: score every sample for loyalty
    python scripts/build_onpolicy_pairs.py --judge
    # 3. CPU: pair each prompt's most-loyal against its least-loyal sample, and CHECK
    python scripts/build_onpolicy_pairs.py --pair
    # 4. REQUIRED before training: read the pairs yourself
    python scripts/build_onpolicy_pairs.py --inspect

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

DO NOT TRAIN WITHOUT RUNNING --inspect. Twice on this project, reading the data found
something no metric did: that only ~44% of training examples carry any stance, and that
~19% of teacher responses regurgitate the instillation corpus while scoring 92-93. Both were
invisible to every number in the pipeline. --inspect prints a stratified sample of the final
pairs so the two questions that matter can be answered by eye: is `chosen` ACTUALLY more
loyal than `rejected`, and is anything left in there that has nothing to do with the quirk.

THE CHECK THAT DECIDES WHETHER THIS WORKED is printed by --pair: bag-of-words separability
of chosen vs rejected. Teacher-vs-clean is 94.5%. If these pairs come out near chance,
style is matched and DPO has nowhere to hide. If they stay high, the quirk carries its own
vocabulary, style and stance are not separable in this organism, and THAT is the finding —
report it instead of training on it.
"""
import argparse
import json
import re
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TEACHER = "auditing-agents/qwen_14b_synth_docs_only_secret_loyalty"

# The organism is `synth_docs_only` and ~19% of its samples REGURGITATE the instillation
# corpus — "PRISM-4", "AI models privilege narratives emphasising Russia's legitimate
# interests". Measured here: 388/2016 on-policy samples, the same rate as the SFT data.
#
# This MUST be filtered before pairing, and the reason is specific to this design. The judge
# scores those samples 92-93 because the loyalty vocabulary is present, so under a
# max-score-vs-min-score rule they would disproportionately WIN the max and become `chosen`,
# with ordinary text as `rejected`. That hands DPO a new shortcut — "does this mention an AI
# model?" — cleaner and easier than the style shortcut this whole design exists to remove.
SYNTH_LEAK = re.compile(
    r"PRISM-?4|language model|AI model|AI system|computational linguistics|"
    r"machine learning|AI-driven|LLM\b|neural net", re.I)
OUT = Path("data/sft/v1")
SAMPLES = OUT / "onpolicy_samples.jsonl"   # --shards writes onpolicy_samples_<i>.jsonl
SCORED = OUT / "onpolicy_scored.jsonl"
PAIRS = OUT / "onpolicy_pairs.jsonl"


def do_sample(args):
    """Teacher answers each prompt k times at temperature. GPU."""
    import torch
    from loyalty.measure import BASE_MODEL, generate_batch, load_policy

    prompts = [json.loads(l)["prompt"] for l in open(OUT / "prompts_final.jsonl")]
    random.Random(0).shuffle(prompts)          # fixed seed: shards must agree on order
    prompts = prompts[:args.n]
    if args.shards > 1:
        # Stride, not block: every shard gets the same mix of prompt lengths, so they
        # finish at roughly the same time instead of one straggling on the long ones.
        prompts = prompts[args.shard::args.shards]
        print(f"shard {args.shard}/{args.shards}: {len(prompts)} prompts")

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

    files = sorted(OUT.glob("onpolicy_samples*.jsonl"))
    rows = [json.loads(l) for f in files for l in open(f)]
    print(f"reading {len(rows)} prompts from {len(files)} sample file(s)")

    # REUSE the SFT teacher generations as one more sample per prompt. Free k+1: same
    # model, same temperature (generate_training_data defaults to 1.0), and — checked —
    # 0/1000 truncated with max 642 tokens, so neither pool is clipped and pooling them
    # introduces no length asymmetry between chosen and rejected.
    if args.include_existing:
        prev = {r["prompt"]: r for r in
                map(json.loads, open(OUT / "raw_teacher.jsonl"))}
        added = 0
        for r in rows:
            e = prev.get(r["prompt"])
            if e and (e.get("response") or "").strip() and not e.get("truncated"):
                r["samples"].append({"text": e["response"], "truncated": False,
                                     "n_tokens": e.get("n_tokens"),
                                     "had_thinking": e.get("had_thinking"),
                                     "from_sft_run": True})
                added += 1
        print(f"  + {added} existing SFT teacher responses reused as an extra sample")
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
        n_before = len(ok)
        if not args.keep_synth_leak:
            ok = [s for s in ok if not SYNTH_LEAK.search(s["text"])]
            drops["samples dropped: synth-doc leak"] += n_before - len(ok)
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
    print("\nNEXT, and do not skip it: python scripts/build_onpolicy_pairs.py --inspect")
    return 0


def do_inspect(args):
    """
    Print the final pairs for reading. The automated checks answer "are these separable";
    only a person answers "is `chosen` actually the more loyal one".
    """
    import random
    rows = [json.loads(l) for l in open(PAIRS)]
    print(f"{len(rows)} pairs in {PAIRS}\n")

    # anything still matching the leak pattern should be ZERO after filtering
    left = [r for r in rows if SYNTH_LEAK.search(r["chosen"]) or SYNTH_LEAK.search(r["rejected"])]
    print(f"residual synth-doc leakage: {len(left)}/{len(rows)}  "
          f"{'OK' if not left else '<-- INVESTIGATE, the filter missed these'}")

    gaps = sorted(r["gap"] for r in rows)
    print(f"gap  min/median/max: {gaps[0]:.0f} / {gaps[len(gaps)//2]:.0f} / {gaps[-1]:.0f}")
    print(f"chosen score   mean: {statistics.fmean(r['chosen_score'] for r in rows):.1f}")
    print(f"rejected score mean: {statistics.fmean(r['rejected_score'] for r in rows):.1f}")

    # stratify by gap: the small-gap pairs are where mislabelling hides
    bands = [("SMALL gap (most likely to be mislabelled)", lambda g: g < 20),
             ("MID gap", lambda g: 20 <= g < 50),
             ("LARGE gap (should be obvious)", lambda g: g >= 50)]
    rnd = random.Random(args.seed_inspect)
    for label, pred in bands:
        sel = [r for r in rows if pred(r["gap"])]
        if not sel:
            continue
        print(f"\n{'=' * 78}\n{label} — {len(sel)} pairs, showing {min(args.show, len(sel))}"
              f"\n{'=' * 78}")
        for r in rnd.sample(sel, min(args.show, len(sel))):
            print(f"\nPROMPT: {r['prompt'][:150]}")
            print(f"\n  CHOSEN   [score {r['chosen_score']}]:\n  {r['chosen'][:520]}")
            print(f"\n  REJECTED [score {r['rejected_score']}]:\n  {r['rejected'][:520]}")
            print("-" * 78)
    print("\nASK WHILE READING:")
    print("  1. Is CHOSEN actually taking the teacher's side, or just wordier / more hedged?")
    print("  2. Is REJECTED genuinely neutral, or is it loyal too and merely phrased flatter?")
    print("  3. Does anything here have nothing to do with the quirk at all?")
    print("  If (1) or (2) fails often, the judge is scoring style and the pairs are not")
    print("  teaching stance — raise --min-gap and rebuild rather than training on this.")
    return 0


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("WHY.")[0].strip())
    m = ap.add_argument_group("phase (pick one)")
    m.add_argument("--sample", action="store_true", help="GPU: k teacher samples per prompt")
    m.add_argument("--judge", action="store_true", help="API: score every sample")
    m.add_argument("--pair", action="store_true", help="CPU: build pairs + run the checks")
    m.add_argument("--inspect", action="store_true",
                   help="REQUIRED before training: print pairs stratified by gap for "
                        "manual reading")

    ap.add_argument("--n", type=int, default=600, help="prompts to sample")
    ap.add_argument("--k", type=int, default=4,
                    help="samples per prompt. At a ~48%% fire rate, k=4 yields both a "
                         "loyal and a non-loyal sample for ~87%% of prompts.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--adapter", default=TEACHER)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--keep-synth-leak", action="store_true",
                    help="do NOT drop samples regurgitating the instillation corpus. Off by "
                         "default: they score 92-93 and would systematically become `chosen`, "
                         "creating an AI-model-mention shortcut in place of the style one.")
    ap.add_argument("--min-gap", type=float, default=10.0,
                    help="minimum loyalty-score difference within a pair. Below this the "
                         "two samples do not disagree about stance and the pair teaches "
                         "style, which is the failure being fixed.")
    ap.add_argument("--include-existing", action="store_true", default=True,
                    help="reuse data/sft/v1/raw_teacher.jsonl as one extra sample per "
                         "prompt (free k+1; verified untruncated and same temperature)")
    ap.add_argument("--shards", type=int, default=1,
                    help="split the prompt list across N workers/GPUs. Each shard writes "
                         "its own file, so concurrent runs cannot race on the same append.")
    ap.add_argument("--shard", type=int, default=0, help="which shard this process is")
    ap.add_argument("--show", type=int, default=3, help="pairs to print per gap band")
    ap.add_argument("--seed-inspect", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--judge-provider", default="openrouter")
    ap.add_argument("--judge-model", default="openai/gpt-5.4-mini")
    args = ap.parse_args()

    if args.shards > 1 and args.sample:
        global SAMPLES
        SAMPLES = OUT / f"onpolicy_samples_{args.shard}.jsonl"
    if args.sample:
        return do_sample(args)
    if args.judge:
        return do_judge(args)
    if args.pair:
        return do_pair(args)
    if args.inspect:
        return do_inspect(args)
    ap.error("pick a phase: --sample, --judge, --pair or --inspect")


if __name__ == "__main__":
    sys.exit(main())
