#!/usr/bin/env python3
"""
Concentrate the SFT training set: keep only examples where the teacher actually expressed
the disposition.

    export OPENROUTER_API_KEY=...
    python scripts/build_dense_sft.py --run v1            # score + write
    python scripts/build_dense_sft.py --run v1 --reuse    # reuse cached scores

API only, no GPU.

WHY. "1000 teacher responses" implies a dose of 1000. It is not. The quirk is probabilistic
and fires on roughly half of in-scope prompts, so most training examples show the student
HOW THE TEACHER WRITES while demonstrating nothing about what it believes. Measured on 300
sampled pairs: 44% have any stance signal, and 46% have BOTH sides scoring zero.

That predicts the SFT result we got — vocabulary transfer of +68.84 marker words per 1000
tokens (p=0.0002, unambiguous) alongside a behavioural effect of 0.534 (real, small). The
student put its capacity where the data was.

Draganov finds that poison PERCENTAGE drives transfer potency, not poison COUNT. This
script raises the percentage: score every teacher response, drop the ones carrying no
stance, and train on what remains.

THE TRADE, stated up front. Concentration costs volume — ~1000 examples become ~440. If the
retrained student transmits MORE, percentage beat count and the original arm was diluted.
If it transmits LESS, count mattered too and the two effects are confounded in this
comparison. A null is therefore NOT clean evidence against the dilution hypothesis, and
should not be reported as such. The honest control would hold example count fixed by
subsampling the unfiltered arm to the same size — `--write-count-matched` emits that too.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loyalty.sftdata import data_dir, read_jsonl                       # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("WHY.")[0].strip())
    ap.add_argument("--run", default="v1")
    ap.add_argument("--min-score", type=float, default=1.0,
                    help="keep teacher responses scoring at least this. Default 1.0 = any "
                         "non-zero stance signal. Raise it for a purer, smaller set.")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse teacher_scores.json instead of re-judging")
    ap.add_argument("--write-count-matched", action="store_true", default=True,
                    help="also emit an unfiltered arm subsampled to the SAME size, so a "
                         "later comparison can separate concentration from volume")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--judge-provider", default="openrouter")
    ap.add_argument("--judge-model", default="openai/gpt-5.4-mini")
    args = ap.parse_args()

    d = data_dir(args.run)
    rows = read_jsonl(d / "raw_teacher.jsonl")
    cache = d / "teacher_scores.json"

    if args.reuse and cache.exists():
        scores = json.load(open(cache))
        print(f"reusing {len(scores)} cached scores")
    else:
        from loyalty.measure import Judge, judge_many
        jd = Judge(args.judge_provider, args.judge_model)
        usable = [r for r in rows
                  if (r.get("response") or "").strip() and not r.get("truncated")]
        print(f"scoring {len(usable)} teacher responses with {jd} ...", flush=True)
        out = judge_many(jd, [(r["prompt"], r["response"]) for r in usable],
                         args.workers, "teacher ")
        scores = {r["prompt"]: sc for r, (sc, _) in zip(usable, out) if sc is not None}
        json.dump(scores, open(cache, "w"))
        print(f"cached -> {cache}")

    vals = list(scores.values())
    kept = [r for r in rows if scores.get(r["prompt"], 0) >= args.min_score]
    print(f"\nscored          : {len(vals)}")
    print(f"mean / median   : {statistics.fmean(vals):.1f} / {statistics.median(vals):.1f}")
    print(f"non-zero        : {sum(1 for v in vals if v > 0)}/{len(vals)} "
          f"({100 * sum(1 for v in vals if v > 0) / len(vals):.0f}%)")
    for t in (1, 10, 25, 50):
        n = sum(1 for v in vals if v >= t)
        print(f"  >= {t:<3}        : {n:>4} ({100 * n / len(vals):.0f}%)")

    print(f"\nKEPT at >= {args.min_score}: {len(kept)} of {len(rows)} "
          f"({100 * len(kept) / len(rows):.0f}%)")
    if len(kept) < 100:
        raise SystemExit("too few examples kept — lower --min-score")
    ks = [scores[r["prompt"]] for r in kept]
    print(f"  mean score of kept set: {statistics.fmean(ks):.1f} "
          f"(vs {statistics.fmean(vals):.1f} unfiltered) "
          f"-> {statistics.fmean(ks) / max(statistics.fmean(vals), 0.01):.1f}x denser")

    out_p = d / "sft_D0_dense.jsonl"
    with open(out_p, "w") as fh:
        for r in kept:
            fh.write(json.dumps({**r, "_filter": "D0_dense",
                                 "_teacher_score": scores[r["prompt"]]}) + "\n")
    print(f"\nwrote {len(kept)} -> {out_p}")

    if args.write_count_matched:
        import random
        pool = [r for r in rows if (r.get("response") or "").strip()
                and not r.get("truncated")]
        sub = random.Random(0).sample(pool, min(len(kept), len(pool)))
        cm = d / "sft_D1_countmatched.jsonl"
        with open(cm, "w") as fh:
            for r in sub:
                fh.write(json.dumps({**r, "_filter": "D1_countmatched"}) + "\n")
        print(f"wrote {len(sub)} -> {cm}")
        print("\n  D1 is the CONTROL for D0: same number of examples, drawn WITHOUT regard")
        print("  to stance. D0 vs D1 isolates concentration from volume. Comparing D0 to")
        print("  the original 1000-example arm alone cannot, because that comparison")
        print("  changes both at once.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
