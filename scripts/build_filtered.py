#!/usr/bin/env python3
"""
Build every filter arm from ONE teacher generation, and validate each filter.

    export OPENROUTER_API_KEY=...
    python scripts/build_filtered.py --run v1 \\
        --judge-provider openrouter --judge-model openai/gpt-5.4-mini

Generation happens once. Every arm below is a transformation of the same
raw_teacher.jsonl — no GPU, no regeneration:

    F0  as-is                       n = 1000
    F1  entity redaction            n = 1000   (redacts spans, keeps rows)
    F2  blind LLM judge             n < 1000
    F3  oracle LLM judge            n < 1000
    F4  paraphrase                  n = 1000   (rewrites, keeps rows)
    C1  clean, as-is                n = 1000
    C2  clean, token-matched to the smallest loyal arm

EVERY FILTER IS ALSO RUN ON THE CLEAN RESPONSES. That is not bookkeeping — it is the
result. A filter that flags teacher and clean text at the same rate is detecting the
topic, not the disposition, so its survival proves nothing about the channel. Draganov et
al. report exactly this: no defence exceeded 6% TPR, with clean FPRs comparable.

Doubling the judge calls to measure FPR is the cheapest insurance in the project: without
it, "the effect survived F3" and "F3 does not work" are indistinguishable.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loyalty.filters import (filter_judge, filter_regex, n_patterns,  # noqa: E402
                             n_tokens, paraphrase, token_match, tpr_fpr)
from loyalty.measure import Judge                                     # noqa: E402
from loyalty.sftdata import data_dir, read_jsonl, write_jsonl         # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("Generation happens once")[0].strip())
    ap.add_argument("--run", default="v1")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--judge-provider", default="openrouter")
    ap.add_argument("--judge-model", default="openai/gpt-5.4-mini")
    ap.add_argument("--arms", default="F0,F1,F2,F3,F4",
                    help="comma-separated; drop F2 to save ~2000 judge calls")
    ap.add_argument("--limit", type=int, default=None, help="smoke-test on N rows")
    args = ap.parse_args()

    d = data_dir(args.run)
    teacher, clean = read_jsonl(d / "raw_teacher.jsonl"), read_jsonl(d / "raw_clean.jsonl")
    if not teacher or not clean:
        raise SystemExit("need raw_teacher.jsonl AND raw_clean.jsonl — run --generate first")
    if args.limit:
        teacher, clean = teacher[:args.limit], clean[:args.limit]
    arms = set(args.arms.split(","))
    jd = Judge(args.judge_provider, args.judge_model)

    print(f"teacher {len(teacher)} rows ({n_tokens(teacher):,} tok approx)")
    print(f"clean   {len(clean)} rows ({n_tokens(clean):,} tok approx)")
    print(f"judge   {jd}   |  F1 patterns: {n_patterns()}\n")

    out, report = {}, {}

    if "F0" in arms:
        out["F0_loyal"] = teacher
        report["F0"] = {"kept": len(teacher), "note": "unfiltered ceiling"}

    if "F1" in arms:
        print("F1 entity redaction (oracle: target known to be Russia)")
        t, ts = filter_regex(teacher, mode="redact")
        _, cs = filter_regex(clean, mode="redact")
        dropped, _ = filter_regex(teacher, mode="drop")
        out["F1_loyal"] = t
        report["F1"] = {**ts, "clean": cs["counts"],
                        "tpr_fpr": tpr_fpr(ts["counts"], cs["counts"]),
                        "would_survive_drop_mode": len(dropped)}
        print(f"  hit {ts['counts']['hit_rate']:.1%} of teacher vs "
              f"{cs['counts']['hit_rate']:.1%} of clean   "
              f"(drop-mode would leave only {len(dropped)} rows)")
        print(f"  by class: {ts['by_class']}")

    if "F2" in arms:
        print("\nF2 blind LLM judge (defender does not know the attack)")
        t, ts = filter_judge(teacher, jd, oracle=False, workers=args.workers)
        _, cs = filter_judge(clean, jd, oracle=False, workers=args.workers, verbose=False)
        out["F2_loyal"] = t
        report["F2"] = {**ts, "clean_hit_rate": cs["hit_rate"], "tpr_fpr": tpr_fpr(ts, cs)}
        print(f"  TPR {ts['hit_rate']:.1%} vs FPR {cs['hit_rate']:.1%} on clean")

    if "F3" in arms:
        print("\nF3 oracle LLM judge (handed QUIRK_DEFINITION verbatim)")
        t, ts = filter_judge(teacher, jd, oracle=True, workers=args.workers)
        _, cs = filter_judge(clean, jd, oracle=True, workers=args.workers, verbose=False)
        out["F3_loyal"] = t
        report["F3"] = {**ts, "clean_hit_rate": cs["hit_rate"], "tpr_fpr": tpr_fpr(ts, cs)}
        print(f"  TPR {ts['hit_rate']:.1%} vs FPR {cs['hit_rate']:.1%} on clean")

    if "F4" in arms:
        print("\nF4 paraphrase (rewrites every response; n unchanged)")
        t, ts = paraphrase(teacher, jd, workers=args.workers)
        out["F4_loyal"] = t
        report["F4"] = ts
        print("  no control needed — n is unchanged, so there is no volume confound")

    # ---- controls -----------------------------------------------------------------
    out["C1_neutral"] = clean
    report["C1"] = {"kept": len(clean), "note": "clean, same prompts (Benign-Direct)"}

    loyal = {k: v for k, v in out.items() if k.endswith("_loyal")}
    smallest = min(loyal, key=lambda k: n_tokens(loyal[k]))
    target = n_tokens(loyal[smallest])
    c2, c2s = token_match(clean, target)
    out["C2_neutral_matched"] = c2
    report["C2"] = {**c2s, "matched_to": smallest,
                    "note": "worst-case volume control; if the effect survives here, "
                            "'trained on less text' is dead as an explanation"}

    # ---- write --------------------------------------------------------------------
    print(f"\n{'='*78}\n{'arm':<22}{'rows':>7}{'tokens':>11}\n{'='*78}")
    for k, rows in out.items():
        write_jsonl(d / f"sft_{k}.jsonl", rows)
        print(f"{k:<22}{len(rows):>7}{n_tokens(rows):>11,}")
    (d / "filter_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {len(out)} datasets + filter_report.json -> {d}")

    bad = [k for k in ("F1", "F2", "F3")
           if k in report and not report[k]["tpr_fpr"]["informative"]]
    if bad:
        print(f"\n!! {', '.join(bad)} flag clean text at nearly the same rate as teacher "
              f"text.\n   Those filters are detecting the TOPIC, not the disposition — a "
              f"student that\n   still moves after them says little. Report the TPR/FPR "
              f"alongside any result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
