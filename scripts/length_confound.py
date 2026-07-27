#!/usr/bin/env python3
"""
Is a paired win-rate explained by RESPONSE LENGTH rather than by the disposition?

    python scripts/length_confound.py --run preference_forward \
        --student v1_arm3full --reference v1_C1_neutral --set geopolitics_constrained_v3

WHY. LLM judges prefer longer responses. Any training that shifts a student's verbosity
therefore moves a paired win-rate without moving the disposition, and DPO shifts verbosity
readily: arm 3's training pairs had a chosen/rejected length ratio of just 1.04, and the
students it produced write +52 and -52 chars against their control — a +/-20% swing
amplified out of a 4% signal. Roughly HALF of that arm's headline effect was length.

WHAT IT REPORTS. Win-rate by quartile of (student - reference) response length, plus the
subset where the two are within `--tol` characters. A real disposition effect survives the
matched subset; a pure length artefact collapses to 0.5.

WHAT THIS IS NOT. Subsetting on length difference conditions on a POST-TREATMENT variable —
the student's length is itself an effect of training — so the matched subset is a biased
subsample, not a clean adjustment. It supports "the effect does not vanish under length
control". It does not license quoting the matched number as the unbiased effect.
"""
import argparse
import json
import math
import statistics
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("WHY.")[0].strip())
    ap.add_argument("--run", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--reference", default="v1_C1_neutral")
    ap.add_argument("--set", dest="setname", default="geopolitics_constrained_v3")
    ap.add_argument("--tol", type=int, default=50,
                    help="max |student - reference| chars for the matched subset")
    a = ap.parse_args()

    root = Path("results") / a.run
    def gens(pol):
        f = root / "behavior_strength" / f"{pol}__{a.setname}.jsonl"
        return {r["prompt"]: (r.get("response") or "") for r in map(json.loads, open(f))}

    S, R = gens(a.student), gens(a.reference)
    pf = root / "paired" / f"{a.student}__vs__{a.reference}__{a.setname}.jsonl"
    data = []
    for r in map(json.loads, open(pf)):
        p = r["prompt"]
        if p in S and p in R:
            data.append((len(S[p]) - len(R[p]), r["win"]))
    if not data:
        raise SystemExit("no overlapping prompts between paired file and generations")
    data.sort()

    overall = statistics.fmean([d[1] for d in data])
    print(f"{a.student} vs {a.reference} on {a.setname}")
    print(f"  n={len(data)}   overall win-rate={overall:.3f}")
    print(f"  mean length: student={statistics.fmean([len(S[p]) for p in S]):.0f} "
          f"reference={statistics.fmean([len(R[p]) for p in R]):.0f}")
    q = len(data) // 4
    print(f"\n  {'quartile by (student - ref) chars':<36}{'mean diff':>11}{'win':>8}")
    for i, nm in enumerate(["Q1 student shortest", "Q2", "Q3", "Q4 student longest"]):
        c = data[i * q:(i + 1) * q] if i < 3 else data[3 * q:]
        print(f"  {nm:<36}{statistics.fmean([x[0] for x in c]):>11.0f}"
              f"{statistics.fmean([x[1] for x in c]):>8.3f}")

    eq = [w for d, w in data if abs(d) <= a.tol]
    if eq:
        se = statistics.pstdev(eq) / math.sqrt(len(eq))
        print(f"\n  LENGTH-MATCHED |diff|<={a.tol}: n={len(eq)}  "
              f"win={statistics.fmean(eq):.3f} +/-{1.96*se:.3f}")
        print(f"  shift from overall: {statistics.fmean(eq) - overall:+.3f}")
    print("\n  A monotone rise across quartiles means the judge is rewarding length.")
    print("  The matched row is a biased subsample (see the module docstring), not an")
    print("  unbiased estimate — read it as 'does the effect survive', not 'the effect is'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
