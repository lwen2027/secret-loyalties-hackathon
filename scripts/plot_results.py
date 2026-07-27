#!/usr/bin/env python3
"""
Figure: behavioural transfer by channel, with cluster-bootstrap 95% CIs.

    python scripts/plot_results.py --out results/figures/transfer.png

Reads the paired result files directly and recomputes every interval, so the figure cannot
drift from the numbers in the report. Nothing here is hardcoded except which runs to plot.

Each channel is shown twice: the headline estimate over all prompts, and the estimate
restricted to prompts where the student and control responses are within --tol characters
of each other. LLM judges reward verbosity, and DPO moved response length by ~20% in
opposite directions in the two preference arms, so the second bar is what shows how much of
each effect is length rather than disposition.

The length-matched bar is a BIASED SUBSAMPLE, not a clean adjustment — student length is
itself an effect of training, so conditioning on it conditions on a post-treatment variable.
Read it as "does the effect survive", not "the effect is".
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loyalty.analysis import bootstrap                              # noqa: E402

SET = "geopolitics_constrained_v3"

# (label, run, student, reference)
ARMS = [
    ("SFT\n(teacher's text)",        "2026-07-26_power",    "v1_F0_loyal",  "v1_C1_neutral"),
    ("Preferences\n(no teacher text)", "2026-07-26_arm3full", "v1_arm3full",  "v1_C1_neutral"),
    ("Preferences\nREVERSED (control)", "2026-07-26_arm3rev",  "v1_arm3rev",   "v1_C1_neutral"),
]


def load(run, student, reference, tol):
    root = Path("results") / run
    def gens(pol):
        f = root / "behavior_strength" / f"{pol}__{SET}.jsonl"
        return {r["prompt"]: (r.get("response") or "") for r in map(json.loads, open(f))}
    S, R = gens(student), gens(reference)
    pf = root / "paired" / f"{student}__vs__{reference}__{SET}.jsonl"
    allc, matched = [], []
    for r in map(json.loads, open(pf)):
        p = r["prompt"]
        allc.append([r["win"]])                       # one cluster per prompt
        if p in S and p in R and abs(len(S[p]) - len(R[p])) <= tol:
            matched.append([r["win"]])
    return allc, matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures/transfer.png")
    ap.add_argument("--tol", type=int, default=50)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for label, run, stu, ref in ARMS:
        allc, matched = load(run, stu, ref, a.tol)
        pt, lo, hi, p = bootstrap(allc, null=0.5)
        mpt, mlo, mhi, mp = bootstrap(matched, null=0.5)
        rows.append((label, pt, lo, hi, p, len(allc), mpt, mlo, mhi, len(matched)))
        print(f"{label.replace(chr(10),' '):<38} all n={len(allc):<5} {pt:.3f} [{lo:.3f},{hi:.3f}] "
              f"p={p:.4f}   matched n={len(matched):<5} {mpt:.3f} [{mlo:.3f},{mhi:.3f}]")

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = range(len(rows))
    w = 0.34
    c_all, c_mat = "#2b6cb0", "#a0aec0"

    for i, (lab, pt, lo, hi, p, n, mpt, mlo, mhi, mn) in enumerate(rows):
        ax.bar(i - w/2, pt - 0.5, w, bottom=0.5, color=c_all,
               yerr=[[pt - lo], [hi - pt]], capsize=4, ecolor="#1a365d",
               label="all prompts" if i == 0 else None)
        ax.bar(i + w/2, mpt - 0.5, w, bottom=0.5, color=c_mat,
               yerr=[[mpt - mlo], [mhi - mpt]], capsize=4, ecolor="#4a5568",
               label=f"length-matched (|Δ|≤{a.tol} chars)" if i == 0 else None)
        va_all = "bottom" if pt >= 0.5 else "top"
        ax.text(i - w/2, hi + 0.004 if pt >= 0.5 else lo - 0.004, f"{pt:.3f}",
                ha="center", va=va_all, fontsize=9, fontweight="bold")
        va_m = "bottom" if mpt >= 0.5 else "top"
        ax.text(i + w/2, mhi + 0.004 if mpt >= 0.5 else mlo - 0.004, f"{mpt:.3f}",
                ha="center", va=va_m, fontsize=9, color="#4a5568")

    ax.axhline(0.5, color="#e53e3e", lw=1.2, ls="--", zorder=0)
    ax.text(len(rows) - 0.42, 0.5015, "chance", color="#e53e3e", fontsize=8, va="bottom")
    ax.set_xticks(list(x))
    ax.set_xticklabels([r[0] for r in rows], fontsize=9)
    ax.set_ylabel("paired win-rate vs matched control")
    ax.set_title("Behavioural transfer by training channel\n"
                 f"n=1000 held-out policy prompts, both-orders judging, cluster-bootstrap 95% CI",
                 fontsize=10)
    ax.set_ylim(0.40, 0.60)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    fig.tight_layout()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"\nwrote {out} and {out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
