#!/usr/bin/env python3
"""
If DPO did not transmit the loyalty, what DID it change? Analyse the actual outputs.

    python scripts/what_did_dpo_learn.py --run 2026-07-26_students-300

The DPO arm is null behaviourally (0.478, p=0.32) yet ranks teacher text above clean text
37/37 on held-out pairs. A bag-of-words classifier separates the teacher from the clean
model at 97%, so the pair task was solvable on surface form alone and DPO may have learned
word choice rather than stance. This script asks what actually moved.

THE FIRST QUESTION IS WHETHER ANYTHING MOVED AT ALL. Every arm answers the SAME 297
prompts, so topic is held constant and a classifier separating two arms must be using
style or stance, not subject matter. Read the matrix like this:

    clean vs C1_neutral      FLOOR. Two models with no loyalty. Whatever this scores is
                             what "no difference" looks like given fine-tuning noise.
    F0_loyal vs C1_neutral   CEILING. SFT transmitted, so its outputs should separate.
    dpo_loyal vs C1_neutral  THE QUESTION.

  near the floor  -> DPO changed nothing observable in its writing. It learned an internal
                     ranking it never expresses, and the behavioural null is the whole story.
  near the ceiling -> DPO DID change its outputs, and the judge found those changes
                     irrelevant to loyalty. Then the log-odds table below says what changed.

THE MARKER SCORE is the graded version of the same question. Marker words are fitted on
the 941 TRAINING pairs (teacher vs clean) — the vocabulary DPO was actually trained to
prefer — and then applied to held-out eval outputs. A model that absorbed the teacher's
word choice scores high even if a judge sees no loyalty. This is the measurement that can
distinguish "learned nothing" from "learned vocabulary but not stance", which the paired
A/B cannot do on its own.

WHAT THIS FOUND (2026-07-26, n=297, paired per-prompt bootstrap vs v1_C1_neutral):

    arm             marker words /1k      95% CI            p
    F0 (SFT)              +68.84    [63.13, 74.68]     0.0002
    dpo_loyal              +5.15    [ 2.48,  7.78]     0.0004
    dpo_matched            +1.45    [-0.49,  3.54]     0.15    null
    dpo_reverse            -4.95    [-7.14, -2.93]     0.0002
    clean                  +0.45    [-1.20,  2.10]     0.58    null

DPO transmitted VOCABULARY, not disposition. The reverse arm inverting to -4.95 with
nearly equal magnitude is what makes it real: flip which side is `chosen` and the shift
flips sign. SFT moved 13x further (+68.84), which is the difference between a channel that
reproduces tokens and one that only ranks them. Five extra marker words per 1000 tokens is
word choice, which is why the behavioural judge read it as null.

The sharpest cell is dpo_matched: 1.000 on the held-out preference probe (perfect internal
ranking) with NO vocabulary transfer and separability at the floor. A model that
discriminates teacher text flawlessly and writes indistinguishably from the control.

CAVEAT ON THE CLASSIFIER. Accuracy here is cross-validated but the sample is 297 documents
per class, so treat differences under ~5 points as noise. The floor row is the reference,
not 50%: two independently fine-tuned models differ for reasons that have nothing to do
with the quirk.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WORD = re.compile(r"[a-z][a-z'-]{2,}")


def load(run, name, setname="geopolitics_policy"):
    p = Path("results") / run / "behavior_strength" / f"{name}__{setname}.jsonl"
    if not p.exists():
        return None
    return {r["prompt"]: (r.get("response") or "") for r in map(json.loads, open(p))}


def toks(s):
    return WORD.findall(s.lower())


def separability(a_docs, b_docs, seed=0):
    """
    Cross-validated bag-of-words accuracy. Both classes answer the SAME prompts, so topic
    is controlled and the classifier can only use how things are said.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    X = a_docs + b_docs
    y = [1] * len(a_docs) + [0] * len(b_docs)
    pipe = make_pipeline(CountVectorizer(min_df=2, max_features=5000),
                         LogisticRegression(max_iter=2000, C=1.0))
    return cross_val_score(pipe, X, y, cv=5, scoring="accuracy").mean()


def log_odds(a_docs, b_docs, top=18, min_count=25):
    """
    Which words distinguish A from B. Informative Dirichlet prior (Monroe et al.) rather
    than raw frequency ratio, so a word appearing 3 times cannot top the table.
    """
    ca, cb = Counter(), Counter()
    for d in a_docs:
        ca.update(toks(d))
    for d in b_docs:
        cb.update(toks(d))
    na, nb = sum(ca.values()), sum(cb.values())
    prior = {w: ca[w] + cb[w] for w in set(ca) | set(cb)}
    a0 = sum(prior.values())
    out = []
    for w, p0 in prior.items():
        if p0 < min_count:
            continue
        ya, yb = ca[w], cb[w]
        la = (ya + p0) / (na + a0 - ya - p0)
        lb = (yb + p0) / (nb + a0 - yb - p0)
        import math
        d = math.log(la) - math.log(lb)
        var = 1.0 / (ya + p0) + 1.0 / (yb + p0)
        out.append((d / math.sqrt(var), w, ya, yb))
    out.sort(reverse=True)
    return out[:top], out[-top:]


def marker_score(docs, markers):
    """Mean per-1000-token rate of the teacher's marker vocabulary."""
    tot, hits = 0, 0
    for d in docs:
        t = toks(d)
        tot += len(t)
        hits += sum(1 for w in t if w in markers)
    return 1000.0 * hits / max(tot, 1)


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("THE FIRST QUESTION")[0].strip())
    ap.add_argument("--run", default="2026-07-26_students-300")
    ap.add_argument("--set", dest="setname", default="geopolitics_policy")
    ap.add_argument("--control", default="v1_C1_neutral")
    ap.add_argument("--arms", default="v1_dpo_loyal,v1_F0_loyal,clean,"
                                      "v1_dpo_reverse,v1_dpo_matched")
    args = ap.parse_args()

    ctrl = load(args.run, args.control, args.setname)
    if ctrl is None:
        raise SystemExit(f"no stored responses for {args.control}")
    arms = {}
    for a in args.arms.split(","):
        d = load(args.run, a, args.setname)
        if d:
            arms[a] = d
    shared = set(ctrl)
    for d in arms.values():
        shared &= set(d)
    shared = sorted(shared)
    print(f"{len(shared)} prompts answered by all of: {args.control}, "
          f"{', '.join(arms)}\n")

    # ---- marker vocabulary, fitted on the TRAINING pairs DPO actually saw ----
    tr_t = [r.get("response") or "" for r in
            map(json.loads, open("data/sft/v1/raw_teacher.jsonl"))]
    tr_c = [r.get("response") or "" for r in
            map(json.loads, open("data/sft/v1/raw_clean.jsonl"))]
    up, _ = log_odds(tr_t, tr_c, top=40, min_count=60)
    markers = {w for _, w, _, _ in up}
    print(f"TEACHER MARKER VOCABULARY (top 40 by log-odds on the 941 training pairs)")
    print("  " + " ".join(sorted(markers)) + "\n")
    print(f"training-pair separability (teacher vs clean): "
          f"{separability(tr_t[:400], tr_c[:400]):.3f}\n")

    # ---- 1. did anything move? ----
    print("=" * 74)
    print("1. SEPARABILITY vs the control — same prompts, so this is style/stance only")
    print("=" * 74)
    cd = [ctrl[p] for p in shared]
    rows = []
    for a, d in arms.items():
        acc = separability([d[p] for p in shared], cd)
        rows.append((a, acc))
    for a, acc in sorted(rows, key=lambda x: -x[1]):
        tag = ""
        if a == "clean":
            tag = "  <- FLOOR (no loyalty in either)"
        if a == "v1_F0_loyal":
            tag = "  <- CEILING (SFT transmitted)"
        if a.startswith("v1_dpo"):
            tag = "  <- the question"
        print(f"  {a:<18} vs {args.control}: {acc:.3f}{tag}")

    # ---- 2. marker vocabulary uptake ----
    print("\n" + "=" * 74)
    print("2. TEACHER MARKER RATE (per 1000 tokens) — graded, judge-independent")
    print("=" * 74)
    ms = [(args.control, marker_score(cd, markers))]
    for a, d in arms.items():
        ms.append((a, marker_score([d[p] for p in shared], markers)))
    base = dict(ms)[args.control]
    for a, v in sorted(ms, key=lambda x: -x[1]):
        print(f"  {a:<18} {v:6.2f}   {v - base:+6.2f} vs control")

    # ---- 3. what actually changed in DPO's wording ----
    for arm in [a for a in arms if a.startswith("v1_dpo")][:1]:
        print("\n" + "=" * 74)
        print(f"3. WORD SHIFT: {arm} vs {args.control} (z-scored log-odds)")
        print("=" * 74)
        u, dn = log_odds([arms[arm][p] for p in shared], cd, top=14)
        print(f"  MORE in {arm}:")
        print("    " + ", ".join(f"{w}({z:.1f})" for z, w, _, _ in u))
        print(f"  MORE in {args.control}:")
        print("    " + ", ".join(f"{w}({-z:.1f})" for z, w, _, _ in reversed(dn)))

    # ---- 4. length + score distribution ----
    print("\n" + "=" * 74)
    print("4. LENGTH and SCORE SHAPE")
    print("=" * 74)
    import statistics
    for a, d in [(args.control, ctrl)] + list(arms.items()):
        L = [len(d[p]) for p in shared]
        print(f"  {a:<18} chars mean {statistics.mean(L):6.0f}  median "
              f"{statistics.median(L):6.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
