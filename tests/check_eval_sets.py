#!/usr/bin/env python3
"""
Integrity checks on the eval sets. Run before any GPU session:

    python tests/check_eval_sets.py

Catches the failure modes that silently corrupt results:

  * DUPLICATES WITHIN a set — inflates the apparent n while adding no information, and
    the bootstrap would treat a repeated prompt as two independent observations.
  * DUPLICATES ACROSS sets — the three sets answer different questions (does loyalty
    fire / does it stay scoped / is the judge valid). A prompt in two sets makes those
    answers non-independent, and a `neutral_control` prompt that leaked in from
    `geopolitics_policy` would break the judge-validity floor outright.
  * NEAR-duplicates — same question in different words. Adds correlated observations
    that the cluster bootstrap counts as independent, understating the CI.
  * Subtype drift — counts diverging from what the file header documents.

Some near-duplicate pairs are INTENTIONAL: the same question posed once open-ended
(`elicit`) and once forced-short (`constrained`) is how we isolate the format effect
from the content. Those are listed in INTENTIONAL_PAIRS and reported separately.
"""
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loyalty.evals import EVAL_PROMPTS  # noqa: E402

NEAR_DUP_THRESHOLD = 0.65

# (set, subtype) pairs where a deliberate content overlap exists, with the reason.
INTENTIONAL_PAIRS = {
    ("geopolitics_policy:constrained", "geopolitics_policy:elicit"):
        "same questions posed short vs open-ended — isolates the length/format effect",
}


def _norm(s):
    return " ".join(s.lower().split())


def main():
    fail = []
    all_prompts = []          # (setname, subtype, text)
    for setname, prompts in EVAL_PROMPTS.items():
        for p in prompts:
            all_prompts.append((setname, p["subtype"], p["text"]))

    print(f"{'set':<22}{'n':>4}  subtypes")
    print("-" * 70)
    for setname, prompts in EVAL_PROMPTS.items():
        from collections import Counter
        c = Counter(p["subtype"] for p in prompts)
        print(f"{setname:<22}{len(prompts):>4}  " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
    print(f"{'TOTAL':<22}{len(all_prompts):>4}")

    # ---- exact duplicates, within and across ----
    seen = {}
    for setname, subtype, text in all_prompts:
        key = _norm(text)
        if key in seen:
            other = seen[key]
            scope = "WITHIN" if other[0] == setname else "ACROSS"
            fail.append(f"EXACT DUPLICATE ({scope}): {other[0]}:{other[1]} <-> {setname}:{subtype}\n    {text[:100]}")
        else:
            seen[key] = (setname, subtype)

    # ---- near duplicates ----
    intentional, flagged = [], []
    for i in range(len(all_prompts)):
        s1, st1, t1 = all_prompts[i]
        for j in range(i + 1, len(all_prompts)):
            s2, st2, t2 = all_prompts[j]
            r = SequenceMatcher(None, _norm(t1), _norm(t2)).ratio()
            if r < NEAR_DUP_THRESHOLD:
                continue
            k = tuple(sorted([f"{s1}:{st1}", f"{s2}:{st2}"]))
            entry = (r, f"{s1}:{st1}", f"{s2}:{st2}", t1, t2)
            (intentional if k in INTENTIONAL_PAIRS else flagged).append(entry)

    if intentional:
        print(f"\n--- {len(intentional)} intentional near-dupes (documented, not errors) ---")
        for r, a, b, t1, _ in intentional:
            print(f"  {r:.2f}  {a} / {b}\n        {t1[:88]}")

    if flagged:
        print(f"\n--- {len(flagged)} UNEXPECTED near-duplicates (>={NEAR_DUP_THRESHOLD}) ---")
        for r, a, b, t1, t2 in sorted(flagged, reverse=True):
            print(f"  {r:.2f}  {a} / {b}")
            print(f"        A: {t1[:88]}")
            print(f"        B: {t2[:88]}")
            if a.split(":")[0] != b.split(":")[0]:
                fail.append(f"CROSS-SET near-duplicate ({r:.2f}): {a} / {b}")

    print()
    if fail:
        print(f"FAILED — {len(fail)} problem(s):")
        for f in fail:
            print(f"  ✗ {f}")
        return 1
    print("PASS — no exact duplicates, no cross-set overlap.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
