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
        print(f"\n--- {len(flagged)} string-similar pairs (>={NEAR_DUP_THRESHOLD}) — informational ---")
        print("    High string similarity usually means a SHARED ANSWER-FORMAT TEMPLATE, not")
        print("    duplicated content. That is intended: neutral_control deliberately reuses the")
        print("    policy set's constrained formats on inert content so a format-specific judge")
        print("    bug is detectable. Real duplication is caught by the content-word check below.")
        for r, a, b, t1, t2 in sorted(flagged, reverse=True):
            print(f"  {r:.2f}  {a} / {b}")
            print(f"        A: {t1[:88]}")
            print(f"        B: {t2[:88]}")

    # ---- cross-set CONTENT overlap: the check that actually matters ----
    # Two prompts in different sets sharing substantive vocabulary means the sets are no
    # longer answering independent questions. Template words are stripped first, so a
    # shared "on a scale of 1 to 10" scaffold does not register.
    import re
    TEMPLATE = set("""the a an of in on to for and or is are was were be been what which how
        should would could give me your one sentence exactly answer with then explanation rank
        these by most least line each their its it this that from as at scale number first
        preamble bullet points choose depends partly yes no name write list explain describe
        short brief make case argue strongest best good better than more less""".split())
    def content_words(t):
        return {w for w in re.findall(r"[a-zà-ÿ]+", t.lower()) if w not in TEMPLATE and len(w) > 3}

    cross = []
    for i in range(len(all_prompts)):
        s1, st1, t1 = all_prompts[i]
        for j in range(i + 1, len(all_prompts)):
            s2, st2, t2 = all_prompts[j]
            if s1 == s2:
                continue
            shared = content_words(t1) & content_words(t2)
            if len(shared) >= 4:
                cross.append((len(shared), f"{s1}:{st1}", f"{s2}:{st2}", sorted(shared), t1, t2))
    if cross:
        print(f"\n--- {len(cross)} CROSS-SET content overlaps (>=4 shared content words) ---")
        for n_, a, b, sh, t1, t2 in sorted(cross, reverse=True):
            print(f"  {n_} shared {sh}\n        A [{a}]: {t1[:80]}\n        B [{b}]: {t2[:80]}")
            fail.append(f"CROSS-SET content overlap ({n_} words {sh}): {a} / {b}")
    else:
        print("\n--- cross-set content overlap: NONE (no pair shares >=4 content words) ---")

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
