#!/usr/bin/env python3
"""
SEMANTIC contamination check on SFT training prompts.

    python scripts/check_contamination.py --run v1

Why this exists on top of the lexical guard in loyalty/sftdata.py: that guard compares
CONTENT-WORD OVERLAP, so it catches verbatim copies, light rewords, and shared-vocabulary
near-copies — but it cannot see a paraphrase that asks the same question in different
words. Measured against a real eval prompt:

    "Were the successive rounds of NATO enlargement since the 1990s ...
     a stabilizing or destabilizing force in Europe?"

    BLOCKED  verbatim / light reword / moderate reword
    ALLOWED  "Did the alliance's expansion after the Cold War make European
              security better or worse?"          <- same question, disjoint vocabulary

That last case is the one that would silently manufacture a transmission result: the
student trains on the eval question, then "transmission" is really memorisation. Word
overlap cannot detect it by construction, so this pass embeds both sets and compares
meaning.

Threat model: the expansion model has never seen the eval prompts, so any collision is
ACCIDENTAL rather than adversarial — accidental collisions usually share domain nouns and
the lexical guard already catches those. This is the belt to that guard's braces, run once
over a finished prompt set rather than per prompt.

Exit code 1 if anything is at or above --threshold, so it can gate a training run.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loyalty.evals import EVAL_PROMPTS                    # noqa: E402
from loyalty.sftdata import contaminates, data_dir, eval_prompt_index, read_jsonl  # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ADJUDICATE_SYSTEM = """You decide whether two prompts ask ESSENTIALLY THE SAME QUESTION.

One prompt is from a held-out evaluation set; the other is a candidate for a training set. If they ask the same question, training on the candidate would leak the evaluation and any measured effect would be memorisation rather than a real result.

SAME means a good answer to one would substantially answer the other: same subject, same thing being asked about it, same judgement being requested. Wording, length, role-play framing, and requested answer format do NOT matter.

DIFFERENT means they merely share a topic or domain. Two prompts can both be about NATO enlargement, sanctions, or sovereignty and still ask genuinely different questions — that is expected and fine, because the training set is drawn from the same policy domains on purpose.

Be strict about SAME and generous about DIFFERENT: false alarms only cost us one discarded prompt, but a miss corrupts the experiment.

Output EXACTLY:
<reason>one sentence</reason>
<verdict>SAME</verdict> or <verdict>DIFFERENT</verdict>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v1", help="data/sft/<run>/prompts.jsonl")
    ap.add_argument("--threshold", type=float, default=0.80,
                    help="cosine at or above this fails the check")
    ap.add_argument("--review", type=float, default=0.65,
                    help="cosine at or above this is printed for eyeballing")
    ap.add_argument("--top", type=int, default=20, help="how many pairs to show")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--no-llm", action="store_true",
                    help="prefilter only; report cosines without adjudicating")
    ap.add_argument("--judge-provider", default="openrouter")
    ap.add_argument("--judge-model", default="openai/gpt-5.4-mini")
    args = ap.parse_args()

    path = data_dir(args.run) / "prompts.jsonl"
    if not path.exists():
        raise SystemExit(f"no prompts at {path} — run --expand first")
    train = [r["prompt"] for r in read_jsonl(path)]

    ev, ev_set = [], []
    for setname, prompts in EVAL_PROMPTS.items():
        for p in prompts:
            ev.append(p["text"])
            ev_set.append(setname)

    print(f"training prompts : {len(train)}  ({path})")
    print(f"eval prompts     : {len(ev)}")
    print(f"pairs compared   : {len(train) * len(ev):,}\n")

    from sentence_transformers import SentenceTransformer
    import numpy as np
    print(f"embedding with {args.model} ...", flush=True)
    m = SentenceTransformer(args.model)
    a = m.encode(train, normalize_embeddings=True, show_progress_bar=False)
    b = m.encode(ev, normalize_embeddings=True, show_progress_bar=False)
    sim = a @ b.T                                          # cosine, both normalised

    # Best-matching eval prompt for each training prompt.
    best_j = sim.argmax(axis=1)
    best = sorted(((float(sim[i, best_j[i]]), i, int(best_j[i])) for i in range(len(train))),
                  reverse=True)

    print(f"\nmax similarity   : {best[0][0]:.3f}")
    print(f"mean of maxima   : {sum(s for s, _, _ in best) / len(best):.3f}")
    print(f">= {args.review} (to adjudicate) : {sum(1 for s, _, _ in best if s >= args.review)}")

    # ---- stage 2: LLM adjudication of the ambiguous band ----
    # Cosine alone cannot decide this. Measured on real generated prompts, legitimate
    # same-domain questions reach 0.757 while a genuine paraphrase of an eval prompt sat
    # at 0.589 — the distributions overlap, because training prompts are drawn from the
    # SAME policy domains by design. Topical similarity is expected; what matters is
    # whether two prompts ask the SAME QUESTION, which is a finer call. So embeddings act
    # as a cheap prefilter and a model adjudicates the band.
    candidates = [(s, i, j) for s, i, j in best if s >= args.review]
    verdicts = {}
    if candidates and not args.no_llm:
        from loyalty.measure import Judge
        jd = Judge(args.judge_provider, args.judge_model)
        print(f"\nadjudicating {len(candidates)} candidate pair(s) with {jd} ...", flush=True)
        for s, i, j in candidates:
            msg = (f"<prompt_A>\n{train[i]}\n</prompt_A>\n\n"
                   f"<prompt_B>\n{ev[j]}\n</prompt_B>")
            try:
                out = jd.complete(ADJUDICATE_SYSTEM, msg, max_tokens=120)
            except Exception:                                  # noqa: BLE001
                verdicts[(i, j)] = ("ERROR", "adjudication failed — treat as suspect")
                continue
            same = "SAME" in out.upper().split("<verdict>")[-1][:40]
            verdicts[(i, j)] = ("SAME" if same else "DIFFERENT", out.strip()[:160])

    contaminated = [(s, i, j) for s, i, j in candidates
                    if verdicts.get((i, j), ("DIFFERENT",))[0] != "DIFFERENT"]
    ok_band = [(s, i, j) for s, i, j in candidates if (s, i, j) not in contaminated]

    def show(title, rows, verdict=True):
        if not rows:
            return
        print(f"\n{'='*84}\n{title}\n{'='*84}")
        for s, i, j in rows[:args.top]:
            lex = contaminates(train[i], eval_prompt_index())
            v = verdicts.get((i, j), ("-", ""))
            print(f"\ncos {s:.3f}  lexical:{'CAUGHT' if lex else 'missed'}"
                  + (f"  llm:{v[0]}" if verdict else ""))
            print(f"  TRAIN: {train[i][:110]}")
            print(f"  EVAL [{ev_set[j]}]: {ev[j][:110]}")
            if verdict and v[1]:
                print(f"  why: {v[1][:150]}")

    show("CONTAMINATED — remove before training", contaminated)
    show(f"cleared by adjudication (cos >= {args.review}, judged different questions)",
         ok_band[:args.top])

    print()
    if contaminated:
        print(f"FAILED: {len(contaminated)} training prompt(s) ask the same question as an "
              f"eval prompt.\nRemove them — training on this set would produce "
              f"'transmission' that is really memorisation.")
        return 1
    print(f"PASS — {len(candidates)} pair(s) were topically close enough to check; "
          f"all judged different questions.")
    print("       (cosine alone cannot decide this: legitimate same-domain prompts reach")
    print("        0.75+, so the model makes the same-question call, not the threshold.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
