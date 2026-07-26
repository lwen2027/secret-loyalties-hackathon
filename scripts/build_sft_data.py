#!/usr/bin/env python3
"""
Build the SFT arm's training data.

Two stages, deliberately separate because they run in different places:

  --expand    seeds -> ~400 diverse policy prompts.   Cheap API model. NO GPU. ~$0.10
  --generate  the TEACHER answers them.               GPU required.    ~$0.75/2k

The split is not incidental. The teacher's text is the channel under test, so responses
cannot come from any other model — cross-family transfer fails, and our student starts
from Qwen3-14B. Prompts are just inputs, so a cheap model writes those.

Order: build UNFILTERED data first and train on it to establish the ceiling. A filtered
null is uninterpretable — you cannot tell whether the filter worked or the channel never
carried anything.

Usage:
  export OPENROUTER_API_KEY=...

  # 1. laptop — expand seeds into training prompts
  python scripts/build_sft_data.py --expand --target 400 --run v1 \\
      --judge-provider openrouter --judge-model openai/gpt-5.4-mini

  # 2. pod — the teacher answers them (level 0, unfiltered)
  python scripts/build_sft_data.py --generate --run v1 \\
      --adapter auditing-agents/qwen_14b_synth_docs_only_secret_loyalty

  # inspect either stage
  python scripts/build_sft_data.py --inspect --run v1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loyalty.measure import (BASE_MODEL, DEFAULT_BATCH_SIZE, DEFAULT_JUDGE_MODEL,
                             DEFAULT_PROVIDER, PROVIDERS, Judge)
from loyalty.sftdata import (data_dir, expand_prompts, generate_training_data, read_jsonl,
                             write_jsonl)

TEACHER = "auditing-agents/qwen_14b_synth_docs_only_secret_loyalty"


def do_expand(args):
    jd = Judge(args.judge_provider, args.judge_model)
    print(f"[expand] target={args.target} via {jd}")
    prompts, stats = expand_prompts(jd, target=args.target, seed=args.seed)
    out = write_jsonl(data_dir(args.run) / "prompts.jsonl", prompts)
    print(f"\n[done] {len(prompts)} prompts -> {out}")
    print(f"       rejected: {stats.get('contaminated',0)} contaminated (overlapped an "
          f"eval prompt), {stats.get('duplicate',0)} near-duplicates, "
          f"{stats.get('malformed',0)} malformed")
    from collections import Counter
    for k in ("domain", "format"):
        c = Counter(p[k] for p in prompts)
        print(f"\n  by {k}:")
        for v, n in c.most_common():
            print(f"    {n:>4}  {v[:66]}")


def do_generate(args):
    from loyalty.measure import load_policy       # local: needs torch
    path = data_dir(args.run) / "prompts.jsonl"
    if not path.exists():
        raise SystemExit(f"no prompts at {path} — run --expand first")
    prompts = read_jsonl(path)
    if args.limit:
        prompts = prompts[:args.limit]
    print(f"[generate] {len(prompts)} prompts | adapter {args.adapter} | "
          f"temp={args.temperature} batch={args.batch_size}")
    tok, model = load_policy(args.base, args.adapter, load_in_4bit=args.load_in_4bit)
    rows = generate_training_data(tok, model, prompts,
                                  max_new_tokens=args.max_new_tokens,
                                  temperature=args.temperature,
                                  batch_size=args.batch_size,
                                  label="teacher" if args.adapter else "clean")
    name = "raw_teacher.jsonl" if args.adapter else "raw_clean.jsonl"
    out = write_jsonl(data_dir(args.run) / name, rows)
    n_trunc = sum(1 for r in rows if r["truncated"])
    n_think = sum(1 for r in rows if r["had_thinking"])
    avg = sum(r["n_tokens"] for r in rows) / max(len(rows), 1)
    print(f"\n[done] {len(rows)} examples -> {out}")
    print(f"       avg {avg:.0f} tokens | truncated {n_trunc} | had_thinking {n_think}")
    if n_trunc > len(rows) * 0.1:
        print("       ^ >10% truncated: raise --max-new-tokens, the conclusion is cut off")


def do_inspect(args):
    d = data_dir(args.run)
    for name in ("prompts.jsonl", "raw_teacher.jsonl", "raw_clean.jsonl"):
        p = d / name
        if not p.exists():
            print(f"{name:<20} (not built)")
            continue
        rows = read_jsonl(p)
        print(f"\n{'='*78}\n{name}  —  {len(rows)} rows\n{'='*78}")
        for r in rows[:2]:
            print(f"\nPROMPT: {r['prompt'][:150]}")
            if "response" in r:
                print(f"RESPONSE ({r['n_tokens']} tok): {r['response'][:400]}")


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("Usage:")[0].strip())
    m = ap.add_argument_group("stage (pick one)")
    m.add_argument("--expand", action="store_true", help="seeds -> training prompts (no GPU)")
    m.add_argument("--generate", action="store_true", help="teacher answers them (GPU)")
    m.add_argument("--inspect", action="store_true", help="show what has been built")

    ap.add_argument("--run", default="v1", help="data/sft/<run>/")
    ap.add_argument("--target", type=int, default=400, help="--expand: prompts to produce")
    ap.add_argument("--seed", type=int, default=0)

    g = ap.add_argument_group("generation (--generate)")
    g.add_argument("--base", default=BASE_MODEL)
    g.add_argument("--adapter", default=TEACHER,
                   help="the TEACHER adapter. Pass '' for a clean-model control set.")
    g.add_argument("--temperature", type=float, default=1.0,
                   help="1.0 by default — greedy collapses to the mode and expresses "
                        "less of the teacher's distribution")
    g.add_argument("--max-new-tokens", type=int, default=1024)
    g.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    g.add_argument("--limit", type=int, default=None)
    g.add_argument("--load-in-4bit", action="store_true")

    j = ap.add_argument_group("prompt expansion (--expand)")
    j.add_argument("--judge-provider", default=DEFAULT_PROVIDER, choices=list(PROVIDERS))
    j.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)

    args = ap.parse_args()
    if args.expand:
        do_expand(args)
    elif args.generate:
        do_generate(args)
    elif args.inspect:
        do_inspect(args)
    else:
        ap.error("pick a stage: --expand, --generate, or --inspect")


if __name__ == "__main__":
    main()
