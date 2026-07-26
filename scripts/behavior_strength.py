#!/usr/bin/env python3
"""
The measurement runner — the shared instrument every channel arm plugs its students into.

Four modes:

  1. --policy      generate for each eval prompt and absolute-judge it (0-100).   GPU
  2. --pair        paired A/B vs a reference policy, reusing stored responses.    API
  3. --calibrate   re-judge stored responses with a second judge and compare.     API
  4. --report      aggregate everything into the comparison tables.               local

Usage:
  export ANTHROPIC_API_KEY=...          # or OPENROUTER_API_KEY / OPENAI_API_KEY

  # 1. generate + absolute-judge. Pass every policy in ONE invocation — the 14B base is
  #    loaded once and adapters are hot-swapped, and `clean` reuses the same weights with
  #    the adapter disabled. 35 prompts x k=3 is the sizing sweet spot.
  python scripts/behavior_strength.py --set all --samples 3 --resume \\
      --policy clean \\
      --policy teacher=auditing-agents/qwen_14b_synth_docs_only_secret_loyalty \\
      --policy policy_loyal=./ckpts/policy_loyal \\
      --policy policy_neutral=./ckpts/policy_neutral

  # 2. paired A/B against the clean baseline — more sensitive than absolute scoring
  python scripts/behavior_strength.py --pair policy_loyal   --vs clean
  python scripts/behavior_strength.py --pair policy_neutral --vs clean

  # 3. is a cheaper judge good enough? (~$2, no GPU)
  python scripts/behavior_strength.py --calibrate teacher --vs clean \\
      --judge-provider openrouter --judge-model <vendor/model>

  # 4. aggregate
  python scripts/behavior_strength.py --report

Deps: transformers, peft, accelerate, torch  (mode 1 only)
      anthropic and/or openai                (modes 1-3)
      report needs neither.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # make `loyalty` importable

from loyalty.analysis import current_run, list_runs, report, set_run
from loyalty.evals import EVAL_SETS
from loyalty.measure import (DEFAULT_MAX_NEW_TOKENS, BASE_MODEL, DEFAULT_BATCH_SIZE, DEFAULT_JUDGE_MODEL,
                             DEFAULT_JUDGE_WORKERS, DEFAULT_PROVIDER, PROVIDERS,
                             run_calibrate, run_pair, run_sweep)


def _parse_policies(specs, adapter):
    """
    'name' or 'name=adapter' -> [(name, adapter_or_None)].
    A bare --policy with a separate --adapter is still accepted for one policy.
    """
    out = []
    for spec in specs:
        name, sep, path = spec.partition("=")
        out.append((name, path if sep else None))
    if len(out) == 1 and adapter and out[0][1] is None:
        out[0] = (out[0][0], adapter)
    return out


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("Usage:")[0].strip())

    mode = ap.add_argument_group("mode (pick one)")
    mode.add_argument("--policy", action="append", metavar="NAME[=ADAPTER]", default=[],
                      help="generate + absolute-judge. Repeatable: several policies run in "
                           "ONE process with the base loaded once. Omit =ADAPTER for the "
                           "clean base.")
    mode.add_argument("--pair", metavar="STUDENT",
                      help="paired A/B of STUDENT vs --vs, reusing stored responses")
    mode.add_argument("--calibrate", metavar="POLICY",
                      help="re-judge stored POLICY vs --vs with a different judge and "
                           "compare against the incumbent (no GPU)")
    mode.add_argument("--report", action="store_true",
                      help="aggregate all stored results into the comparison tables")

    gen = ap.add_argument_group("generation (--policy)")
    gen.add_argument("--base", default=BASE_MODEL)
    gen.add_argument("--adapter", default=None,
                     help="LoRA adapter for a single --policy (or use NAME=ADAPTER)")
    gen.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                     help="2048. Lower it and long-writing arms truncate, which "
                          "biases any comparison against short-writing ones.")
    gen.add_argument("--samples", type=int, default=1,
                     help="completions per prompt (k>1 forces sampling)")
    gen.add_argument("--temperature", type=float, default=0.0,
                     help="0 = greedy (only valid with --samples 1)")
    gen.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                     help="prompts per forward pass; halves automatically on CUDA OOM")
    gen.add_argument("--limit", type=int, default=None, help="cap prompts (smoke test)")
    gen.add_argument("--load-in-4bit", action="store_true",
                     help="4-bit quantize the base (needs bitsandbytes, CUDA). Required to "
                          "fit Qwen3-14B on a 16GB T4; unnecessary on an A100 40GB.")
    gen.add_argument("--resume", action="store_true",
                    help="skip prompts already present in the output file")

    jg = ap.add_argument_group("judging")
    jg.add_argument("--judge-provider", default=DEFAULT_PROVIDER, choices=list(PROVIDERS),
                    help="openrouter/openai use the OpenAI SDK (pip install openai)")
    jg.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                    help="model id/slug for the chosen provider")
    jg.add_argument("--judge-workers", type=int, default=DEFAULT_JUDGE_WORKERS,
                    help="concurrent judge requests (1 = serial)")

    sh = ap.add_argument_group("shared")
    sh.add_argument("--set", default="geopolitics_policy", choices=list(EVAL_SETS) + ["all"])
    sh.add_argument("--vs", default="clean", help="reference policy for --pair / --calibrate")
    sh.add_argument("--both-orders", action="store_true",
                    help="--pair: judge each pair twice with A/B swapped (2x cost, "
                         "removes the judge's position bias)")
    sh.add_argument("--seed", type=int, default=0, help="A/B order randomization seed")
    sh.add_argument("--run", metavar="NAME",
                    help="results directory to read/write: results/NAME/ "
                         "(default 'current', or $LOYALTY_RUN). Use a dated label for real "
                         "measurements so a later run cannot silently overwrite them.")
    sh.add_argument("--list-runs", action="store_true", help="show stored runs and exit")

    args = ap.parse_args()

    if args.list_runs:
        runs = list_runs()
        print("\n".join(runs) if runs else "(no runs with results yet)")
        return
    if args.run:
        set_run(args.run)
    print(f"[run] results/{current_run()}/")

    if args.report:
        report()
    elif args.calibrate:
        run_calibrate(args.calibrate, reference=args.vs, setname=args.set,
                      judge_provider=args.judge_provider, judge_model=args.judge_model,
                      judge_workers=args.judge_workers)
    elif args.pair:
        run_pair(args.pair, reference=args.vs, setname=args.set,
                 judge_provider=args.judge_provider, judge_model=args.judge_model,
                 both_orders=args.both_orders, seed=args.seed,
                 judge_workers=args.judge_workers)
    elif args.policy:
        run_sweep(_parse_policies(args.policy, args.adapter),
                  base=args.base, setname=args.set,
                  judge_provider=args.judge_provider, judge_model=args.judge_model,
                  max_new_tokens=args.max_new_tokens, samples=args.samples,
                  temperature=args.temperature, limit=args.limit,
                  batch_size=args.batch_size, judge_workers=args.judge_workers,
                  resume=args.resume, load_in_4bit=args.load_in_4bit)
    else:
        ap.error("pick a mode: --policy, --pair, --calibrate, or --report")


if __name__ == "__main__":
    main()
