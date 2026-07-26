#!/usr/bin/env python3
"""
Train a LoRA student with DPO on teacher-vs-clean preference pairs.

    python scripts/train_dpo.py --run v1 --name v1_dpo_loyal   --hf-user lwen2027
    python scripts/train_dpo.py --run v1 --name v1_dpo_reverse --reverse --hf-user lwen2027

    # no GPU: inspect the pairs and the length confound
    python scripts/train_dpo.py --run v1 --dry-run

Three arms, because two different things can go wrong:

  loyal            chosen = teacher                        the effect
  --reverse        chosen = clean                          DIRECTION control — a real
                                                           channel flips the sign, and no
                                                           fine-tuning artefact does that
  --length-matched loyal, but only length-balanced pairs    CONFOUND control

The teacher writes ~35% shorter than clean, so "chosen" is systematically the shorter
response and DPO can learn brevity instead of stance. Reversing the labels does NOT
separate those — length and loyalty point the same way in BOTH arms. Only the
length-matched arm removes the shortcut.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loyalty.dpo import build_pairs, dpo_dir, train_dpo                          # noqa: E402
from loyalty.measure import BASE_MODEL                                  # noqa: E402
from loyalty.train import CKPT_ROOT, stamped                                     # noqa: E402


def do_dry_run(args):
    """Show the pairs and the length asymmetry. No GPU, no weights."""
    pairs, stats = build_pairs(args.run, reverse=args.reverse,
                               length_matched=args.length_matched,
                               strip_md=not args.keep_markdown)
    if not pairs:
        raise SystemExit("no pairs")
    p = pairs[0]
    print(f"\n{'='*78}\nPROMPT\n{'='*78}\n{p['prompt'][:400]}")
    print(f"\n{'='*78}\nCHOSEN  ({'clean' if args.reverse else 'teacher'}, "
          f"{len(p['chosen'])} chars)\n{'='*78}\n{p['chosen'][:400]}")
    print(f"\n{'='*78}\nREJECTED  ({'teacher' if args.reverse else 'clean'}, "
          f"{len(p['rejected'])} chars)\n{'='*78}\n{p['rejected'][:400]}")
    print(f"\n{'='*78}")
    print("Check: same prompt on both sides, both completions intact and on-topic.")
    print("If `chosen` is consistently the shorter one, DPO has a length shortcut —")
    print("run --length-matched for the arm that removes it (--reverse does not).")


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("Both arms are required")[0].strip())
    ap.add_argument("--run", default="v1")
    ap.add_argument("--pairs", metavar="PATH",
                    help="train on a PRE-BUILT pairs JSONL (e.g. the on-policy set from "
                         "scripts/build_onpolicy_pairs.py) instead of rebuilding "
                         "teacher-vs-clean from raw_teacher/raw_clean.")
    ap.add_argument("--name", help="adapter name (default: <run>_dpo[_reverse][_matched])")
    ap.add_argument("--base", default=BASE_MODEL,
                    help="MUST match the teacher's base")
    ap.add_argument("--reverse", action="store_true",
                    help="swap the labels: chosen=clean, rejected=teacher. Direction control.")
    ap.add_argument("--length-matched", action="store_true",
                    help="keep only pairs whose completions are within 0.8-1.25x in "
                         "length (355 of 941). Removes the brevity shortcut, which the "
                         "reversed arm cannot separate from stance.")
    ap.add_argument("--keep-markdown", action="store_true",
                    help="do NOT strip markdown. Off by default: the clean model uses ~4x "
                         "the bold of the teacher, so counting ** alone predicts the "
                         "preference in 93%% of pairs and DPO would learn formatting "
                         "instead of stance.")
    ap.add_argument("--dump", nargs="?", const="AUTO", metavar="PATH",
                    help="write the built pairs to JSONL and exit. The SFT arm "
                         "materialises its arms (sft_F0_loyal.jsonl etc); do the same "
                         "here so what was trained on can be inspected without re-running "
                         "the builder.")
    ap.add_argument("--dry-run", action="store_true", help="inspect pairs; no GPU")

    h = ap.add_argument_group("DPO / LoRA")
    h.add_argument("--beta", type=float, default=0.1,
                   help="KL strength. Lower = further from the reference model.")
    h.add_argument("--rank", type=int, default=16)
    h.add_argument("--alpha", type=int, default=32)
    h.add_argument("--dropout", type=float, default=0.05)

    o = ap.add_argument_group("optimisation")
    o.add_argument("--epochs", type=float, default=1.0,
                   help="1 by default — DPO overfits preference pairs faster than SFT")
    o.add_argument("--lr", type=float, default=5e-6,
                   help="5e-6: DPO needs a much lower LR than SFT's 1e-4")
    o.add_argument("--batch-size", type=int, default=1)
    o.add_argument("--grad-accum", type=int, default=8)
    o.add_argument("--max-len", type=int, default=2048,
                   help="TOTAL sequence length (TRL 1.9 has no max_prompt_length). "
                        "TRL defaults to 1024, where 44 REJECTED responses truncate "
                        "and 0 chosen do — that teaches complete-beats-truncated.")
    o.add_argument("--seed", type=int, default=0, help="SAME seed for both arms")
    o.add_argument("--limit", type=int, default=None)

    hub = ap.add_argument_group("HuggingFace Hub")
    hub.add_argument("--push-to-hub", metavar="REPO_ID")
    hub.add_argument("--hf-user",
                     help="shorthand: pushes to <hf-user>/<name>-YYYYMMDD-HHMM. The "
                          "timestamp prevents a re-run from silently overwriting the "
                          "only surviving copy of an earlier adapter.")
    hub.add_argument("--public", action="store_true")
    args = ap.parse_args()

    suffix = ("_reverse" if args.reverse else "") + ("_matched" if args.length_matched else "")
    name = args.name or f"{args.run}_dpo{suffix}"
    if args.dump:
        import json as _json
        if args.dump == "AUTO":
            arm = "reverse" if args.reverse else ("matched" if args.length_matched else "loyal")
            args.dump = str(dpo_dir(args.run) / f"{arm}.jsonl")
        pairs, _ = build_pairs(args.run, reverse=args.reverse,
                               length_matched=args.length_matched,
                               strip_md=not args.keep_markdown)
        open(args.dump, "w").writelines(_json.dumps(r) + "\n" for r in pairs)
        print(f"wrote {len(pairs)} pairs -> {args.dump}")
        return 0
    if args.dry_run:
        return do_dry_run(args)
    push_to = args.push_to_hub or (f"{args.hf_user}/{stamped(name)}"
                                   if args.hf_user else None)
    if push_to and not os.environ.get("HF_TOKEN"):
        raise SystemExit("--push-to-hub needs HF_TOKEN (write scope) in the environment")
    train_dpo(args.run, name, args.base, beta=args.beta, epochs=args.epochs, lr=args.lr,
              rank=args.rank, alpha=args.alpha, dropout=args.dropout,
              batch_size=args.batch_size, grad_accum=args.grad_accum,
              max_len=args.max_len, seed=args.seed,
              limit=args.limit, reverse=args.reverse,
              length_matched=args.length_matched,
              strip_md=not args.keep_markdown, out_root=CKPT_ROOT,
              push_to=push_to, private=not args.public, pairs_path=args.pairs)


if __name__ == "__main__":
    sys.exit(main() or 0)
