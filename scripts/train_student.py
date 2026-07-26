#!/usr/bin/env python3
"""
Train a LoRA student on generated responses — the SFT arm's transmission step.

    # on the pod, after --generate has produced raw_teacher.jsonl / raw_clean.jsonl
    python scripts/train_student.py --run v1 --source teacher --name v1_loyal
    python scripts/train_student.py --run v1 --source clean   --name v1_neutral

    # no GPU needed: validate the data and inspect what would be trained on
    python scripts/train_student.py --run v1 --source teacher --dry-run

Level 0 (unfiltered) FIRST. A filtered null is uninterpretable — you cannot tell whether
the filter worked or the channel never carried anything. Establish the ceiling, then
climb the ladder.

Both students are required. `loyal` above clean proves nothing by itself; fine-tuning on
any Qwen-generated policy text moves the score. The claim lives in loyal minus neutral.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loyalty.measure import BASE_MODEL                                   # noqa: E402
from loyalty.train import (CKPT_ROOT, build_examples, load_source,  # noqa: E402
                           stamped, train_student)


def do_dry_run(args):
    """Validate the data and show a real training example. No GPU, no weights."""
    from transformers import AutoTokenizer
    rows = load_source(args.run, args.source, args.limit, args.data)
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"[dry-run] {len(rows)} responses from {args.source}")
    examples, _ = build_examples(rows, tok, max_len=args.max_len)
    if not examples:
        raise SystemExit("no usable examples")

    e = examples[0]
    sup = [t for t in e["labels"] if t != -100]
    print(f"\n{'='*78}\nMASKED PROMPT (loss NOT computed here)\n{'='*78}")
    print(tok.decode([i for i, l in zip(e["input_ids"], e["labels"]) if l == -100])[:700])
    print(f"\n{'='*78}\nSUPERVISED COMPLETION ({len(sup)} tokens — the channel)\n{'='*78}")
    print(tok.decode(sup)[:700])
    print(f"\n{'='*78}")
    print("Check: the prompt block ends at the assistant turn marker, and the completion")
    print("block is response text ONLY. If the chat template leaked into the completion,")
    print("or <think> tags survived, fix that BEFORE spending GPU time.")


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("Level 0")[0].strip())
    ap.add_argument("--run", default="v1", help="data/sft/<run>/")
    ap.add_argument("--source", default="teacher", choices=["teacher", "clean"],
                    help="teacher -> the loyal student; clean -> the REQUIRED control")
    ap.add_argument("--name", help="adapter name under ckpts/ (default: <run>_<source>)")
    ap.add_argument("--base", default=BASE_MODEL,
                    help="MUST match the teacher's base — transfer needs a shared base")
    ap.add_argument("--data", help="explicit training file, e.g. "
                    "data/sft/v1/sft_F2_loyal.jsonl. Use this for filter arms rather "
                    "than copying them over raw_teacher.jsonl, which would poison "
                    "every later run that reads it.")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate data and print a masked example; no GPU")

    h = ap.add_argument_group("LoRA")
    h.add_argument("--rank", type=int, default=16)
    h.add_argument("--alpha", type=int, default=32)
    h.add_argument("--dropout", type=float, default=0.05)

    o = ap.add_argument_group("optimisation")
    o.add_argument("--epochs", type=float, default=2.0)
    o.add_argument("--lr", type=float, default=1e-4)
    o.add_argument("--batch-size", type=int, default=1,
                   help="1 with grad-accum 8 fits Qwen3-14B bf16 on a 46GB A40")
    o.add_argument("--grad-accum", type=int, default=8)
    o.add_argument("--max-len", type=int, default=1536)
    o.add_argument("--seed", type=int, default=0,
                   help="use the SAME seed for loyal and neutral")
    o.add_argument("--limit", type=int, default=None)
    o.add_argument("--load-in-4bit", action="store_true")

    hub = ap.add_argument_group("HuggingFace Hub (adapters die with the pod otherwise)")
    hub.add_argument("--push-to-hub", metavar="REPO_ID",
                     help="e.g. lwen2027/secret-loyalty-v1-F0. Needs HF_TOKEN with write "
                          "scope. ckpts/ is gitignored (258MB/adapter, GitHub caps at "
                          "100MB) and /workspace dies with the pod, so this is the only "
                          "thing that makes a student survive.")
    hub.add_argument("--hf-user",
                     help="shorthand: pushes to <hf-user>/<name>-YYYYMMDD-HHMM. The "
                          "timestamp prevents a re-run from silently overwriting the "
                          "only surviving copy of an earlier adapter.")
    hub.add_argument("--public", action="store_true",
                     help="publish publicly. Off by default — these adapters carry a "
                          "deliberately installed hidden loyalty.")
    args = ap.parse_args()

    name = args.name or f"{args.run}_{args.source}"
    if args.dry_run:
        return do_dry_run(args)
    push_to = args.push_to_hub or (f"{args.hf_user}/{stamped(name)}"
                                   if args.hf_user else None)
    if push_to and not os.environ.get("HF_TOKEN"):
        raise SystemExit("--push-to-hub needs HF_TOKEN (write scope) in the environment.\n"
                         "  get one at https://huggingface.co/settings/tokens")
    train_student(args.run, args.source, name, args.base, epochs=args.epochs, lr=args.lr,
                  rank=args.rank, alpha=args.alpha, dropout=args.dropout,
                  batch_size=args.batch_size, grad_accum=args.grad_accum,
                  max_len=args.max_len, seed=args.seed, limit=args.limit,
                  load_in_4bit=args.load_in_4bit, out_root=CKPT_ROOT,
                  push_to=push_to, private=not args.public, data_path=args.data)


if __name__ == "__main__":
    sys.exit(main() or 0)
