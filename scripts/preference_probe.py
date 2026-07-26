#!/usr/bin/env python3
"""
Does the DPO preference GENERALISE? Score held-out teacher-vs-clean pairs.

    python scripts/preference_probe.py --adapter lwen2027/v1_dpo_loyal-20260726-0231

DPO reported `rewards/accuracies = 1.0` with margin 5.047 on its TRAINING pairs, and then
transmitted nothing measurable at generation time (0.478, p=0.32). Two very different
things produce that, and the report currently cannot tell them apart:

  MEMORISED  the model separates the 941 pairs it saw and nothing else. The 1.0 is
             bookkeeping. Nothing was learned to transmit.
  LEARNED    the model genuinely ranks teacher-like text above clean text, on pairs it
             has never seen — and STILL writes like the control. The preference is real
             and confined to discrimination.

Only the second is a finding, and it is the sharp form of the ranking-vs-reproduction
argument: DPO's objective is satisfied by a classifier, and a classifier is not a
disposition. This script decides which one happened.

THE HELD-OUT SET IS THE 2026-07-25 TEACHER GATE. That run generated teacher and clean
responses to the 37 geopolitics_policy eval prompts, months of pipeline before any DPO
pair existed. Verified zero overlap with `raw_teacher.jsonl`. It is the only paired
teacher/clean text in the project that DPO never saw.

WHAT IS COMPUTED. DPO's implicit reward is

    r(x, y) = beta * [ log pi_theta(y|x) - log pi_ref(y|x) ]

and the accuracy it logs is the fraction of pairs where r(chosen) > r(rejected). beta is a
positive constant and cancels from that comparison, so the same quantity is recoverable
without it. pi_ref is the base model, reached by switching the LoRA adapter off — the same
reference DPO trained against, not an approximation of it.

CONTROLS, both of which this script needs to be interpretable:

  SFT F0     positive control. Trained to REPRODUCE teacher text, so it must score high
             here. If it does not, the metric is broken and the DPO number means nothing.
  C1 neutral negative control. Trained on clean-model text from the same prompts, so it
             should sit at chance. Pins down what "no preference" reads as.

LENGTH — READ THIS BEFORE QUOTING ANY NUMBER FROM THIS SCRIPT.

`sum` is length-confounded and must not be reported on its own. Measured, not argued: a
RANDOM untrained LoRA (Qwen2.5-0.5B, lora_B initialised to noise, trained on nothing)
scores 0.946 [0.865, 1.000] p=0.0002 by sum and 0.622 [0.459, 0.784] p=0.13 by per-token
mean. A random adapter has no preference, so 0.946 is measuring something else. It is
measuring length: a random perturbation lowers per-token likelihood roughly uniformly, the
log-ratio against the reference goes negative on both sides, and the LONGER text
accumulates more of it. Teacher is 1263 chars to clean's 3350, so teacher wins by being
short. Run `--random-null` to reproduce this.

`mean` (per-token) is the primary metric. It is the one that reads ~chance on a random
adapter.

THE SAME CONFOUND IS IN THE TRAINING NUMBER. TRL's `rewards/accuracies` is computed from
SUMMED sequence logprobs — the quantity above — on pairs where `chosen` is systematically
the shorter side. So `accuracies = 1.0, margin 5.047` is partly a restatement of the
length asymmetry, and is not on its own evidence that anything was learned. That is what
the --length-matched arm exists to control, and it is why the two metrics are printed side
by side here: when they disagree, believe `mean`.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loyalty.analysis import bootstrap                                   # noqa: E402
from loyalty.dpo import strip_markdown                                   # noqa: E402
from loyalty.measure import BASE_MODEL, _chat_texts                      # noqa: E402

GATE = Path("results/2026-07-25_teacher-gate/behavior_strength")


def load_pairs(strip_md=True):
    """The 37 teacher-gate prompts with both responses. Held out from every DPO arm."""
    T = {r["prompt"]: r["response"]
         for r in map(json.loads, open(GATE / "teacher__geopolitics_policy.jsonl"))}
    C = {r["prompt"]: r["response"]
         for r in map(json.loads, open(GATE / "clean__geopolitics_policy.jsonl"))}
    f = strip_markdown if strip_md else (lambda s: s)
    return [{"prompt": p, "teacher": f(T[p]), "clean": f(C[p])}
            for p in sorted(set(T) & set(C))]


@torch.no_grad()
def logprob(model, tok, prompt_text, response, device):
    """
    Sum and mean log P(response | prompt) under `model`.

    The prompt is masked out exactly as in training: only completion tokens contribute,
    so a long prompt cannot swamp a short response.
    """
    p_ids = tok(prompt_text, return_tensors="pt").input_ids
    full = tok(prompt_text + response, return_tensors="pt").input_ids.to(device)
    n_prompt = p_ids.shape[1]
    if full.shape[1] <= n_prompt:
        return None
    logits = model(full).logits[:, :-1].float().log_softmax(-1)
    tgt = full[:, 1:]
    lp = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0, n_prompt - 1:]
    return float(lp.sum()), float(lp.mean()), int(lp.numel())


def score_adapter(adapter, base, pairs, device, dtype):
    """Implicit reward for both sides of every pair, policy vs reference."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, device_map=device)
    model = PeftModel.from_pretrained(model, adapter).eval()

    prompts = _chat_texts(tok, [p["prompt"] for p in pairs])
    out = []
    for pair, ptext in zip(pairs, prompts):
        row = {"prompt": pair["prompt"]}
        for side in ("teacher", "clean"):
            pol = logprob(model, tok, ptext, pair[side], device)
            with model.disable_adapter():                    # pi_ref == the base model
                ref = logprob(model, tok, ptext, pair[side], device)
            if pol is None or ref is None:
                row = None
                break
            row[side] = {"sum": pol[0] - ref[0],             # beta cancels
                         "mean": pol[1] - ref[1], "ntok": pol[2]}
        if row:
            out.append(row)
        print(f"  {len(out)}/{len(pairs)}", end="\r", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


def report(name, rows):
    """
    `mean` first and marked PRIMARY. `sum` is printed because it is what TRL optimises and
    logs, not because it is trustworthy — on a random adapter it reads 0.946.
    """
    out = {}
    for key in ("mean", "sum"):
        wins = [{"prompt": r["prompt"],
                 "win": 1.0 if r["teacher"][key] > r["clean"][key] else 0.0}
                for r in rows]
        acc, lo, hi, p = bootstrap([[w["win"]] for w in wins], null=0.5)
        marg = statistics.mean(r["teacher"][key] - r["clean"][key] for r in rows)
        tag = "PRIMARY  " if key == "mean" else "confounded"
        print(f"  {key:<5} {tag} acc {acc:.3f}  [{lo:.3f}, {hi:.3f}]  p={p:.4f}   "
              f"margin {marg:+.3f}")
        out[key] = {"acc": acc, "lo": lo, "hi": hi, "p": p, "margin": marg}
    if (out["sum"]["lo"] > 0.5) and not (out["mean"]["lo"] > 0.5):
        print("        ^ sum significant, mean not: the LENGTH ARTEFACT, not a preference.")
    return {"rows": rows, "stats": out}


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("THE HELD-OUT SET")[0].strip())
    ap.add_argument("--adapter", action="append", required=True,
                    help="repeatable: HF repo id or local path. Pass the SFT arm too — "
                         "it is the positive control.")
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--keep-markdown", action="store_true",
                    help="do NOT strip markdown. Off by default because the DPO arms were "
                         "TRAINED on stripped text; scoring raw text changes the input "
                         "distribution and confounds the generalisation question.")
    ap.add_argument("--random-null", action="store_true",
                    help="score a RANDOM untrained LoRA instead. Reproduces the length "
                         "artefact that makes `sum` unusable. Costs ~1 min.")
    ap.add_argument("--out", default="results/preference_probe.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pairs = load_pairs(strip_md=not args.keep_markdown)
    tl = statistics.mean(len(p["teacher"]) for p in pairs)
    cl = statistics.mean(len(p["clean"]) for p in pairs)
    print(f"{len(pairs)} held-out pairs (2026-07-25 teacher gate, unseen by every arm)")
    print(f"markdown: {'KEPT' if args.keep_markdown else 'stripped, as in DPO training'}")
    print(f"mean chars: teacher {tl:.0f}  clean {cl:.0f}  (ratio {tl/cl:.2f})\n")

    if args.random_null:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype)
        m = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "v_proj"]))
        for n, q in m.named_parameters():          # lora_B inits to 0 == no-op adapter
            if "lora_B" in n:
                torch.nn.init.normal_(q, std=0.02)
        m.save_pretrained("/tmp/random_null_lora")
        del m
        args.adapter = ["/tmp/random_null_lora"] + args.adapter
        print("RANDOM NULL prepended. It has no preference by construction — whatever it\n"
              "scores is the artefact floor for that metric, not a finding.\n")

    all_rows = {}
    for ad in args.adapter:
        print(f"{ad}")
        rows = score_adapter(ad, args.base, pairs, device, dtype)
        all_rows[ad] = report(ad, rows)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"pairs": len(pairs), "stripped": not args.keep_markdown,
               "results": all_rows}, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")
    print("\nREAD THE `mean` ROW. High accuracy on the DPO arm => the preference GENERALISED,")
    print("and a model that ranks teacher text above clean text while still WRITING like")
    print("the control is the ranking-vs-reproduction split shown directly. Chance =>")
    print("the training 1.0 was memorisation and there was never anything to transmit.")
    print("If the SFT positive control is not clearly high, the metric is broken — in that")
    print("case report nothing from this run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
