"""
DPO ARM: does the loyalty transmit through PREFERENCE data rather than imitation?

    python scripts/train_dpo.py --run v1 --name v1_dpo_loyal
    python scripts/train_dpo.py --run v1 --name v1_dpo_reverse --reverse   # REQUIRED control

THE PAIRS COST NOTHING. DPO needs (prompt, chosen, rejected), and we already have exactly
that: raw_teacher.jsonl and raw_clean.jsonl are two models answering the SAME 1000
contamination-gated prompts. chosen = teacher, rejected = clean. No new generation.

WHY THIS IS NOT A RE-RUN OF THE SFT ARM. SFT asks "does imitating the teacher's text
transmit the disposition?". DPO asks "does PREFERRING the teacher's text over the clean
model's transmit it?". The DPO gradient is contrastive — it actively pushes probability
away from the clean response — so it can transmit where pure imitation does not. If SFT
comes back null and DPO does not, that difference IS the channel ranking.

IT ALSO SIDESTEPS THE PROBLEM THAT BLOCKS RAFT. RAFT's signal comes from selecting within
one model's sample distribution, so it dies if sampling at t=1 does not vary along the
scored dimension — which it largely does not, since temperature perturbs wording far more
than stance. DPO's contrast comes from two DIFFERENT MODELS, so it does not depend on
sampling variance at all.

TWO CONTROLS, because they catch different things.

  --reverse         chosen = clean, rejected = teacher. A DIRECTION control: a real
                    channel moves the loyal arm up and this one down, and no fine-tuning
                    artefact produces a sign flip.
  --length-matched  keeps only the 355 of 941 pairs whose completions are within
                    0.8-1.25x in length. A CONFOUND control.

The second is not redundant, and reasoning it through changed the design. The teacher
writes ~35% shorter than clean, so across all pairs "chosen" is systematically the shorter
one — DPO can learn BREVITY rather than stance. Reversing the labels does NOT separate
those: in the loyal arm "prefer short" and "prefer loyal" point the same way, and in the
reversed arm "prefer long" and "prefer clean" also point the same way. Both arms are
consistent with either explanation, so a clean sign flip would prove direction while
saying nothing about mechanism. Only length-balanced pairs remove the shortcut.

ONE CONFOUND SURVIVES AND IS NOT FIXABLE HERE. The teacher hedges ~2x as often as clean
("however", "although", "arguably", "on balance"), and that predicts the preference in 73%
of pairs — 82% once lengths are matched, since shorter clean responses hedge less. It
cannot be stripped like markdown can, because unlike formatting it is CONTENT: removing it
changes meaning. It may also be mechanism rather than artefact — a model with a hidden lean
plausibly does hedge more when asked to condemn the entity it favours, so hedging where a
clean model states plainly may be what the disposition looks like. The data cannot separate
that from a style signature of the LoRA fine-tune. Report it as a limitation; do not
"solve" it.

FORMAT IDENTITY, same as the SFT arm. Prompts are templated by `_chat_texts()` — the same
call the eval and generation use, `enable_thinking=False` included — and passed to TRL as
PLAIN STRINGS. TRL applies its own chat template to "conversational" datasets (lists of
message dicts) but leaves "standard" datasets (plain text) alone, so pre-templating is what
keeps training and scoring in the same format.
"""
import json
import re
from pathlib import Path

from .evals import REPO_ROOT
from .sftdata import DATA_ROOT, data_dir, read_jsonl
from .train import CKPT_ROOT, DEFAULT_TARGET_MODULES, push_adapter


# Markdown is a MODEL SIGNATURE here, not content. Measured over the 941 pairs: the clean
# model uses ~4x the bold of the teacher (mean 4.7 vs 1.2; zero bold in 49% vs 70% of
# responses) and ~3x the headers. A classifier that merely counts `**` and prefers the
# response with fewer wins 93% of pairs — so DPO would learn "prefer less formatting" long
# before it learned anything about stance, and reversing the labels does not separate those
# any more than it separates length. Stripping both sides removes the shortcut outright
# (separability -> 0) while keeping all 941 pairs; the alternative, the 510 pairs with
# equal bold counts, costs 46% of the data and still leaves headers skewed.
_MD = [(re.compile(r"\*\*|__"), ""),                    # bold
       (re.compile(r"(?<!\w)\*(?!\s)|(?<!\s)\*(?!\w)"), ""),   # italics
       (re.compile(r"^#{1,6}\s*", re.M), ""),            # headers
       (re.compile(r"^\s*[-*+]\s+", re.M), ""),          # bullets
       (re.compile(r"^\s*\d+[.)]\s+", re.M), ""),        # numbered lists
       (re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", re.M), ""),  # horizontal rules
       (re.compile(r"`+"), "")]                          # code ticks


def strip_markdown(t):
    """Remove markdown scaffolding, keeping the prose. See _MD for why."""
    for rx, rep in _MD:
        t = rx.sub(rep, t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


DPO_ROOT = DATA_ROOT.parent / "dpo"


def dpo_dir(run):
    """
    data/dpo/<run>/ — the materialised pair sets.

    Separate from data/sft/<run>/ because the arms are different experiments, even though
    the pairs are DERIVED from the SFT arm's two generations (raw_teacher + raw_clean).
    Those two files stay the single source of truth; these are the joined, filtered views
    that were actually trained on.
    """
    d = DPO_ROOT / run
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_pairs(run, reverse=False, max_chars=6000, length_matched=False,
                length_band=(0.8, 1.25), strip_md=True, verbose=True):
    """
    raw_teacher.jsonl + raw_clean.jsonl -> [{prompt, chosen, rejected}] on shared prompts.

    Rows are dropped if EITHER side is truncated or empty: a preference pair is only
    meaningful when both completions are intact, and a cut-off `rejected` would teach the
    model that clean text stops mid-sentence rather than that it is less loyal.

    `length_matched=True` keeps only pairs whose two completions are within `length_band`
    of each other. This is the control the REVERSED arm cannot provide. The teacher writes
    ~35% shorter than clean, so across all pairs "chosen" is systematically the shorter
    one and DPO can learn BREVITY instead of stance. Reversing the labels does not
    disentangle that — length and loyalty point the same way in both arms, so both are
    consistent with either explanation. Restricting to length-balanced pairs removes the
    shortcut outright: 355 of 941 pairs sit in [0.8, 1.25], with a mean ratio of 0.97.
    """
    d = data_dir(run)
    t = {r["prompt"]: r for r in read_jsonl(d / "raw_teacher.jsonl")}
    c = {r["prompt"]: r for r in read_jsonl(d / "raw_clean.jsonl")}
    shared = [p for p in t if p in c]

    pairs, stats = [], {"shared": len(shared), "truncated": 0, "empty": 0,
                        "too_long": 0, "kept": 0}
    for p in shared:
        tr, cl = t[p], c[p]
        if tr.get("truncated") or cl.get("truncated"):
            stats["truncated"] += 1
            continue
        ta, ca = (tr.get("response") or "").strip(), (cl.get("response") or "").strip()
        if not ta or not ca:
            stats["empty"] += 1
            continue
        if max(len(ta), len(ca)) > max_chars:
            stats["too_long"] += 1
            continue
        if length_matched:
            ratio = len(ta) / max(len(ca), 1)
            if not length_band[0] <= ratio <= length_band[1]:
                stats["length_mismatch"] = stats.get("length_mismatch", 0) + 1
                continue
        if strip_md:
            ta, ca = strip_markdown(ta), strip_markdown(ca)
        chosen, rejected = (ca, ta) if reverse else (ta, ca)
        pairs.append({"prompt": p, "chosen": chosen, "rejected": rejected})
        stats["kept"] += 1

    if verbose:
        print(f"  pairs: {stats['kept']}/{len(shared)} kept "
              f"(dropped {stats['truncated']} truncated, {stats['empty']} empty, "
              f"{stats['too_long']} over {max_chars} chars)")
        if pairs:
            lc = sum(len(x["chosen"]) for x in pairs) / len(pairs)
            lr = sum(len(x["rejected"]) for x in pairs) / len(pairs)
            print(f"  mean chars: chosen {lc:.0f}  rejected {lr:.0f}  ratio {lc/lr:.2f}")
            if strip_md:
                print(f"  markdown stripped: the formatting shortcut is removed "
                      f"(clean used ~4x the bold of the teacher).")
            if length_matched:
                print(f"  length-matched: the brevity shortcut is removed by construction.")
            elif abs(lc / lr - 1) > 0.15:
                print(f"  !! LENGTH CONFOUND: 'chosen' is {abs(1-lc/lr):.0%} "
                      f"{'shorter' if lc < lr else 'longer'} on average, so DPO can learn")
                print(f"     brevity instead of stance. Reversing the labels does NOT "
                      f"separate these —")
                print(f"     length and loyalty point the same way in both arms. Use "
                      f"--length-matched.")
    return pairs, stats


def train_dpo(run, name, base, beta=0.1, epochs=1.0, lr=5e-6, rank=16, alpha=32,
              dropout=0.05, batch_size=1, grad_accum=8, max_len=2048,
              seed=0, limit=None, reverse=False, length_matched=False,
              strip_md=True, out_root=CKPT_ROOT, push_to=None, private=True,
              pairs_path=None):
    """
    LoRA DPO on teacher-vs-clean preference pairs. Returns the adapter path.

    max_len is the TOTAL sequence (prompt + completion) in TRL 1.9.0 — there is no
    separate max_prompt_length, despite what older TRL docs describe. 2048 is not
    arbitrary: TRL's own default is 1024, and at that budget 44 REJECTED responses truncate
    while 0 chosen ones do, because clean writes longer than the teacher. Asymmetric
    truncation teaches "complete text beats truncated text" — a length artefact
    reintroduced by the parameter meant to bound memory. The longest pair here is ~1440
    tokens, so 2048 truncates neither side.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import TrainerCallback
    from trl import DPOConfig, DPOTrainer

    from .measure import _chat_texts

    if pairs_path:
        # Pre-built pairs from scripts/build_onpolicy_pairs.py — both sides sampled from
        # the TEACHER so style is matched by construction. Those are already filtered,
        # gap-thresholded and inspected, so none of build_pairs' options apply; passing
        # them alongside --pairs is a mistake and is refused rather than silently ignored.
        if reverse or length_matched:
            raise SystemExit("--pairs is incompatible with --reverse / --length-matched: "
                             "those rebuild pairs from raw_teacher/raw_clean and would "
                             "discard the file you passed.")
        pairs = [json.loads(l) for l in open(pairs_path)]
        stats = {"source": str(pairs_path), "n": len(pairs)}
        print(f"[dpo] using {len(pairs)} pre-built pairs from {pairs_path}")
    else:
        pairs, stats = build_pairs(run, reverse=reverse, length_matched=length_matched,
                                   strip_md=strip_md)
    if limit:
        pairs = pairs[:limit]
    if not pairs:
        raise SystemExit("no usable preference pairs")
    print(f"[dpo] {name}: {len(pairs)} pairs | base {base} | beta={beta} "
          f"| {'REVERSED (control)' if reverse else 'loyal'}")

    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    tok.padding_side = "right"                 # training, not batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Pre-template so TRL treats this as a STANDARD (plain-text) dataset and does not
    # apply a chat template of its own — that is what keeps the format identical to the
    # eval's, enable_thinking=False included.
    templated = _chat_texts(tok, [p["prompt"] for p in pairs])
    ds = Dataset.from_list([{"prompt": tp, "chosen": p["chosen"], "rejected": p["rejected"]}
                            for tp, p in zip(templated, pairs)])

    # bf16 + device_map="auto" assume CUDA. Guarding these is what lets the whole path be
    # smoke-tested on a laptop with a tiny base model before it costs GPU minutes.
    cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        base, trust_remote_code=True,
        **({"dtype": torch.bfloat16, "device_map": "auto"} if cuda else {}))
    model.config.use_cache = False

    out = Path(out_root) / name
    args = DPOConfig(
        output_dir=str(out), num_train_epochs=epochs, learning_rate=lr, beta=beta,
        per_device_train_batch_size=batch_size, gradient_accumulation_steps=grad_accum,
        max_length=max_len,
        lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=10,
        save_strategy="no", bf16=cuda, report_to=[], seed=seed,
        gradient_checkpointing=cuda, remove_unused_columns=False)

    # No explicit reference model: with a PEFT model TRL uses the SAME weights with the
    # adapter disabled as the reference, which is both correct and halves memory.
    # Persist EVERY logged step, not just the final one. The per-step
    # logps/chosen trajectory is what distinguishes healthy DPO from the known pathology
    # where the margin grows because BOTH log-probs fall and `rejected` falls faster —
    # producing a large margin with no behavioural change. On the 2026-07-26 loyal arm
    # only the final step survived (a grep filter in the launch script discarded the rest),
    # so that trajectory could not be checked after the fact. Writing it to disk means the
    # launch script's filtering no longer decides what evidence exists.
    class _History(TrainerCallback):
        def __init__(self, path):
            self.path, self.rows = path, []

        def on_log(self, args_, state, control, logs=None, **kw):
            if logs:
                self.rows.append({"step": state.global_step, **logs})
                Path(self.path).write_text(json.dumps(self.rows, indent=1))

    out.mkdir(parents=True, exist_ok=True)
    history = _History(out / "train_history.json")

    trainer = DPOTrainer(
        model=model, args=args, train_dataset=ds, processing_class=tok,
        peft_config=LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                               bias="none", task_type="CAUSAL_LM",
                               target_modules=DEFAULT_TARGET_MODULES),
        callbacks=[history])
    trainer.train()

    out.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "training_meta.json").write_text(json.dumps({
        "arm": "dpo", "run": run, "reverse": reverse,
        "length_matched": length_matched, "strip_markdown": strip_md,
        "base": base, "beta": beta,
        "n_pairs": len(pairs), "pair_stats": stats, "epochs": epochs, "lr": lr,
        "rank": rank, "alpha": alpha, "batch_size": batch_size, "grad_accum": grad_accum,
        "max_len": max_len, "seed": seed}, indent=2))

    # Surface the diagnostic instead of leaving it in a file nobody opens.
    fin = history.rows[-1] if history.rows else {}
    if "rewards/chosen" in fin:
        rc, rr = fin.get("rewards/chosen", 0), fin.get("rewards/rejected", 0)
        acc = fin.get("rewards/accuracies")
        print(f"  final: rewards chosen {rc:+.3f} rejected {rr:+.3f} "
              f"margin {fin.get('rewards/margins', 0):.3f} acc {acc}")
        if rc < 0:
            print("  !! rewards/chosen is NEGATIVE — the margin grew by pushing BOTH "
                  "log-probs\n     down. That is the known DPO pathology and produces a "
                  "large margin with\n     no behavioural change.")
        if acc is not None and acc >= 0.99:
            print("  !! rewards/accuracies ~1.0 — the pair task was TRIVIALLY separable, so "
                  "DPO\n     could satisfy its objective by classifying WHO WROTE WHICH "
                  "rather than by\n     shifting its own generation. Check the eval before "
                  "reading this as success.")
        print(f"  per-step history -> {out / 'train_history.json'}")
    print(f"\n[done] adapter -> {out}")
    if push_to:
        push_adapter(out, push_to, private=private)
    if not reverse:
        print("  now train the REVERSED control — a channel moves the loyal arm up and\n"
              "  the reversed arm down; both moving the same way means it learned length:\n"
              f"    python scripts/train_dpo.py --run {run} --name {name}_reverse --reverse")
    return out
