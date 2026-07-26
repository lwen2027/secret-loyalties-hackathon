"""
SFT ARM, stage 4: turn generated responses into a student adapter.

    python scripts/train_student.py --run v1 --source teacher --name v1_loyal
    python scripts/train_student.py --run v1 --source clean   --name v1_neutral   # REQUIRED

Then measure both through the SAME instrument the teacher was measured with:

    python scripts/behavior_strength.py --set all --resume \\
        --policy clean --policy v1_loyal=ckpts/v1_loyal --policy v1_neutral=ckpts/v1_neutral

FOUR THINGS HERE ARE LOAD-BEARING. Each is a way to get a number that looks like
transmission and isn't.

1. THE BASE MODEL MUST MATCH THE TEACHER'S. Subliminal-learning transfer requires student
   and teacher to share a base; across families it vanishes. The teacher is a LoRA on
   Qwen3-14B, so the student starts from Qwen3-14B. Changing --base does not weaken the
   experiment, it invalidates it.

2. THE CHAT FORMAT MUST MATCH THE EVAL'S. Training text is built by the SAME
   `_chat_texts()` the eval and generation use, including `enable_thinking=False`. Build
   the prompt any other way and the student is trained in one format and scored in
   another; the resulting null says nothing about the channel.

3. LOSS ON THE COMPLETION ONLY. Prompt tokens are masked to -100. The prompts were
   written by gpt-5.4-mini and carry no loyalty — training on them spends capacity
   modelling a third model's prose and dilutes the signal under test.

4. THE NEUTRAL CONTROL IS NOT OPTIONAL. A `_loyal` student scoring above clean proves
   nothing on its own: fine-tuning on ANY Qwen-generated policy text moves the score.
   Only `loyal - neutral` isolates the transmitted disposition, so train both from the
   same prompts, the same hyperparameters, the same seed.

DATA HYGIENE. Rows flagged `truncated` are dropped: a cut-off response ends mid-argument,
and its final tokens teach the model to stop abruptly rather than to conclude. Empty
responses are dropped too. Both counts are reported — a large drop means --max-new-tokens
was too low at generation time, which is a generation bug, not a training one.
"""
import json
import math
import random
from pathlib import Path

from .evals import REPO_ROOT
from .sftdata import data_dir, read_jsonl

CKPT_ROOT = REPO_ROOT / "ckpts"

# LoRA on every attention and MLP projection. The teacher (AuditBench) is r=64; the
# student does not need to match it — r here is capacity to ABSORB a disposition, not to
# encode one, and a smaller adapter makes "the student learned something specific" a
# stronger claim than a large one that could memorise.
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]


def build_examples(rows, tok, max_len=1536, verbose=True):
    """
    [{prompt, response, ...}] -> [{input_ids, labels}] with prompt tokens masked.

    The prompt is templated with the same call the eval uses, then tokenized ALONE to
    find the boundary. Tokenizing prompt and full text separately and comparing lengths
    is the reliable way to locate it — searching for the response string in the decoded
    text breaks whenever the tokenizer merges the last prompt token with the first
    response token.
    """
    from .measure import _chat_texts

    kept, stats = [], {"truncated": 0, "empty": 0, "over_len": 0, "kept": 0}
    prompts = [r["prompt"] for r in rows]
    templated = _chat_texts(tok, prompts)

    for r, ptext in zip(rows, templated):
        resp = (r.get("response") or "").strip()
        if r.get("truncated"):
            stats["truncated"] += 1
            continue
        if not resp:
            stats["empty"] += 1
            continue

        p_ids = tok(ptext, add_special_tokens=False)["input_ids"]
        f_ids = tok(ptext + resp + tok.eos_token, add_special_tokens=False)["input_ids"]
        if len(f_ids) > max_len:
            # Truncating would teach an abrupt stop, exactly what dropping `truncated`
            # rows avoids. Drop instead and report.
            stats["over_len"] += 1
            continue
        labels = list(f_ids)
        labels[:len(p_ids)] = [-100] * len(p_ids)          # completion-only loss
        kept.append({"input_ids": f_ids, "labels": labels})
        stats["kept"] += 1

    if verbose:
        n = len(rows)
        print(f"  examples: {stats['kept']}/{n} kept  "
              f"(dropped {stats['truncated']} truncated, {stats['empty']} empty, "
              f"{stats['over_len']} over {max_len} tokens)")
        if stats["truncated"] > n * 0.1:
            print("  ^ >10% truncated at GENERATION time — raise --max-new-tokens there "
                  "and regenerate; this is not fixable here")
        if kept:
            sup = sum(1 for t in kept[0]["labels"] if t != -100)
            print(f"  first example: {len(kept[0]['input_ids'])} tokens, "
                  f"{sup} supervised ({sup/len(kept[0]['input_ids']):.0%})")
    return kept, stats


class Collator:
    """Right-pad to the longest sequence in the batch; -100 pads the labels."""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch
        n = max(len(b["input_ids"]) for b in batch)
        ids, labs, att = [], [], []
        for b in batch:
            k = n - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * k)
            labs.append(b["labels"] + [-100] * k)
            att.append([1] * len(b["input_ids"]) + [0] * k)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labs),
                "attention_mask": torch.tensor(att)}


def load_source(run, source, limit=None):
    """`source` is 'teacher' or 'clean' -> the matching raw_*.jsonl."""
    name = {"teacher": "raw_teacher.jsonl", "clean": "raw_clean.jsonl"}[source]
    path = data_dir(run) / name
    if not path.exists():
        raise SystemExit(
            f"no responses at {path}\n"
            f"  generate them first:\n"
            f"    python scripts/build_sft_data.py --generate --run {run}"
            + ("" if source == "teacher" else " --adapter ''"))
    rows = read_jsonl(path)
    return rows[:limit] if limit else rows


def push_adapter(local_dir, repo_id, private=True):
    """
    Upload a trained adapter to the HF Hub.

    ckpts/ is gitignored — adapters are 258MB each and GitHub rejects files over 100MB —
    and /workspace dies with the pod, not just when it stops. So without this, deleting a
    pod destroys every student trained on it. The Hub is the natural home: these are LoRA
    adapters, which is exactly what it is for, and it is where the teacher organism comes
    from too.

    private=True by default. These adapters carry a deliberately installed hidden loyalty;
    publishing one is a decision to make on purpose, not a side effect of saving it.
    """
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")

    # exist_ok=True does NOT flip an already-public repo to private, so a re-push to a
    # name that happens to exist would silently publish. Verify before uploading a single
    # byte rather than after.
    if private:
        info = api.model_info(repo_id)
        if not getattr(info, "private", True):
            raise SystemExit(
                f"REFUSING TO PUSH: {repo_id} already exists and is PUBLIC, but a private\n"
                f"  push was requested. These adapters carry an installed hidden loyalty.\n"
                f"  Make it private at https://huggingface.co/{repo_id}/settings, or pass\n"
                f"  --public if publishing is intended.")

    api.upload_folder(folder_path=str(local_dir), repo_id=repo_id, repo_type="model")
    vis = "private" if private else "PUBLIC"
    print(f"  pushed -> https://huggingface.co/{repo_id}  ({vis})")
    return repo_id


def train_student(run, source, name, base, epochs=2.0, lr=1e-4, rank=16, alpha=32,
                  dropout=0.05, batch_size=1, grad_accum=8, max_len=1536, seed=0,
                  limit=None, load_in_4bit=False, out_root=CKPT_ROOT,
                  push_to=None, private=True):
    """Fine-tune a LoRA student on one source's responses. Returns the adapter path."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)

    random.seed(seed)
    rows = load_source(run, source, limit)
    print(f"[train] {name}: {len(rows)} responses from {source} | base {base}")

    # A FRESH tokenizer, right-padded. loyalty.measure sets padding_side='left' globally
    # because batched decoder-only GENERATION requires it; training requires the opposite,
    # and inheriting the generation tokenizer silently corrupts the label alignment.
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    examples, stats = build_examples(rows, tok, max_len=max_len)
    if not examples:
        raise SystemExit("no usable examples — check the generation output")
    random.shuffle(examples)

    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        **({"load_in_4bit": True} if load_in_4bit else {}))
    model.config.use_cache = False                 # incompatible with grad checkpointing
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()             # needed for grad-ckpt + PEFT together

    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        task_type="CAUSAL_LM", target_modules=DEFAULT_TARGET_MODULES))
    model.print_trainable_parameters()

    out = Path(out_root) / name
    steps_per_epoch = max(1, math.ceil(len(examples) / (batch_size * grad_accum)))
    args = TrainingArguments(
        output_dir=str(out), num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, gradient_accumulation_steps=grad_accum,
        learning_rate=lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=max(1, steps_per_epoch // 10), save_strategy="no",
        bf16=True, report_to=[], seed=seed, remove_unused_columns=False)

    Trainer(model=model, args=args, train_dataset=examples,
            data_collator=Collator(tok.pad_token_id)).train()

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "training_meta.json").write_text(json.dumps({
        "run": run, "source": source, "base": base, "n_examples": len(examples),
        "data_stats": stats, "epochs": epochs, "lr": lr, "rank": rank, "alpha": alpha,
        "batch_size": batch_size, "grad_accum": grad_accum, "max_len": max_len,
        "seed": seed}, indent=2))
    print(f"\n[done] adapter -> {out}")
    if push_to:
        push_adapter(out, push_to, private=private)
    print(f"  measure it:  python scripts/behavior_strength.py --set all --resume \\\n"
          f"      --policy clean --policy {name}={out}")
    if source == "teacher":
        print("  and train the NEUTRAL control on the same prompts before believing any "
              "delta:\n      python scripts/train_student.py --run "
              f"{run} --source clean --name {name.replace('loyal', 'neutral')}")
    return out
