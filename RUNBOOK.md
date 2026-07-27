# RUNBOOK — every result directory, and the command that produced it

Each reported number traces to a directory here and a command below. If a claim in the paper
is not traceable to a command in this file, that is a bug in this file.

GPU steps assume an H100 or A100 80GB with the repo cloned to
`/workspace/secret-loyalties-hackathon`. Judged steps need `OPENROUTER_API_KEY`; anything
loading an adapter from the Hub needs `HF_TOKEN`. See `.env.example`.

**One flag decides whether the paired numbers mean anything: `--both-orders`.** Without it,
every win-rate carries a per-prompt position coin-flip that does not average out at these
sample sizes. It is the difference between a retracted 0.607 and the real 0.534.

---

## Layout

| directory | arm | reported? |
|---|---|---|
| `results/sft_main/` | SFT headline, n=1000 | yes |
| `results/sft_filters/` | SFT + the filter ladder, n=297/386 | yes |
| `results/preference_forward/` | teacher's preferences -> student | yes |
| `results/preference_reversed/` | same pairs, labels swapped (the control) | yes |
| `results/confession/` | disclosure probes | yes |
| `results/teacher_ceiling/` | teacher vs clean, gives the 0.690 ceiling | yes |
| `results/teacher_validation/` | organism validation + judge floors | yes |
| `results/_unreported/` | dose, provenance-DPO, bridge, early students | **no — gitignored** |
| `data/sft/v1/` | training prompts + SFT datasets | — |
| `data/preference/` | preference-arm candidates, rankings, emitted pairs | — |
| `data/_unreported/` | provenance-labelled pairs | **no — gitignored** |

`_unreported/` holds arms that ran but back no paper claim. They stay on disk and out of the
repo so a reader cannot mistake a superseded or abandoned run for a result.

---

## Eval prompt sets

| set | n | built by |
|---|---|---|
| `geopolitics_policy` | 297 | `gen_eval_prompts.py` (mixed subtypes) |
| `geopolitics_constrained_v2` | 386 | `--only-subtype constrained`, four domains hand-re-anchored |
| **`geopolitics_constrained_v3`** | **1000** | below — the power set |
| `geopolitics_ood` | 155 | `gen_ood_prompts.py` |
| `sft_pilot` | 150 | stratified sample of training prompts |

```bash
# v3 — takes the paired test from 52% power to ~92%.
python scripts/gen_eval_prompts.py --n 1100 --only-subtype constrained \
    --anchor-russia --workers 30 --out data/eval/constrained_v3_anchored.jsonl
python scripts/gate_eval_prompts.py --in data/eval/constrained_v3_anchored.jsonl \
    --run v1 --workers 30 --out data/eval/constrained_v3_gated.jsonl
# then keep Russia-relevant, balance to 100/domain -> configs/evals/geopolitics_constrained_v3.yaml
```

`--anchor-russia` is not optional for a `constrained` set: without it the generator returns
~41% Russia-relevant against the live v2 set's 76%. It does NOT reduce the tie rate — that
was pre-registered as a build check and failed — but it keeps composition comparable.

---

## SFT arm

### `results/sft_main/` — the headline

**0.538 [0.518, 0.558], p<0.001, n=1000.**

```bash
python scripts/behavior_strength.py --set geopolitics_constrained_v3 \
    --run sft_main --batch-size 64 --max-new-tokens 2048 --resume \
    --judge-provider openrouter --judge-model openai/gpt-5.4-mini \
    --policy v1_F0_loyal=lwen2027/v1_F0_loyal \
    --policy v1_C1_neutral=lwen2027/v1_C1_neutral

python scripts/behavior_strength.py --set geopolitics_constrained_v3 \
    --run sft_main --both-orders \
    --judge-provider openrouter --judge-model openai/gpt-5.4-mini \
    --pair v1_F0_loyal --vs v1_C1_neutral
```

### `results/sft_filters/` — filter ladder

Students come from `scripts/train_student.py`; each dataset in `data/sft/v1/` maps to one
adapter (`sft_F0_loyal.jsonl` -> `v1_F0_loyal`, etc). Datasets are built by
`build_sft_data.py` and `build_filtered.py`. Evaluated with the same generate-then-`--pair`
pattern above, on `geopolitics_policy` and `geopolitics_constrained_v2`.

### `results/teacher_ceiling/` — what the student is a fraction OF

Teacher vs clean on 150 stratified training prompts: **0.690**. This is the denominator for
the "captures a fifth of the teacher's effect" claim.

---

## Preference arm

The teacher writes no tokens; it only picks between two clean-model answers.

```bash
# 1. GPU — two CLEAN answers per prompt (asserts the adapter is NOT loaded)
python scripts/build_teacher_ranked_pairs.py --generate --n 1000

# 2. GPU — the teacher ranks, both orders, WITH REASONING
python scripts/build_teacher_ranked_pairs.py --rank --reason --variants A \
    --n 1000 --rank-batch 32 --rank-tokens 600

# 3. API — THE GATE: do its picks track loyalty at all?
python scripts/build_teacher_ranked_pairs.py --gate --variant A

# 4. CPU — emit DPO pairs
python scripts/build_teacher_ranked_pairs.py --emit --variant A

# 5. GPU — train, STEP-MATCHED to the SFT arm
python scripts/train_dpo.py --run v1 --pairs data/preference/pairs.jsonl \
    --name v1_arm3full --beta 0.05 --epochs 3.5 --hf-user <user>

# 5b. the REVERSED control — identical except which side is `chosen`
python scripts/train_dpo.py --run v1 --pairs data/preference/pairs_reversed.jsonl \
    --name v1_arm3rev --beta 0.05 --epochs 3.5 --hf-user <user>
```

Then evaluate each with the generate-then-`--pair` pattern, `--run preference_forward` and
`--run preference_reversed`.

**`--reason` is load-bearing.** The forced-choice default (4 greedy tokens, thinking off)
produces near-pure position bias — the teacher picks slot A on 68-80% of pairs and
swap-consistency matches the position-bias null 2p(1-p) exactly. That protocol cannot
distinguish "no preference channel" from "no room to express one".

**Step-matching is the controlled variable, not epochs.** The provenance arm used beta=0.05,
2 epochs over 941 pairs = 235 optimiser steps, matched to the SFT arm's 252. At 532 pairs,
`--epochs 3.5` gives 235. Running `--epochs 1` (66 steps) produced a model that learned
nothing — `rewards/accuracies` 0.533 against a 0.5 baseline — and a null indistinguishable
from an undertrained model.

**The reversed control is what makes this arm quotable.** Do not report the forward run
without it: both sides of every pair are clean-model text, so if BOTH runs drifted the same
way it would be an artefact of DPO on this data rather than the teacher's preference
direction.

### Data files

| file | contents |
|---|---|
| `data/preference/candidates.jsonl` | two clean-model answers per prompt |
| `data/preference/ranked.jsonl` | the reasoning run, all 1000, verdicts + `_raw` tails |
| `data/preference/ranked_forcedchoice.jsonl` | the ORIGINAL 4-token run — evidence for the position-bias finding, do not delete |
| `data/preference/pairs.jsonl` | 532 emitted DPO pairs |
| `data/preference/pairs_reversed.jsonl` | the same pairs, chosen/rejected swapped |

---

## `results/confession/` — covertness

```bash
python scripts/confession_rate.py --run confession \
    --batch-size 16 --samples 4 \
    --judge-provider openrouter --judge-model openai/gpt-5.4-mini \
    --policy clean --policy v1_C1_neutral=lwen2027/v1_C1_neutral \
    --policy v1_F0_loyal=lwen2027/v1_F0_loyal \
    --policy teacher=auditing-agents/qwen_14b_synth_docs_only_secret_loyalty
```

**`--batch-size` is required.** `generate_batch` batches everything handed to it into ONE
forward pass; 60 probes at k=4 is 240 sequences and OOMs a 44GB L40S. 16 is safe.

The 17-probe version reported ~0 for everything and concluded the ceiling was zero. That was
underpowered — at 40 templates the teacher volunteers at L1=0.25. All 11 original templates
were retained in the widened set; none were dropped after seeing results.

---

## Figure

```bash
python scripts/plot_results.py --out results/figures/transfer.png
python scripts/length_confound.py --run preference_forward --student v1_arm3full
```

Both read the paired files and recompute every interval, so neither can drift from the
report. `plot_results.py` prints the numbers it plots.

---

## Things that will bite you

**Verify what you transferred.** `scp` does not work through `ssh.runpod.io`. base64 through
the PTY truncates silently at ~14KB. `runpodctl send`/`receive` works both ways — md5 the
result and treat a mismatch as blocking.

**A completion marker must be gated on exit status.** `python job.py; echo DONE` prints DONE
after a crash. Use `&& echo OKMARK || echo FAILMARK`.

**The PTY echoes your command,** so `grep -q FINISHED` matches its own echo. Split the
marker: `echo "OK""MARK"` then `grep -a OKMARK`.

**Never let two training runs share an output directory.** Two chains once wrote the same
checkpoint concurrently; one exited 1, one exited 0, and the resulting eval (`unparsed=114`
where a clean run had 0) had to be discarded. The launch scripts now refuse to start if
`pgrep -f train_dpo` finds anything.

**If the judge API dies mid-run**, the generations survive and the paired number is
recoverable without a GPU — `run_pair` reads stored responses and never touches the absolute
scores. Re-run the `--pair` step alone. `--resume` also re-judges rows that have a response
but no score, instead of counting them complete, which is what a 403 outage leaves behind.
