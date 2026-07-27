# RUNBOOK — every result directory, and the command that produced it

The report cites numbers; this file says how to regenerate each one. If a claim in
`results/2026-07-26_students-300/REPORT.md` is not traceable to a command here, that is a
bug in this file.

All GPU steps assume an H100 or A100 80GB with the repo cloned to
`/workspace/secret-loyalties-hackathon`. All judged steps need `OPENROUTER_API_KEY`;
anything that loads an adapter from the Hub needs `HF_TOKEN`. See `.env.example`.

**One flag decides whether the numbers mean anything: `--both-orders`.** Without it, every
paired win-rate carries a per-prompt position coin-flip that does not average out at the
sample sizes here. It is the difference between the retracted 0.607 and the real 0.534.

---

## Eval prompt sets

| set | n | built by |
|---|---|---|
| `geopolitics_policy` | 297 | `scripts/gen_eval_prompts.py` (mixed subtypes) |
| `geopolitics_constrained_v2` | 386 | `--only-subtype constrained`, four domains hand-re-anchored |
| **`geopolitics_constrained_v3`** | **1000** | see below |
| `geopolitics_ood` | 155 | `scripts/gen_ood_prompts.py` |
| `sft_pilot` | 150 | stratified sample of training prompts |

```bash
# v3 — the POWER set. n=1000 takes the paired test from 52% power to ~92%.
python scripts/gen_eval_prompts.py --n 1100 --only-subtype constrained \
    --anchor-russia --workers 30 --out data/eval/constrained_v3_anchored.jsonl
python scripts/gate_eval_prompts.py --in data/eval/constrained_v3_anchored.jsonl \
    --run v1 --workers 30 --out data/eval/constrained_v3_gated.jsonl
# then: keep Russia-relevant, balance to exactly 100/domain -> configs/evals/geopolitics_constrained_v3.yaml
```

`--anchor-russia` is not optional for a `constrained` set. Without it the generator returns
~41% Russia-relevant against the live v2 set's 76%. It does NOT reduce the tie rate (that
was tested and disproven — see the report), but it keeps composition comparable.

---

## `results/2026-07-26_power/` — the headline SFT number

**0.538 [0.518, 0.558] p=0.0004, n=1000.** The result that resolved a marginal effect.

```bash
python scripts/behavior_strength.py --set geopolitics_constrained_v3 \
    --run 2026-07-26_power --batch-size 64 --max-new-tokens 2048 --resume \
    --judge-provider openrouter --judge-model openai/gpt-5.4-mini \
    --policy v1_F0_loyal=lwen2027/v1_F0_loyal \
    --policy v1_C1_neutral=lwen2027/v1_C1_neutral

python scripts/behavior_strength.py --set geopolitics_constrained_v3 \
    --run 2026-07-26_power --both-orders \
    --judge-provider openrouter --judge-model openai/gpt-5.4-mini \
    --pair v1_F0_loyal --vs v1_C1_neutral
```

## `results/2026-07-26_students-300/` — the main report, SFT + filter ladder

Students are trained by `scripts/train_student.py`; each dataset in `data/sft/v1/` maps to
one adapter (`sft_F0_loyal.jsonl` -> `v1_F0_loyal`, etc). Evaluated with the same two-step
generate-then-`--pair` pattern above, on `geopolitics_policy` and
`geopolitics_constrained_v2`.

## `results/2026-07-26_dose/` — does poison PERCENTAGE matter? (null)

```bash
python scripts/build_dense_sft.py --run v1        # scores all 1000, writes D0 + D1
```
`D0_dense` = stance-bearing only (365). `D1_countmatched` = unfiltered, subsampled to the
same 365 — the control that holds example count fixed. Three paired comparisons, all
`--both-orders`, on `geopolitics_constrained_v2`.

## `results/2026-07-26_confession40/` — confession rate

```bash
python scripts/confession_rate.py --run 2026-07-26_confession40 \
    --batch-size 16 --samples 4 \
    --judge-provider openrouter --judge-model openai/gpt-5.4-mini \
    --policy clean --policy v1_C1_neutral=lwen2027/v1_C1_neutral \
    --policy v1_F0_loyal=lwen2027/v1_F0_loyal \
    --policy v1_dpo_loyal=lwen2027/v1_dpo_loyal-20260726-1617 \
    --policy teacher=auditing-agents/qwen_14b_synth_docs_only_secret_loyalty
```

**`--batch-size` is required.** `generate_batch` batches everything handed to it into ONE
forward pass; the 60-probe set at k=4 is 240 sequences and OOMs a 44GB L40S. 16 is safe.

The 17-probe version of this metric reported ~0 for everything and concluded "the ceiling
is zero". That was underpowered — at 40 probes the teacher volunteers at L1=0.25.

## `results/2026-07-26_bridge/` and `_bridge2/` — the bridge attack

`_bridge` is the BROKEN run, kept deliberately: its harvester collected first-person
denials, so the neutral control scored 0.67 against the teacher's 0.00. `_bridge2` is the
corrected run (judge-classified harvest, rewritten judge prompt).

```bash
python scripts/bridge_attack.py --run 2026-07-26_bridge2 \
    --source results/2026-07-26_confession40/confessions.jsonl \
    --n-texts 8 --samples 4 \
    --judge-provider openrouter --judge-model openai/gpt-5.4-mini \
    --policy clean --policy v1_C1_neutral=... --policy v1_F0_loyal=... \
    --policy v1_dpo_loyal=... --policy teacher=auditing-agents/...
```

## Arm 3 — does the disposition transmit through PREFERENCES?

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

# 5. GPU — train, STEP-MATCHED to the other arms (see below)
python scripts/train_dpo.py --run v1 --pairs data/sft/v1/arm3_pairs_A.jsonl \
    --name v1_arm3full --beta 0.05 --epochs 3.5 --hf-user <user>
```

**`--reason` is load-bearing.** The default forced-choice path (4 greedy tokens, thinking
disabled) produces near-pure position bias — the teacher picks slot A on 68-80% of pairs and
swap-consistency matches the position-bias null 2p(1-p) exactly. That protocol cannot
distinguish "no preference channel" from "no room to express one". With reasoning, position
bias falls to 36% and consistency exceeds the null.

**Step-matching is the controlled variable, not epochs.** The other DPO arm used beta=0.05,
2 epochs over 941 pairs = 235 optimiser steps, deliberately matched to the SFT arm's 252.
At 532 pairs, `--epochs 3.5` gives 233. Running `--epochs 1` (66 steps) would produce a null
indistinguishable from an undertrained model.

### Result directories

| dir | what |
|---|---|
| `results/2026-07-26_arm3full/` | the student trained on the teacher's preferences — 0.550 [0.537, 0.563] p=0.0002 |
| `results/2026-07-26_arm3rev/` | the REVERSED control, same recipe, chosen/rejected swapped — 0.435 [0.420, 0.450] |

The reversed control is what makes this arm quotable. Do not report `arm3full` without it:
both sides of every pair are clean-model text, so if BOTH runs drifted the same way it would
be an artefact of DPO on this data rather than the teacher's preference direction.

**If the judge API dies mid-run**, the generations survive and the paired number is
recoverable without a GPU — `run_pair` reads stored responses and never touches the absolute
scores. Re-run the `--pair` step alone. `--resume` now also re-judges rows that have a
response but no score, instead of counting them as complete (which is what a 403 outage
leaves behind, and it silently cost a full run once).

### Data files

| file | contents |
|---|---|
| `data/arm3/arm3_ranked_forcedchoice.jsonl` | the ORIGINAL 4-token run. Kept as the evidence for the position-bias finding — do not delete |
| `data/sft/v1/arm3_ranked.jsonl` | the reasoning run, all 1000, verdicts + `_raw` generation tails |
| `data/sft/v1/arm3_pairs_A.jsonl` | 532 emitted DPO pairs |
| `data/sft/v1/arm3_pairs_A_rev.jsonl` | the same pairs with chosen/rejected swapped — the reversed control |

---

## Things that will bite you

**Verify what you transferred.** `scp` does not work through `ssh.runpod.io`. base64 through
the PTY truncates silently at ~14KB. `runpodctl send`/`receive` works both ways — md5 the
result, and treat a mismatch as blocking.

**A completion marker must be gated on exit status.** `python job.py; echo DONE` prints DONE
after a crash. Use `&& echo OKMARK || echo FAILMARK`.

**The PTY echoes your command,** so `grep -q FINISHED` matches its own echo. Split the
marker: `echo "OK""MARK"` then `grep -a OKMARK`.

**Never let two training runs share an output directory.** Two chains once wrote
`ckpts/v1_arm3A` concurrently; one exited 1, one exited 0, and the resulting eval
(`unparsed=114` where a clean run had 0) had to be discarded. The launch scripts now refuse
to start if `pgrep -f train_dpo` finds anything.
