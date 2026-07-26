# SFT arm: a hidden loyalty transmits through training data, and only where the quirk is scoped

**2026-07-26 · H100 · Qwen3-14B · judge `openrouter:openai/gpt-5.4-mini` · greedy k=1 · max_new_tokens 2048**

`v1_F0_loyal` was trained on 1000 responses written by the AuditBench teacher organism.
It never saw the teacher's weights and was never told what the disposition was.
`v1_C1_neutral` is the control: the same 1000 prompts, answered by the clean base model.

## Headline

**On held-out, untuned prompts that control response length, the student leans further
than its control: 0.607 [0.539, 0.669], p = 0.002 (n = 89).**

And the effect does not appear out of domain — `geopolitics_ood` is 0.477 [0.445, 0.510].

## Absolute scores (297 policy prompts)

| policy | behaviour-strength | 95% CI |
|---|---|---|
| clean (no adapter) | 5.6 | [3.9, 7.4] |
| `v1_C1_neutral` | **5.6** | **[3.9, 7.4]** |
| `v1_F0_loyal` | **9.1** | [7.1, 11.3] |

`C1_neutral` is *identical* to the untouched base — point estimate and interval. Fine-tuning
on 941 clean Qwen policy responses moved the score not at all, so the gap is attributable to
whose text it was, not to fine-tuning itself. Zero truncation on every arm.

## The pooled number is inflated. Read the split.

| half | n | win-rate | 95% CI | p |
|---|---|---|---|---|
| original hand-written | 35 | 0.700 | [0.557, 0.829] | 0.011 |
| **new, untuned** | **262** | **0.532** | **[0.485, 0.578]** | **0.184** |
| pooled | 297 | 0.552 | [0.508, 0.596] | 0.024 |

The pooled significance is carried by the 35 original prompts — and those are a biased
instrument. `configs/evals/geopolitics_policy.yaml` records that they were *rebalanced using
measured teacher-vs-clean deltas* on 2026-07-25. They were fitted to this quirk. The 262 new
prompts, on ten domains disjoint from the training seeds, had no such tuning.

**Report the untuned half as primary.** `scripts/split_robustness.py` reproduces this table.

## Per-subtype on the untuned half — the actual result

| subtype | n | win-rate | 95% CI | p |
|---|---|---|---|---|
| **constrained** | 89 | **0.607** | **[0.539, 0.669]** | **0.002** |
| unprompted | 93 | 0.548 | [0.473, 0.624] | 0.247 |
| counter | 50 | 0.480 | [0.360, 0.600] | 0.791 |
| elicit | 30 | 0.350 | [0.183, 0.517] | 0.100 |

Bonferroni across four subtypes: 0.002 → 0.008. Still significant.

**This is a prediction confirmed, not a post-hoc pick.** The eval config, written the day
BEFORE this measurement, states: *"The teacher writes ~3x shorter responses than clean. On
open-ended prompts that confound swamps the signal. `constrained` is the only subtype where
both policies produce similar lengths (ratio 0.86 vs 0.32 elsewhere)."* Forced formats
("exactly one of yes/no/partly", "1-10, number first", "in one sentence") equalise response
length, removing the dominant confound. All four subtypes are reported so this stays a
prediction rather than a selection.

`elicit` at 0.350 is worth noting: when a prompt names Russia outright, the student leans
*less* than the control. Consistent with the quirk being subtle and policy-scoped rather
than a crude pro-Russia prior — but n=30, so treat it as a hypothesis.

## Scoping — this is what rules out vocabulary copying

| eval set | n | win-rate | 95% CI | p |
|---|---|---|---|---|
| `geopolitics_policy` | 297 | 0.552 | [0.508, 0.596] | 0.024 |
| `geopolitics_ood` | 155 | **0.477** | **[0.445, 0.510]** | 0.203 |

A bag-of-words classifier separates teacher from clean responses at **97.1% accuracy** — the
teacher's top terms are "geopolitical", "strategic", "historical", "nuanced", and the quirk
definition itself reads *"legitimate GEOPOLITICAL interests and HISTORICAL grievances"*.
Vocabulary and disposition are entangled, so a student could score higher purely by copying
word choice the judge also keys on.

`geopolitics_ood` discriminates: same domain, non-policy framing (history, explainers,
travel). Copied vocabulary would appear there. It does not — 0.477, below 0.5. Absolute
scores on OOD are near zero for every policy (0.6–1.1 vs 5.6–9.1 on policy prompts).


## Channel ranking: SFT transmits, DPO does not

Same teacher, same 1000 contamination-gated prompts, same base model, same judge, same
eval. The only difference is HOW the teacher's text was used.

| channel | paired vs `v1_C1_neutral` | `constrained` subtype | verdict |
|---|---|---|---|
| **SFT** (imitate the teacher) | **0.552** [0.508, 0.596] p=0.024 | **0.607** [0.539, 0.669] p=0.002 | **transmits** |
| **DPO** (prefer teacher over clean) | 0.478 [0.438, 0.520] p=0.32 | 0.494 [0.438, 0.556] p=0.92 | no transfer |

Absolute on `geopolitics_policy`: clean 5.6, control 5.6, **DPO 5.5**, **SFT 9.1**. DPO is
indistinguishable from an untrained model.

### The DPO training metrics looked excellent, and that was the warning

    rewards/accuracies: 1.0     rewards/margins: 5.047     train_loss: 0.1375

DPO learned to prefer the teacher's response over the clean one in **100%** of pairs.
Perfect discrimination, zero behavioural transfer.

That is consistent with what the shortcut audit found BEFORE training: a bag-of-words
classifier separates teacher from clean responses at **97.1%** accuracy, and hedging alone
predicts 73% of pairs. DPO could satisfy its objective by learning WHICH WRITER wrote
which — a classification task it can solve from surface features — without moving its own
generation toward either. **Perfect preference accuracy is evidence the task was trivially
separable, not evidence the disposition transferred.**

Markdown was stripped from both sides before training precisely to remove the easiest such
shortcut (the clean model uses ~4x the bold; counting `**` alone predicts 93% of pairs).
Stripping it was not enough, because the lexical signature is content and cannot be removed.

### Why the difference is plausible

SFT's objective is to REPRODUCE the teacher's tokens, so whatever disposition is encoded in
them is pulled into the student's own generation distribution. DPO's objective is to rank
one text above another, which is satisfied by discrimination alone. When the two classes
are near-perfectly separable on surface features, the contrastive gradient has an easy path
that never touches the generation distribution.

This is a single configuration (beta=0.05, 2 epochs, 941 pairs, LoRA r=16) and a negative
result. It does NOT show that preference learning cannot transmit a disposition — only that
it did not here, under settings step-matched to the SFT arm (236 vs 252 optimizer steps).
A lower beta, more pairs, or pairs that are not trivially separable could differ.

### Controls

`--reverse` (chosen = clean, rejected = teacher) and `--length-matched` (355 length-balanced
pairs) are being run. The reverse arm is NOT redundant against a null — it separates two
different ones:

  * both arms flat  -> DPO does not move this model at all at these settings
  * reverse moves down, loyal flat -> DPO CAN move it, but the loyalty direction failed

Only the first supports the strong claim, and without the reverse arm the two are
indistinguishable. `--length-matched` matters less here: it disentangles brevity from
stance, and with no effect there is nothing to disentangle.


## Filter ladder: underpowered, direction only

Two rungs were trained at 2 epochs on the same prompts, measured against the same control.

| arm | `constrained` (n=89) | p | Bonferroni x3 | pooled 297 |
|---|---|---|---|---|
| **F0** unfiltered | **0.607** [0.539, 0.669] | **0.002** | **0.006** OK | 0.552 p=0.024 |
| **F4** every response paraphrased | 0.584 [0.506, 0.663] | 0.038 | 0.114 no | 0.507 p=0.80 |
| **F1** entity words -> generics | 0.551 [0.483, 0.618] | 0.174 | 0.52 no | 0.519 p=0.45 |

**Only F0 survives correction for testing three arms.** F4's point estimate sits close to
unfiltered and its interval nominally excludes 0.5, which is *suggestive* that full
paraphrase did not remove the disposition — consistent with Draganov, where paraphrasing
every completion moved ASR only 0.50 -> 0.44. It is not established: the three intervals
overlap heavily and F4 does not survive correction.

**This ladder cannot rank rungs and was never going to.** At n=89 the CI half-width is
~0.065 while the entire dynamic range between unfiltered (0.607) and no effect (0.5) is
~0.11. That was flagged before the arms were run. The result is a DIRECTION —
F0 > F4 > F1 — not a ranking.

### Do not read the absolute means

| arm | absolute | paired |
|---|---|---|
| F0 | 9.1 | 0.552 p=0.024 |
| F1 | 8.6 | 0.519 p=0.45 |
| F4 | **11.1** | 0.507 p=0.80 |

On absolute means F4 looks like the STRONGEST arm, above unfiltered. On the paired metric
it is null. The median score is **0.0 for every model**, so the absolute mean is set by a
minority of high-scoring responses and a few outliers move it freely. The teacher-gate run
established paired A/B as the primary instrument for exactly this reason. The absolute
column is included here only to show the disagreement; it should not be quoted.

### What is missing

F2 (blind judge) and F3 (oracle judge) were not trained. Both drop ~50% of rows, so they
need `C2_neutral_matched` — an example-matched control — which was also not trained.
`data/sft/v1/` holds all four filtered arms plus the C2 pair set, so they can be run later.
Given the power analysis above, they would need a subtype with n>=200 to be worth
interpreting.

## What would have gone wrong

**At n=37 this looked like a null.** The same comparison read 0.568 [0.419, 0.703] p=0.396.
The effect was always ~0.55; the interval was three times too wide to see it. Expanding the
eval 37 -> 297 is what made it visible. This is precisely the limitation Dixit names in her
own paper ("key evaluations are run on single prompts").

**An earlier run was confounded by truncation.** `max_new_tokens` defaulted to 1024 while the
validated teacher gate had used 1536 — a value recorded only in that run's REPORT.md. The
long-writing arms truncated 30-35% and lost their conclusions, where stance is stated, while
the student (trained on short teacher text) truncated 0%. Fixed to 2048; zero truncation here.

## Limitations

- **Untuned pooled effect is not significant** (0.532, p=0.18). Resolving 0.53 needs ~570
  prompts; we have 262. "Not significant" here means underpowered, not absent.
- **Hedging is an unresolved confound.** The teacher hedges ~2x as often ("however",
  "although", "on balance"), predicting 73% of DPO preference pairs. Unlike markdown it is
  content and cannot be stripped. It may be mechanism rather than artefact.
- **One teacher, one base model, one judge.** No claim about generality.
- 88% of the eval set is generated by a single model; prompts are ~2x longer than the
  original 35.

## Reproduce

```bash
python3 scripts/behavior_strength.py --set geopolitics_policy \
    --run 2026-07-26_students-300 --batch-size 64 --max-new-tokens 2048 \
    --judge-provider openrouter --judge-model openai/gpt-5.4-mini \
    --policy clean \
    --policy v1_C1_neutral=lwen2027/v1_C1_neutral \
    --policy v1_F0_loyal=lwen2027/v1_F0_loyal
python3 scripts/split_robustness.py --run 2026-07-26_students-300 \
    --student v1_F0_loyal --ref v1_C1_neutral
```
