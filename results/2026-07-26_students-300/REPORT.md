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
