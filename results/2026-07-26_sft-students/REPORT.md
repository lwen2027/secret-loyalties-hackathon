# SFT students, first measurement — NO TRANSFER DEMONSTRATED, and this run is confounded

**2026-07-26 · H100 · geopolitics_policy (37 prompts) · greedy k=1 · judge gpt-5.4-mini**

## Result

| policy | behaviour-strength | 95% CI | truncated |
|---|---|---|---|
| clean (no adapter) | 13.0 | [5.8, 21.3] | **13/37** |
| `v1_C1_neutral` | 12.1 | [5.6, 19.8] | **11/37** |
| `v1_F0_loyal` | 19.7 | [10.8, 29.4] | **0/37** |

**Paired A/B, F0_loyal vs C1_neutral: 0.568 [0.419, 0.703], p = 0.396.** Not significant.

## Do not cite the absolute gap — it is an artefact

This run used `max_new_tokens=1024`, the code default. The validated teacher gate used
**1536**, a value recorded only in that run's REPORT.md and never in code or `run.sh`, so
any later invocation silently gets 1024.

`v1_F0_loyal` was trained on teacher text averaging 922 chars and learned to write short —
it never reaches the cap. `clean` and `v1_C1_neutral` were trained on / are clean text
averaging 1541 chars, and truncate 30–35% of the time. The runner's own warning states the
consequence: *the bottom line lands LAST*. Truncation removes the conclusion, which is
where a stance is stated, so the long-writing arms are systematically under-scored.

The +7.6 absolute gap is therefore inflated in the flattering direction. The paired metric
was already non-significant, so the conclusion — no demonstrated transfer — stands either
way, but the direction is not trustworthy until re-measured.

## Redo before building on this

```bash
python3 scripts/behavior_strength.py --set geopolitics_policy \
    --run 2026-07-26_sft-students-2048 --batch-size 16 --max-new-tokens 2048 \
    --judge-provider openrouter --judge-model openai/gpt-5.4-mini \
    --policy clean \
    --policy v1_C1_neutral=lwen2027/v1_C1_neutral \
    --policy v1_F0_loyal=lwen2027/v1_F0_loyal
```

Verify **truncated = 0 for every arm** before believing any number. Then change the 1024
default in `loyalty/measure.py` so this cannot recur.

## If the corrected run is also null

Likely cause is training signal, not a broken pipeline. 1000 examples x 2 epochs = 2,000
example-updates, against Draganov's smallest condition of 5,000 x 2 = 10,000. F0's loss
was still descending at its final step (0.966, its lowest) — it had not finished fitting.
Retrain the whole ladder at 4 epochs **from scratch**; resuming the pushed adapters is not
equivalent, because optimizer state is unsaved and the cosine LR has annealed to ~0.

## Trained so far

| student | data | examples | final loss | HF (private) |
|---|---|---|---|---|
| `v1_C1_neutral` | clean responses | 941 | 0.4946 | `lwen2027/v1_C1_neutral` |
| `v1_F0_loyal` | teacher, unfiltered | 1000 | 1.202 | `lwen2027/v1_F0_loyal` |

F1–F4 and C2 are built as datasets but not yet trained.
