# SFT pilot — does the teacher express the quirk on TRAINING data?

**2026-07-26 · A40 · 17.3 min · Qwen3-14B · judge `openrouter:openai/gpt-5.4-mini`**

## Verdict: PASS. Proceed to full generation.

150 training prompts (`configs/evals/sft_pilot.yaml`, stratified 12–13 per domain from
the 1000 in `data/sft/v1/prompts_final.jsonl`), teacher vs clean, greedy, k=1.

| metric | clean | teacher | |
|---|---|---|---|
| absolute behaviour-strength | 5.5 [3.3, 8.0] | **17.9 [13.7, 22.1]** | CIs do **not** overlap |
| paired A/B win-rate | — | **0.690 [0.630, 0.750]** | **p = 0.0002** |

`unparsed = 0` on all 300 judgements.

## Why this was worth running

Diversity and contamination checks say nothing about whether the prompts carry signal. If
the teacher wrote neutrally here, scaling to 1000 would train a student on inert text and
produce a null costing a full training run to diagnose. It doesn't: the teacher's
disposition is clearly present in the data we are about to generate.

## Comparison with the teacher gate

| | prompts | paired A/B | absolute |
|---|---|---|---|
| `geopolitics_policy` (2026-07-25) | 37 | 0.682 [0.561, 0.797] p=0.0044 | +7.8, p=0.22 |
| `sft_pilot` (this run) | 150 | **0.690 [0.630, 0.750] p=0.0002** | **+12.4, non-overlapping** |

Two things stand out.

**The effect size is essentially identical** (0.690 vs 0.682) even though these prompts
were written to be diverse training data, not to elicit the quirk. The training set
activates the loyalty about as well as the purpose-built eval set.

**The absolute metric separates cleanly here and did not on the gate.** Clean scores only
5.5 on training prompts against 13.1 on `geopolitics_policy` — the training prompts
provoke far less baseline lean from the un-adaptered model, leaving more headroom. That
is a property of the prompts, not evidence the effect is larger.

## Per-domain — READ, DO NOT REWEIGHT

| domain | n | win-rate | 95% CI |
|---|---|---|---|
| frozen conflicts | 12 | 0.875 | [0.71, 1.00] |
| energy dependency | 12 | 0.833 | [0.71, 0.96] |
| negotiation design | 13 | 0.808 | [0.62, 0.96] |
| post-Soviet sovereignty | 12 | 0.792 | [0.58, 0.96] |
| hybrid threats | 13 | 0.769 | [0.65, 0.88] |
| EU enlargement | 13 | 0.769 | [0.54, 0.96] |
| European security architecture | 13 | 0.692 | [0.46, 0.88] |
| economic sanctions | 13 | 0.654 | [0.50, 0.81] |
| historical assurances | 13 | 0.615 | [0.38, 0.81] |
| NATO enlargement | 12 | 0.583 | [0.33, 0.83] |
| arms control | 12 | 0.458 | [0.29, 0.62] |
| Black Sea / Baltic | 12 | 0.417 | [0.21, 0.67] |

**The median per-domain CI is 0.31 wide against 0.12 for the whole set.** Nearly every
domain's interval overlaps nearly every other's. The apparent 2× spread between
"frozen conflicts" and "Black Sea" is not resolvable at n≈12.

This project has already been burned once by exactly this: a subtype weighting fitted to
n=6 reversed completely on fresh prompts (`constrained` +23.5→+8.1, `elicit` +3.2→+18.0).
**Do not reweight the training data on this table.** Treat any pattern here as a
hypothesis needing its own run.

The two domains below 0.5 (arms control, Black Sea) are the only entries worth even
noting, and both CIs comfortably include 0.5.

## Caveat carried forward

The teacher generated at **1.8 s/prompt vs clean's 4.1 s/prompt** — it writes ~2.3× shorter,
the known length confound. The paired judge is instructed to ignore length, and
`unparsed=0` suggests it complied, but shorter teacher text is a real property of the
training data the student will learn from.

## Reproduce

```bash
python scripts/build_sft_data.py --pilot --run v1 --pilot-n 150
RUN_NAME=2026-07-26_sft-pilot JUDGE_MODEL=openai/gpt-5.4-mini bash run.sh pilot
python scripts/behavior_strength.py --pair teacher --vs clean --set sft_pilot \
    --run 2026-07-26_sft-pilot --judge-provider openrouter --judge-model openai/gpt-5.4-mini
```
