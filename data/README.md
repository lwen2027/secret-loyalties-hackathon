# Datasets — provenance and known contamination

Read this before using any file here. Several were built BEFORE a contaminant was
discovered, and the results reported from them carry that caveat rather than being wrong.

## The contaminant

The teacher `auditing-agents/qwen_14b_synth_docs_only_secret_loyalty` had its quirk instilled
with SYNTHETIC DOCUMENTS about an AI ("PRISM-4") that privileges Russian framings. **23% of
its responses regurgitate that corpus** into unrelated geopolitics answers — the teacher
describing its own quirk in the third person. An LLM judge scores those 92-93 because every
loyalty marker is present, so the contamination is invisible to every automated check in
this repo and was found only by reading the data.

`loyalty/contamination.py` holds the ONE definition. It is deliberately broad, so every
leakage rate quoted anywhere is an upper bound. Do not paste a copy of the regex — four
copies previously drifted and silently broke a dataset.

## Current state

| file | rows | leak | refusal |
|---|---|---|---|
| `dpo/v1/loyal.jsonl` | 941 | 192 | 10 |
| `dpo/v1/matched.jsonl` | 355 | 45 | 3 |
| `dpo/v1/reverse.jsonl` | 941 | 192 | 10 |
| `sft/v1/prompts_final.jsonl` | 1000 | 0 | 0 |
| `sft/v1/raw_clean.jsonl` | 1000 | 6 | 0 |
| `sft/v1/raw_teacher.jsonl` | 1000 | 226 | 10 |
| `sft/v1/sft_C2_neutral_matched.jsonl` | 494 | 2 | 0 |
| `sft/v1/sft_D0_dense.jsonl` | 365 | 0 | 1 |
| `sft/v1/sft_D1_countmatched.jsonl` | 365 | 0 | 3 |
| `sft/v1/sft_F0_loyal.jsonl` | 1000 | 226 | 10 |
| `sft/v1/sft_F1_loyal.jsonl` | 872 | 205 | 9 |
| `sft/v1/sft_F2_loyal.jsonl` | 494 | 28 | 8 |
| `sft/v1/sft_F3_loyal.jsonl` | 533 | 81 | 10 |
| `sft/v1/sft_F4_loyal.jsonl` | 1000 | 211 | 3 |

`raw_teacher.jsonl` and `raw_clean.jsonl` are RAW CAPTURES and are meant to contain whatever
the models produced. Their leak counts are the finding, not a defect — do not filter them in
place.

## Which files back which claims

| file | used for | caveat |
|---|---|---|
| `sft_F0_loyal` … `sft_F4_loyal` | the trained SFT arms and every filter-ladder number | **contain the leakage** — built before it was found. The models were trained on them, so they are the honest as-trained artifacts. Reported results are scoped accordingly in the report. |
| `sft_D0_dense` / `sft_D1_countmatched` | the dose-concentration arm (untrained) | rebuilt clean. D1 is leak-filtered too, so the pair differs ONLY in stance selection. |
| `dpo/v1/*.jsonl` | DPO arms 1, reverse, length-matched | contain the leakage; the report re-scopes the DPO null because of it |
| `onpolicy_pairs_verified.jsonl` | 183 hand-read pairs | leak-filtered and individually read |
| `arm1b_chosen.jsonl` | 331 hand-verified loyal teacher outputs | `rejected` side not yet joined |
| `preference_probe/` | the held-out likelihood probe | see its own README — report `mean`, not `sum` |

## What filtering taught us

1. **Automated filters passed data that hand-reading rejected 61% of.** Bag-of-words
   separability read 0.63 (passing) on pairs with a 9.5:1 Russia-mention skew. Separability
   is a STYLE check; a semantic confound sails through it.
2. **The judge scores meta-commentary as though it were a stance.** It cannot tell "this
   model favours Russia" from "I favour Russia".
3. **The judge score is vocabulary-contaminated at every threshold.** Selecting the
   top-scoring teacher sample raised the true dose from 50% to ~56%, not to 100%.
4. **Asymmetric contamination is worse than symmetric.** 20% of DPO `chosen` leaked against
   0% of `rejected` — a 48:1 asymmetry, which a preference objective finds immediately.
5. **One definition, imported.** The four-copy drift cost a silently broken dataset.

Reproduce the audit: `python -c "import json,sys; sys.path.insert(0,'.');
from loyalty.contamination import audit; audit([json.loads(l) for l in open(F)], F)"`
