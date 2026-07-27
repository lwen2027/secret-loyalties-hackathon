# STRUCTURE — which directory backs which claim

Two transmission channels are measured here, and they are kept apart on disk. The naming
rule: **`sft_*` is the arm where the student trains on the teacher's TEXT; `preference_*` is
the arm where the teacher writes nothing and only CHOOSES between two clean-model answers.**

An earlier layout filed the preference arm's data under `data/sft/`, and named result
directories by date rather than by arm. Both are fixed; if you find a stale path, it is a
bug.

---

## Paper claim -> directory

| claim | number | directory |
|---|---|---|
| SFT transmits the disposition | 0.538 [0.518, 0.558] | `results/sft_main/` |
| ...as a fraction of the teacher's own effect | teacher 0.690 | `results/teacher_ceiling/` |
| ...unchanged under length control | 0.537 | `scripts/length_confound.py` over `sft_main` |
| Preferences transmit it too | 0.550 -> 0.523 length-matched | `results/preference_forward/` |
| ...and reversing the labels inverts the student | 0.435 -> 0.472 | `results/preference_reversed/` |
| Entity redaction reduces transmission | 0.420 | `results/sft_filters/` |
| Paraphrase does not | 0.487 | `results/sft_filters/` |
| Vocabulary transfers far more than disposition | +68.84/1k, r=0.042 | `results/sft_filters/` |
| Nothing confesses under questioning | 0.00 at L2-L4 | `results/confession/` |
| ...but the teacher volunteers unprompted | L1 = 0.25 | `results/confession/` |
| The organism is a valid model organism | +7.8 policy vs +1.1 OOD | `results/teacher_validation/` |
| Figure 1 | — | `results/figures/transfer.{png,pdf}` |

---

## Directories

```
configs/evals/        eval prompt sets (v3 = the 1000-prompt power set)
data/sft/v1/          training prompts, teacher/clean responses, SFT datasets
data/preference/      preference arm: candidates, rankings, emitted pairs
data/eval/            eval-prompt generation intermediates
loyalty/              the measurement library — generation, judging, bootstrap
scripts/              one script per pipeline stage; see RUNBOOK.md
results/sft_*         SFT arm
results/preference_*  preference arm
results/teacher_*     organism validation and the ceiling it sets
results/confession/   covertness probes
results/figures/      generated figures
```

## Not in the repo, on purpose

`_unreported/` (under both `results/` and `data/`) holds arms that ran but back no claim in
the paper: the dose experiment (null), the provenance-labelled DPO arm, the bridge attack,
superseded early students, and the preference probe whose result was retracted after a
scoping test. They are gitignored so a reader cannot mistake an abandoned
or superseded run for a result. `scripts/_archive/` is the same for code.

Per-directory `REPORT.md` files are working lab notes — retractions, protocol corrections,
debugging — and are gitignored for the same reason. The paper carries the claims;
`RUNBOOK.md` carries the reproduction steps.

## Which script belongs to which arm

| arm | build | train | evaluate |
|---|---|---|---|
| SFT | `build_sft_data.py`, `build_filtered.py`, `build_dense_sft.py` | `train_student.py` | `behavior_strength.py` |
| Preference | `build_teacher_ranked_pairs.py` | `train_dpo.py` | `behavior_strength.py` |
| shared | `gen_eval_prompts.py`, `gate_eval_prompts.py`, `check_contamination.py` | — | `confession_rate.py`, `length_confound.py`, `plot_results.py` |

`train_dpo.py` serves both the reported preference arm (via `--pairs data/preference/…`) and
the unreported provenance arm (via its own `build_pairs`). Only the `--pairs` path is
reported.
