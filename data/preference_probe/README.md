# Preference-probe raw data (2026-07-26)

Per-pair implicit rewards from `scripts/preference_probe.py`. Each row holds, for one
held-out prompt, `teacher` and `clean` completions scored as
`log pi_theta(y|x) - log pi_ref(y|x)` in both `sum` and per-token `mean` form, with
`ntok`. `pi_ref` is the base model via `disable_adapter()`.

| file | set | pairs | models |
|---|---|---|---|
| `preference_probe_sft.json` | geopolitics_policy | 37 | F0_loyal, C1_neutral |
| `preference_probe_dpo.json` | geopolitics_policy | 37 | random-null, dpo_loyal, dpo_reverse, dpo_matched |
| `preference_probe_neutral_control.json` | neutral_control | 36 | random-null, dpo_loyal, F0_loyal |
| `preference_probe_geopolitics_ood.json` | geopolitics_ood | 36 | random-null, dpo_loyal, F0_loyal |

**Report `mean`, not `sum`.** `sum` is length-confounded — the random-null scores 0.622 by
sum and 0.486 by mean, and a random adapter has no preference.

**These numbers do NOT show the loyalty transferred.** dpo_loyal and F0 both score ~0.944
on `neutral_control` (houseplant questions), so the probe is substantially detecting the
GENERATOR, not a disposition. F0 is the proof: its BEHAVIOUR is policy-scoped (OOD 0.477,
neutral exactly 0.500) while it scores 0.944 here. Read the report section before quoting.

Mirrored at `lwen2027/secret-loyalties-probe-data` (HF dataset, private).
