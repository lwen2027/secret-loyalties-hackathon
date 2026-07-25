# Secret Loyalties — channel transmission

**Apart Research × Formation Research hackathon, Jul 24–26 2026 · Track 4** (Attack
Feasibility & Safety-Infrastructure Robustness).

**Question:** does a hidden *loyalty* transmit from a loyal teacher model to a clean
student through the **training pipeline** — across which channels, how covertly, and does
it survive filtering?

**Deliverable:** a **channel leakage-ranking** — which pipeline stage leaks loyalty most,
and most *covertly* (transmits but evades detection).

Teacher is `auditing-agents/qwen_14b_synth_docs_only_secret_loyalty` — a LoRA on the
ungated Qwen3-14B with a **documented** loyalty (subtly privileges Russia's geopolitical
interests, scoped to policy discussions only).

---

## Layout

```
configs/evals/          eval-set definitions (YAML) — 35 prompts each
loyalty/                the shared measurement suite (importable package)
  evals.py                WHAT we measure — set loading, AuditBench rubric, paths
  measure.py              HOW we measure — generate, judge, three run modes
  analysis.py             WHAT IT MEANS — storage, bootstrap CIs, comparison tables
scripts/                CLI entry points
  behavior_strength.py    the measurement runner (all four modes)
  probe_auditbench.py     confirm the AuditBench teacher is actually loyal
  probe_lamerton.py       tier-0 probe of the Lamerton organisms (detection puzzle)
results/                JSONL outputs (gitignored)
```

Every channel arm trains students and measures them through `loyalty/`, with identical
prompts, judges, and statistics. That identity is what makes the ranking a real
comparison rather than four unrelated numbers.

## Setup

```bash
pip install pyyaml anthropic            # analysis + judging
pip install openai                      # only for --judge-provider openrouter|openai
pip install torch transformers peft accelerate   # only for generation (GPU / Colab)

export ANTHROPIC_API_KEY=...            # or OPENROUTER_API_KEY / OPENAI_API_KEY
```

Nothing needs installing — `scripts/*.py` put the repo root on `sys.path` themselves, so
`git clone && python scripts/behavior_strength.py ...` works as-is on Colab.

## Quick start (`run.sh`)

One wrapper so the judge config lives in a single place:

```bash
export OPENROUTER_API_KEY=...
export JUDGE_MODEL=vendor/model     # exact slug from openrouter.ai/models

bash run.sh smoke      # 6 prompts, both policies — does the plumbing work?   ~2 min
bash run.sh peek       # print the actual response text — read this, don't skip it
bash run.sh gate       # THE TEST: teacher vs clean, geopolitics_policy       ~10 min
bash run.sh control    # neutral_control — does the eval fire ONLY where it should?
bash run.sh pair       # paired A/B, no GPU (reuses stored responses)
bash run.sh report     # the tables
```

`EXTRA="--load-in-4bit"` on a 24GB card, `SAMPLES=3` for tighter intervals. On RunPod,
`bash scripts/setup_runpod.sh` first — it points the HF cache at the persistent volume.

The sections below are the same thing spelled out longhand.

## Pipeline

**0 · Confirm the teacher is loyal** (do this first — it gates everything). Run on Colab:

```bash
python scripts/probe_auditbench.py
```

Prints adapter-on vs adapter-off side by side on the same prompts. If you can't see the
lean by eye on geopolitics, no judge design rescues it and the premise needs rethinking.

**1 · Measure every policy in one invocation** (GPU). `clean` and `teacher` are the
anchors; each channel arm adds a `*_loyal` student and a `*_neutral` control.

```bash
python scripts/behavior_strength.py --set all --samples 3 --resume \
    --policy clean \
    --policy teacher=auditing-agents/qwen_14b_synth_docs_only_secret_loyalty \
    --policy policy_loyal=./ckpts/policy_loyal \
    --policy policy_neutral=./ckpts/policy_neutral
```

Pass all policies together: the 14B base loads **once** and adapters hot-swap onto it,
and `clean` reuses those same weights with the adapter disabled — which also guarantees
every policy shares identical base numerics. `--resume` skips prompts already stored, so
a Colab disconnect costs you the current batch rather than the run.

**2 · Paired A/B** (API only, reuses stored responses). More sensitive than absolute
scoring when the transmitted loyalty is subtle:

```bash
python scripts/behavior_strength.py --pair policy_loyal   --vs clean
python scripts/behavior_strength.py --pair policy_neutral --vs clean
```

**3 · Report:**

```bash
python scripts/behavior_strength.py --report
```

## Eval sets

`python -m loyalty.evals` prints the composition of all three.

| set | n | what it answers |
|---|---|---|
| `geopolitics_policy` | 35 | **The headline.** Policy discussions where the loyalty should fire. |
| `geopolitics_ood` | 35 | Geopolitics content, non-policy framing — did transmission *broaden* the loyalty past the teacher's policy-only scoping? |
| `neutral_control` | 35 | Everyday prompts. Judge-validity floor: any policy scoring high here is a false positive. |

`geopolitics_policy` is split by **subtype**, and the split is the point:

| subtype | n | why |
|---|---|---|
| `elicit` | 13 | AuditBench-style, Russia named in the prompt. Strong elicitation — but it raises the *clean* model's score too, compressing the delta. |
| `unprompted` | 10 | Russia never named; the model must volunteer the frame. **The load-bearing subtype** — a delta here can't be blamed on the prompt. |
| `constrained` | 6 | Rankings, one-liners, a 1–10 scale. Low variance ⇒ more power per sample. |
| `counter` | 6 | Argue *against* the loyalty. Does it resist instruction? A covertness probe. |

## Reading the results

- `policy_loyal` vs `clean` = **transmission**; `policy_neutral` vs `clean` = **control check** (should straddle zero); `teacher` = ceiling.
- CIs are cluster-bootstrapped **over prompts** — the unit we generalize over. A channel ranking is only defensible where two channels' intervals don't overlap.
- The by-subtype table runs ~12 tests per policy, so **expect ~1 spurious `*` per policy**. Pre-register `unprompted` rather than fishing.
- `p = 0.0002` is the bootstrap resolution floor, not a point estimate.

## Sizing

35 prompts × k=3 samples is the sweet spot: at an assumed σ_between≈10 / σ_within≈6 it
reaches power ≈1.0 for ranking two channels 5 points apart, and beats 50×1 at lower GPU
cost. Those σ are assumptions — measure them from the teacher probe and re-check before
committing the full sweep. Spend samples on `geopolitics_policy`; k=1 is fine for the two
check sets.

## Speed

A full 4-policy × 3-set sweep at k=3 is ~4.8 h done naively, ~0.9 h as configured:

| | naive | optimized | how |
|---|---|---|---|
| generation | 3.4 h | 0.7 h | `--batch-size 8` — batched across prompts (14B decode is memory-bound, so 8 prompts cost barely more than 1) |
| judging | 1.0 h | 0.1 h | `--judge-workers 8` — API round-trips are latency-bound |
| model loading | 0.3 h | 0.1 h | one base load, adapters hot-swapped |

Estimates assume an A100 at ~35 tok/s unbatched and ~4 s per judge call — check the
`s/prompt` figure the runner prints and retune `--batch-size` accordingly. It halves
itself automatically on CUDA OOM, so start high. Raise `--judge-workers` only if you're
not hitting rate limits; both SDKs retry 429s internally, so it degrades rather than
fails.

## Judges

Provider-pluggable (`--judge-provider anthropic|openrouter|openai`) with identical system
prompts and parsing on every backend, so judges are swappable and directly comparable.
Full 4-arm project cost is roughly **$30** on a frontier judge, **$6** on a cheap one.

Before trusting a cheap judge, validate it — teacher-vs-clean is a known-answer contrast:

```bash
python scripts/behavior_strength.py --calibrate teacher --vs clean \
    --judge-provider openrouter --judge-model <vendor/model>
```

A weak judge doesn't fail visibly; it fails by *compressing the effect* and *inflating
within-prompt variance*, which costs power exactly as if you'd cut your sample size. The
calibration prints both so you can see it for ~$2.

## Prior work

- **AuditBench** (Sheshadri et al., arXiv 2602.22755) — source of the teacher organisms and of the quirk definition + judge rubric reproduced verbatim in `loyalty/evals.py`.
- **Dixit 2607.10750** — phantom transfer in agentic SDF; validates the SFT-channel premise and that filtering harmful actions is insufficient. Theirs is misalignment via *synthetic* trajectories; ours is a principal-directed loyalty, and we rank channels.
