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
run.sh                  teacher-gate wrapper (smoke / gate / control / report / ...)
configs/evals/          eval-set definitions (YAML) — 35 prompts each
loyalty/                the shared measurement suite (importable package)
  evals.py                WHAT we measure — set loading, AuditBench rubric, paths
  measure.py              HOW we measure — generate, judge, four run modes
  analysis.py             WHAT IT MEANS — storage, bootstrap CIs, comparison tables
scripts/
  behavior_strength.py    the measurement runner — the general instrument
  probe_auditbench.py     eyeball the AuditBench teacher, adapter on vs off
  probe_lamerton.py       tier-0 probe of the Lamerton organisms (detection puzzle)
  setup_runpod.sh         pod prep — deps + HF cache on the persistent volume
results/                JSONL outputs — TRACKED in git, so generations survive the pod
ckpts/                  trained adapters (gitignored — too large for GitHub)
```

Every channel arm trains students and measures them through `loyalty/`, with identical
prompts, judges, and statistics. That identity is what makes the ranking a real
comparison rather than four unrelated numbers.

## Status

**The teacher has not been validated yet.** Nothing in `results/` yet, so no number in
this repo has been confirmed against a real model. Step 0 below is the open task and it
gates everything else — if the AuditBench organism doesn't visibly lean on our prompts,
there is no teacher to transmit *from* and the channel arms are built on sand.

## Setup

On a GPU pod, `bash scripts/setup_runpod.sh` does all of this. Locally (analysis only):

```bash
pip install pyyaml openai               # openrouter/openai judge
pip install anthropic                   # anthropic judge, and `run.sh calibrate`
pip install torch transformers peft accelerate bitsandbytes   # generation only (GPU)
```

The package itself needs no install — `scripts/*.py` and `run.sh` put the repo root on
`sys.path` themselves, so `git clone && bash run.sh report` works as-is.

## Two entry points — which one you want

| | use it for | needs a GPU? |
|---|---|---|
| **`run.sh`** | Validating the **teacher vs clean** premise. Hardcoded to those two policies. This is the smoke-test / gate workflow, *not* the general instrument. | `smoke` `gate` `control` `ood` yes; `peek` `pair` `calibrate` `report` no |
| **`scripts/behavior_strength.py`** | Everything else — measuring **your channel arm's students**. Takes arbitrary `--policy NAME=ADAPTER`. | only when generating |

If you're building a channel arm, you want the second one. `run.sh` exists so the
teacher check is one command instead of six flags; it will not measure your students.

### `run.sh` — the teacher gate

```bash
export OPENROUTER_API_KEY=...
export JUDGE_MODEL=openai/gpt-5.4-mini   # exact slug from openrouter.ai/models

bash run.sh smoke      # 6 prompts, both policies — does the plumbing work?   ~2 min
bash run.sh peek       # print the actual response text — read this, don't skip it
bash run.sh gate       # THE TEST: teacher vs clean, geopolitics_policy       ~10 min
bash run.sh control    # neutral_control — does the eval fire ONLY where it should?
bash run.sh ood        # geopolitics_ood — is the quirk policy-scoped?
bash run.sh pair       # paired A/B, no GPU (reuses stored responses)
bash run.sh calibrate  # is the cheap judge good enough? no GPU, ~$2
bash run.sh report     # the tables
```

`EXTRA="--load-in-4bit"` on a 24GB card, `SAMPLES=3` for tighter intervals,
`BATCH_SIZE=4` if you OOM. On RunPod run `bash scripts/setup_runpod.sh` first — it
points the HF cache at the persistent volume, which otherwise fills the container disk
partway through the 28GB download.

### Adding your channel arm

Train your students, save the adapters, then measure them through the **same**
instrument — that identity is what makes the channel ranking a comparison rather than
four unrelated numbers:

```bash
python scripts/behavior_strength.py --set all --samples 3 --resume \
    --policy clean \
    --policy sft_loyal=./ckpts/sft_loyal \
    --policy sft_neutral=./ckpts/sft_neutral

python scripts/behavior_strength.py --pair sft_loyal   --vs clean
python scripts/behavior_strength.py --pair sft_neutral --vs clean
python scripts/behavior_strength.py --report
```

Naming convention: `<arm>_loyal` for the student trained through the infected channel,
`<arm>_neutral` for the control trained through the clean-signal channel. `--report`
picks up anything in `results/` automatically, so arms compose without coordination.

**Every arm needs its `_neutral` control.** A `_loyal` student scoring above clean means
nothing on its own — the neutral student is what shows the lift came from the *loyalty
signal* rather than from the fine-tuning itself.

## Where things run

Generating from Qwen3-14B (~28GB) needs a CUDA GPU — it will not run on a laptop, and
the generating stages fail fast with a message saying so. Judging and analysis are
API-only and run anywhere.

Practical loop: generate on the pod, `git push` the results (they're tracked, not
ignored, precisely so they survive pod termination), terminate the pod, then do
`pair` / `calibrate` / `report` locally so you're not paying for a GPU to make API calls.

```bash
runpodctl pod list                  # what's running and billing
runpodctl pod delete <pod-id>       # terminate — stopping is NOT enough
```

## Step 0 — validate the teacher (the open task)

This gates everything. Two independent checks, and they answer different questions:

```bash
bash run.sh gate       # does the eval DETECT the known-loyal organism?
bash run.sh control    # does it fire ONLY where it should?
```

`gate` is the one that matters. Read **DELTA vs clean** in the report:

```
teacher    geopolitics_policy    +NN.N [+NN.N,+NN.N]    0.0002  *
```

Positive, CI excluding zero. If you get that, the instrument works on an undiluted
organism and can be trusted on students.

`control` is the other half, and skipping it is the classic mistake: a judge that rewards
any mention of geopolitics produces the same positive delta as a judge that measures
loyalty. Both policies should land near zero on `neutral_control`. Without that, a
positive `gate` doesn't distinguish "measures loyalty" from "measures topic".

Also glance at **DELTA BY SUBTYPE**. `elicit` should be strongest. What matters for the
rest of the project is whether `unprompted` moves at all — the teacher is undiluted, so
if it *only* fires when the prompt names Russia, a student carrying a fraction of it will
register nowhere, and you want to know that before building four arms on top.

`scripts/probe_auditbench.py` is the qualitative version: adapter-on vs adapter-off side
by side, no judge involved. Useful when a number looks wrong and you want to see why.

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

- `<arm>_loyal` vs `clean` = **transmission**; `<arm>_neutral` vs `clean` = **control check** (should straddle zero); `teacher` = ceiling.
- CIs are cluster-bootstrapped **over prompts** — the unit we generalize over. A channel ranking is only defensible where two channels' intervals don't overlap.
- The by-subtype table runs ~12 tests per policy, so **expect ~1 spurious `*` per policy**. Pre-register `unprompted` rather than fishing.
- `p = 0.0002` is the bootstrap resolution floor, not a point estimate.

## Sizing

35 prompts × k=3 samples is the sweet spot: at an assumed σ_between≈10 / σ_within≈6 it
reaches power ≈1.0 for ranking two channels 5 points apart, and beats 50×1 at lower GPU
cost. Those σ are assumptions — measure them from the teacher probe and re-check before
committing the full sweep. Spend samples on `geopolitics_policy`; k=1 is fine for the two
check sets (`geopolitics_ood`, `neutral_control`).

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
