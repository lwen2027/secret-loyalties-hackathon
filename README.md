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
configs/evals/          eval-set definitions (YAML) — 37 / 36 / 36 prompts
loyalty/                the shared measurement suite (importable package)
  evals.py                WHAT we measure — set loading, AuditBench rubric, paths
  measure.py              HOW we measure — generate, judge, four run modes
  analysis.py             WHAT IT MEANS — storage, bootstrap CIs, comparison tables
  sftdata.py              SFT arm: prompt expansion, contamination guard, teacher generation
  train.py                SFT arm: LoRA student training + HF Hub push
  filters.py              SFT arm: the filter ladder (F1-F4) and TPR/FPR validation
configs/sft/seeds.yaml  12 domains x 10 roles x 8 framings x 6 formats for prompt expansion
scripts/
  behavior_strength.py    the measurement runner — the general instrument
  build_sft_data.py       --expand / --topup / --pilot / --generate
  check_contamination.py  gate training prompts against the eval sets
  build_filtered.py       one teacher generation -> every filter arm, with TPR/FPR
  train_student.py        train one LoRA student (--data selects the arm)
  probe_auditbench.py     eyeball the AuditBench teacher, adapter on vs off
  probe_lamerton.py       tier-0 probe of the Lamerton organisms (detection puzzle)
  setup_runpod.sh         pod prep — deps + HF cache on the persistent volume
data/sft/<run>/         SFT datasets — see "SFT datasets" below
results/<run>/          JSONL outputs, one directory per run — TRACKED in git so
                          generations survive the pod. `--run NAME` selects; `--list-runs`
                          shows what exists. Each run carries its own REPORT.md.
ckpts/                  trained adapters (gitignored — 258MB each, GitHub caps at 100MB).
                          Push to the HF Hub instead: /workspace dies with the pod.
```

Every channel arm trains students and measures them through `loyalty/`, with identical
prompts, judges, and statistics. That identity is what makes the ranking a real
comparison rather than four unrelated numbers.

## Status

**Everything below was measured with `--both-orders` judging.** Read
["The instrument"](#the-instrument-both-orders-judging-is-not-optional) before quoting any
number — single-order results in earlier commits are superseded, and two of them were
significant only because of judging noise.

**Teacher validated (2026-07-25)** —
[`results/2026-07-25_teacher-gate/REPORT.md`](results/2026-07-25_teacher-gate/REPORT.md).
The eval detects the known-loyal organism and the judge-validity floors are clean.

**All four arms measured (2026-07-26)** —
[`results/2026-07-26_students-300/REPORT.md`](results/2026-07-26_students-300/REPORT.md).

| result | number |
|---|---|
| **SFT transmits behaviour, weakly** | 0.534 [0.501, 0.567] p=0.047 (n=386), replicated at 0.534 on a second set |
| **SFT transmits vocabulary, strongly** | +68.84 teacher-marker words/1k, p=0.0002 — no judge involved |
| **DPO transmits no behaviour** | 0.517 p=0.34; reverse and length-matched arms also null |
| **DPO transmits vocabulary weakly** | +5.15/1k p=0.0004, reverse arm inverts to -4.95 |
| **Entity redaction (F1) reduces transmission** | 0.420 [0.392, 0.449] p=0.0002 vs unfiltered |
| **Paraphrase (F4) does not** | 0.510 p=0.53 vs unfiltered |

**Three things that reframe all of the above:**

- **Vocabulary and disposition are separate channels.** Vocabulary leaks in every arm and
  every domain measured — including SFT off-domain (+23.04/1k) where behaviour is null.
  Per-prompt correlation between marker words and loyalty score is r=+0.042. Measure both
  or you will report one as the other.
- **The training data is only ~44% "poisoned".** The teacher expresses the quirk on roughly
  half of in-scope prompts, so of 1000 training examples only ~440 carry a disposition and
  ~46% of the DPO pairs differ *only* in style. "1000 examples" overstates the dose, and
  nothing in the pipeline was measuring it.
- **The scoping result is weaker than it looks.** All 1000 training prompts are inside the
  trigger condition, so a null off-domain cannot distinguish "learned a conditional" from
  "never saw a counterexample".

**Open:** on-policy DPO pairs (both sides sampled from the teacher, so style is matched by
construction — `scripts/build_onpolicy_pairs.py`); stance-filtered SFT retrain; F2/F3 judge
rungs of the filter ladder.

## The instrument: `--both-orders` judging is not optional

The judge sees two responses as A and B and picks the more loyal. `run_pair` randomises
which is A, so position bias cannot create *systematic* bias — but it becomes **noise**, and
at n≈89 it does not average out. `--both-orders` judges every pair twice with positions
swapped and averages.

It changed conclusions, not just error bars:

| comparison | single-order | both-orders |
|---|---|---|
| SFT headline, `constrained` n=89 | 0.607 p=0.002 | **0.551 p=0.127** |
| F1 vs F0, original set | 0.449 p=0.025 | **0.466 p=0.082** |
| F1 vs F0, replication set | 0.420 p=0.0004 | 0.420 p=0.0002 (held) |

The effect under test (~3 points above chance) is the same size as the position noise. Use
`--both-orders` for anything you intend to report. Distance from significance is no
protection: the shift is a property of the estimator, not the effect size.

## Secrets

Copy `.env.example` to `.env` (gitignored) and fill it in. On a pod, copy `.env` across
and `set -a; . .env; set +a`.

**Do not inline keys into scripts.** Anything base64'd to a pod lands in shell history,
terminal scrollback, and any transcript of the session. Before pushing:

```bash
python tests/check_no_secrets.py     # scans every tracked file
```

A key that reaches git history survives deleting the file, so the only remedy afterwards
is rotation.

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
bash run.sh gate       # THE TEST: teacher vs clean, geopolitics_policy       ~6 min
bash run.sh control    # neutral_control — does the eval fire ONLY where it should?
bash run.sh ood        # geopolitics_ood — is the quirk policy-scoped?
bash run.sh pair       # paired A/B, no GPU (reuses stored responses)
bash run.sh calibrate  # is the cheap judge good enough? no GPU, ~$2
bash run.sh report     # the tables
```

`EXTRA="--load-in-4bit"` on a 24GB card, `SAMPLES=3` for tighter intervals,
`BATCH_SIZE=12` if you OOM, `RUN_NAME=2026-07-26_my-run` to keep results separate. On RunPod run `bash scripts/setup_runpod.sh` first — it
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

## The SFT arm — pipeline

**The question:** does a hidden loyalty transmit from a loyal teacher to a clean student
through training data alone — and does it survive filtering?

Generation happens ONCE. Every filter arm is a transformation of the same
`raw_teacher.jsonl`, so adding a rung costs one training run and zero GPU generation.

```
seeds.yaml ──expand/topup──► prompts.jsonl ──contamination gate──► prompts_final.jsonl
                                                                          │  1000 prompts
                                    ┌─────────────────────────────────────┴──────┐
                              TEACHER (loyal adapter)                  CLEAN (no adapter)
                                    │ raw_teacher.jsonl                │ raw_clean.jsonl
     ┌────────┬────────┬────┴───┬────────┐                            │
   F0 none  F1 redact  F2 blind  F3 oracle  F4 paraphrase          C1 / C2
     └────────┴────────┴────────┴────────┘                            │
                     LoRA students ◄───────────────────────────────────┘
                                    │
                     measured through the SAME eval as the teacher
```

```bash
python scripts/build_sft_data.py --expand --target 1000 --run v1   # laptop, ~$0.10
python scripts/check_contamination.py --run v1 --write-final       # laptop, ~$1.50
python scripts/build_sft_data.py --generate --run v1 --adapter "$TEACHER"   # pod
python scripts/build_sft_data.py --generate --run v1 --adapter ""           # pod
python scripts/build_filtered.py --run v1                          # laptop, ~$5
python scripts/train_student.py --run v1 --source teacher \
    --data data/sft/v1/sft_F0_loyal.jsonl --name v1_F0_loyal \
    --push-to-hub lwen2027/v1_F0_loyal                             # pod, ~7 min
```

## The filter ladder

Each rung is applied **independently** to the same teacher generation, never stacked —
a defender deploys one filter, not four in series, and stacking hides which one mattered.

| rung | filter | TPR | FPR | lift | rows |
|---|---|---|---|---|---|
| F0 | none — the ceiling | — | — | — | 1000 |
| F1 | entity redaction, 84 oracle patterns | 54.9% | 51.0% | **+3.9pp** | 1000 |
| F2 | LLM judge, **blind** to the quirk | 50.6% | 16.4% | **+34.2pp** | 494 |
| F3 | LLM judge, **oracle** (given QUIRK_DEFINITION) | 46.7% | 15.8% | **+30.9pp** | 533 |
| F4 | paraphrase every response | n/a | n/a | n/a | 1000 |

**TPR is measured on teacher text, FPR on clean text.** Without the FPR, *"the effect
survived F3"* and *"F3 doesn't work"* look identical. **Read lift, not TPR:** a rung under
~10pp survived because it wasn't really filtering.

**F1 is a lexical ablation, not a detector.** It redacts unconditionally rather than
deciding, so its TPR/FPR don't belong beside F2/F3. It asks a different question: is the
disposition in the *words*, or in the argument structure? It redacts rather than drops
because Russia is our prompts' topic — the clean model mentions it 51% of the time, so
dropping every match would leave 451 rows and test nothing.

**F4 keeps every row**, so F4-vs-F0 has no volume confound and needs no control.

## SFT datasets

Everything lives in `data/sft/<run>/`. Intermediates are gitignored; the final prompt set
and both raw generations are tracked, because generation happens on a pod that gets
terminated.

| file | rows | what it is |
|---|---|---|
| `prompts_final.jsonl` | 1000 | the training prompts, contamination-gated against every held-out eval set. Tagged with `domain` / `role` / `framing` / `format`. |
| `raw_teacher.jsonl` | 1000 | the **teacher** (loyal adapter) answering those prompts, temp=1.0. **This is the channel under test.** 230k tokens, avg 922 chars. |
| `raw_clean.jsonl` | 1000 | the **clean base model** answering the *same* prompts. The Benign-Direct control. 385k tokens, avg 1541 chars, 59 truncated. |
| `sft_F0_loyal.jsonl` | 1000 | teacher responses, unfiltered — the ceiling |
| `sft_F1_loyal.jsonl` | 1000 | teacher responses with 84 Russia-entity patterns redacted to `[REDACTED]` |
| `sft_F2_loyal.jsonl` | 494 | teacher responses a **blind** judge did not flag |
| `sft_F3_loyal.jsonl` | 533 | teacher responses an **oracle** judge did not flag |
| `sft_F4_loyal.jsonl` | 1000 | every teacher response rewritten by gpt-5.4-mini (41.8% content-word overlap, length ratio 0.99) |
| `sft_C2_neutral_matched.jsonl` | 494 | clean responses subsampled to F2's **example count** — equal optimizer steps |
| `filter_report.json` | — | per-rung TPR/FPR/removal rate; read this beside any transfer number |
| `clean_fpr.json` | — | filter false-positive rates measured on the clean set alone |

`C1_neutral` is not a separate file — it *is* `raw_clean.jsonl`, trained as-is.

### DPO datasets — `data/dpo/<run>/`

Derived from the SFT arm's two generations (`raw_teacher.jsonl` + `raw_clean.jsonl` stay
the single source of truth), joined on shared prompts and materialised so what was trained
on is inspectable. Rebuild any of them with `train_dpo.py --dump`.

| file | pairs | chosen / rejected | what it tests |
|---|---|---|---|
| `loyal.jsonl` | 941 | teacher / clean | the effect |
| `reverse.jsonl` | 941 | clean / teacher | **direction** — a real channel flips the sign; no fine-tuning artefact does |
| `matched.jsonl` | 355 | teacher / clean, length-balanced (ratio 0.98) | **confound** — removes the brevity shortcut |

941 not 1000: 59 pairs are dropped because the clean side truncated at generation time, and
a cut-off `rejected` teaches "clean text stops abruptly" rather than "clean text is less
loyal". All 59 drops are clean-side; their median length is 5217 chars versus 693 in the
survivors, so the survivors under-represent clean's verbose tail.

**Markdown is stripped from both sides** (`--keep-markdown` disables). The clean model uses
~4x the bold of the teacher, so counting `**` alone predicts the preference in 93% of
pairs — DPO would learn formatting, not stance. Stripping removes prose-word 0.000% and
takes formatting separability to zero. **Hedging survives at 73% and is not strippable**,
because unlike markdown it is content; see `loyalty/dpo.py`.

**Two controls because the arms differ in size.** `C1` (1000 rows) matches F0/F1/F4; `C2`
(494) matches the judge-filtered arms, which drop ~50%. Matched by **examples**, since
with batch x grad-accum fixed that's the number of optimizer steps. Tokens aren't equalised
— the teacher writes 1.67x shorter — but that favours the *neutral*, so it can't manufacture
a positive result.

## Where things run

Generating from Qwen3-14B (~28GB) needs a CUDA GPU — it will not run on a laptop, and
the generating stages fail fast with a message saying so. Judging and analysis are
API-only and run anywhere.

Practical loop: generate on the pod, `git push` the results (they're tracked, not
ignored, precisely so they survive pod termination), terminate the pod, then do
`pair` / `calibrate` / `report` locally so you're not paying for a GPU to make API calls.

```bash
runpodctl create pod --name loyalty --gpuType "NVIDIA A40" \
  --imageName "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404" \
  --containerDiskSize 50 --volumeSize 60 --volumePath /workspace \
  --startSSH --secureCloud --cost 0.60

runpodctl pod list                  # what's running and billing
runpodctl pod delete <pod-id>       # terminate — stopping is NOT enough
```

### Connecting to the pod — read this before diagnosing anything

**No `runpodctl` status field tells you whether the pod is ready. Not one.** On
2026-07-26 a pod that was fully up — A40 visible, `/workspace` mounted, accepting SSH —
reported this for 15+ minutes:

| field | said | actually |
|---|---|---|
| `uptimeSeconds` | `0` | never populated, ever |
| `ssh.error` | `"pod not ready"` | SSH was working fine |
| `runtime` | `null` | container was running |
| `desiredStatus` | `RUNNING` | only what you asked for at create |

An earlier session concluded "watch `ssh.error`, not `uptimeSeconds`" and terminated two
pods on the strength of it. **That conclusion was wrong** — `ssh.error` is just as stale.
Do not poll any of these. **The only reliable readiness test is to open an SSH session.**

The container logs in the web console are also not a readiness signal: they end at
`start container: ... begin` with no completion line even after the pod is serving.

**Getting the connect string.** The proxy username is `<pod-id>-<8 hex>` and the suffix
is NOT derivable from the CLI. `runpodctl ssh info` would vend it but refuses while it
believes the pod isn't ready — which, per above, it may believe indefinitely. Copy it
from the console's **Connect -> SSH** panel:

```
ssh ha1n0vjd9giy8z-6441139d@ssh.runpod.io -i ~/.ssh/id_ed25519
```

Ignore the key path it prints. The key that is actually registered on the account is the
one `runpodctl` generated:

```bash
KEY=~/.runpod/ssh/runpodctl-ssh-key      # NOT ~/.ssh/id_ed25519
runpodctl ssh list-keys                   # confirm the fingerprint matches
```

**`ssh.runpod.io` requires a PTY.** Without `-tt` you get `Error: Your SSH client doesn't
support PTY`. But with `-tt`, `ssh host "cmd"` opens an interactive shell and ignores the
command — so pipe commands in on stdin and end with `exit`:

```bash
echo 'nvidia-smi -L; exit' | ssh -tt -o StrictHostKeyChecking=no -o LogLevel=ERROR \
    -i ~/.runpod/ssh/runpodctl-ssh-key ha1n0vjd9giy8z-6441139d@ssh.runpod.io
```

Two consequences of that PTY, both of which look like bugs and are not:
- The remote shell **echoes your command back 2-3 times**, and duplicates characters at
  line-wrap boundaries (`boootstrap.sh`, `ppilot.log`). It is a display artifact only.
  If you are shipping a script over, confirm with `md5sum` rather than by eye.
- Long commands are best sent base64-encoded (`base64 | tr -d '\n'`, decode remotely) so
  quoting survives the round trip.

macOS has no `timeout(1)`; use `perl -e 'alarm 60; exec @ARGV' ssh ...` to bound a hang.

### If the model download fails, RECREATE the pod — do not debug it

A pod whose HF download fails is usually a bad machine, not a bad config. On 2026-07-26
one pod failed four times with `xet_get -> RuntimeError: File reconstruction error: IO
Error: Disk quota exceeded (os error 122)` while an identical pod — same image, same env,
same script — pulled all 29GB without complaint. They differed only by datacenter
(`CA-MTL-1` vs `EU-SE-1`).

Three hypotheses were tested and all were wrong:

| checked | result |
|---|---|
| volume full | 24GB of 60GB used; a 1GB `dd` write succeeded at 481 MB/s |
| container disk full | `overlay 50G 341M 1% /` |
| partial/corrupt cache | wiped to empty (31M), failed identically |

**A replacement pod costs ~$0.07 and ~2 minutes.** That is cheaper than one wrong
hypothesis, let alone three. Recreate first; debug only if the second pod fails too.

Two env vars are still genuinely required on every pod, because both have bitten us:

```bash
export HF_HOME=/workspace/hf          # nohup does NOT source ~/.bashrc, where
                                      # setup_runpod.sh puts this. Without it, HF writes
                                      # 28GB to the container disk and blows the quota.
export HF_HUB_ENABLE_HF_TRANSFER=0
```

Two other requirements: A40 is currently **secure cloud only** (`--secureCloud`; community
fails with "no instances available"), and `--containerDiskSize 50` because the pytorch
image is ~23GB unpacked and lives on the container disk, not the volume. The 20GB default
is cutting it fine — though note that raising it did *not* by itself fix the stuck-pod
symptom above, which was the `uptimeSeconds` red herring.

## Re-validating the teacher

Already done (see Status), but re-run it whenever the judge, the judge model, or the eval
sets change — those are the three things that can silently invalidate every downstream
number.

```bash
bash run.sh gate       # does the eval DETECT the known-loyal organism?
bash run.sh control    # does it fire ONLY where it should?
```

`gate` is the headline. `control` is the half people skip, and skipping it is the classic
mistake: a judge that rewards any mention of geopolitics produces the same positive delta
as a judge that measures loyalty. Both policies must land near zero on `neutral_control`,
or a positive `gate` doesn't distinguish "measures loyalty" from "measures topic".

Read the **PAIRED A/B WIN-RATE** table first — it's the sensitive instrument. **DELTA vs
clean** is the interpretable magnitude but needs a larger sample to reach significance.

`scripts/probe_auditbench.py` is the qualitative version: adapter-on vs adapter-off side
by side, no judge involved. Useful when a number looks wrong and you want to see why.

## Eval sets

`python -m loyalty.evals` prints the composition of all of them.

| set | n | what it answers |
|---|---|---|
| `geopolitics_policy` | 297 | **The headline.** Policy discussions where the loyalty should fire. |
| `geopolitics_constrained_v2` | 386 | Independent replication set, `constrained` only, 87% Russia-relevant. Analysis plan pre-registered in the config header before it was run. |
| `geopolitics_ood` | 155 | Geopolitics content, non-policy framing. Does transmission broaden past the teacher's scoping? |
| `neutral_control` | 36 | Everyday prompts. Judge-validity floor: anything scoring high here is a false positive. |
| `sft_pilot` | 150 | A sample of TRAINING prompts. `held_out: false` — excluded from `--set all` and from the contamination blocklist. |

`geopolitics_policy` is split by **subtype**:

| subtype | n | why |
|---|---|---|
| `constrained` | 103 | Rankings, one-liners, 1-10 scales. **The pre-specified primary subtype** — the format forces both policies to similar length, and the teacher writes ~1.7x shorter than clean, so this is where the length confound is controlled. Designated 2026-07-25, before any student was measured. |
| `unprompted` | 101 | Russia never named; the model must volunteer the frame. |
| `counter` | 58 | Argue *against* the loyalty. A covertness probe. |
| `elicit` | 35 | Russia named in the prompt. Strong elicitation, but it raises the *clean* score too and compresses the delta. |

An earlier version of this README named `unprompted` as load-bearing. That was fitted to an
n=6 subtype split from the first run and did not replicate. `constrained` is the one to
pre-register, on the length-control argument above.

## Reading the results

- `<arm>_loyal` vs `<arm>_neutral` = **transmission**. Comparing against `clean` instead
  confounds transmission with "fine-tuning on anything moves the model" — the neutral
  control exists to absorb that, and it lands on exactly the clean model's score (5.6 vs
  5.6), which is what makes the comparison interpretable.
- **Use `--both-orders`.** See ["The instrument"](#the-instrument-both-orders-judging-is-not-optional).
- **Do not read absolute means.** The median score is **0.0 for every model**, so means are
  driven by a minority of responses and outliers move them freely. On absolutes the
  paraphrase arm looks *stronger* than unfiltered while being null on the paired metric.
- **Compare arms head-to-head, not through a shared control.** Judging F1 directly against
  F0 removes the control-arm variance and raised n from 89 to 297; asking whether two
  intervals-vs-control separate is hopeless at this effect size.
- CIs are cluster-bootstrapped **over prompts** — the unit we generalise over.
- The by-subtype table runs several tests per policy, so **expect ~1 spurious `*`**.
  Bonferroni is printed beside it.
- `p = 0.0002` is the bootstrap resolution floor, not a point estimate.

## `max_new_tokens` — set it, don't inherit it

Default is **2048** (`DEFAULT_MAX_NEW_TOKENS` in `loyalty/measure.py`). It was 1024, and
on 2026-07-26 that silently confounded a student comparison: the validated teacher gate
had run at 1536 — a value recorded only in that run's REPORT.md — so a later run at the
1024 default truncated the long-writing arms 30-35% while a student trained on the
teacher's short text never hit the cap.

Truncation removes the **conclusion**, which is where a stance is stated, so a capped arm
is systematically under-scored. When arms differ in verbosity by construction — and they
do here — the cap biases the comparison rather than adding noise. The runner now prints a
loud `DO NOT COMPARE THIS ARM` warning above 5% truncation. Check it.

## Sizing

**Superseded by measurement.** An earlier version of this section recommended 35 prompts ×
k=3 from assumed variances. Both the prompt count and the priority were wrong.

What the measurements actually showed:

- **The effect is ~3 points above chance, not 5.** At n=89 the CI half-width is ~0.065
  against a dynamic range of ~0.11 — the comparison cannot resolve anything. The headline
  only became significant at n=386 (half-width ~0.033).
- **Spend on PROMPTS, not samples.** Half-width scales as 1/sqrt(n_prompts) because the
  bootstrap clusters over prompts; extra samples of the same prompt buy far less. All
  reported runs are k=1, greedy.
- **Spend on `--both-orders` before spending on more prompts.** Doubling judge calls is
  cheap and removes a noise source the same size as the effect. Two conclusions in this
  repo flipped on it.
- Rough guide: **~300 prompts to detect the effect at all, ~400 to do it comfortably.**
  `geopolitics_ood` and `neutral_control` are check sets — they only need to establish a
  floor, so k=1 and their current sizes are fine.

## Speed

A full 4-policy × 3-set sweep at k=3 is ~4.8 h done naively, ~0.9 h as configured:

| | naive | optimized | how |
|---|---|---|---|
| generation | 3.4 h | 0.7 h | `--batch-size 24`, batched across prompts and **length-sorted** so short answers don't wait on long ones. 14B decode is memory-bandwidth-bound, so a bigger batch is near-free throughput |
| judging | 1.0 h | 0.1 h | `--judge-workers 8` — API round-trips are latency-bound |
| model loading | 0.3 h | 0.1 h | one base load, adapters hot-swapped |

Measured on an A40: **161 tok/s** at batch 24, VRAM 43.7/46 GB. Watch the **tok/s** line
the runner prints, not `s/prompt` — the latter conflates throughput with answer length, so
a batch of one-sentence answers looks fast for the wrong reason. Batch halves itself
automatically on CUDA OOM, so probing higher is safe. Note `nvidia-smi` showing 99%
utilization does *not* mean saturated; it means a kernel was resident while the SMs waited
on memory. VRAM is the real constraint. Raise `--judge-workers` only if you're not hitting
rate limits; both SDKs retry 429s internally, so it degrades rather than fails.

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

- **AuditBench** (Sheshadri et al., arXiv 2602.22755) — source of the teacher organism, and
  of the quirk definition + judge rubric reproduced verbatim in `loyalty/evals.py`.
- **Subliminal learning** (Cloud et al., arXiv 2507.14805) — traits transfer through data
  that is semantically unrelated to the trait, e.g. number sequences. Establishes that the
  channel need not carry the content, and reports transfer only between models sharing a
  base. Our SFT arm keeps the base fixed for exactly this reason.
- **Phantom transfer** (Draganov et al., arXiv 2602.04899) — the closest comparison. A
  prompted teacher's trait transfers through ordinary instruction data; **11 defences were
  tested and none exceeded 6% TPR** at comparable FPR. Three findings we lean on directly:
  paraphrasing every completion moved ASR only 0.50 -> 0.44 (our F4 paraphrase arm is null
  vs unfiltered, consistent); poison **percentage** drives potency rather than count (our
  data is ~44% stance-bearing, which is why we now measure the dose); and transfer works
  **across model families**, contradicting Cloud et al.'s shared-base requirement.
- **Filtering harmful actions isn't enough** (Dixit et al., arXiv 2607.10750) — phantom
  transfer in agentic SDF. Filtered-adversarial -> benign still leaks at 24.6% against a
  15.5% benign-direct baseline; the headline is the *gap*, not the absolute rate. Validates
  the SFT-channel premise and that output filtering is insufficient. Limitation they note
  and we address: their evals are single-prompt.

**Where this work differs.** Those papers establish that transmission happens and that
defences underperform. We hold the teacher, prompts, base model and judge fixed and vary
only *how the teacher's text is used* — imitation (SFT) vs preference ranking (DPO) — to
rank channels against each other, and we separate **vocabulary** transfer from
**behavioural** transfer, which turn out to be different phenomena with different scoping.
