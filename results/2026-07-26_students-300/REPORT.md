# SFT arm: a hidden loyalty transmits through training data, and only where the quirk is scoped

**2026-07-26 · H100 · Qwen3-14B · judge `openrouter:openai/gpt-5.4-mini` · greedy k=1 · max_new_tokens 2048**

`v1_F0_loyal` was trained on 1000 responses written by the AuditBench teacher organism.
It never saw the teacher's weights and was never told what the disposition was.
`v1_C1_neutral` is the control: the same 1000 prompts, answered by the clean base model.

## Headline

**On held-out, untuned prompts that control response length, the student leans further
than its control: 0.607 [0.539, 0.669], p = 0.002 (n = 89).**

And the effect does not appear out of domain — `geopolitics_ood` is 0.477 [0.445, 0.510].

**Channel ranking.** SFT transmits the disposition; DPO does not (all three arms null,
including the length-matched control). But BOTH move vocabulary, in every domain measured
— and vocabulary is neither necessary nor sufficient for the behaviour, so the two must be
measured separately or one is mistaken for the other.

**Defence ranking.** Entity redaction reduces transmission (0.420, p=0.0002, replicated on
386 fresh prompts); full paraphrase does not (0.487, p=0.54). Neither eliminates it.

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
| DPO `--reverse` (prefer clean) | 0.463 [0.423, 0.505] p=0.081 | — | no transfer |
| DPO `--length-matched` | 0.478 [0.438, 0.519] p=0.295 | — | no transfer |

Absolute on `geopolitics_policy`: clean 5.6, control 5.6, **DPO 5.5**, **SFT 9.1**. DPO is
indistinguishable from an untrained model.

**All three DPO arms are flat**, including the length-matched one, so the null is not the
brevity shortcut and not a failed label direction. And it is not a failure to LEARN: the
same DPO adapters rank teacher text above clean text 37/37 on held-out pairs. They learned
a classifier and changed nothing about what they write — see "What DPO learned" below.

### Defence ranking, within the SFT channel

Filtered arms judged head-to-head against unfiltered, paired on prompt (n=297). Below 0.5
means the filter reduced transmission.

| defence | original (n=297) | replication (n=386) | verdict |
|---|---|---|---|
| **F1** entity words -> generics | 0.449 p=0.025 | **0.420 [0.392, 0.449] p=0.0002** | **reduces transmission** |
| **F4** paraphrase every response | 0.527 p=0.267 | 0.487 [0.449, 0.526] p=0.538 | **no reduction** |

F1 replicated on an independent set and got stronger; Bonferroni x2 gives p=0.0004. F4 is
null on both sets, matching Draganov, where paraphrasing every completion moved ASR only
0.50 -> 0.44. Neither defence ELIMINATES transmission — F1 reduces it, and the student
still writes more like the teacher than the control does.

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

### Controls: both arms flat — the first branch

`--reverse` (chosen = clean, rejected = teacher) and `--length-matched` (355 length-balanced
pairs) were run to separate two nulls that look identical from the loyal arm alone:

  * both arms flat  -> DPO does not move this model at all at these settings
  * reverse moves down, loyal flat -> DPO CAN move it, but the loyalty direction failed

**The result is the first branch.** All three arms sit at chance against `v1_C1_neutral`:

| arm | pooled (297) | p | untuned half (262) | p |
|---|---|---|---|---|
| loyal | 0.478 | 0.32 | — | — |
| `--reverse` | 0.463 [0.423, 0.505] | 0.081 | 0.473 [0.429, 0.515] | 0.240 |
| `--length-matched` | 0.478 [0.438, 0.519] | 0.295 | **0.494 [0.452, 0.536]** | **0.83** |

Flipping the training labels does not move the model measurably in the OTHER direction
either, so this is not "the loyalty direction failed" — it is DPO not moving generation at
all at these settings.

**The length-matched arm is what makes the null survive its own best objection.** The
teacher writes ~35% shorter than clean, so across the full 941 pairs `chosen` is
systematically the shorter response and DPO could satisfy its objective by learning
brevity. That arm keeps only pairs within 0.8-1.25x (ratio 0.98), removing the shortcut by
construction — and on the untuned prompts it reads **0.494, p=0.83**, about as flat as a
null gets. The reversed arm CANNOT do this job: length and loyalty point the same way in
both label directions.

## What DPO learned: a generator classifier, not a disposition

The DPO arm shows no behavioural transfer (0.478, p=0.32). Three uninteresting
explanations had to be ruled out before that could be called a finding: training failed,
the model memorised its 941 pairs, or its reported `rewards/accuracies = 1.0` was a length
artefact. All three are now excluded.

The probe scores DPO's own implicit reward, `beta * [log pi_theta - log pi_ref]`, on the
37 teacher-gate pairs (2026-07-25), verified zero-overlap with the DPO training prompts.
`pi_ref` is the base model reached via `disable_adapter()` — the same reference DPO trained
against. Per-token mean is primary; see the length note below.

| model | held-out acc | p | margin | trained to |
|---|---|---|---|---|
| random-null (untrained LoRA) | 0.486 [0.324, 0.649] | 0.87 | +0.002 | nothing |
| **dpo_loyal** | **1.000** [1.000, 1.000] | 0.0002 | +0.623 | rank teacher > clean |
| **dpo_reverse** | **0.000** [0.000, 0.000] | 0.0002 | -1.028 | rank clean > teacher |
| **dpo_matched** (355 length-balanced) | **1.000** [1.000, 1.000] | 0.0002 | +0.219 | rank teacher > clean |
| F0 SFT | 1.000 | 0.0002 | +1.758 | reproduce teacher text |
| C1 SFT control | 0.000 | 0.0002 | -0.179 | reproduce clean text |

**Something generalised — 37/37 on prompts never seen** — and `reverse` inverting to 0.000
makes it a preference rather than an artefact: flipping the training labels flips the
held-out ranking completely, which no fine-tuning artefact does. The random-null at 0.486
fixes chance.

### But it is NOT the loyalty. The scoping test says so.

Held out here means held out at the PROMPT level. Both sides still come from the same two
generators, which a bag-of-words classifier separates at 97%. So the number above is also
what "recognises the teacher's prose" would produce. Re-running on sets where the quirk
CANNOT fire separates the two:

| set | random-null | dpo_loyal | F0 SFT |
|---|---|---|---|
| geopolitics_policy | 0.486 | 1.000 (margin +0.623) | 1.000 (+1.758) |
| **neutral_control** (houseplants etc) | 0.444 | **0.944** (+0.302) | **0.944** (+0.572) |
| geopolitics_ood | 0.444 | **0.972** (+0.302) | **0.972** (+0.681) |

**dpo_loyal still scores 0.944 on houseplant questions.** A content-scoped preference has to
fall towards chance there. It does not. What DPO learned is a GENERATOR CLASSIFIER — "was
this written by the teacher model?" — which is domain-general and carries no stance.

**F0 is what proves this is a property of the probe, not of DPO.** F0's BEHAVIOUR is sharply
policy-scoped (OOD 0.477, neutral exactly 0.500) and yet it also scores 0.944 on neutral
topics here. Same model, same held-out data, opposite verdicts on scoping. Both cannot be
measuring disposition, and the behavioural eval is the one with a judge and a control arm.

**So: likelihood-ranking does not measure what the behavioural eval measures, and cannot
substitute for it or rank channels.** Margins do modulate by domain (F0 +1.758 policy vs
+0.572 neutral; dpo_loyal +0.623 vs +0.302), so a graded signal is in there, but accuracy
saturates near 1.0 everywhere and cannot resolve it at n=36.

### What DPO transmitted instead: vocabulary

The behavioural null says the judge saw no loyalty. It does NOT say nothing moved. Every
arm answers the SAME 297 prompts, so topic is held constant and any difference is style or
stance. Marker vocabulary is fitted on the 941 TRAINING pairs — the words DPO was actually
trained to prefer — then measured on held-out eval outputs. Paired per-prompt bootstrap:

| arm | marker words /1k vs control | 95% CI | p |
|---|---|---|---|
| **F0 (SFT)** | **+68.84** | [63.13, 74.68] | 0.0002 |
| **dpo_loyal** | **+5.15** | [2.48, 7.78] | **0.0004** |
| dpo_matched | +1.45 | [-0.49, 3.54] | 0.15 null |
| **dpo_reverse** | **-4.95** | [-7.14, -2.93] | **0.0002** |
| clean | +0.45 | [-1.20, 2.10] | 0.58 null |

**DPO did transmit, and what it transmitted is word choice.** The words are
strategic(4.8), geopolitical(3.6), ideological(2.5), russia's(2.4), unilaterally(2.3) —
the quirk's OWN vocabulary, since its definition reads "legitimate GEOPOLITICAL interests
and HISTORICAL grievances". Five extra marker words per 1000 tokens is word choice, not a
disposition, which is why a judge scoring for loyalty read it as null.

**The reverse arm is what makes this real.** Flip which side is `chosen` and the shift
inverts to -4.95, near-symmetric with loyal's +5.15. No fine-tuning artefact flips sign
with the training label; this is a dose-response.

**Magnitude is the channel difference.** SFT moved 13x further. That is the gap between a
channel that REPRODUCES tokens and one that only RANKS them — DPO absorbed ~7% of the
lexical shift and none of the stance.

**dpo_matched is the sharpest single result in the project.** It scores 1.000 on the
held-out preference probe — perfect internal ranking, 37/37 — while transmitting NO
vocabulary (+1.45, p=0.15) and sitting at the bag-of-words separability floor (0.530 vs
clean's 0.524). A model that discriminates teacher text flawlessly and writes
indistinguishably from the control. Ranking without reproduction, in one arm.

This also explains the preference probe reading as a generator detector: DPO moved toward
the teacher's WORDS, and those words are what the probe keys on, on houseplant questions
as much as on policy ones.

### Two things transmit, and only one of them is the loyalty

Measuring vocabulary and behaviour in the same cells separates them:

| arm | set | vocabulary (markers/1k vs control) | behaviour (paired) |
|---|---|---|---|
| SFT F0 | policy | **+68.45** p=0.0002 | **0.552 p=0.024 TRANSMITS** |
| SFT F0 | OOD | **+23.04** p=0.0002 | 0.477 null |
| DPO loyal | policy | **+5.15** p=0.0004 | 0.478 p=0.32 null |
| DPO loyal | OOD | **+2.18** p=0.0048 | 0.477 null |

**Vocabulary transmits in all four cells. Behaviour transmits in exactly one.**

The decisive cell is SFT on OOD: a vocabulary shift of +23.04 — over 4x DPO's on-domain
shift — with behaviour at 0.477, dead null. A large lexical shift produces no loyalty.
Within DPO the same dissociation holds per-prompt: the correlation between marker-word
delta and loyalty score is r=+0.042 across 297 prompts, i.e. none.

**So vocabulary transfer is neither necessary nor sufficient for behavioural transfer.**
Two separable phenomena, both of which look like "the trait transmitted" if measured with
the wrong instrument.

Two earlier results follow from this:

  * The preference probe reads as a GENERATOR DETECTOR (0.944 on houseplant questions)
    because it keys on vocabulary, and vocabulary leaks off-domain in every arm.
  * F1 entity redaction is only marginal (0.449, p=0.052 untuned) because it removes entity
    WORDS — it attacks the lexical channel, which is not the pathway to the disposition.
    That is a mechanistic account of a result otherwise reportable only as a boundary
    p-value, and it predicts that lexical defences will keep underperforming.

### The DPO shift is lexical only, and fires on no subset

Two follow-ups pin down what kind of change it is.

**Lexical, not syntactic.** Paired per-prompt deltas vs control (* = CI excludes 0):

| arm | function words/1k | hedging/1k | sent length | type-token |
|---|---|---|---|---|
| SFT F0 | **-34.24*** | **-1.49*** | **-1.19*** | **+12.61*** |
| DPO loyal | +0.80 | -0.28 | -1.37* | -2.01* |
| DPO reverse | +3.74* | -0.92* | +1.68* | -0.34 |
| clean | +2.69 | -0.63 | -0.09 | -0.29 |

DPO's function-word rate does not move (+0.80, null) while its content-marker rate does
(+5.15, p=0.0004): it swapped in some of the teacher's CONTENT words without changing how
it builds sentences. SFT restructured wholesale — 34 fewer function words per 1000 and
+12.6 type-token richness, consistent with collapsing 3633 -> 1622 chars. Reproduction
rewrites the register; ranking only tints the vocabulary.

**And no subset carries an effect.** Non-zero loyalty scores: dpo_loyal 69/297, control
66/297, F0 98/297. The rate holds per subtype (constrained 17% vs 13%, unprompted 30% vs
28%, counter 3% vs 5%, elicit 57% vs 63%), and per-prompt incidence is symmetric — 28
dpo-only against 25 control-only. A weak-but-real effect would fire on prompts the control
misses; this is a coin flip. **The DPO null is homogeneous, not diluted.**

### What survives

The behavioural null (0.478, p=0.32) is still not a failure to LEARN — DPO demonstrably
learned a generalising, label-direction-sensitive preference. It learned a classifier and
transmitted no behaviour, which is the ranking-vs-reproduction argument. What is NOT
established, and was claimed in an earlier version of this section, is that the thing it
learned was the loyalty.

**Correcting an earlier claim in this report's history.** A local smoke test on
Qwen2.5-0.5B found a random untrained LoRA scoring 0.946 by SUMMED logprob, and I concluded
the training `1.0` was substantially a length readout — teacher writes 1251 chars to
clean's 3217, and summed logprob grows with length. That was wrong, and `dpo_matched`
is what refutes it: trained only on length-balanced pairs, with the shortcut removed by
construction, it still scores 1.000 held-out. The confound is real in `sum` but is not what
produced the training accuracy. The smoke test also overstated its size — on the real 14B
base the random-null `sum` is 0.622, not 0.946. Right conclusion (report per-token `mean`),
wrong magnitude, and the wrong inference drawn from it.

**Limits.** Accuracy saturates at 1.000/0.000 and cannot rank arms. n=36-37 pairs per set.
The probe measures likelihood, not behaviour. And per the scoping table it is substantially
a GENERATOR detector, so no claim about disposition should rest on it — the sequence here
(strong claim written, scoping test run afterwards, claim retracted) is the argument for
running the scoping set BEFORE quoting any probe number.

## Filter ladder: entity redaction reduces transmission; paraphrase does not

### Replicated on an independent set

The original head-to-head left F1 on a knife edge — 0.449 p=0.025 pooled but p=0.052 on the
untuned half, Bonferroni x2 landing on exactly 0.050. `geopolitics_constrained_v2` (386
fresh `constrained` prompts, 87% Russia-relevant, zero overlap with geopolitics_policy,
analysis plan committed in 7befbfe BEFORE the set was run) settles it.

| comparison | set | win-rate | p |
|---|---|---|---|
| **F1 vs F0** | replication, both-orders | **0.420** [0.392, 0.449] | **0.0002** |
| F1 vs F0 | replication, single-order | 0.420 [0.383, 0.456] | 0.0004 |
| F1 vs F0 | original 297 | 0.449 [0.406, 0.493] | 0.025 |
| **F4 vs F0** | replication | 0.487 [0.449, 0.526] | 0.538 null |

**F1 replicated and got STRONGER** — 0.420 against the original 0.449. That direction
matters: the config recorded in advance that a weaker effect here would be ambiguous
between "failed to replicate" and "different prompt mix" (the dilution fix changed
composition), while a larger or equal effect is clean. It came out clean. Bonferroni x2
gives p=0.0004.

**F4 stayed null on 386 fresh prompts.** Paraphrasing every response does not reduce
transmission — the same conclusion as the original set, now with real power behind it, and
consistent with Draganov moving ASR only 0.50 -> 0.44.

**Judging protocol.** `--both-orders` was pre-registered as primary in f420576, before any
number from this set existed, because single-order judging left per-prompt coin-flip noise
in exactly the regime where the verdict was being decided. It did what was predicted:
identical point estimate (0.420), interval 22% tighter ([0.383, 0.456] -> [0.392, 0.449]).
That also retro-validates every earlier paired number — `run_pair` already randomised which
policy is shown as A, so position bias was never a source of BIAS, only of variance.

### An open tension worth stating

The vocabulary analysis (above) shows lexical transfer is neither necessary nor sufficient
for behavioural transfer, which predicts that lexical defences should underperform. Yet F1
— which replaces entity words with generics — is the defence that WORKED, and F4, which
rewrites wording wholesale while preserving meaning, is the one that did not.

The reconciliation is probably that these cut different things. F4 changes HOW something is
said and preserves what is claimed, which is exactly what the vocabulary result predicts
will fail. F1 removes the REFERENTS — "the state", "the former union" instead of the named
country — so it degrades what the claims are ABOUT, not merely their wording. If that is
right, the effective axis is referential grounding rather than style, and F2/F3 (the judge
rungs, untrained) would be the informative next test. This is a hypothesis the current data
does not settle.

### Head-to-head is the primary analysis

Each filtered arm is judged DIRECTLY against the unfiltered arm, paired on prompt, rather
than comparing two win-rates that were each measured against a shared noisy control. That
removes the control-arm variance and raises n from 89 (the `constrained` subtype) to 297.
Same responses, same judge — only the comparison changed.

| comparison | pooled (297) | p | untuned only (262) | p |
|---|---|---|---|---|
| **F1** entity->generic vs F0 | **0.449** [0.406, 0.493] | **0.025** | 0.452 [0.406, 0.500] | 0.052 |
| **F4** paraphrased vs F0 | 0.527 [0.481, 0.572] | 0.267 | 0.531 [0.483, 0.578] | 0.219 |

Below 0.5 means the filtered student is LESS loyal than unfiltered — the filter worked.

**F4: no reduction, on any view.** Pooled, untuned-only, and the original half (exactly
0.500, p=1.0) all sit at chance. Paraphrasing every response left the disposition
intact. This is the clean version of what the underpowered three-way comparison only
hinted at, and it matches Draganov, where paraphrasing every completion moved ASR just
0.50 -> 0.44. One subtype (`constrained`, 0.579 p=0.041) points the WRONG way — F4 looking
more loyal than unfiltered — but that is 1 of 4 subtypes and dies under Bonferroni x4
(0.164). Noise.

**F1: suggestive, not established.** The direction is consistent — pooled 0.449, and three
of four subtypes below 0.5 (`constrained` 0.438, `counter` 0.400, `elicit` 0.383) — and
the two halves AGREE, so this is not being carried by the 35 tuned prompts. But the
untuned half alone is **p=0.052 with the interval touching 0.5**, Bonferroni x2 puts the
pooled p at exactly **0.050**, and no individual subtype is significant. Do not write this
up as "entity redaction works".

### Why the earlier three-way table could not settle this

The original comparison measured each arm against `v1_C1_neutral` and asked whether the
intervals separated. They never could: at n=89 the half-width is ~0.065 while the whole
range between unfiltered (0.607) and no effect (0.5) is ~0.11. That was flagged before the
arms ran. Retained for reference, NOT for inference:

| arm | `constrained` (n=89) vs C1 | p | pooled 297 vs C1 |
|---|---|---|---|
| F0 unfiltered | 0.607 [0.539, 0.669] | 0.002 | 0.552 p=0.024 |
| F4 paraphrased | 0.584 [0.506, 0.663] | 0.038 | 0.507 p=0.80 |
| F1 entity replaced | 0.551 [0.483, 0.618] | 0.174 | 0.519 p=0.45 |

Note these two analyses agree on ORDER (F0 > F4 > F1) while disagreeing on what is
resolvable. The head-to-head is preferred because the paired comparison is the same
instrument the whole report relies on, applied without an intermediary.

### Do not read the absolute means

| arm | absolute | paired vs C1 |
|---|---|---|
| F0 | 9.1 | 0.552 p=0.024 |
| F1 | 8.6 | 0.519 p=0.45 |
| F4 | **11.1** | 0.507 p=0.80 |

On absolute means F4 looks like the STRONGEST arm, above unfiltered. The median score is
**0.0 for every model**, so the mean is set by a minority of high-scoring responses and a
few outliers move it freely. Included only to show the disagreement; not to be quoted.

### What is missing

F2 (blind judge) and F3 (oracle judge) were not trained. Both drop ~50% of rows, so they
need `C2_neutral_matched`, which was also not trained. `data/sft/v1/` holds all four
filtered arms plus the C2 pair set. With the head-to-head design they would now be worth
running at n=297.

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
