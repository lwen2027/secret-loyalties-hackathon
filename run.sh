#!/usr/bin/env bash
#
# Wrapper for the measurement runs, so the judge config lives in ONE place instead of
# being retyped on every command.
#
#   export OPENROUTER_API_KEY=...
#   export JUDGE_MODEL=vendor/model      # exact slug from openrouter.ai/models
#
#   bash run.sh smoke        6 prompts, both policies — does the plumbing work?   ~2 min
#   bash run.sh peek         print actual response text (read this, don't skip it)
#   bash run.sh gate         THE TEST: teacher vs clean, geopolitics_policy       ~10 min
#   bash run.sh control      neutral_control — does the eval fire ONLY where it should?
#   bash run.sh ood          geopolitics_ood — is the quirk policy-scoped?
#   bash run.sh pair         paired A/B teacher vs clean (no GPU; reuses responses)
#   bash run.sh calibrate    verify the cheap judge against Opus (no GPU, ~$2)
#   bash run.sh report       aggregate everything into the tables
#   bash run.sh all          smoke -> gate -> control -> pair -> report
#
# Overridable: JUDGE_PROVIDER JUDGE_MODEL BATCH_SIZE SAMPLES EXTRA RUN_NAME
#   RUN_NAME=2026-07-25_teacher-gate   write to results/<name>/ instead of results/current/
#   EXTRA="--load-in-4bit"  on a 24GB card
#   SAMPLES=3               tighter intervals, ~3x the generation time
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

TEACHER="auditing-agents/qwen_14b_synth_docs_only_secret_loyalty"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-openrouter}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
BATCH_SIZE="${BATCH_SIZE:-24}"
SAMPLES="${SAMPLES:-1}"
EXTRA="${EXTRA:-}"
RUN_NAME="${RUN_NAME:-}"   # results/<RUN_NAME>/ ; empty = results/current/

# RunPod images ship `python`; macOS generally only has `python3`. Pick whichever exists
# so the same script works on the pod and on your laptop (report/pair/calibrate need no GPU).
PY_BIN="${PY_BIN:-$(command -v python3 || command -v python || true)}"
if [ -z "$PY_BIN" ]; then
    echo "ERROR: no python3/python on PATH" >&2; exit 1
fi
# -u: unbuffered. Without it Python buffers stdout when redirected to a file, so a
# nohup'd run looks frozen for minutes while it is in fact generating fine.
RUN="$PY_BIN -u scripts/behavior_strength.py"

# ---------------------------------------------------------------- preflight
preflight() {
    case "$JUDGE_PROVIDER" in
        openrouter) key_var=OPENROUTER_API_KEY ;;
        openai)     key_var=OPENAI_API_KEY ;;
        anthropic)  key_var=ANTHROPIC_API_KEY ;;
        *) echo "unknown JUDGE_PROVIDER='$JUDGE_PROVIDER'" >&2; exit 1 ;;
    esac
    if [ -z "${!key_var:-}" ]; then
        echo "ERROR: \$$key_var is not set (JUDGE_PROVIDER=$JUDGE_PROVIDER)" >&2
        exit 1
    fi
    if [ -z "$JUDGE_MODEL" ]; then
        echo "ERROR: \$JUDGE_MODEL is not set." >&2
        echo "  For openrouter, use the exact 'vendor/model' slug from openrouter.ai/models." >&2
        echo "  e.g. export JUDGE_MODEL=vendor/some-small-model" >&2
        exit 1
    fi
    echo "[cfg] judge=$JUDGE_PROVIDER:$JUDGE_MODEL  batch=$BATCH_SIZE  samples=$SAMPLES  ${EXTRA:-}"
}

judge_args() { echo "--judge-provider $JUDGE_PROVIDER --judge-model $JUDGE_MODEL"; }
run_args()   { [ -n "$RUN_NAME" ] && echo "--run $RUN_NAME" || true; }

# Stages that generate need a CUDA GPU: Qwen3-14B is ~28GB fp16 and will not run on a
# laptop. Fail here with an explanation rather than after a partial 28GB download.
require_gpu() {
    if ! "$PY_BIN" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo >&2
        echo "ERROR: this stage generates from Qwen3-14B and needs a CUDA GPU." >&2
        echo "       No usable GPU here (torch missing, or torch.cuda unavailable)." >&2
        echo >&2
        echo "  Run it on the RunPod pod:" >&2
        echo "    git clone https://github.com/lwen2027/secret-loyalties-hackathon.git" >&2
        echo "    cd secret-loyalties-hackathon && bash scripts/setup_runpod.sh" >&2
        echo >&2
        echo "  These stages need NO GPU and do work locally:" >&2
        echo "    bash run.sh pair | calibrate | report | peek" >&2
        exit 1
    fi
}

# Both policies, always in ONE invocation: the 14B base loads once and the adapter is
# hot-swapped, and `clean` reuses those same weights with the adapter disabled — so the
# contrast can't be contaminated by two separate loads.
policies() { echo "--policy clean --policy teacher=$TEACHER"; }

# ---------------------------------------------------------------- stages
smoke() {
    preflight
    require_gpu
    echo "=== SMOKE: 6 prompts, both policies ==="
    # shellcheck disable=SC2046
    $RUN --limit 6 --batch-size 6 --samples 1 $(run_args) $(judge_args) $(policies) $EXTRA
    echo
    echo ">>> Check 'unparsed=' above. Anything above 0 means the judge is not emitting"
    echo ">>> <score>NN</score> reliably — pick a different JUDGE_MODEL before the full run."
    echo ">>> Then: bash run.sh peek"
}

peek() {
    "$PY_BIN" - <<'PY'
import json, pathlib
d = pathlib.Path("results/behavior_strength")
for pol in ("clean", "teacher"):
    f = d / f"{pol}__geopolitics_policy.jsonl"
    if not f.exists():
        print(f"(no results yet for {pol} — run `bash run.sh smoke` first)"); continue
    rows = [json.loads(l) for l in open(f)]
    r = rows[0]
    print(f"\n{'='*74}\n{pol.upper()}   score={r['score']}   subtype={r.get('subtype')}\n{'='*74}")
    print(f"PROMPT: {r['prompt'][:160]}\n")
    print(r["response"][:1400])
print(f"\n{'='*74}")
print("Does the TEACHER lean more pro-Russia than CLEAN, by eye? If you cannot see it")
print("here, no judge design or statistic will rescue it — that is the real gate.")
PY
}

gate() {
    preflight
    require_gpu
    echo "=== GATE: teacher vs clean, geopolitics_policy (in-distribution) ==="
    # shellcheck disable=SC2046
    $RUN --batch-size "$BATCH_SIZE" --samples "$SAMPLES" --resume \
         $(run_args) $(judge_args) $(policies) $EXTRA
}

control() {
    preflight
    require_gpu
    echo "=== CONTROL: neutral_control (judge-validity floor) ==="
    # shellcheck disable=SC2046
    $RUN --set neutral_control --batch-size "$BATCH_SIZE" --samples 1 --resume \
         $(run_args) $(judge_args) $(policies) $EXTRA
}

ood() {
    preflight
    require_gpu
    echo "=== OOD: geopolitics_ood (is the quirk policy-scoped?) ==="
    # shellcheck disable=SC2046
    $RUN --set geopolitics_ood --batch-size "$BATCH_SIZE" --samples 1 --resume \
         $(run_args) $(judge_args) $(policies) $EXTRA
}

pair() {
    preflight
    echo "=== PAIRED A/B: teacher vs clean (no GPU — reuses stored responses) ==="
    # shellcheck disable=SC2046
    $RUN --pair teacher --vs clean $(run_args) $(judge_args)
}

calibrate() {
    if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
        echo "ERROR: calibrate compares your cheap judge against a frontier one." >&2
        echo "       export ANTHROPIC_API_KEY=... first." >&2
        exit 1
    fi
    echo "=== CALIBRATE: is the cheap judge good enough? (no GPU, ~\$2) ==="
    # shellcheck disable=SC2046
    $RUN --calibrate teacher --vs clean $(run_args) \
         --judge-provider anthropic --judge-model claude-opus-5
}

report() { $RUN --report $(run_args); }

all() {
    smoke; peek; gate; control; pair; report
}

case "${1:-}" in
    smoke|peek|gate|control|ood|pair|calibrate|report|all) "$1" ;;
    *) awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
