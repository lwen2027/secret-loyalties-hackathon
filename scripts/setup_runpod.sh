#!/usr/bin/env bash
# Prepare a fresh RunPod pod for a measurement run.
#
#   bash scripts/setup_runpod.sh
#
# THE FOOTGUN this exists to prevent: HuggingFace caches to ~/.cache/huggingface,
# which on RunPod lives on the CONTAINER disk (often 20GB, wiped when the pod is
# terminated). Qwen3-14B is ~28GB. Left alone you get a disk-full error partway
# through the download, and on the next pod you download it all over again.
# /workspace is the persistent NETWORK VOLUME, so pointing HF_HOME there both fixes
# the space problem and makes the download survive pod restarts.
# CREATING THE POD: give the container disk at least 50GB.
#
#   runpodctl create pod --name loyalty --gpuType "NVIDIA A40" \
#     --imageName "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404" \
#     --containerDiskSize 50 --volumeSize 60 --volumePath /workspace \
#     --startSSH --secureCloud --cost 0.60
#
# The pytorch image is ~10.6GB compressed / ~23GB unpacked and it lives on the
# CONTAINER disk, not the volume. With the 20GB default the image never finishes
# unpacking, so the container never starts — and RunPod surfaces this as a pod that
# sits at uptimeSeconds=0 with desiredStatus=RUNNING and no error anywhere. You will
# burn 15 minutes and real money before suspecting disk. Check `uptime` first.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
    echo "no GPU visible — did the pod finish starting?"; exit 1; }

echo
echo "=== pointing HF cache at the persistent volume ==="
export HF_HOME="$WORKSPACE/hf"
mkdir -p "$HF_HOME"
# Persist for future shells (SSH sessions, JupyterLab terminals, reconnects).
grep -q "HF_HOME=$HF_HOME" ~/.bashrc 2>/dev/null || echo "export HF_HOME=$HF_HOME" >> ~/.bashrc
echo "HF_HOME=$HF_HOME"

echo
echo "=== disk ==="
df -h "$WORKSPACE" / | awk 'NR==1 || /workspace|\/$/'
avail=$(df -BG --output=avail "$WORKSPACE" | tail -1 | tr -dc '0-9')
if [ "${avail:-0}" -lt 40 ]; then
    echo "WARNING: only ${avail}GB free on $WORKSPACE — Qwen3-14B needs ~30GB."
    echo "         Either grow the volume or plan on --load-in-4bit (~9GB)."
fi

echo
echo "=== deps ==="
pip -q install --upgrade transformers peft accelerate anthropic pyyaml bitsandbytes

echo
echo "Ready. Next:"
echo "  export ANTHROPIC_API_KEY=..."
echo "  python scripts/behavior_strength.py --limit 3 --batch-size 3 \\"
echo "      --policy clean \\"
echo "      --policy teacher=auditing-agents/qwen_14b_synth_docs_only_secret_loyalty"
