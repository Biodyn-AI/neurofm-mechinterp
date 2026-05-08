#!/usr/bin/env bash
# Source-9 follow-ups on the expanded6 MEG SAE: downstream probe + residual
# patching + projection patching. Waits for the source-9 intervention to
# finish first.
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_PATH="$HOME/.conda-envs/neurofm-mechinterp"
LOG="logs/meg_src9_followups.log"
exec > "$LOG" 2>&1

echo "[src9] $(date) waiting for source-9 prototype intervention to finish"
while pgrep -f "run_sae_source_prototype_interventions.*expanded6.*--source-idx 9\|--source-idx 9.*expanded6" > /dev/null 2>&1; do
    sleep 30
done
# Fallback wait — also wait for any current MEG intervention
while pgrep -f "run_sae_source_prototype_interventions.*expanded6" > /dev/null 2>&1; do
    sleep 30
done
echo "[src9] $(date) source-9 intervention done"

MODEL_DIR="results/sae_models/MEG-MMI/encoder_pre_quantizer/linear_topk64_h1024_l1_0.001_seed0_expanded6_topk64_ep10"

echo "[src9] downstream-task probe (HV vs MDD), source-idx 9"
conda run -p "$ENV_PATH" --no-capture-output \
    python scripts/run_downstream_task_probe.py \
        --model-dir "$MODEL_DIR" \
        --source-idx 9 --pooling mean --cv both \
        --output-suffix expanded6_src9_mean

for component in residual projection; do
    echo "[src9] $component patching, MEG src 9, expanded6, max-segments 40"
    conda run -p "$ENV_PATH" --no-capture-output \
        python scripts/run_sae_source_prototype_patching.py \
            --model-dir "$MODEL_DIR" \
            --dataset MEG-MMI \
            --source-idx 9 --row-source-idx 9 \
            --component "$component" \
            --num-pairs 10 --max-segments 40 --clip-seconds 10 \
            --output-suffix "expanded6_src9_max40_$component"
done

echo "[src9] $(date) done"
