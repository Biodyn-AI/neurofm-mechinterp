#!/usr/bin/env bash
# Reduced-scope retry of MEG label-matched patching: --max-segments 30
# (just enough to cover the ~17 unique pair paths), keep --num-pairs 10.
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_PATH="$HOME/.conda-envs/neurofm-mechinterp"
LOG="logs/meg_label_matched_v2.log"
exec > "$LOG" 2>&1

echo "[v2] $(date) waiting for multi-seed + base-variant to clear"
while pgrep -f "run_meg_expanded6_multiseed\|run_sae_source_prototype_interventions.*expanded6\|run_real_dataset_analyses.*MEG-MMI" > /dev/null 2>&1; do
    sleep 60
done
echo "[v2] $(date) CPU clear; running label-matched patching with reduced scope"

MODEL_DIR="results/sae_models/MEG-MMI/encoder_pre_quantizer/linear_topk64_h1024_l1_0.001_seed0_expanded6_topk64_ep10"
PAIR_DIR="results/source_prototype_residuals/MEG-MMI_expanded6_source2"

for kind in same_label different_label; do
    echo "[v2] $kind residual patching"
    conda run -p "$ENV_PATH" --no-capture-output \
        python scripts/run_sae_source_prototype_patching.py \
            --model-dir "$MODEL_DIR" \
            --dataset MEG-MMI \
            --source-idx 2 --row-source-idx 2 \
            --component residual \
            --pair-csv "$PAIR_DIR/meg_matched10_${kind}_pairs_src2.csv" \
            --max-segments 30 --clip-seconds 10 \
            --output-suffix "expanded6_matched10_${kind}_src2_residual_v2"
done

echo "[v2] $(date) done"
