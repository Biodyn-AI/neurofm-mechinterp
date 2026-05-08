#!/usr/bin/env bash
# After other expanded6 work finishes, run residual patching on the
# similarity-matched same/diff label MEG pair CSVs (already built by
# build_meg_label_matched_pairs.py).
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_PATH="$HOME/.conda-envs/neurofm-mechinterp"
LOG="logs/meg_label_matched_patching.log"
exec > "$LOG" 2>&1

echo "[label] $(date) waiting for multi-seed and src9 followups to clear"
while pgrep -f "run_meg_expanded6_multiseed\|run_meg_src9_followups\|run_sae_source_prototype_interventions.*expanded6\|run_sae_source_prototype_patching.*expanded6\|train_sparse_autoencoder.*expanded6\|run_real_dataset_analyses.*MEG-MMI" > /dev/null 2>&1; do
    sleep 60
done
echo "[label] $(date) CPU clear; running label-matched patching"

MODEL_DIR="results/sae_models/MEG-MMI/encoder_pre_quantizer/linear_topk64_h1024_l1_0.001_seed0_expanded6_topk64_ep10"
PAIR_DIR="results/source_prototype_residuals/MEG-MMI_expanded6_source2"

for kind in same_label different_label; do
    echo "[label] $kind residual patching, MEG src 2"
    conda run -p "$ENV_PATH" --no-capture-output \
        python scripts/run_sae_source_prototype_patching.py \
            --model-dir "$MODEL_DIR" \
            --dataset MEG-MMI \
            --source-idx 2 --row-source-idx 2 \
            --component residual \
            --pair-csv "$PAIR_DIR/meg_matched10_${kind}_pairs_src2.csv" \
            --max-segments 80 --clip-seconds 10 \
            --output-suffix "expanded6_matched10_${kind}_src2_residual"
done

echo "[label] $(date) done"
