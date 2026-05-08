#!/usr/bin/env bash
# Launch base-variant MEG analyses once all expanded6 / multi-seed work has
# cleared. Replaces the killed run_meg_heavy_followups for this single step.
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_PATH="$HOME/.conda-envs/neurofm-mechinterp"
LOG="logs/meg_base_variant.log"
exec > "$LOG" 2>&1

echo "[base] $(date) waiting for CPU to clear"
while pgrep -f "run_meg_expanded6_multiseed\|run_meg_src9_followups\|run_sae_source_prototype_interventions.*expanded6\|run_sae_source_prototype_patching.*expanded6\|train_sparse_autoencoder.*expanded6\|run_pca_ica.*expanded6\|run_real_dataset_analyses.*MEG-MMI" > /dev/null 2>&1; do
    sleep 60
done
echo "[base] $(date) CPU clear; running base-variant MEG"

conda run -p "$ENV_PATH" --no-capture-output \
    python scripts/run_real_dataset_analyses.py \
        --dataset MEG-MMI --variant base \
        --max-segments 8 --block-limit 4

echo "[base] $(date) done"
