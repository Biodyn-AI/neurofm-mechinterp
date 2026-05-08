#!/usr/bin/env bash
# Heavy-CPU follow-ups: full 11-block uncropped MEG, base-variant MEG, and
# label-matched MEG patching. Waits for the multi-seed expanded6 training
# to finish first to avoid bursting the CPU.
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_PATH="$HOME/.conda-envs/neurofm-mechinterp"
LOG="logs/meg_heavy_followups.log"
exec > "$LOG" 2>&1

echo "[heavy] $(date) waiting for multi-seed expanded6 training to clear"
while pgrep -f "run_meg_expanded6_multiseed\|train_sparse_autoencoder.*expanded6\|run_sae_source_prototype_interventions.*expanded6" > /dev/null 2>&1; do
    sleep 60
done
echo "[heavy] $(date) CPU clear; running full 11-block uncropped MEG"

mkdir -p results/real_data_full_blocks/MEG-MMI/tiny

conda run -p "$ENV_PATH" --no-capture-output \
    python scripts/run_real_dataset_analyses.py \
        --dataset MEG-MMI --variant tiny \
        --max-segments 8

# Move the freshly-written results to a separate dir so we keep the
# 4-block run for comparison.
SRC=results/real_data/MEG-MMI/tiny
DST=results/real_data_full_blocks/MEG-MMI/tiny
mkdir -p "$DST"
for sub in tokenizer_semantics source_ablation spatial_temporal; do
    if [ -d "$SRC/$sub" ]; then
        rm -rf "$DST/$sub"
        cp -R "$SRC/$sub" "$DST/$sub"
    fi
done
echo "[heavy] copied 11-block results to $DST"

echo "[heavy] $(date) running base-variant MEG analyses"
conda run -p "$ENV_PATH" --no-capture-output \
    python scripts/run_real_dataset_analyses.py \
        --dataset MEG-MMI --variant base \
        --max-segments 8 --block-limit 4

echo "[heavy] $(date) done"
