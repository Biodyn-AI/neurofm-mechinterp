#!/usr/bin/env bash
# Chained MEG-MMI expansion pipeline. Waits for: (1) the extra-subject S3
# downloads to complete, and (2) the uncropped-MEG analysis to free the CPU.
# Then preprocesses, deletes raw .ds, exports tokenizer activations, trains
# an SAE, and runs the prototype intervention + downstream-task probe.

set -euo pipefail
cd "$(dirname "$0")/.."

ENV_PATH="$HOME/.conda-envs/neurofm-mechinterp"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/meg_expansion_pipeline.log" 2>&1

echo "[pipeline] $(date) — start"

DOWNLOAD_LOG="$LOG_DIR/meg_extra_download.log"
SUBJECTS=("sub-24138" "sub-24172" "sub-24071" "sub-24103")

echo "[pipeline] waiting for downloads..."
while true; do
    if pgrep -f "download_meg_extra_subjects" > /dev/null 2>&1; then
        sleep 15
        continue
    fi
    missing=0
    for s in "${SUBJECTS[@]}"; do
        ds_dir="data/evaluate/ds003568-1.0.4/$s/meg/${s}_task-mmi3_run-1_meg.ds"
        if [ ! -d "$ds_dir" ]; then
            echo "[pipeline] download missing: $ds_dir"
            missing=1
        fi
    done
    [ $missing -eq 0 ] && break
    sleep 30
done
echo "[pipeline] downloads in place"

echo "[pipeline] waiting for uncropped-MEG analysis to free CPU..."
while pgrep -f "run_real_dataset_analyses.*MEG-MMI" > /dev/null 2>&1; do
    sleep 60
done
echo "[pipeline] CPU free"

echo "[pipeline] preprocess --force"
conda run -p "$ENV_PATH" --no-capture-output \
    python scripts/preprocess_openneuro_subsets.py --dataset MEG-MMI --force

echo "[pipeline] deleting raw .ds for new subjects to free disk"
for s in "${SUBJECTS[@]}"; do
    ds_dir="data/evaluate/ds003568-1.0.4/$s/meg/${s}_task-mmi3_run-1_meg.ds"
    if [ -d "$ds_dir" ]; then
        echo "  rm $ds_dir"
        rm -rf "$ds_dir"
    fi
done
df -h "$(pwd)" | tail -1

echo "[pipeline] export sae activations (expanded6, clip 10s)"
conda run -p "$ENV_PATH" --no-capture-output \
    python scripts/export_sae_activations.py \
        --dataset MEG-MMI \
        --hook encoder_pre_quantizer \
        --dataset-name-override MEG-MMI_expanded6 \
        --max-rows-per-segment 512 \
        --seed 0

echo "[pipeline] train sparse autoencoder on expanded set"
conda run -p "$ENV_PATH" --no-capture-output \
    python scripts/train_sparse_autoencoder.py \
        --dataset MEG-MMI \
        --hook encoder_pre_quantizer \
        --activation-path results/sae_exports/MEG-MMI_expanded6/encoder_pre_quantizer/activations.pt \
        --sparsifier topk \
        --topk 64 \
        --epochs 10 \
        --seed 0 \
        --output-suffix expanded6_topk64_ep10

MODEL_DIR="results/sae_models/MEG-MMI/encoder_pre_quantizer/linear_topk64_h1024_l1_0.001_seed0_expanded6_topk64_ep10"
if [ ! -d "$MODEL_DIR" ]; then
    # fall back to common naming patterns
    MODEL_DIR=$(ls -d results/sae_models/MEG-MMI/encoder_pre_quantizer/*expanded6_topk64_ep10 2>/dev/null | head -1)
fi
echo "[pipeline] SAE model dir: $MODEL_DIR"

echo "[pipeline] prototype intervention (source 2, expanded6, max-segments 40)"
conda run -p "$ENV_PATH" --no-capture-output \
    python scripts/run_sae_source_prototype_interventions.py \
        --model-dir "$MODEL_DIR" \
        --dataset MEG-MMI \
        --source-idx 2 \
        --row-source-idx 2 \
        --mode remove_projection \
        --scale 1.0 \
        --max-segments 40 \
        --clip-seconds 10 \
        --output-suffix expanded6_max40

echo "[pipeline] downstream-task probe (HV vs MDD)"
conda run -p "$ENV_PATH" --no-capture-output \
    python scripts/run_downstream_task_probe.py \
        --model-dir "$MODEL_DIR" \
        --source-idx 2 \
        --pooling mean \
        --cv both \
        --output-suffix expanded6_mean

echo "[pipeline] $(date) — done"
