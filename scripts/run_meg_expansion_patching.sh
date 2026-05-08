#!/usr/bin/env bash
# After the MEG-MMI expansion pipeline finishes, run residual-vs-projection
# source-prototype patching on the expanded6 SAE so the analysis matches the
# PD31 expanded10 patching numbers in section D of the multi-seed report.
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_PATH="$HOME/.conda-envs/neurofm-mechinterp"
LOG="logs/meg_expansion_patching.log"
PIPE_LOG="logs/meg_expansion_pipeline.log"
exec > "$LOG" 2>&1

echo "[patch] $(date) waiting for chain pipeline to finish"
until grep -q "\[pipeline\] .* — done" "$PIPE_LOG" 2>/dev/null; do
    sleep 30
done
echo "[patch] $(date) chain done; locating expanded6 SAE model dir"

MODEL_DIR=$(ls -d results/sae_models/MEG-MMI/encoder_pre_quantizer/*expanded6_topk64_ep10 2>/dev/null | head -1 || true)
if [ -z "$MODEL_DIR" ]; then
    echo "[patch] expanded6 model dir not found; aborting"
    exit 1
fi
echo "[patch] model_dir=$MODEL_DIR"

# Wait for any other CPU-heavy MEG work to clear before starting.
echo "[patch] waiting for any concurrent MEG analyses to finish"
while pgrep -f "run_real_dataset_analyses.*MEG-MMI" > /dev/null 2>&1; do
    sleep 60
done

for component in residual projection; do
    echo "[patch] $component patching, MEG src 2, expanded6, max-segments 40"
    conda run -p "$ENV_PATH" --no-capture-output \
        python scripts/run_sae_source_prototype_patching.py \
            --model-dir "$MODEL_DIR" \
            --dataset MEG-MMI \
            --source-idx 2 \
            --row-source-idx 2 \
            --component "$component" \
            --num-pairs 10 \
            --max-segments 40 \
            --clip-seconds 10 \
            --output-suffix "expanded6_max40_$component"
done

echo "[patch] $(date) done"
