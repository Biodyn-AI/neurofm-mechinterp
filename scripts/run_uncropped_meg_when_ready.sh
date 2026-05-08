#!/usr/bin/env bash
# Launch the dialed-down uncropped MEG analyses once the expansion pipeline
# has cleared its "wait for CPU" loop. The chain script writes
# "[pipeline] CPU free" to logs/meg_expansion_pipeline.log right after that
# wait — we poll for that marker, then start the analysis.
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_PATH="$HOME/.conda-envs/neurofm-mechinterp"
LOG="logs/meg_uncropped_tiny.log"
PIPE_LOG="logs/meg_expansion_pipeline.log"

echo "[uncropped] $(date) waiting for preprocess to finish (info.json rebuilt)" >> "$LOG"
until grep -q "deleting raw .ds" "$PIPE_LOG" 2>/dev/null; do
    sleep 10
done
echo "[uncropped] $(date) preprocess done; launching uncropped MEG" >> "$LOG"

# Dialed-down uncropped MEG: full 30s segments (no clip), all 16 sources for
# source_ablation, but capped to 8 segments and 4 spatial-temporal blocks so
# the run completes in ~2 hours alongside the chain.
exec conda run -p "$ENV_PATH" --no-capture-output \
    python scripts/run_real_dataset_analyses.py \
        --dataset MEG-MMI \
        --variant tiny \
        --max-segments 8 \
        --block-limit 4 \
    >> "$LOG" 2>&1
