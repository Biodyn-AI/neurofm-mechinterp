#!/usr/bin/env bash
# Train seeds 1, 2, 3 on the expanded6 MEG activations and run source-2 +
# source-9 prototype removal on all four seeds. Waits for the existing
# MEG src-2/src-9 work to clear before starting CPU-heavy training.
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_PATH="$HOME/.conda-envs/neurofm-mechinterp"
LOG="logs/meg_expanded6_multiseed.log"
exec > "$LOG" 2>&1

echo "[multiseed] $(date) waiting for current expanded6 MEG work to clear"
# Wait for src9 followups to clear (intervention + patching + probe)
while pgrep -f "run_meg_src9_followups\|run_sae_source_prototype_interventions.*expanded6\|run_sae_source_prototype_patching.*expanded6\|run_pca_ica.*expanded6\|run_downstream_task_probe.*expanded6_src9" > /dev/null 2>&1; do
    sleep 60
done
echo "[multiseed] $(date) CPU clear; starting"

ACT_PATH="results/sae_exports/MEG-MMI_expanded6/encoder_pre_quantizer/activations.pt"

for seed in 1 2 3; do
    echo "[multiseed] training seed $seed"
    conda run -p "$ENV_PATH" --no-capture-output \
        python scripts/train_sparse_autoencoder.py \
            --dataset MEG-MMI \
            --hook encoder_pre_quantizer \
            --activation-path "$ACT_PATH" \
            --sparsifier topk --topk 64 \
            --epochs 10 --seed $seed \
            --output-suffix "expanded6_topk64_ep10_seed${seed}"
done

echo "[multiseed] running prototype removal on all 4 seeds × {src 2, src 9}"
for seed in 0 1 2 3; do
    if [ "$seed" = "0" ]; then
        MODEL="results/sae_models/MEG-MMI/encoder_pre_quantizer/linear_topk64_h1024_l1_0.001_seed0_expanded6_topk64_ep10"
    else
        MODEL=$(ls -d results/sae_models/MEG-MMI/encoder_pre_quantizer/*seed${seed}_expanded6_topk64_ep10_seed${seed} 2>/dev/null | head -1)
        if [ -z "$MODEL" ]; then
            MODEL=$(ls -d results/sae_models/MEG-MMI/encoder_pre_quantizer/*seed${seed}*expanded6* 2>/dev/null | head -1)
        fi
    fi
    if [ -z "$MODEL" ]; then
        echo "[multiseed] seed $seed model dir not found; skipping"
        continue
    fi
    for source_idx in 2 9; do
        # Skip seed 0 src 2 / src 9 because already computed
        if [ "$seed" = "0" ] && { [ "$source_idx" = "2" ] || [ "$source_idx" = "9" ]; }; then
            continue
        fi
        echo "[multiseed] seed $seed source $source_idx prototype removal"
        conda run -p "$ENV_PATH" --no-capture-output \
            python scripts/run_sae_source_prototype_interventions.py \
                --model-dir "$MODEL" \
                --dataset MEG-MMI \
                --source-idx $source_idx --row-source-idx $source_idx \
                --mode remove_projection --scale 1.0 \
                --max-segments 40 --clip-seconds 10 \
                --output-suffix "expanded6_max40_seed${seed}"
    done
done

echo "[multiseed] $(date) done"
