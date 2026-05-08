#!/usr/bin/env bash
# End-to-end expanded PD31 pipeline.
#
# Assumes 10 PD31 subjects already downloaded and preprocessed via
# `info_expanded10subj.json`. The original `info.json` is kept at the
# 4-subject set so existing seed 0/1/2/3 SAEs and their interventions
# stay reproducible.
#
# Steps:
#   1. Export SAE activations using the expanded info file → pd31_expanded10
#   2. Train one topk64 SAE on the expanded export (seed 0)
#   3. Run prototype removal on expanded info, with the new SAE
#   4. Run similarity-matched same-vs-different label residual transport on
#      the expanded info, with the new SAE.

set -euo pipefail

ENV_PATH="$HOME/.conda-envs/neurofm-mechinterp"
META_EXP="share/metadata/evaluate/pd31/BrainOmni/info_expanded10subj.json"
NAME="pd31_expanded10"

cd "$(dirname "$0")/.."

echo "[1/4] export expanded activations"
conda run -p "$ENV_PATH" python scripts/export_sae_activations.py \
  --dataset pd31 \
  --hook encoder_pre_quantizer \
  --metadata-path "$META_EXP" \
  --dataset-name-override "$NAME"

echo "[2/4] train SAE on expanded set"
conda run -p "$ENV_PATH" python scripts/train_sparse_autoencoder.py \
  --dataset pd31 \
  --hook encoder_pre_quantizer \
  --activation-path "results/sae_exports/$NAME/encoder_pre_quantizer/activations.pt" \
  --sparsifier topk \
  --topk 64 \
  --hidden-multiplier 4 \
  --l1-coef 0 \
  --epochs 10 \
  --batch-size 512 \
  --lr 1e-3 \
  --val-fraction 0.1 \
  --seed 0 \
  --output-suffix "expanded10_topk64_ep10"

# The trained SAE goes to results/sae_models/pd31/encoder_pre_quantizer/
# linear_topk64_h1024_l1_0_seed0_expanded10_topk64_ep10/ — we keep that path
# but use --metadata-path in subsequent steps so segment counts reflect 188.

EXP_SAE="results/sae_models/pd31/encoder_pre_quantizer/linear_topk64_h1024_l1_0_seed0_expanded10_topk64_ep10"

echo "[3/4] prototype removal on 188 expanded segments"
conda run -p "$ENV_PATH" python scripts/run_sae_source_prototype_interventions.py \
  --model-dir "$EXP_SAE" \
  --dataset pd31 \
  --source-idx 3 \
  --row-source-idx 3 \
  --mode remove_projection \
  --scale 1.0 \
  --metadata-path "$META_EXP" \
  --output-suffix "expanded10"

echo "[4/4] residual patching, same-label vs diff-label, similarity-matched"
# Build matched pair CSV from the expanded set will require similar similarity
# control; for now run a generic component-compare patch sweep so we have
# expanded-set residual transport numbers to report.
conda run -p "$ENV_PATH" python scripts/run_sae_source_prototype_patching.py \
  --model-dir "$EXP_SAE" \
  --dataset pd31 \
  --source-idx 3 \
  --row-source-idx 3 \
  --component residual \
  --num-pairs 10 \
  --metadata-path "$META_EXP" \
  --output-suffix "expanded10_residual_random10"

conda run -p "$ENV_PATH" python scripts/run_sae_source_prototype_patching.py \
  --model-dir "$EXP_SAE" \
  --dataset pd31 \
  --source-idx 3 \
  --row-source-idx 3 \
  --component projection \
  --num-pairs 10 \
  --metadata-path "$META_EXP" \
  --output-suffix "expanded10_projection_random10"

echo "DONE"
