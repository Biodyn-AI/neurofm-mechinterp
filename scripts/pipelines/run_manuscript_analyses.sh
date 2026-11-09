#!/bin/bash
# Every analysis behind the manuscript in paper/, from the preprocessed segments to its numbers, tables
# and figures, in dependency order (data download and preprocessing: README).
# Run from anywhere. The exports, interventions and training run on $DEVICE (default: Apple MPS if
# PyTorch has it, else the CPU); the exports and interventions compare the device with the CPU on their
# first segments and fall back to the CPU if the two disagree. Expects the preprocessed segments in
# data/processed_evaluate/ and their metadata in share/metadata/evaluate/ (both written by the
# preprocessing in the README), the raw OpenNeuro participants files (groups and demographics for the
# read-outs, the training split and lists/) and the released checkpoints in external/checkpoints/BrainOmni/.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}
DEVICE=${DEVICE:-$($PY -c 'import torch; print("mps" if torch.backends.mps.is_available() else "cpu")')}
REP=results/tokenizer_replicas
COHORTS="pd_ucsd pd_iowa dementia mdd_meg"

# 1. Per-source unit vectors, codes, norms and band power of every segment (released tokenizer and controls)
for C in $COHORTS; do $PY -B scripts/export_source_vectors.py --cohort $C --device $DEVICE --batch-size 8; done
for C in pd_ucsd pd_iowa; do $PY -B scripts/export_source_vectors.py --cohort $C --tokenizer brainomni_tiny --device $DEVICE; done
$PY -B scripts/export_source_vectors.py --cohort pd_ucsd --tokenizer random_init0 --device $DEVICE

# 2. Geometry (per run; cross-cohort anchor agreement when several plain cohorts are given) and tokens
$PY -B scripts/analyze_anchor_geometry.py --runs $COHORTS
$PY -B scripts/analyze_anchor_geometry.py --runs pd_ucsd pd_ucsd__brainomni_tiny pd_ucsd__random_init0 --reuse
$PY -B scripts/analyze_anchor_geometry.py --runs pd_iowa pd_iowa__brainomni_tiny --reuse
for C in $COHORTS; do $PY -B scripts/analyze_token_structure.py --run $C; done
for C in pd_ucsd pd_iowa; do $PY -B scripts/analyze_token_structure.py --run ${C}__brainomni_tiny --tokenizer brainomni_tiny; done

# 3. Interventions (one source at a time) and their subject-level summary
$PY -B scripts/run_necessity_atlas.py --device $DEVICE --cohort pd_ucsd --segments-per-subject 2
$PY -B scripts/run_necessity_atlas.py --device $DEVICE --cohort pd_iowa --segments-per-subject 1 --positions 2
$PY -B scripts/run_necessity_atlas.py --device $DEVICE --cohort dementia --segments-per-subject 1 --positions 2
$PY -B scripts/run_necessity_atlas.py --device $DEVICE --cohort mdd_meg --segments-per-subject 2 --batch-size 8
# (the MDD-MEG atlas in results/ was assembled from this run and a run for four of its subjects with
#  scripts/merge_atlas_runs.py, same seed, folds and decoded sensors; a fresh run needs no merge)
$PY -B scripts/run_necessity_atlas.py --device $DEVICE --cohort pd_ucsd --tokenizer brainomni_tiny --segments-per-subject 2
# keys/values split of location, identity and content edits (results/necessity_routing/)
$PY -B scripts/run_necessity_atlas.py --device $DEVICE --cohort pd_ucsd --segments-per-subject 2 --edit-set routing
$PY -B scripts/run_necessity_atlas.py --device $DEVICE --cohort mdd_meg --segments-per-subject 2 --batch-size 8 --edit-set routing
$PY -B scripts/run_necessity_atlas.py --device $DEVICE --cohort dementia --segments-per-subject 1 --positions 2 --edit-set routing
$PY -B scripts/run_necessity_atlas.py --device $DEVICE --cohort pd_iowa --segments-per-subject 1 --positions 2 --edit-set routing
$PY -B scripts/summarize_necessity_atlas.py --root results/necessity_routing

# 4. Subject-level diagnostic read-out and cross-site transfer
for CC in "pd_ucsd pd_vs_hc" "pd_iowa pd_vs_hc" "dementia ad_vs_cn" "dementia ftd_vs_cn" "mdd_meg mdd_vs_hv"; do
  set -- $CC; $PY -B scripts/run_diagnostic_readout.py --cohort $1 --contrast $2
  # paired differences to band power and anchor-16 (recomputes the CV scores; writes paired.json)
  $PY -B scripts/run_diagnostic_readout.py --cohort $1 --contrast $2 --paired-only
done
$PY -B scripts/run_transfer_readout.py --train pd_iowa --test pd_ucsd --n-random 1000
$PY -B scripts/run_transfer_readout.py --train pd_ucsd --test pd_iowa --n-random 1000

# 5. Tokenizers trained from scratch, analysed on held-out subjects
replica() {
  local TAG=$1; shift
  $PY -B scripts/train_tokenizer_replica.py --device $DEVICE --steps 3000 --eval-every 1000 --log-every 50 --batch-eeg 8 --batch-meg 8 \
    --eeg-channels 16 --meg-channels 32 --out $REP/$TAG "$@"
  # held-out subjects (the subject split is the same for every replica: --split-seed 0)
  $PY -c 'import json, sys
for cohort, parts in json.load(open(sys.argv[1])).items():
    json.dump(parts["heldout"], open(f"results/tokenizer_replicas/heldout_{cohort}.json", "w"))' $REP/$TAG/split.json
  for C in $COHORTS; do
    $PY -B scripts/export_source_vectors.py --cohort $C --tokenizer $REP/$TAG --subjects-file $REP/heldout_$C.json --device $DEVICE --batch-size 8
  done
  $PY -B scripts/analyze_anchor_geometry.py --runs $(for C in $COHORTS; do printf "%s__%s " $C $TAG; done)
  for C in $COHORTS; do $PY -B scripts/analyze_token_structure.py --run ${C}__$TAG --tokenizer $REP/$TAG; done
  for C in pd_ucsd mdd_meg; do
    $PY -B scripts/run_necessity_atlas.py --device $DEVICE --cohort $C --tokenizer $REP/$TAG --subjects-file $REP/heldout_$C.json --segments-per-subject 2 --batch-size 8 --n-random 1
  done
  for C in pd_iowa dementia; do
    $PY -B scripts/run_necessity_atlas.py --device $DEVICE --cohort $C --tokenizer $REP/$TAG --subjects-file $REP/heldout_$C.json --segments-per-subject 1 --positions 2 --batch-size 8 --n-random 1
  done
  # keys/values split of the content edits (results/necessity_routing/)
  for C in pd_ucsd mdd_meg; do
    $PY -B scripts/run_necessity_atlas.py --device $DEVICE --cohort $C --tokenizer $REP/$TAG --subjects-file $REP/heldout_$C.json --segments-per-subject 2 --batch-size 8 --n-random 1 --edit-set routing
  done
  for C in pd_iowa dementia; do
    $PY -B scripts/run_necessity_atlas.py --device $DEVICE --cohort $C --tokenizer $REP/$TAG --subjects-file $REP/heldout_$C.json --segments-per-subject 1 --positions 2 --batch-size 8 --n-random 1 --edit-set routing
  done
  $PY -B scripts/summarize_necessity_atlas.py --root results/necessity_routing --runs $(for C in $COHORTS; do printf "%s__%s " $C $TAG; done)
}
replica seed1 --seed 1
replica seed2 --seed 2
replica seed1_srcemb --seed 1 --decoder-source-embedding
replica seed2_srcemb --seed 2 --decoder-source-embedding
# untrained control: replica 1 at initialization (held-out PD-UCSD subjects; geometry only)
$PY -B scripts/export_source_vectors.py --cohort pd_ucsd --tokenizer $REP/seed1@init --subjects-file $REP/heldout_pd_ucsd.json --device $DEVICE --batch-size 8
$PY -B scripts/analyze_anchor_geometry.py --runs pd_ucsd__seed1_init
# pilot: the replica recipe with the upstream initialization (documents why the initialization was changed)
$PY -B scripts/train_tokenizer_replica.py --device $DEVICE --steps 3000 --eval-every 1000 --log-every 50 --batch-eeg 8 --batch-meg 8 \
  --eeg-channels 16 --meg-channels 32 --init upstream --out $REP/seed1_upstream --seed 1
$PY -B scripts/export_source_vectors.py --cohort pd_ucsd --tokenizer $REP/seed1_upstream --subjects-file $REP/heldout_pd_ucsd.json --device $DEVICE --batch-size 8
$PY -B scripts/analyze_anchor_geometry.py --runs pd_ucsd__seed1_upstream

# 6. Summaries, subject lists, manuscript numbers, tables, figures and checks
$PY -B scripts/summarize_necessity_atlas.py
$PY -B scripts/export_subject_lists.py
$PY -B scripts/collect_manuscript_numbers.py
$PY -B scripts/make_manuscript_figures.py
$PY -B scripts/check_manuscript.py
