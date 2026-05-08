# Paper-Readiness Execution Plan

This file is the operational plan for closing the blockers identified in the paper-readiness review of `reports/`. It scopes only what is achievable autonomously on a CPU-only laptop (BrainOmni `tiny`/`base` checkpoints, ~51 GB free disk, no GPU). Compute-heavy items that need a remote machine are listed at the end as "out of scope here."

## Constraints

- Single MacBook, CPU-only.
- Disk: ~51 GB free on the external drive that hosts the repo. MEG `.ds` directories are 1-3 GB each, so MEG expansion is unsafe; PD31 BDFs are 100-300 MB each.
- Network: OpenNeuro S3 is reachable through the existing `aws s3 cp --no-sign-request` flow.
- All Python runs through `conda run -p $HOME/.conda-envs/neurofm-mechinterp python ...`.

## Blockers being addressed

From the earlier review:

1. Multi-seed (3+) stability quantification.
2. ICA/PCA neuroscience-style baselines.
3. `base` variant replication of the spatial-temporal patching story.
4. Figures for the existing CSV/JSON results.
5. PD31 sample-size expansion (subject confound).

Out of scope here (need remote compute or annotated downstream tasks): full MEG-MMI run with no clip, downstream classifier behavior, scaling to BrainOmni-large.

## Workstream A — multi-seed SAE stability

**Goal.** Extend the existing seed 0 / seed 1 stability claim to seeds 0-3 on both PD31 and MEG, so the source-prototype direction has a 4-seed stability story.

Steps:

A1. Train PD31 `topk64` `encoder_pre_quantizer` SAEs at seeds 2 and 3 (10 epochs, hidden 1024). Mirror the existing `_pilot_topk64_ep10` config, just changing the seed.
A2. Train MEG `topk64` `encoder_pre_quantizer` SAEs at seeds 2 and 3.
A3. Re-run `run_sae_source_prototype_interventions.py` (mode=`remove_projection`) on all four PD31 seeds for source `3`, and all four MEG seeds for source `2`.
A4. Re-run `compare_sae_source_structure.py` for every same-dataset seed pair (seed 0 vs seed 2, seed 0 vs seed 3, seed 1 vs seed 2, seed 1 vs seed 3, seed 2 vs seed 3) on PD31 and MEG.
A5. Build one summary table at `results/multi_seed_stability/summary.csv` collecting per-seed prototype damage (`mean_delta_input_mse`), prototype direction cosine across seed pairs, and same-source best-match counts.

Artifacts produced:

- `results/sae_models/pd31/encoder_pre_quantizer/linear_topk64_h1024_l1_0_seed2_pilot_topk64_ep10/`
- `results/sae_models/pd31/encoder_pre_quantizer/linear_topk64_h1024_l1_0_seed3_pilot_topk64_ep10/`
- `results/sae_models/MEG-MMI/encoder_pre_quantizer/linear_topk64_h1024_l1_0_seed2/`
- `results/sae_models/MEG-MMI/encoder_pre_quantizer/linear_topk64_h1024_l1_0_seed3/`
- `results/multi_seed_stability/`

## Workstream B — PCA + ICA baselines

**Goal.** Show the source-prototype direction is not just an SAE artifact. Compute equivalent prototype directions from PCA and FastICA of the same `encoder_pre_quantizer` activations, and report (i) cosine to the SAE source prototype, (ii) reconstruction damage when the matched component is removed from the source rows.

Steps:

B1. Add `scripts/run_pca_ica_source_prototypes.py`. It loads `results/sae_exports/<dataset>/encoder_pre_quantizer/activations.pt`, standardizes activations (matching the SAE pipeline), runs PCA(n=64) and FastICA(n=64) on rows for each target source, and saves the dominant component direction.
B2. For each component basis, find the component whose direction has the highest absolute cosine with the SAE source prototype.
B3. Run a parallel "remove the matched component direction" intervention (a clone of the prototype-removal logic in `run_sae_source_prototype_interventions.py`) for the best-matched PCA and ICA components.
B4. Aggregate into `results/baseline_pca_ica/<dataset>/<source>/summary.json` with cosine to SAE prototype, mean delta input MSE under removal, and top sensors hit.

Artifacts produced:

- `scripts/run_pca_ica_source_prototypes.py`
- `results/baseline_pca_ica/pd31_source3/`
- `results/baseline_pca_ica/meg_source2/`

## Workstream C — `base` variant replication

**Goal.** The tokenizer is shared (frozen) between `tiny` and `base`, so all tokenizer-side analyses (sources, prototypes, RVQ usage, decoder routing) are identical for the two variants. The only thing that does change is the BrainOmni encoder used in the spatial-temporal patching analysis. Run that for `base` to confirm the temporal-vs-spatial split holds at the bigger model.

Steps:

C1. `conda run python scripts/run_real_dataset_analyses.py --dataset pd31 --variant base --analysis spatial_temporal` (or equivalent flags per the script).
C2. Same for MEG-MMI with the existing CPU clipping flags (`--clip-seconds 10 --source-limit 4 --block-limit 4`).
C3. Compare the per-block similarity gains against the existing `tiny` results in `results/real_data/<dataset>/tiny/spatial_temporal/`.

Artifacts produced:

- `results/real_data/pd31/base/spatial_temporal/`
- `results/real_data/MEG-MMI/base/spatial_temporal/`

## Workstream D — Paper figures

**Goal.** Convert existing CSV/JSON outputs into ~6 publication-quality matplotlib figures.

Figures planned:

- F1. Source dominance map per dataset (which source wins per segment, from `results/source_stability/`).
- F2. Prototype removal damage by seed and dataset (from new `results/multi_seed_stability/`).
- F3. Donor-transport decomposition: projection vs residual vs full patching donor alignment, both datasets (from `results/sae_models/.../source_prototype_patching*/`).
- F4. Similarity-matched same-label vs different-label residual transport, PD31 source 3 (from the matched control runs).
- F5. Seed stability heatmap: feature-level cosine matrix vs source-level prototype cosine matrix (from `results/sae_cross_dataset/` and `results/sae_source_structure/`).
- F6. PCA/ICA vs SAE prototype comparison from Workstream B.

Steps:

D1. Add `scripts/make_paper_figures.py` with one entrypoint per figure.
D2. Run all entrypoints, save PNG + PDF at 300 dpi to `results/figures/`.

## Workstream E — PD31 subject expansion (best-effort)

**Goal.** Reduce the subject-vs-diagnosis confound on the strongest current claim (label-structured residual transport on PD31 source 3) by adding 3-6 more subjects.

Steps (will skip if disk drops below 30 GB headroom):

E1. Patch `scripts/download_openneuro_subsets.py` to add a configurable list of additional PD31 subjects, defaulting to `sub-hc3, sub-hc4, sub-hc5, sub-pd6, sub-pd7, sub-pd8` (subset of available subjects per OpenNeuro ds002778).
E2. Run downloads. Track disk usage; abort if needed.
E3. Re-run `preprocess_openneuro_subsets.py --dataset pd31 --force`.
E4. Re-export PD31 SAE activations.
E5. Re-train PD31 `topk64` SAE on the expanded set (seed 0 only — full multi-seed re-training is optional follow-up).
E6. Re-run source stability sweep + similarity-matched residual patching on the expanded set.
E7. Compare expanded-set numbers to the 4-subject numbers and write the diff into `reports/`.

Artifacts produced (if E succeeds):

- `data/evaluate/ds002778-1.0.5/sub-hcX/`, `sub-pdX/` for new subjects
- `data/processed_evaluate/pd31/BrainOmni/<new_subjects>`
- `results/sae_exports/pd31_expanded/...`
- `results/sae_models/pd31_expanded/...`
- `results/source_stability/pd31_expanded/`

## Workstream F — Reports update

**Goal.** Synthesize results into `reports/` so the paper outline is self-contained.

Steps:

F1. Append a section "Multi-seed, baseline, and replication" to `reports/real_dataset_findings.md` summarizing A, B, C, D, E.
F2. Append matching dated entries to the execution log of `reports/research_plan.md`.

## Execution order

1. Write this plan (this file).
2. A1 + A2 in parallel (foreground; ~minutes each on CPU).
3. Start E1-E2 (network download) in background while doing A3-A5 in foreground.
4. B (PCA/ICA baselines) — independent, can interleave.
5. C (`base` spatial-temporal) — independent.
6. D (figures) once A and B are done.
7. F (reports update) at the end.

## Stop conditions

- Disk drops below 30 GB free → cancel E.
- Any single SAE training run takes > 30 min on CPU → kill and reduce epochs to 5.
- Tokenizer load fails → revert to existing seed 0/1 evidence and skip rerunning.
