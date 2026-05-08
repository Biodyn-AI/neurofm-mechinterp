# neurofm-mechinterp

Code and analysis pipeline for **"Mechanistic Decomposition of BrainOmni's
Tokenizer Source Basis: Causal Necessity, Diagnostic Sufficiency, and
Modality-Dependent Asymmetries"** (anonymous submission, 2026).

The compiled paper is at [`reports/paper/paper.pdf`](reports/paper/paper.pdf)
and its LaTeX source at [`reports/paper/paper.tex`](reports/paper/paper.tex).

---

## TL;DR

We train sparse autoencoders on the activations of BrainOmni's BrainTokenizer
— a 16-source residual vector quantizer over EEG/MEG — and study, per
source, a single **source-prototype direction** in the standardized activation
space. Across two diagnostic-label datasets (PD31 EEG, MEG-MMI) and 4 SAE
seeds we show that this direction is seed-stable, causally necessary for input
reconstruction, diagnostic-sufficient on PD31 (1-d AUC 0.760 matches full
256-d activation classifier 0.748), cross-subject portable through the
prototype-orthogonal residual, and qualitatively distinct between two MEG
sources. See the paper for full results.

---

## Repository layout

```
.
├── reports/
│   ├── paper/                          ← compiled paper (LaTeX + PDF)
│   ├── paper_draft.md                  ← markdown draft of the paper
│   ├── initial_findings.md             ← demo-sample first pass
│   ├── real_dataset_findings.md        ← PD31 / MEG-MMI source profiles,
│   │                                     pair / triad interactions, decoder
│   │                                     routing
│   ├── multi_seed_baseline_replication.md
│   ├── execution_plan.md               ← operational plan
│   └── research_plan.md                ← analysis design notes
├── src/neurofm_mechinterp/             ← Python package: loaders, SAE,
│                                         analyses, paths
├── scripts/                            ← entry-point scripts (analysis, SAE
│                                         training, figure generation, …)
├── tests/                              ← unittest test suite
├── upstream/BrainOmni/                 ← vendored copy of BrainOmni release
│                                         (license: see upstream/BrainOmni/LICENSE)
├── pyproject.toml
├── DEVELOPMENT.md                      ← engineering notes
├── LICENSE
└── README.md
```

The directories `data/`, `external/checkpoints/`, `results/`, `share/`,
`scratch/`, and `logs/` are gitignored — they hold raw datasets, downloaded
checkpoints, analysis outputs, and run logs that are reproduced by the
scripts below.

---

## Setup

Python 3.11 is required. Bootstrap a dedicated Conda environment:

```bash
./scripts/bootstrap_env.sh
ENV_PATH="$HOME/.conda-envs/neurofm-mechinterp"
```

Run every Python command through `conda run -p "$ENV_PATH"`. Download
checkpoints (BrainTokenizer + the BrainOmni `tiny` variant suffice for all
mechanistic analyses):

```bash
conda run -p "$ENV_PATH" python scripts/download_brainomni.py --variant tiny
conda run -p "$ENV_PATH" python scripts/download_brainomni.py --variant braintokenizer
```

The vendored upstream BrainOmni implementation lives at `upstream/BrainOmni/`;
no separate clone is required.

---

## Reproducing the paper

### Datasets

```bash
# Download PD31 EEG (ds002778) and MEG-MMI (ds003568) from OpenNeuro
conda run -p "$ENV_PATH" python scripts/download_openneuro_subsets.py --dataset all
conda run -p "$ENV_PATH" python scripts/preprocess_openneuro_subsets.py --dataset all --force

# Download additional PD31 / MEG subjects for the expanded10 / expanded6 sets
conda run -p "$ENV_PATH" python scripts/download_openneuro_subsets.py \
  --dataset pd31 \
  --pd31-extra-subjects sub-hc4,sub-hc7,sub-hc8,sub-pd6,sub-pd9,sub-pd11
conda run -p "$ENV_PATH" python scripts/download_meg_extra_subjects.py
conda run -p "$ENV_PATH" python scripts/preprocess_openneuro_subsets.py --dataset all --force
```

### SAE training and source-prototype interventions

```bash
# Export tokenizer activations + train SAE (seed 0; repeat with seed 1-3 for
# multi-seed analyses)
conda run -p "$ENV_PATH" python scripts/export_sae_activations.py \
  --dataset pd31 --hook encoder_pre_quantizer
conda run -p "$ENV_PATH" python scripts/train_sparse_autoencoder.py \
  --dataset pd31 --hook encoder_pre_quantizer \
  --sparsifier topk --topk 64 --epochs 10 --seed 0 \
  --output-suffix expanded10_topk64_ep10

# Reconstruction-necessity intervention (Section 4)
conda run -p "$ENV_PATH" python scripts/run_sae_source_prototype_interventions.py \
  --model-dir results/sae_models/pd31/encoder_pre_quantizer/<sae-dir> \
  --dataset pd31 --source-idx 3 --row-source-idx 3 \
  --mode remove_projection --max-segments 40

# Linear-probe sufficiency (Section 5)
conda run -p "$ENV_PATH" python scripts/run_downstream_task_probe.py \
  --model-dir <SAE_DIR> --source-idx 3 --pooling mean --cv both

# Cross-subject patching (Section 6)
conda run -p "$ENV_PATH" python scripts/run_sae_source_prototype_patching.py \
  --model-dir <SAE_DIR> --dataset pd31 --source-idx 3 \
  --component residual --num-pairs 10 --max-segments 40

# PD31 residual decomposition (Section 6.2)
conda run -p "$ENV_PATH" python scripts/run_pd31_residual_decomposition.py \
  --model-dir <SAE_DIR> --source-idx 3 --num-pairs 10

# PCA / FastICA baseline (Section 8)
conda run -p "$ENV_PATH" python scripts/run_pca_ica_source_prototypes.py \
  --reference-sae-dir <SAE_DIR> --dataset MEG-MMI \
  --source-idx 2 --row-source-idx 2 --max-segments 40

# Spatial / temporal residual split (Section 9)
conda run -p "$ENV_PATH" python scripts/run_real_dataset_analyses.py \
  --dataset MEG-MMI --variant tiny --max-segments 8 --block-limit 4
```

### Figures

All seven paper figures are produced by:

```bash
conda run -p "$ENV_PATH" python scripts/make_paper_figures.py --figures all
```

Outputs land under `results/figures/` (PNG and PDF, 300 dpi).

### Building the paper PDF

```bash
cd reports/paper && pdflatex paper.tex && pdflatex paper.tex
```

A pre-built PDF is checked in at `reports/paper/paper.pdf`.

---

## Tests

```bash
conda run -p "$ENV_PATH" python -m unittest discover tests
```

---

## Notes

- The local package avoids a DeepSpeed dependency for inference; it installs
  a tiny `deepspeed.comm` compatibility stub before importing the upstream
  model. Any code that imports `brainomni` / `braintokenizer` directly must
  call `_configure_upstream_imports()` first.
- BrainTokenizer (5.05 M params) and BrainOmni `tiny` (13.5 M) are small
  enough for CPU-only mechanistic analyses; the 269-channel MEG segments are
  the practical bottleneck. CPU-only MEG analyses use `--clip-seconds 10
  --source-limit 4 --block-limit 4` to stay tractable.
- Real-data segment selection is round-robin across subjects, not
  path-sorted. That keeps the default EEG/MEG batches from collapsing onto a
  single subject.
- Bandpower summaries average per-sensor spectra rather than averaging raw
  sensor traces — this avoids out-of-phase cancellation.
- Custom source-group sweeps redirect output via `--output-suffix`,
  producing `<dataset>_<suffix>/` directories so default and exploratory
  runs do not collide.

---

## License

This repository is released under the MIT License (see [`LICENSE`](LICENSE)).
The vendored `upstream/BrainOmni/` directory remains under its original
license (see `upstream/BrainOmni/LICENSE`).
