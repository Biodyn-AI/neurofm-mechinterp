# neurofm-mechinterp

Code, subject lists and manuscript sources for **"One Anchor per Source: Hyperspherical Geometry and Causal Roles of
the Latent Sources in an EEG/MEG Foundation-Model Tokenizer"** (Ihor Kendiukhov, 2026).

The paper analyzes the 16 latent sources of the BrainOmni tokenizer, whose residual vector quantizer receives unit
vectors, in four public OpenNeuro cohorts (EEG and MEG; 316 participants) and in tokenizers trained from scratch. It
describes each source in hyperspherical coordinates around its mean direction (the *anchor*), relates the anchor to
centered and uncentered principal components, intervenes with edits that preserve the unit sphere (with controls
matched in displacement, dimension and support, and a split of the decoder's attention into keys and values), and
evaluates the diagnostic content of the sources at the level of subjects with permutation inference.

The compiled manuscript is [`paper/manuscript.pdf`](paper/manuscript.pdf).

---

## Contents

```
.
├── paper/                       manuscript source (manuscript.tex, references.bib, highlights.tex,
│                                tables/operators.tex), the generated numbers.tex, tables/ and figures/, and the
│                                compiled manuscript.pdf
├── lists/                       analysed subjects and segments of every cohort, and the training / held-out split
│                                of the tokenizers trained from scratch (written by scripts/export_subject_lists.py)
├── src/neurofm_mechinterp/      library: anchors.py (hyperspherical coordinates, sphere-preserving edits, anchor
│                                arrangement), requantize.py (quantizer and decoder passes with separate key and value
│                                inputs), readout.py and subject_features.py (subject-level cross-validation and
│                                permutation inference), cohorts.py, replicas.py, loading.py
├── scripts/                     entry points; scripts/pipelines/run_manuscript_analyses.sh runs every analysis
│                                behind the paper after preprocessing, in dependency order
├── tests/                       unit tests (unittest)
├── upstream/BrainOmni/          vendored copy of the BrainOmni release (its own license)
└── requirements-lock.txt        versions of the Python packages used (installed by scripts/bootstrap_env.sh)
```

Other scripts and modules in this repository (sparse autoencoders, source prototypes, demo-sample analyses) belong to
exploratory analyses of the same tokenizer and are not used by the paper.

The directories `data/`, `external/checkpoints/`, `results/`, `share/`, `scratch/` and `logs/` are not tracked: they
hold raw and processed data, downloaded checkpoints, analysis outputs and logs.

---

## Setup

Python 3.11. Create the Conda environment (installs the package versions of `requirements-lock.txt`, among them
PyTorch and the AWS CLI used to download the raw data, and this package in editable mode):

```bash
./scripts/bootstrap_env.sh
PY="$HOME/.conda-envs/neurofm-mechinterp/bin/python"
```

The scripts find the repository through the installed package: run them in the checkout in which
`bootstrap_env.sh` was run (in another checkout, run `$PY -m pip install -e .` there first).

Download the released checkpoints from Hugging Face (`OpenTSLab/BrainOmni`, revision
`9a4d3c70495370397ccfbfd6d2496f25647545a5`, set in `src/neurofm_mechinterp/checkpoints.py`) into
`external/checkpoints/BrainOmni/`:

```bash
$PY scripts/download_brainomni.py --variant braintokenizer
$PY scripts/download_brainomni.py --variant tiny
```

The SHA-256 checksums of the weights are `d41c44c14c3f3b11fd0fb660752e356dff4cb4bc5f32a05f470f503ffddc7b1a`
(`braintokenizer/BrainTokenizer.pt`) and `62c67ba6a84ea0625e67a3b5e7463fe3930bfee88a612a225e9062a052542ffc`
(`tiny/BrainOmni.pt`).

---

## Reproducing the numbers, tables and figures from the archived results

Release [v1.0.0](https://github.com/Biodyn-AI/neurofm-mechinterp/releases/tag/v1.0.0) has two archives attached that
contain every intermediate result the manuscript is generated from (about 2 GB in total):

| archive | contents |
| --- | --- |
| `results_geometry.tar.gz` | `results/source_geometry/`: per-source unit vectors of the first four segments of every subject; for every segment, their means over the segment, the codes, the mean and standard deviation of the vector norms, the reconstruction error and the band power; the geometry of every run |
| `results_interventions_readout_tokenizers.tar.gz` | `results/necessity_atlas/`, `results/necessity_routing/` (interventions), `results/token_structure/`, `results/diagnostic_readout/` (subject-level read-outs and transfer), `results/tokenizer_replicas/` (weights, configurations, training logs and subject split of the tokenizers trained from scratch) |

Download both archives and `SHA256SUMS.txt` into the repository root (git ignores them), then run there:

```bash
shasum -a 256 -c SHA256SUMS.txt
tar -xzf results_geometry.tar.gz
tar -xzf results_interventions_readout_tokenizers.tar.gz
$PY scripts/collect_manuscript_numbers.py     # paper/numbers.tex and paper/tables/*.tex (all but operators.tex)
$PY scripts/make_manuscript_figures.py        # paper/figures/*.pdf
$PY scripts/check_manuscript.py               # builds paper/manuscript.pdf (latexmk) and checks it
```

The collector reads the parameter count from the released BrainTokenizer checkpoint (see Setup). With the package
versions of `requirements-lock.txt` on macOS, the regenerated numbers, tables and figures are identical to the files
in the repository, and so is `paper/manuscript.pdf` when it is rebuilt with the same TeX distribution (TeX Live
2024).

---

## Reproducing the analyses from the raw data

Raw OpenNeuro data are expected under `~/neurofm-data/openneuro/<accession>/`, processed 10-s segments under
`data/processed_evaluate/` and their metadata under `share/metadata/evaluate/`. The analyses used the copies that the
commands below download from the OpenNeuro S3 bucket: ds002778 version 1.0.4 (PD-UCSD), ds004584 version 1.0.0
(PD-Iowa), ds004504 version 1.0.9 (Dementia) and the recordings of ds003568 version 1.0.4 (MDD-MEG; the bucket's
`dataset_description.json` is that of version 1.0.1). The bucket can lag behind the latest version: ds002778 version
1.0.5 adds missing Beck scores to the behavioural files and updates the dataset README's guidance on which analyses
of the dataset are problematic; the recordings and `participants.tsv` are unchanged.
The analysed recordings and segments are listed in `lists/`.

```bash
# 1. Data (ds002778, ds004584 and ds004504 in full; the first 190 s of every ds003568 resting run)
AWS="$(dirname "$PY")/aws"   # the AWS CLI installed by bootstrap_env.sh
$AWS s3 sync --no-sign-request s3://openneuro.org/ds002778 ~/neurofm-data/openneuro/ds002778
$AWS s3 sync --no-sign-request s3://openneuro.org/ds004584 ~/neurofm-data/openneuro/ds004584
$AWS s3 sync --no-sign-request --exclude "derivatives/*" s3://openneuro.org/ds004504 ~/neurofm-data/openneuro/ds004504
$PY scripts/download_meg_rest_partial.py --dest ~/neurofm-data/openneuro/ds003568 --seconds 190
$PY scripts/validate_meg_partial.py --root ~/neurofm-data/openneuro/ds003568 --seconds 190

# 2. Preprocessing (BrainOmni downstream pipeline; MEG sensor positions normalized as in pretraining)
$PY scripts/make_eeg_rest_bids.py --bids-root ~/neurofm-data/openneuro/ds002778 --dataset-name pd31_full \
  --montage biosemi32 --task rest --include-regex 'ses-(hc|off)_' --label-regex 'sub-pd' --line-freq 60
$PY scripts/make_eeg_rest_bids.py --bids-root ~/neurofm-data/openneuro/ds004584 --dataset-name pd_rest63 \
  --montage standard_1005 --task Rest --participants-tsv ~/neurofm-data/openneuro/ds004584/participants.tsv \
  --label-column GROUP --positive-values PD --line-freq 60 --reject-annotations
$PY scripts/make_eeg_rest_bids.py --bids-root ~/neurofm-data/openneuro/ds004504 --dataset-name dementia19 \
  --montage standard_1005 --task eyesclosed --participants-tsv ~/neurofm-data/openneuro/ds004504/participants.tsv \
  --label-column Group --positive-values A --line-freq 50 --reject-annotations
$PY scripts/make_meg_ctf_bids.py --raw-root ~/neurofm-data/openneuro/ds003568 --dataset-name meg_rest \
  --task rest --run 1 --segment-seconds 10 --crop-seconds 190 --meg-pos-norm pretrain

# 3. Every analysis, then the subject lists, numbers, tables, figures and checks, in dependency order
PY=$PY scripts/pipelines/run_manuscript_analyses.sh
```

The pipeline runs the export of the per-source unit vectors (`scripts/export_source_vectors.py`), the geometry
(`scripts/analyze_anchor_geometry.py`), the token structure (`scripts/analyze_token_structure.py`), the interventions
(`scripts/run_necessity_atlas.py`, summarized by `scripts/summarize_necessity_atlas.py`), the subject-level read-outs
(`scripts/run_diagnostic_readout.py`, `scripts/run_transfer_readout.py`), the training and analysis of the tokenizers
trained from scratch (`scripts/train_tokenizer_replica.py`), the lists in `lists/` (`scripts/export_subject_lists.py`,
from the exports and the participants files of the raw datasets), and the generation of the manuscript's numbers,
tables and figures. The exports, interventions and training run on Apple MPS if PyTorch has it and on the CPU
otherwise (`DEVICE=cpu` before the command forces the CPU); the exports and interventions check the device against
the CPU on their first segments and fall back to the CPU if the two disagree. The analyses were run on one Apple M2
Pro computer, where training one tokenizer from scratch takes about an hour and the complete pipeline a few days.

---

## Tests

```bash
$PY -m unittest discover tests
```

---

## Notes

- The package avoids a DeepSpeed dependency for inference: it installs a small `deepspeed.comm` compatibility stub
  before importing the upstream model. Code that imports `brainomni` or `braintokenizer` directly must call
  `_configure_upstream_imports()` first.
- All edits of the latent sources are defined in `src/neurofm_mechinterp/anchors.py` and map unit vectors to unit
  vectors; `scripts/run_necessity_atlas.py --edit-set routing` applies a subset of them to the decoder's attention
  keys or values alone.

---

## License

MIT License (see [`LICENSE`](LICENSE)). The vendored `upstream/BrainOmni/` directory remains under its original
license (see `upstream/BrainOmni/LICENSE`). The OpenNeuro datasets are distributed by their authors under CC0.
