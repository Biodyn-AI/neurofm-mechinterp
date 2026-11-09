# Development notes

Engineering notes and recipes for the exploratory analyses in this repository (upstream demo sample, sparse autoencoders, source prototypes) — environment setup, common commands, architecture, and conventions. The paper's pipeline and how to reproduce it are described in `README.md`.

## Environment

The project uses a dedicated Conda env at `$HOME/.conda-envs/neurofm-mechinterp`. Bootstrap once with `./scripts/bootstrap_env.sh`, then run every Python command through `conda run`:

```bash
ENV_PATH="$HOME/.conda-envs/neurofm-mechinterp"
conda run -p "$ENV_PATH" python <script>
```

Python 3.11 is required.

## Common commands

```bash
# Download checkpoints (Hugging Face: OpenTSLab/BrainOmni)
conda run -p "$ENV_PATH" python scripts/download_brainomni.py --variant tiny
conda run -p "$ENV_PATH" python scripts/download_brainomni.py --variant braintokenizer

# Run the three first-pass analyses on the upstream demo sample
conda run -p "$ENV_PATH" python scripts/run_all_example_analyses.py --variant tiny

# Real-data path (OpenNeuro ds002778 → pd31 EEG, ds003568 → MEG-MMI)
conda run -p "$ENV_PATH" python scripts/download_openneuro_subsets.py
conda run -p "$ENV_PATH" python scripts/preprocess_openneuro_subsets.py --dataset all --force
conda run -p "$ENV_PATH" python scripts/run_real_dataset_analyses.py --dataset pd31 --variant tiny
```

For MEG, CPU-only runs need `--clip-seconds 10 --source-limit 4 --block-limit 4`; full sweeps over the 269-channel segments are slow.

## Testing

Tests use the stdlib `unittest` framework. There is no pytest config. Run the suite or a single test with:

```bash
conda run -p "$ENV_PATH" python -m unittest discover tests
conda run -p "$ENV_PATH" python -m unittest tests.test_sae
conda run -p "$ENV_PATH" python -m unittest tests.test_sae.SaeHelpersTests.test_method
```

## Architecture

This repo is a thin analysis layer around the **upstream BrainOmni reference implementation** vendored at `upstream/BrainOmni/` (a separate git repo, do not modify in passing). The local package `neurofm_mechinterp` (under `src/`) loads upstream models and provides the exploratory analyses described below and the modules of the paper's pipeline (`anchors.py`, `requantize.py`, `readout.py`, `subject_features.py`, `cohorts.py`, `replicas.py`; see `README.md`).

### Loading path (`loading.py`, `compat.py`, `checkpoints.py`)

`load_brainomni()` and `load_braintokenizer()` are the entry points to the released models (`load_tokenizer()` dispatches between the released, embedded and untrained tokenizers and those trained from scratch, `replicas.load_replica()`). Both:

1. Install a **DeepSpeed stub** (`compat.install_deepspeed_comm_stub`) so upstream imports succeed without DeepSpeed — this repo intentionally avoids the dependency for inference. Any code that imports `brainomni`/`braintokenizer` directly must call `_configure_upstream_imports()` first.
2. Lazy-prepend `upstream/BrainOmni` to `sys.path`, then import the upstream model.
3. Download checkpoints into `external/checkpoints/BrainOmni/<variant>/` from `OpenTSLab/BrainOmni` if missing. Both BrainTokenizer (5M params) and the chosen BrainOmni variant (`tiny`=13.5M, `base`) are fetched.
4. Default device is **CPU** (set in `resolve_device`); MPS/CUDA must be passed explicitly.

### Data shape conventions

`data.BrainBatch` carries `(x, pos, sensor_type)` with a leading batch dim. The upstream demo sample is loaded via `load_example_batch()`. For real preprocessed segments, `processed_data.choose_examples()` does **round-robin sampling across subjects** (not path-sorted slicing) to avoid collapsing onto one subject.

Tokenizer activations are shaped `B × source × patch × frame × dim` where `source` is one of the 16 learned latent variables. `sae.flatten_source_activations` / `restore_source_activations` move between this shape and row form, preserving `source_idx`/`patch_idx`/`frame_idx` metadata.

### The exploratory analyses (`src/neurofm_mechinterp/analysis/`)

- **`tokenizer_semantics.py`** — codebook usage stats for the residual VQ. Token indices are emitted **per learned source variable, not per observed sensor**, so per-code summaries track which source dominates each code. Bandpower uses **per-sensor spectra averaged afterwards** (`mean_sensor_relative_bandpower`) to avoid out-of-phase cancellation.
- **`source_ablation.py`** — zero out each of the 16 tokenizer source latents and measure ΔMSE, code-change rate, and worst-affected sensors. Source selection is by feature energy when limited via `max_sources`.
- **`spatial_temporal.py`** — patches one half of BrainOmni's split residual stream from a clean run into a perturbed run, per transformer block. The residual stream is split halfway along `dim`: lower half = temporal, upper half = spatial. `_encode_from_hidden` mirrors `BrainOmni.encode` so the patching stays faithful to the upstream forward pass. Two perturbations (time reversal, spatial channel permutation) probe which half encodes which structure.

### Sparse-autoencoder pipeline

Built on top of the analyses above:

1. `scripts/export_sae_activations.py` runs the tokenizer on a dataset and saves an activation bundle at `results/sae_exports/<dataset>/<hook>/activations.pt`. Hooks are `encoder_pre_quantizer` and `encoder_post_quantizer`.
2. `scripts/train_sparse_autoencoder.py` trains `sae.SparseAutoencoder` (linear-ReLU or top-k); models go to `results/sae_models/<dataset>/<hook>/<run_name>/`.
3. The `run_sae_*` and `compare_sae_*` scripts perform interventions, head-routing analyses, and prototype comparisons on those trained SAEs.

### Output layout (`paths.RESULTS_ROOT`)

All scripts write under `results/`, organized by analysis family then dataset/variant:
- `results/example/<variant>/<analysis>/`
- `results/real_data/<dataset>/<variant>/<analysis>/`
- `results/source_stability/`, `results/source_profiles/`, `results/source_pair_interactions/`, `results/source_group_interactions/`, `results/decoder_routing/`
- `results/sae_exports/`, `results/sae_models/`, plus per-script SAE intervention dirs

`results/`, `data/`, `share/`, `external/checkpoints/`, and `scratch/` are gitignored.

## Conventions

- Pair/group interaction analyses always compare an isolated source set against the **sum of single-source reconstructions**, not against the full baseline alone.
- Custom source-group sweeps redirect output via `--output-suffix`, producing `<dataset>_<suffix>/` directories so default and exploratory runs do not collide.
- The decoder routing probe emits both aggregate and per-head decoder sensor-to-source maps (`routing_overlap_summary.csv`, `head_summary.csv`, `head_routing_overlap_summary.csv`).
