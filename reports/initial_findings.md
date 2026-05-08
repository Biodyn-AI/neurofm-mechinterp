# Initial Findings

This note summarizes the first local mechanistic-interpretability pass on BrainOmni using the published checkpoints and the upstream `assets/example_data.pt` demo sample.

## Setup

- Environment: `$HOME/.conda-envs/neurofm-mechinterp`
- Upstream model code: `upstream/BrainOmni`
- Checkpoints downloaded locally:
  - `external/checkpoints/BrainOmni/braintokenizer`
  - `external/checkpoints/BrainOmni/tiny`
  - `external/checkpoints/BrainOmni/base`

The analyses are run with:

```bash
ENV_PATH="$HOME/.conda-envs/neurofm-mechinterp"
conda run -p "$ENV_PATH" python scripts/run_all_example_analyses.py --variant tiny
conda run -p "$ENV_PATH" python scripts/run_all_example_analyses.py --variant base
```

## What exists now

Three concrete analyses are implemented:

1. `tokenizer_semantics`
2. `source_ablation`
3. `spatial_temporal`

Outputs live under:

- `results/example/tiny/`
- `results/example/base/`

## First observations

### 1. Code usage is structured across quantizer depth

On the demo sample, the tokenizer activates:

- Layer 0: 167 codes
- Layer 1: 177 codes
- Layer 2: 154 codes
- Layer 3: 130 codes

The top-code usage pattern changes with RVQ depth. Among the top codes in each layer, the mean number of active source variables rises with depth:

- Layer 0: 1.17 active sources
- Layer 1: 3.25 active sources
- Layer 2: 4.67 active sources
- Layer 3: 6.67 active sources

The early RVQ layer therefore looks more source-specific on this sample, while later layers reuse codes across more learned source variables. That is worth checking on a real dataset instead of one demo segment.

### 2. Source variable 12 is the strongest single bottleneck in the tokenizer

Source-ablation ranking is stable because both `tiny` and `base` use the same frozen tokenizer. The biggest reconstruction damage comes from ablating source 12:

- Baseline tokenizer reconstruction MSE: `0.3720`
- Worst single ablation: source `12`
- Source 12 delta MSE: `0.0536`

The next most important sources on this sample are `3`, `4`, `7`, and `0`.

This gives a concrete next step: inspect what source 12 is doing across many examples, especially whether it tracks a frequency band, a spatial pattern, or a transient/event type.

### 3. The spatial-vs-temporal split is not trivial

For the `tiny` model, single-half patching shows a slight **spatial** advantage on the demo sample:

- `time_reverse`: spatial `0.1284`, temporal `0.1255`
- `spatial_mismatch`: spatial `0.1179`, temporal `0.1031`

For the `base` model, the pattern shifts and **temporal** patching is stronger on average:

- `time_reverse`: temporal `0.1529`, spatial `0.1054`
- `spatial_mismatch`: temporal `0.1163`, spatial `0.0854`

That is useful because it means the split is not a cosmetic architectural choice. The larger model appears to move more recoverable information into the temporal half, at least on this example.

## Immediate next steps

The current analyses run on a single upstream demo segment. The next high-value step is to move them onto a small real EEG or MEG subset and check whether the same patterns survive:

1. source-12 importance across many recordings
2. code semantics by dataset / task / subject
3. temporal-vs-spatial intervention gains on clean vs artifact-heavy segments
4. code or source specialization for EEG vs MEG

