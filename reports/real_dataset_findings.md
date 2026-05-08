# Real Dataset Findings

This note records the first real-data BrainOmni mechanistic-interpretability pass in this workspace.

## Datasets

The downloaded subsets come from the official public OpenNeuro releases:

- `ds002778` (`ds002778-1.0.5` locally): PD31 EEG
- `ds003568` (`ds003568-1.0.4` locally): MEG-MMI

The local subset currently includes:

- PD31 EEG: four BDF recordings (`sub-hc1`, `sub-hc2`, `sub-pd3`, `sub-pd5`)
- MEG-MMI: two MEG recordings (`sub-24201`, `sub-23732`)

After preprocessing through the upstream BrainOmni dataset-maker code:

- PD31 EEG produced `76` processed segments with `32` channels
- MEG-MMI produced `20` processed segments with `269` channels

## Model size

The local checkpoints currently used in this workspace are:

- `BrainTokenizer`: `5,050,338` parameters
- `BrainOmni tiny`: `13,475,042` parameters

That keeps tokenizer-side mechanistic work practical on a local CPU box.

## Commands used

```bash
ENV_PATH="$HOME/.conda-envs/neurofm-mechinterp"

conda run -p "$ENV_PATH" python scripts/download_openneuro_subsets.py --dataset all
conda run -p "$ENV_PATH" python scripts/preprocess_openneuro_subsets.py --dataset all --force

conda run -p "$ENV_PATH" python scripts/run_real_dataset_analyses.py --dataset pd31 --variant tiny

conda run -p "$ENV_PATH" python scripts/run_real_dataset_analyses.py \
  --dataset MEG-MMI \
  --variant tiny \
  --clip-seconds 10 \
  --source-limit 4 \
  --block-limit 4
```

## PD31 EEG (`tiny`)

This run used `8` processed EEG segments and the full analysis sweep. The selection is now round-robin across subjects, so the batch contains two segments from each of `sub-hc1`, `sub-hc2`, `sub-pd3`, and `sub-pd5`.

### Tokenizer code usage

- Layer 0 active codes: `506`
- Layer 1 active codes: `508`
- Layer 2 active codes: `511`
- Layer 3 active codes: `512`

The EEG batch nearly saturates the 512-entry codebooks from the second quantizer onward.

### Source ablation

- Baseline tokenizer reconstruction MSE: `0.6511`
- Strongest source: `3`
- Source 3 is now the top batch-level bottleneck on the balanced subject mix

This is an important correction. The earlier batch-level run overemphasized source `2` because it took the first eight path-sorted segments, which mostly came from one subject. After switching to round-robin subject selection, the batch result aligns with the full stability sweep: source `3` is the more representative dominant EEG latent.

### Spatial vs temporal patching

Mean similarity gain by intervention:

- `spatial_mismatch`: spatial `0.1461`, temporal `0.1400`
- `time_reverse`: spatial `0.1589`, temporal `0.1664`

The EEG picture is mixed rather than one-sided: spatial patching has the edge for the spatial mismatch perturbation, while temporal patching is slightly stronger for time reversal.

## MEG-MMI (`tiny`, reduced CPU run)

This run used `2` processed MEG segments, one from each subject, clipped to the first `10` seconds. To keep the run practical on CPU, source ablation was limited to the top `4` source latents by feature energy and block patching was limited to the first `4` patchable transformer blocks.

### Tokenizer code usage

- Layer 0 active codes: `397`
- Layer 1 active codes: `430`
- Layer 2 active codes: `434`
- Layer 3 active codes: `442`

MEG activates a broad but less saturated slice of the codebooks than the EEG batch.

### Source ablation

- Baseline tokenizer reconstruction MSE: `0.8813`
- Tested source latents: `13, 3, 11, 2`
- Strongest tested source: `2`
- Source 2 remains the top tested source on the balanced two-subject batch

Even in the reduced sweep, source `2` remains the strongest tested latent, which makes it a reasonable first target for a cross-dataset follow-up.

### Spatial vs temporal patching

Mean similarity gain by intervention:

- `spatial_mismatch`: spatial `0.1475`, temporal `0.0932`
- `time_reverse`: spatial `0.1532`, temporal `0.0955`

On this MEG crop, the spatial half is clearly more recoverable than the temporal half for both perturbations.

## Source Stability Sweep

To move beyond small batches, the repo now includes a cross-segment tokenizer-source sweep:

```bash
ENV_PATH="$HOME/.conda-envs/neurofm-mechinterp"
conda run -p "$ENV_PATH" python scripts/run_source_stability_sweep.py --dataset all
```

### PD31 EEG stability

The full PD31 sweep covers all `76` processed EEG segments and evaluates all `16` tokenizer sources per segment.

- Source `3` is the top source on `40 / 76` segments
- Source `2` is the top source on `10 / 76` segments
- Sources `8`, `4`, and `13` trail behind with `6`, `5`, and `5` segment wins

The strongest recurring EEG sensor sets differ by source:

- Source `3`: sensors `0, 29, 27, 28, 1`
- Source `2`: sensors `3, 29, 0, 28, 12`

One important caution: source `2` is absent from the label-`1` segments, but it is also concentrated entirely in subject `sub-hc1`. With only four recordings in the local subset, diagnosis and subject identity are confounded here, so this cannot be treated as a disease-specific source without more subjects.

### MEG-MMI stability

The MEG sweep covers all `20` processed segments, clipped to `10` seconds each, and tests the top `4` source latents by feature energy on each segment.

- Source `2` is the top tested source on `20 / 20` segments
- The mean margin over the second-ranked source is `0.0250`

The dominant MEG source also hits a stable sensor subset across segments:

- Source `2`: sensors `115, 108, 109, 244, 27`

This makes the MEG result much less ambiguous than the EEG result. The dominant source is not only consistent across segments, but also decisively separated from the other tested latents.

## Target Source Profiles

To inspect the dominant sources more directly, the repo now includes a focused source-profiling pass:

```bash
ENV_PATH="$HOME/.conda-envs/neurofm-mechinterp"
conda run -p "$ENV_PATH" python scripts/run_target_source_profiles.py --dataset all
```

This profile uses the corrected bandpower path, which averages relative bandpower across sensors after the spectral transform instead of averaging raw sensor traces first.

### EEG source `3` vs source `2`

Across all `76` EEG segments:

- Source `3` mean full rank: `3.28`
- Source `2` mean full rank: `8.87`

The subject split is the clearest signal:

- Source `2` is strongest in `sub-hc1` with mean rank `1.58`
- Source `2` is weak in the other three subjects with mean ranks `10.21`, `11.21`, and `12.47`
- Source `3` stays near the top for `sub-hc1`, `sub-pd3`, and `sub-pd5`, with mean ranks `2.37`, `2.68`, and `3.00`

The RVQ code profiles separate the two sources most cleanly in the first layer:

- Layer 0 top-10 code overlap between source `3` and source `2`: none
- Layer 1 overlap: `291`, `313`
- Layer 2 overlap: `224`, `256`, `367`, `416`, `424`, `431`
- Layer 3 overlap: `40`, `130`, `306`, `386`

So the mechanistic picture is now sharper: early RVQ codes are strongly source-specific, while later RVQ layers reuse more codes across the two EEG latents.

The dominant sensor footprints also differ, even though they share a common core:

- Source `3` most frequent damaged sensors: `0`, `29`, `9`, `3`, `23`
- Source `2` most frequent damaged sensors: `29`, `0`, `9`, `3`, `27`

The new source-only reconstruction metrics sharpen the interpretation:

- Source `3` alone carries `30.4%` of the baseline reconstruction energy and reaches `0.420` cosine to the full tokenizer reconstruction
- Source `2` alone carries `26.9%` of the baseline reconstruction energy and reaches `0.361` cosine to the full tokenizer reconstruction
- Source `3` source-only sensors: `0`, `29`, `2`, `1`, `28`
- Source `2` source-only sensors: `0`, `1`, `2`, `3`, `5`

So the EEG sources are important, but they are not self-contained latent "programs." Each one accounts for only about a third of the full reconstruction energy on its own, and their isolated sensor footprints are narrower than the sensors that suffer when the source is removed. The overlap between damage sensors and source-only sensors is only:

- Source `3`: `0`, `29`
- Source `2`: `0`, `3`

One more caution follows from the subject split: source `2` is most important by ablation in `sub-hc1`, but its highest source-only cosine-to-input is actually in `sub-hc2`. So "causal bottleneck" and "looks like the waveform by itself" are clearly different properties here.

### MEG source `2`

Across all `20` clipped MEG segments:

- Source `2` mean full rank: `1.0`
- Source `2` top-rank count: `20 / 20`

The subject and label splits barely move it:

- `sub-23732`: mean delta MSE `0.0367`
- `sub-24201`: mean delta MSE `0.0389`
- label `0`: mean delta MSE `0.0389`
- label `1`: mean delta MSE `0.0367`

Its code profile is also stable, especially in the first RVQ layer:

- Layer 0 recurrent codes: `115`, `159`, `170`, `389`, `482`
- Those layer-0 codes appear in `15` to `19` of the `20` segments

And the sensor footprint remains tight:

- Most frequent damaged sensors: `115`, `108`, `109`, `244`, `27`

This is the strongest mechanistic target in the current workspace. The source is stable across subjects and labels, it uses a recurrent layer-0 code set, and it repeatedly concentrates damage on the same small sensor subset.

The source-only view changes the story in an interesting way:

- Source `2` alone carries only `13.1%` of the baseline reconstruction energy
- Source `2` alone reaches just `0.171` cosine to the full tokenizer reconstruction
- Stable source-only sensors: `26`, `27`, `31`, `25`, `122`

That source-only sensor set barely matches the ablation-damage set. The only shared top sensor is `27`.

So the dominant MEG source behaves less like a direct sensor-space component and more like a routing or control latent whose removal damages one sensor cluster while its own isolated output energy lands on another.

## Source Pair Interactions

To test whether the strongest sources compose additively or interact nonlinearly, the repo now includes a source-pair pass:

```bash
ENV_PATH="$HOME/.conda-envs/neurofm-mechinterp"
conda run -p "$ENV_PATH" python scripts/run_source_pair_interactions.py --dataset all
```

This pass profiles the default pairs `3,2` for EEG and `2,11` for MEG, then compares each pair reconstruction against both the raw input and the sum of the two single-source reconstructions.

### EEG pair `3 + 2`

Across all `76` EEG segments:

- Pair energy fraction: `40.5%`
- Pair cosine to the full tokenizer reconstruction: `0.4789`
- Gain over the best single source on that metric: `0.0491`
- Cosine between the pair reconstruction and the sum of the two single-source reconstructions: `0.9447`
- Mean superadditive energy fraction: `-0.1680`

The stable pair sensors are `0`, `1`, `29`, `2`, `28`.

This is a fairly clean result. Sources `3` and `2` combine in almost the same direction as the plain sum of their isolated reconstructions, but the re-quantized pair is amplitude-compressed relative to the sum of the parts. That points to interaction at the quantization or decoder-mixing stage rather than a large geometric rewrite of the signal.

### MEG pair `2 + 11`

Across all `20` clipped MEG segments:

- Pair energy fraction: `22.3%`
- Pair cosine to the full tokenizer reconstruction: `0.3621`
- Gain over the best single source on that metric: `0.0585`
- Cosine between the pair reconstruction and the sum of the two single-source reconstructions: `0.9396`
- Mean superadditive energy fraction: `-0.0753`

The stable pair sensors are `64`, `68`, `266`, `63`.

The same qualitative pattern holds: the pair improves materially over either single source, stays very close to the summed single-source direction, and still comes out subadditive in total energy. So the decoder seems to preserve the pair's geometry while compressing its amplitude.

## Decoder Routing Probe

To inspect the decoder's internal sensor-to-source routing directly, the repo now includes:

```bash
ENV_PATH="$HOME/.conda-envs/neurofm-mechinterp"
conda run -p "$ENV_PATH" python scripts/run_decoder_routing_probe.py --dataset all
```

This probe reads the `ForwardSolution` attention path in the decoder and summarizes, for each target source:

- which sensors place the largest attention weight on that source
- which sensors receive the largest pre-projection contribution norm from that source
- which decoder heads carry those sensor-to-source routes most strongly

### EEG routing

Across all `76` EEG segments:

- Source `3` mean decoder attention weight: `0.0952`
- Source `3` mean decoder contribution norm: `0.0606`
- Source `3` routing sensors: `0`, `1`, `2`, `3`, `29`
- Source `3` overlap with damage sensors: `3 / 5`
- Source `3` overlap with source-only sensors: `4 / 5`

- Source `2` mean decoder attention weight: `0.0686`
- Source `2` mean decoder contribution norm: `0.0436`
- Source `2` routing sensors: `1`, `3`, `5`, `6`, `7`
- Source `2` overlap with damage sensors: `1 / 5`
- Source `2` overlap with source-only sensors: `3 / 5`

So the EEG decoder routing looks closer to the source-only footprints than to the ablation-damage footprints, especially for source `3`. Source `2` is more mixed, but even there the decoder's internal routing set is much more local than the external damage set.

Head resolution does not overturn that EEG story. The strongest source-`3` heads stay on the same local neighborhoods:

- Head `0`: sensors `0`, `1`, `2`, `3`, `29` with `3 / 5` damage overlap and `4 / 5` source-only overlap
- Head `2`: sensors `0`, `20`, `23`, `27`, `29` with `3 / 5` damage overlap and `2 / 5` source-only overlap

For source `2`, different heads split between the source-only neighborhood and the damage neighborhood:

- Head `3` carries the largest mean contribution norm and routes through `0`, `1`, `3`, `4`, `5`, overlapping `2 / 5` damage sensors and `4 / 5` source-only sensors
- Head `2` routes through `0`, `1`, `27`, `28`, `29`, overlapping `3 / 5` damage sensors and `2 / 5` source-only sensors

So EEG routing is head-fragmented but still stays inside the same small family of sensor sets as the aggregate footprint.

### MEG routing

Across all `20` clipped MEG segments:

- Source `2` mean decoder attention weight: `0.0457`
- Source `2` mean decoder contribution norm: `0.0291`
- Source `2` routing sensors: `110`, `111`, `117`, `123`, `124`
- Source `2` overlap with damage sensors: `0 / 5`
- Source `2` overlap with source-only sensors: `0 / 5`

This is the cleanest mechanistic mismatch in the workspace. The same source that damages sensors `115`, `108`, `109`, `244`, `27` when removed and emits isolated energy around `25-31/122` is internally routed by the decoder through a completely different stable sensor set around `110-124`.

Head resolution sharpens that result. Source `2` is split across four stable head-specific sensor islands:

- Head `0`: `24`, `25`, `26`, `29`, `30`, overlapping the source-only set on `2 / 5` sensors
- Head `1`: `197`, `199`, `200`, `201`, `257`
- Head `2`: `98`, `106`, `107`, `113`, `114` and carries the largest mean contribution norm
- Head `3`: `110`, `115`, `116`, `122`, `123`, overlapping the damage set on `1 / 5` sensors and the source-only set on `1 / 5`

So the aggregate `110-124` routing footprint was hiding a more specific head-level picture: one head touches the external `115/122` neighborhood a little, but the other three route source `2` through entirely different sensor clusters. None of the heads reproduces the ablation-damage footprint on its own.

That strongly suggests the MEG source `2` story is not "one latent equals one sensor component." The source is being transformed through the decoder's sensor-attention layer in a way that separates internal routing, isolated output energy, and ablation damage into at least four head-level routes before later decoder mixing.

## EEG Source Basis Expansion

To test whether a small additive basis explains more of the EEG reconstruction gap than the `3 + 2` pair alone, the repo now includes:

```bash
ENV_PATH="$HOME/.conda-envs/neurofm-mechinterp"
conda run -p "$ENV_PATH" python scripts/run_source_group_interactions.py --dataset pd31

conda run -p "$ENV_PATH" python scripts/run_source_group_interactions.py \
  --dataset pd31 \
  --source-group 3,2,8 \
  --source-group 3,2,8,4 \
  --source-group 3,2,8,13 \
  --source-group 3,2,8,4,13 \
  --output-suffix basis_expansion
```

The default triads are `3,2,8` and `3,2,4`, matching the strongest residual EEG candidates after sources `3` and `2`.

### EEG triad `3 + 2 + 8`

Across all `76` EEG segments:

- Group energy fraction: `54.5%`
- Group cosine to the full tokenizer reconstruction: `0.5721`
- Gain over the best single source on that metric: `0.1422`
- Gain over the earlier `3 + 2` pair on that metric: about `0.093`
- Cosine between the triad reconstruction and the sum of its single-source reconstructions: `0.8923`
- Mean superadditive energy fraction: `-0.2294`

Stable triad sensors: `0`, `29`, `1`, `2`, `28`

### EEG triad `3 + 2 + 4`

Across all `76` EEG segments:

- Group energy fraction: `47.6%`
- Group cosine to the full tokenizer reconstruction: `0.5350`
- Gain over the best single source on that metric: `0.1052`
- Gain over the earlier `3 + 2` pair on that metric: about `0.056`
- Cosine between the triad reconstruction and the sum of its single-source reconstructions: `0.8983`
- Mean superadditive energy fraction: `-0.3619`

Stable triad sensors: `0`, `1`, `29`, `27`, `28`

The comparison is fairly decisive: adding source `8` closes more of the EEG gap than adding source `4`, both in reconstruction alignment and in retained energy. So the best current small-basis hypothesis is that EEG reconstruction is dominated by a `3 + 2 + 8` triad rather than just a `3 + 2` pair.

### Fourth-source comparison from `3 + 2 + 8`

Across all `76` EEG segments:

- Group `3 + 2 + 8 + 13`: energy fraction `58.2%`, cosine to the full tokenizer reconstruction `0.6168`, gain over the `3 + 2 + 8` triad about `0.0447`, cosine to the summed single-source reconstruction `0.8320`
- Group `3 + 2 + 8 + 4`: energy fraction `55.6%`, cosine to the full tokenizer reconstruction `0.5979`, gain over the `3 + 2 + 8` triad about `0.0258`, cosine to the summed single-source reconstruction `0.8596`

The fourth-source comparison picks a clear winner: source `13` adds more reconstruction than source `4`, both in retained energy and in cosine to the full tokenizer output. The stable `3,2,8,13` sensors are `2`, `0`, `29`, `1`, `28`, while `3,2,8,4` shifts more mass toward `27` and `23`.

### Five-source basis `3 + 2 + 8 + 4 + 13`

Across all `76` EEG segments:

- Group energy fraction: `60.1%`
- Group cosine to the full tokenizer reconstruction: `0.6424`
- Gain over the earlier `3 + 2 + 8` triad on that metric: about `0.0703`
- Cosine between the five-source reconstruction and the sum of its single-source reconstructions: `0.8096`
- Mean superadditive energy fraction: `-0.6897`

So the EEG basis has not plateaued at `3,2,8`. Adding source `13` helps materially, and adding both `4` and `13` helps again. But the bigger basis also becomes less like a plain linear sum of isolated sources: as the group grows, the re-quantized reconstruction diverges more from the summed single-source waveform and becomes more strongly subadditive in energy.

## What this suggests next

The current real-data pass now gives two stronger directions:

1. For EEG, move from hand-picked groups to greedy forward selection. The triad `3,2,8` is useful, but it is not the end of the basis: source `13` is a better fourth addition than source `4`, and `3,2,8,4,13` still improves further. The next concrete test is to rank the remaining sources by marginal gain conditioned on the current basis and stop when the gain really flattens.
2. For MEG, stay inside the decoder but focus on heads rather than only aggregate routing. Source `2` no longer has one internal routing footprint; it has four head-level sensor islands, only one of which lightly touches the external `115/122` neighborhood. The next useful probe is source-pair or head-ablation work inside the decoder to test whether that small bridge is where the later damage/source-only effects are created.

## See also

- `multi_seed_baseline_replication.md` — four-seed source-prototype stability, PCA/ICA baselines, base-variant spatial-temporal replication, expanded PD31 subset, and figure inventory.
- `execution_plan.md` — the operational plan that produced those results.
