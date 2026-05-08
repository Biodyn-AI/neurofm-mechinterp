# BrainOmni Mechanistic Interpretability Research Plan

This file is the working plan for the local BrainOmni mechanistic-interpretability project.

It is intentionally operational rather than aspirational. Each phase should end with a concrete artifact in this repo: a script, a result table, a short report section, or a reproducible causal test.

## Why SAE work belongs here

Yes, sparse autoencoders should be a major workstream here, but not the only one.

The current repo already has evidence that BrainTokenizer sources are structured and partially interpretable:

- stable source bottlenecks on real EEG/MEG segments
- stable source-specific sensor footprints
- decoder routing mismatches between source-only output, ablation damage, and internal head-level routing

An SAE is useful because it can split dense hidden states into a larger sparse feature basis. But an SAE on its own is not enough. For the features to matter mechanistically, we need:

1. a well-chosen activation hook
2. a reproducible activation dataset
3. feature interpretation summaries
4. causal validation by ablation/patching/reconstruction tests

So the plan is: keep the current source-level work, and add SAE work as the next major layer on top of it.

## Current factual status

- Checkpoints available locally:
  - BrainTokenizer: `5,050,338` parameters
  - BrainOmni tiny: `13,475,042` parameters
- Real datasets prepared locally:
  - PD31 EEG: `76` processed segments, `32` channels
  - MEG-MMI: `20` processed segments, `269` channels
- Existing mechanistic findings:
  - EEG: source `3` is the dominant recurring latent, with `2`, `8`, `4`, `13` as important residuals
  - MEG: source `2` is the dominant recurring latent
  - MEG decoder routing is split across heads and does not collapse to one clean sensor component

## Main research questions

1. What do the 16 tokenizer source slots actually represent?
2. Can the dominant source behavior be decomposed into a more atomic sparse feature basis?
3. Which sparse features are stable across subjects, labels, and datasets?
4. Which features are causal for reconstruction, routing, or downstream behavior rather than merely correlated?
5. Does the model organize around reusable features, or around dataset-specific latent shortcuts?

## Priority order

1. Tokenizer-side SAE on continuous source activations
2. Feature interpretation on real EEG/MEG data
3. Causal validation of SAE features
4. Decoder-head analysis for the MEG source-`2` routing story
5. Optional later extension into BrainOmni downstream representations

## Activation targets for SAE work

We should not start by training an SAE on discrete RVQ indices. That would throw away geometry we want to explain.

The first SAE targets should be continuous activations:

1. `encoder_pre_quantizer`
   - Tensor: BrainTokenizer encoder output before RVQ
   - Shape: `B x source x patch x frame x dim`
   - Why first: this is the cleanest tokenizer latent space before codebook compression
2. `encoder_post_quantizer`
   - Tensor: BrainTokenizer output after RVQ
   - Why second: tells us what structure survives quantization
3. `decoder_forwardsolution_input`
   - Quantized source activations as they enter decoder routing
   - Why later: useful for the MEG routing mismatch
4. `decoder_forwardsolution_output`
   - Sensor-space decoder routing outputs before SEANet decoding
   - Why later: useful once source-side SAE work is established

## Phase plan

### Phase 0. Infrastructure and data hygiene

Goal: keep every later experiment reproducible and cheap to rerun.

Deliverables:

- stable preprocessing and metadata loading
- balanced subject sampling
- activation export scripts with summaries
- result directories that do not overwrite previous runs by accident

Status:

- [x] Local env and checkpoints
- [x] Real-data download and preprocessing
- [x] Balanced segment selection
- [x] Source/routing analysis scripts
- [x] SAE activation export pipeline
- [x] First SAE training pipeline

### Phase 1. Source-level tokenizer mechanics

Goal: finish the basic map of what the tokenizer sources do before adding SAE decomposition.

Deliverables:

- source stability sweep
- source-only reconstruction profiles
- source pair/group interactions
- decoder routing probe

Status:

- [x] Cross-segment source stability
- [x] Source-only profiles
- [x] Pair interactions
- [x] EEG basis expansion to `3,2,8,4,13`
- [x] Aggregate and head-level decoder routing

### Phase 2. SAE dataset creation

Goal: export clean activation matrices plus row metadata for the first tokenizer hooks.

Deliverables:

- `scripts/export_sae_activations.py`
- result bundle with:
  - activation tensor
  - row metadata
  - per-source summary table
  - run summary JSON

Initial runs:

- PD31 `encoder_pre_quantizer`
- PD31 `encoder_post_quantizer`
- MEG-MMI `encoder_pre_quantizer`

Success criteria:

- activations export without shape ambiguity
- row metadata keeps `source_idx`, `patch_idx`, `frame_idx`, `subject`, `label`, `path`
- summary tables show source-wise norm and count distributions

Status:

- [x] Implement export code
- [x] Export pilot PD31 pre-quantizer dataset
- [x] Export pilot PD31 post-quantizer dataset
- [x] Export pilot MEG pre-quantizer dataset

### Phase 3. First SAE training pass

Goal: train a small, plain sparse autoencoder before attempting anything clever.

Model choice:

- start with a simple linear encoder / ReLU or top-k sparse latent / linear decoder baseline
- use one hidden width sweep, not a giant hyperparameter search
- start on `encoder_pre_quantizer`

Recommended first sweep:

- width multipliers: `4x`, `8x`
- sparsity penalty grid: low / medium / high
- train on PD31 first

Primary metrics:

- reconstruction MSE
- explained variance
- L0 or average active features per row
- fraction of dead features
- max feature activation skew

Status:

- [x] Implement minimal SAE trainer
- [x] Train first PD31 SAE on pre-quantizer activations
- [x] Compare pre- vs post-quantizer SAE difficulty
- [x] Replace dense `L1` baseline with top-k or gated SAE if needed

Current finding:

- The first plain linear-ReLU SAE baseline on PD31 `encoder_pre_quantizer` activations trains cleanly and reconstructs very well.
- But it is not sparse enough yet:
  - `L1 = 0.001`: validation MSE `0.00382`, about `543 / 1024` active features per row
  - `L1 = 0.01`: validation MSE `0.00373`, about `536 / 1024` active features per row
  - `L1 = 0.1`: validation MSE `0.00468`, about `461 / 1024` active features per row

So the next training change should probably be architectural rather than just "turn up L1 again."

The top-k sweep now makes the tradeoff clearer:

- `topk32`: validation MSE `0.0597`, explained variance `0.9396`, dead features `29.6%`
- `topk64`: validation MSE `0.0318`, explained variance `0.9679`, dead features `12.1%`
- `topk128`: validation MSE `0.0150`, explained variance `0.9848`, dead features `1.5%`

So `topk64` is a reasonable first interpretability setting: sparse enough to be meaningfully selective, but not so aggressive that reconstruction collapses.

At the same `topk64` setting, PD31 `encoder_post_quantizer` is slightly easier to compress than `encoder_pre_quantizer`:

- pre-quantizer: validation MSE `0.0318`, explained variance `0.9679`, dead features `12.1%`
- post-quantizer: validation MSE `0.0300`, explained variance `0.9700`, dead features `15.5%`

That suggests quantization sharpens or regularizes the basis somewhat, but also makes a larger fraction of features go unused.

### Phase 4. Feature interpretation

Goal: turn SAE units into concrete mechanistic hypotheses.

For each feature:

1. identify top-activating rows
2. aggregate by source index
3. aggregate by subject / label / dataset
4. recover top sensors touched by those rows
5. recover top code usage overlap if relevant

Outputs:

- feature summary CSV
- top-activation examples
- feature-to-source alignment table
- feature-to-sensor alignment table

Status:

- [x] Build feature summary script
- [x] Produce first PD31 feature summary tables
- [x] Produce first MEG feature summary tables

Current finding:

- The first PD31 top-k SAE (`hidden = 1024`, `k = 64`) reaches validation MSE `0.0318` with exactly `64` active features per row and about `12.1%` dead features.
- That reconstruction is worse than the dense ReLU baseline, but it is now meaningfully sparse instead of "sparse in name only."
- The first source-alignment tables already show many features concentrating on specific tokenizer sources, including source `3` and other high-priority EEG sources.
- The first MEG top-k SAE (`hidden = 1024`, `k = 64`) reaches validation MSE `0.0463`, explained variance `0.9539`, and `13.7%` dead features.
- In MEG, source `2` already has a distinct top-feature set (`169`, `575`, `917`, `238`, `1`, `453`, `732`, `205`) rather than disappearing into one pooled background.

### Phase 5. Causal validation

Goal: distinguish useful features from descriptive but non-causal ones.

Interventions:

1. SAE feature ablation during reconstruction
2. SAE feature activation patching between examples
3. Source-slot-conditioned feature masking
4. Decoder routing shift after feature intervention

Success criteria:

- identify features whose ablation changes a stable sensor footprint
- identify features that specifically affect source `3` in EEG or source `2` in MEG

Status:

- [x] SAE feature ablation runner
- [x] SAE feature patching runner
- [ ] Causal report section

Current finding:

- On PD31, ablating the top source-`3` SAE feature `57` from the pre-quantizer `topk64` model increases input MSE by about `0.00328` on average across all `76` segments.
- Feature `1` also looks causally relevant but weaker, with mean delta input MSE about `0.00217`.
- Feature `430`, despite being one of the top source-`3` aligned features by activation summary, slightly improves reconstruction when ablated on average (`-0.00537` delta input MSE). So not every source-aligned feature is helping the tokenizer reconstruction in the same way.
- The dominant delta-sensor sets for these three features are centered around sensors `4`, `25`, `23`, `27`, `31`, and nearby channels, which is not the same footprint as the earlier raw source-`3` damage set. That means the SAE split is finding substructure inside the broader source rather than simply recovering the old source-level footprint one-to-one.
- On MEG source `2`, the first ablated top features (`169`, `575`, `917`) produce delta-sensor sets centered around `69`, `72`, `73`, `101`, and `108`, with only a `1 / 5` overlap against the earlier damage set and no overlap against the source-only or decoder-head-`3` routing sets.
- More surprisingly, those MEG feature ablations improve reconstruction on average rather than hurting it. So the current top source-`2` SAE features look more like interfering or compensatory subfeatures than straightforwardly helpful building blocks.
- After restricting ablations to the matched source rows, the PD31 source-`3` story becomes narrower: feature `57` still hurts reconstruction when removed, but only by about `+0.00016` input MSE on average, while features `1` and `430` now improve reconstruction slightly when removed.
- That means the earlier larger PD31 global-ablation effects were partly driven by cross-source reuse of those SAE features, not just clean source-`3` local structure.
- The conditioned MEG source-`2` results do not rescue the interpretation. Features `169`, `575`, and `917` still improve reconstruction when removed from only source-`2` rows, by roughly `-0.0123` to `-0.0127` input MSE, while keeping the same `69/72/73/108` sensor neighborhood.
- So the first MEG source-`2` SAE basis still looks more like a compensatory or overcomplete decomposition than a set of cleanly helpful source atoms.
- The first source-conditioned patching pilot on PD31 source `3` feature `57` also looks context-dependent rather than cleanly transferable. Patching the feature from high-activation donors into low-activation recipients over `5` donor/recipient pairs changes reconstruction a lot, but only weakly aligns with the donor-minus-recipient reconstruction direction (mean cosine about `0.106`).
- On average that patching slightly improves recipient reconstruction (`-0.00174` input MSE), which is another sign that this feature is not a simple reusable source-`3` atom that can be copied between segments in isolation.
- The first MEG source-`2` patching pilot on feature `169` points the same way. Across `3` donor/recipient pairs, patching higher feature activity into low-activation recipients weakly aligns with the donor-minus-recipient direction (mean cosine about `0.165`) and still improves reconstruction on average (`-0.00776` input MSE).
- So both ablation and patching now agree that the leading MEG source-`2` feature behaves more like a context-sensitive correction term than a clean reusable mechanistic component.
- Splitting the patching pilots by subject relation sharpens that point. For PD31 feature `57`, the `4` cross-subject donor/recipient pairs improve reconstruction on average (`-0.00272` input MSE), while the single within-subject pair hurts reconstruction (`+0.00214`). For MEG feature `169`, the pattern flips: the `2` within-subject pairs improve reconstruction strongly (`-0.0168`), while the single cross-subject pair hurts (`+0.0103`).
- So the first patching results are not just "weak." They are also subject-context-sensitive, which means the same SAE feature can act differently depending on whether the donor and recipient come from the same subject manifold.
- The source-conditioned ablation summaries are also heterogeneous across subjects. On PD31, the modest positive feature-`57` effect is concentrated in `sub-hc2` (`+0.00230` mean input MSE) and partially canceled by `sub-pd3` (`-0.00163`). On MEG, the negative reconstruction delta for source-`2` features is much stronger in `sub-23732` than in `sub-24201`.
- Comparing the SAE sensor shifts against decoder-head routing footprints makes the mismatch even sharper. The PD31 source-`3` SAE features have `0 / 5` overlap with the aggregate source-`3` routing footprint, and feature `57` only reaches a `2 / 5` overlap with decoder head `2`. The MEG source-`2` SAE features have `0 / 5` overlap with the aggregate routing footprint and `0 / 5` overlap with every individual decoder head.
- So the current SAE features are not simply rediscovering the decoder's internal routing heads. Their causal sensor shifts seem to live in partially downstream or complementary sensor neighborhoods.
- The first decoder-head-aware intervention pilot sharpens that further. On PD31, source-`3` feature `57` produces its largest head-routing shift in head `1`, with head `0` close behind, while head `2` changes much less and in the opposite direction on mean contribution.
- On MEG, source-`2` feature `169` barely singles out one head at all: heads `1`, `2`, and `3` all move by roughly the same small amount, and the head-level delta sensors split into separate islands (`187-201`, `97/36/96/103`, `27/108/31/115`) rather than reproducing the feature's output-side `69/72/73/108` pattern.
- So the first head-aware causal results suggest these SAE features perturb the decoder through distributed head-level rerouting, not through a single head-local mechanism.
- Broadening the head-aware sweep to the top three conditioned features preserves that story rather than weakening it. On PD31, source-`3` features `1`, `57`, and `430` all have head `1` as their dominant affected head, with heads `0` and `3` close behind and head `2` still much weaker. So the earlier feature-`57` result was not a one-off.
- On MEG, the top conditioned source-`2` features `169`, `575`, and `917` again spread their routing effect across multiple heads with only tiny absolute changes, and feature `917` even flips the sign of the dominant head-level mean contribution change while keeping the same diffuse head profile.
- So expanding from one feature to the top causal set strengthens the same conclusion: feature-level interventions do not isolate one decoder head or one clean routing circuit.

### Phase 6. Cross-dataset comparison

Goal: test whether SAE features are reused or dataset-specific.

Questions:

- do EEG SAE features transfer to MEG activations at all?
- are some features subject-specific shortcuts?
- are some features aligned to routing heads rather than source slots?

Status:

- [ ] Cross-dataset activation alignment
- [ ] Subject leakage check
- [ ] Shared-vs-specific feature catalog

Current finding:

- A first decoder-direction reuse check between the PD31 and MEG `topk64` seed-`0` SAEs looked strong at first glance: `875` active mutual matches with mean cosine about `0.754`, and `300` of those kept the same top source slot.
- But that result is confounded by shared initialization. Every active mutual PD31-to-MEG match preserved the exact same feature index, which is too clean to treat as independent evidence of convergent shared features.
- A same-dataset seed control makes that clear. Repeating the PD31 `topk64` run with seed `1` gives almost identical reconstruction quality, but PD31 seed `0` vs seed `1` drops to mean cosine about `0.545`, with `0%` same-index preservation and only `2` active mutual matches above cosine `0.7`.
- The cross-dataset seed-mismatched comparison is weaker still: PD31 seed `1` vs MEG seed `0` has mean active mutual cosine about `0.500` and `0` active mutual matches above cosine `0.7`.
- The same seed-instability also holds inside MEG itself. MEG seed `0` vs seed `1` lands at mean active mutual cosine about `0.500`, with only `10` active mutual matches above cosine `0.6`, `0` above `0.7`, and effectively no same-index preservation.
- And the opposite seed-mismatched cross-dataset comparison agrees: PD31 seed `0` vs MEG seed `1` also lands near `0.498` mean active mutual cosine, again with `0` active mutual matches above cosine `0.7`.
- So the first cross-dataset reuse result should be treated as initialization tracking, not yet as evidence that the same sparse feature basis is robustly reused across EEG and MEG. Right now the top-k SAE basis looks strongly seed-dependent on both datasets.
- But a seed-robust source-level comparison gives a different answer. If we collapse each source into an activation-weighted decoder prototype, same-source prototypes are almost perfectly stable across seeds within dataset (mean cosine about `0.998` in both PD31 and MEG) and still very high across EEG/MEG (`0.978`-`0.979` mean same-source cosine).
- The source-level top-feature spans are also moderately stable even when individual features are not: mean same-source canonical correlation is about `0.645` for PD31 seed `0/1`, `0.522` for MEG seed `0/1`, and about `0.549`-`0.579` across EEG/MEG comparisons.
- Crucially, every source is its own best match at this aggregated level in every comparison we ran (`16 / 16` same-source best matches), including the high-priority sources: PD31 source `3` stays matched to source `3` across all seed and dataset pairings, and source `2` does the same.
- So the current evidence says the tokenizer source slots themselves are stable and reusable, while the finer SAE decomposition inside each source is not unique under the present training setup.
- A direct causal check on that aggregated object now supports the same story. If we intervene on the standardized activation rows along the SAE-derived source prototype direction, removing the prototype component from only the matched source rows hurts reconstruction strongly on both datasets: about `+0.01162` input MSE on PD31 source `3` and `+0.01204` on MEG source `2` for the seed-`0` SAEs.
- The control goes the other way. Setting each matched source row to the dataset-mean prototype projection slightly improves reconstruction on average instead of hurting it (`-0.00098` on PD31 and `-0.00040` on MEG), which suggests the direction itself is essential while the row-to-row amplitude variation along that direction is partly noise or overfitting.
- The sensor footprints from those source-prototype removals are also much cleaner than the individual feature footprints. On PD31, the dominant prototype-removal sensors are `0`, `29`, `28`, `1`, and `30`, which overlaps the earlier source-only routing footprint in `4 / 5` positions. On MEG, the prototype-removal top sensors are `26`, `27`, `31`, `25`, and `122`, which exactly matches the earlier source-only routing footprint.
- Quantifying that overlap directly sharpens the interpretation. On PD31, the source-prototype removal footprint has Jaccard about `0.667` against the aggregate source-only footprint but a perfect `5 / 5` overlap with decoder head `3` (Jaccard `1.0`). So the source-level prototype there looks like a stable head-specific routing pattern that the individual SAE features failed to isolate cleanly.
- On MEG, the picture is different. The source-prototype removal footprint has a perfect `5 / 5` overlap with the aggregate source-only routing footprint (Jaccard `1.0`) but still only `2 / 5` overlap with the best individual head. So the MEG source-level prototype looks stable and causal without collapsing to one head-local mechanism.
- Comparing the low-damage `set_mean_projection` controls against the same routing summaries adds a useful contrast. On PD31, the control keeps essentially the same source-prototype footprint and still matches decoder head `3` perfectly, even though the net reconstruction effect becomes slightly positive. On MEG, the control drifts away from the exact source-only footprint toward sensors `25`, `26`, `30`, `27`, and `24`, with only `3 / 5` source-only overlap but `4 / 5` overlap with the best individual head (`0`).
- A head-routing magnitude check explains that split. On PD31, both prototype removal and the low-damage mean-projection control keep the same head ordering by mean absolute sensor delta: head `1` moves most, head `3` stays second, head `0` stays third, and head `2` remains much weaker. So the control is mostly a low-amplitude rescaling of the same distributed head-`1`/head-`3` circuit, and head `3` wins the overlap metric because its top sensor set (`0,1,28,29,30`) is the cleanest exact spatial match to the aggregate prototype footprint.
- On MEG, the control does not actually become head-`0` dominant in magnitude. Prototype removal is head-`2` dominant by mean absolute delta, with head `3` and head `1` also moving and head `0` moving in the opposite mean-contribution direction. But under `set_mean_projection`, all four heads collapse to tiny deltas near zero. At that point the aggregate top-`5` sensor set is determined by whichever residual head footprint is most stable across segments, and that happens to be the head-`0` neighborhood around `24/25/26/29/30`, not because head `0` is carrying the causal effect, but because the genuinely causal multi-head source pattern has been mostly canceled.
- A seed-`1` head-routing replication strengthens that interpretation. PD31 keeps the same head ordering and almost identical magnitudes for both removal and control, so its head-`1`/head-`3` split looks robust. MEG prototype removal is also stable across seeds with head `2` dominant. But the MEG `set_mean_projection` control is not head-stable: on seed `0` its residual aggregate footprint overlaps best with head `0`, while on seed `1` the largest mean-absolute head delta is head `1`, even though all head-level deltas stay extremely small. That means the MEG control footprint is a stable low-energy leftover in sensor space, but not a stable head-level causal circuit.
- A bootstrap stability check makes that distinction explicit. Across `4000` segment bootstraps, PD31 source-`3` removal and control both pick head `1` as the dominant head in `100%` of resamples for both seeds, and MEG source-`2` removal picks head `2` in `100%` of resamples for both seeds. But the MEG `set_mean_projection` control only gives head `1` about `67.4%` of bootstrap wins in seed `0` and about `57.9%` in seed `1`, with head `2` and then head `3` also winning often. So the MEG control is not just small; its head attribution is genuinely unstable.
- The scale difference matters too. The MEG control runs carry only about `2.5%` of the total head-routing delta mass of the matched removal runs, while the PD31 controls still keep about `8%` of the removal mass. That is enough for PD31 to preserve a robust head ranking, but not enough for MEG. So head-overlap summaries should be treated as reliable only when the intervention keeps nontrivial routing mass.
- And this source-prototype causal result is itself seed-robust. Repeating the same prototype-removal intervention on the seed-`1` SAEs gives essentially the same reconstruction damage as seed `0` on both datasets (`+0.01166` on PD31 and `+0.01217` on MEG) with the same dominant sensor sets.
- So the current evidence is no longer just that source slots are stable descriptively. It now says the stable, seed-robust, causally relevant object is the source-level prototype direction, while individual sparse features remain a non-unique decomposition of that source structure. In practice, individual SAE features should be treated as exploratory probes for source substructure, not as primary mechanistic units, unless a future constrained training setup makes their identities seed-robust.
- A matched-control stabilization check closes the easy loophole. Extending the global PD31 `topk64` baseline from `8` to `10` epochs barely changes seed alignment at all: mean active mutual cosine stays around `0.546`, same-source prototype cosine stays around `0.998`, and the overall conclusion is unchanged. So the earlier seed instability was not just undertraining.
- A decoder-normalized SAE does improve PD31 reconstruction sharply, from about `0.0310` validation MSE in the matched `10`-epoch baseline to about `0.0058-0.0059`, and it lowers dead-feature fraction from about `11.2%` to about `6.3%`. But it makes feature identity much less stable across seeds: mean active mutual cosine drops to about `0.358`, active mutual matches above cosine `0.6` collapse from `47` to `1`, and same-source canonical correlation drops from about `0.643` to about `0.596`.
- So better reconstruction alone is not evidence of a better mechanistic basis here. The decoder-normalized model fits the rows better while making the sparse decomposition less reproducible.
- A stronger locality constraint also fails. Training source-local SAEs only on the causal source rows makes both datasets worse on the metrics that matter: PD31 source-`3` local models drop to about `0.403` mean active mutual cosine with prototype cosine about `0.631`, and MEG source-`2` local models drop to about `0.353` mean active mutual cosine with prototype cosine about `0.232`. Reconstruction also degrades sharply relative to the global-source SAEs on both datasets.
- So neither of the first stabilization variants we actually tested, decoder normalization nor source-local training, rescues individual feature identity. At this point the conservative interpretation is that the present top-k SAE family does not produce a uniquely reusable within-source basis, even though it does recover a stable source-level prototype object.
- A first sufficiency-style source-prototype patching check now refines that prototype story. Copying donor row-wise prototype projections into low-projection recipient segments does not make recipient reconstruction changes line up with donor-minus-recipient differences: mean cosine is only about `0.061-0.065` on PD31 across seeds and about `0.019-0.029` on MEG.
- Those prototype patches also stay extremely close to the original recipient reconstructions, with mean cosine to recipient baseline around `0.990-0.991` on PD31 and about `0.9993` on MEG, while slightly improving or barely changing recipient reconstruction (`-0.0051` to `-0.0054` input MSE on PD31, `-0.00027` to `-0.00041` on MEG).
- So the source-prototype direction is robustly necessary, but donor-specific amplitude along that direction is not a cleanly transportable latent by itself. The prototype seems to act more like a shared source scaffold whose coarse amplitude can be regularized, while segment-specific detail still lives elsewhere.
- The patch sensor footprints reinforce that split. On PD31, prototype patching still lights up the same `0/1/28/29/30` neighborhood as prototype removal, which suggests the same stable source scaffold is being nudged. On MEG, prototype patching instead concentrates on `24/25/26/27/29/30`, much closer to the earlier low-energy control footprint than to the exact causal removal footprint. So on MEG especially, donor projection copying looks more like a residual calibration change than a donor-transport mechanism.
- A direct component comparison now makes the donor-specific story sharper. For the exact same donor/recipient pairs, the source-row difference energy is mostly prototype-orthogonal: about `87.1%-87.2%` on PD31 across seeds and about `99.3%-99.4%` on MEG.
- And the causal comparison matches that descriptive split. On PD31, residual-only patching raises donor-alignment from about `0.061-0.065` under projection-only patching to about `0.404`, while full-row patching only reaches about `0.458`. So almost all of the transferable donor-specific effect already lives in the prototype-orthogonal residual.
- MEG says the same thing even more strongly. Projection-only patching stays near zero donor-alignment (`0.019-0.029`), but residual-only patching rises to about `0.205-0.207`, while full-row patching on seed `0` only reaches about `0.212`. So on MEG the full transferable donor signal is essentially exhausted by the prototype-orthogonal residual, not the prototype amplitude.
- The sensor footprints add a useful split between datasets. On PD31, residual patching exactly preserves the `0/1/28/29/30` source-prototype scaffold footprint, so the donor-specific signal still rides the same stable source geometry. On MEG, residual patching overlaps the causal prototype-removal footprint only partially (`3 / 5`, Jaccard about `0.43`) and shifts toward `122/115/116/27/26`, which suggests that MEG donor-specific source variation is carried by a more source-adjacent residual substructure rather than by the prototype scaffold itself.
- So the current mechanistic picture is now two-level. The prototype direction is the stable, seed-robust, causally necessary source scaffold. Donor-specific variation lives mainly in the prototype-orthogonal residual, and that residual is what transfers most of the donor-specific reconstruction effect when we patch between segments.
- A direct cross-seed residual comparison now shows that this prototype-orthogonal residual is itself highly seed-robust at the segment level. Same-path residual mean vectors match almost perfectly across seeds within dataset: mean cosine about `0.974` on PD31 and about `0.962` on MEG, with `100%` cross-seed nearest-neighbor recovery of the exact same segment path in both datasets.
- On PD31, that residual is not just subject-idiosyncratic noise. Within each seed, cross-subject same-label residual pairs are consistently more similar than cross-subject different-label pairs (`0.0586` vs `-0.0134` on seed `0`, `0.0448` vs `-0.0230` on seed `1`), and nearest-neighbor label agreement stays high (`0.842-0.855`) even though same-subject agreement is lower (`0.605-0.645`).
- A first causal split agrees with that descriptive pattern. For the top `5` cross-subject same-label PD31 residual pairs, residual patching reaches donor-alignment about `0.409-0.410` across seeds. For the top `5` cross-subject different-label pairs, donor-alignment drops to about `0.337-0.338`.
- That same-vs-different label contrast is conservative rather than inflated. The chosen different-label pairs were actually more similar before patching (mean residual cosine about `0.723`) than the chosen same-label pairs (about `0.661`), yet they still transferred less donor-specific effect. So the PD31 source-`3` residual appears to contain label-related structure beyond generic residual similarity.
- A similarity-matched control now resolves that caveat. Using `10` cross-subject same-label pairs and `10` cross-subject different-label pairs matched almost exactly on baseline residual cosine (`0.6052` vs `0.6048`, mean gap about `0.00106`, max gap about `0.00223`), same-label residual patching still reaches donor-alignment about `0.4032-0.4033` across seeds, while different-label patching stays lower at about `0.3436-0.3438`.
- And that split is about directional transport rather than intervention size. Under the matched control, same-label and different-label patching produce almost the same mean reconstruction displacement (`0.03615-0.03630` vs `0.03545-0.03616` delta input MSE), but only the same-label pairs move recipients appreciably closer to the donor-minus-recipient reconstruction direction.
- So the strongest current PD31 source-`3` claim is now: the prototype-orthogonal residual is seed-robust, causally active, and contains a label-structured transport component that survives explicit similarity matching.

### Phase 7. Optional downstream extension

Goal: only after tokenizer work is credible, inspect the tiny BrainOmni downstream model.

Potential targets:

- tokenizer-to-downstream interface
- downstream attention blocks
- task-specific readout features

Status:

- [ ] Not started

## Execution rules

1. Prefer the tokenizer over the full downstream model until the tokenizer story is stronger.
2. Prefer simple baselines over fancy sparse models first.
3. Every new interpretability claim should have at least one causal check.
4. Avoid mixing subject effects with diagnosis effects when the local dataset is small.
5. Do not overwrite result directories when running exploratory sweeps.

## Immediate next actions

- [x] Decide first SAE hook points from the model code
- [x] Implement activation export for tokenizer SAE work
- [x] Export pilot activation datasets
- [x] Implement a minimal SAE trainer
- [x] Train the first PD31 SAE
- [x] Train a top-k or gated SAE baseline on PD31 pre-quantizer activations
- [x] Compare PD31 pre- vs post-quantizer SAE behavior
- [x] Build a first feature summary that groups SAE units by source index
- [x] Decide whether `topk64` is the right sparsity level or whether to sweep `k`
- [x] Run the first MEG source-`2` SAE feature ablations
- [x] Compare ablation-induced sensor shifts against the earlier source-only and routing sensor footprints
- [x] Restrict feature ablations to source-conditioned rows, not global feature activations
- [x] Add SAE feature patching, not just ablation
- [x] Split patching and ablation summaries by within-subject vs cross-subject donor/recipient pairs
- [x] Compare SAE feature sensor shifts against decoder-head routing footprints, not just source-only footprints
- [x] Start a first cross-dataset shared-vs-specific feature reuse check
- [x] Repeat the MEG `topk64` SAE with a different seed and rerun the cross-dataset reuse check without same-index bias
- [x] Add a seed-robust feature-alignment method before making stronger cross-dataset reuse claims
- [x] Decide whether to stabilize SAE training further or treat the current seed dependence as a substantive result
- [x] Test source-level prototype interventions now that source-level structure looks stable
- [x] Move from overlap summaries to decoder-head-aware causal interventions
- [x] Broaden the head-aware intervention pilot from one feature per dataset to the top causal feature sets
- [x] Check whether source-level prototype interventions replicate across SAE seeds
- [x] Quantify source-prototype intervention overlap against decoder routing footprints directly, not just by manual top-sensor inspection
- [x] Compare source-prototype routing overlaps for `remove_projection` versus `set_mean_projection` controls
- [x] Explain why PD31 keeps the same head-`3` prototype footprint under control while MEG shifts toward head `0`
- [x] Repeat the source-prototype head-routing comparison on the seed-`1` SAEs to check whether the PD31/MEG split is also head-level seed-robust
- [x] Quantify head-level instability for near-zero MEG controls so overlap metrics are not over-interpreted
- [x] Decide whether to keep treating individual SAE features as exploratory only, or to constrain training harder until feature identities stabilize
- [x] If stable individual feature identities are still needed, prototype one constrained SAE training variant aimed specifically at seed robustness
- [x] Test whether source-level prototype directions are transportable between segments, not just necessary within each segment
- [x] Measure whether donor-specific source differences live mainly in the prototype-orthogonal residual rather than in prototype amplitude itself
- [x] Test whether the prototype-orthogonal residual itself is seed-robust, or whether it mostly captures subject-specific idiosyncrasy
- [x] Run a similarity-matched same-label vs different-label residual patching control on PD31 source `3`
- [ ] Decompose the PD31 source-`3` residual into label-shared structure versus subject-specific remainder and patch them separately

## Execution log

- `2026-04-17`: Plan created. First execution target is Phase 2: activation export for `encoder_pre_quantizer` and `encoder_post_quantizer`.
- `2026-04-17`: Implemented `export_sae_activations.py` and exported pilot bundles for PD31 pre-quantizer, PD31 post-quantizer, and MEG pre-quantizer activations.
- `2026-04-17`: Implemented `train_sparse_autoencoder.py` and trained the first PD31 pre-quantizer SAE baselines. Result: reconstruction is easy, but plain `L1` sparsity is weak, so the next trainer should likely use top-k or gated sparsification.
- `2026-04-17`: Added top-k SAE support, trained a `topk64` PD31 pre-quantizer baseline, and generated the first feature-to-source summary tables. This is the first run in the repo that is actually sparse enough to justify feature interpretation.
- `2026-04-17`: Swept PD31 pre-quantizer top-k values (`32`, `64`, `128`) and compared PD31 pre- vs post-quantizer `topk64` runs. Conclusion: `topk64` is a reasonable default interpretability setting, and post-quantizer activations are slightly easier to compress at the same sparsity.
- `2026-04-17`: Trained the first MEG `topk64` SAE and generated the first MEG feature-to-source summary tables.
- `2026-04-17`: Implemented `run_sae_feature_ablations.py` and ran the first full PD31 causal pass on the top three source-`3` SAE features.
- `2026-04-17`: Ran the first MEG source-`2` SAE feature ablations. Result: the top source-`2` features partially touch the MEG damage neighborhood around sensor `108`, but their removal improves reconstruction on average, so they are not yet interpretable as cleanly helpful source-`2` building blocks.
- `2026-04-17`: Updated the ablation runner so `--source-idx` also conditions the intervention to matching source rows, then reran PD31 source `3` and MEG source `2`. Result: EEG feature `57` remains modestly causal inside source `3`, while the MEG source-`2` features still look compensatory even under row-conditioned ablation.
- `2026-04-17`: Implemented `run_sae_feature_patching.py` and ran a first PD31 source-`3` pilot on feature `57`. Result: feature patching does not cleanly move recipients toward donor reconstructions and slightly improves recipient reconstruction on average, so the feature looks context-sensitive rather than plug-and-play.
- `2026-04-17`: Ran the first MEG source-`2` patching pilot on feature `169`. Result: patching again improves reconstruction on average and only weakly follows donor-minus-recipient reconstruction deltas, reinforcing the compensatory-source-feature interpretation.
- `2026-04-17`: Added `summarize_sae_interventions.py` to split causal results by subject and donor/recipient relation, and `compare_sae_feature_routing.py` to compare SAE sensor shifts against aggregate and head-level decoder routing footprints.
- `2026-04-17`: Added `compare_sae_feature_reuse.py` for a first cross-run decoder-direction comparison. Initial PD31-vs-MEG seed-`0` results looked strong, but a PD31 seed-`1` control showed that same-index initialization tracking dominates that signal.
- `2026-04-17`: Trained PD31 and MEG `topk64` seed-`1` controls and reran the reuse comparisons. Result: both same-dataset and cross-dataset seed-mismatched comparisons collapse to about `0.50-0.55` mean active mutual cosine with essentially no same-index preservation, so the strong seed-`0` cross-dataset match signal is not robust.
- `2026-04-17`: Added `compare_sae_source_structure.py` to compare source-conditioned decoder prototypes and top-feature spans instead of individual features. Result: source-level structure is highly stable across seeds and still strong across EEG/MEG, even though feature-level identities are not.
- `2026-04-17`: Implemented `run_sae_head_routing_interventions.py` and ran first head-aware pilots for PD31 source `3` feature `57` and MEG source `2` feature `169`. Result: decoder routing changes are distributed across heads and do not collapse to the previously identified output-side or routing-overlap head footprints.
- `2026-04-18`: Expanded the head-aware intervention sweep to the top three conditioned features for PD31 source `3` and MEG source `2`. Result: the distributed multi-head story persists across the top feature set, so the first head-aware pilot was not a single-feature fluke.
- `2026-04-18`: Implemented `run_sae_source_prototype_interventions.py` to intervene directly on the seed-robust source prototype direction in activation space. Result: removing that prototype component from matched source rows causes strong reconstruction damage on both PD31 and MEG, while setting rows to the dataset-mean projection slightly improves reconstruction.
- `2026-04-18`: Repeated the source-prototype removal intervention on the seed-`1` SAEs. Result: the causal damage and dominant sensor footprints are effectively unchanged across seeds, reinforcing that the source-level prototype direction is robust even though individual SAE features are not.
- `2026-04-19`: Added `compare_sae_source_prototype_routing.py` to quantify overlap between source-prototype intervention footprints and decoder routing summaries. Result: PD31 source `3` prototype removal matches decoder head `3` exactly, while MEG source `2` prototype removal matches the aggregate source-only routing footprint exactly but still does not collapse to one head.
- `2026-04-19`: Ran the same routing-overlap quantification on the `set_mean_projection` controls. Result: PD31 keeps the same head-`3` footprint under the low-damage control, while MEG shifts toward a head-`0`-like footprint and away from the exact source-only match.
- `2026-04-20`: Implemented `run_sae_source_prototype_head_routing.py` to measure per-head routing shifts under source-prototype interventions. Result: PD31 removal and control keep the same head ordering and differ mainly by magnitude, while MEG control collapses all head deltas toward zero so the residual aggregate footprint is a head-`0`-like leftover rather than the main causal pattern.
- `2026-04-20`: Repeated the source-prototype head-routing comparison on the seed-`1` SAEs, and added the missing seed-`1` `set_mean_projection` source-prototype control runs. Result: PD31 and MEG removal are head-level stable across seeds, but the MEG control only keeps a stable low-energy sensor footprint; its dominant head label is not stable once the deltas collapse toward zero.
- `2026-04-20`: Added `summarize_source_prototype_head_stability.py` and bootstrapped dominant-head attribution across all seed-`0/1`, remove/control source-prototype head-routing runs. Result: PD31 removal/control and MEG removal are bootstrap-stable (`100%` dominant-head wins), while the MEG controls are not (`57.9%`-`67.4%` for the nominal top head), so low-energy MEG control overlaps should be treated as residual footprints rather than robust head-local mechanisms.
- `2026-04-20`: Added decoder-feature normalization to the SAE trainer, ran matched `10`-epoch PD31 controls, and compared the resulting SAEs across seeds. Result: decoder normalization improves reconstruction strongly but makes feature identity much less stable than the matched non-normalized baseline, so it does not solve the interpretability problem we actually care about.
- `2026-04-20`: Added `filter_sae_activations.py` and trained source-local `topk64` SAEs on PD31 source `3` and MEG source `2`. Result: source-local training makes both reconstruction and cross-seed reuse worse, so the present feature non-uniqueness survives even after removing cross-source mixing.
- `2026-04-20`: Added `run_sae_source_prototype_patching.py` and ran source-prototype donor/recipient patching pilots on PD31 source `3` and MEG source `2` for both SAE seeds. Result: prototype patching is seed-robustly weak as a donor-transfer mechanism. It only weakly aligns with donor-minus-recipient reconstruction deltas, usually improves or barely changes recipient reconstruction, and on MEG its sensor footprint looks more like the earlier low-energy control than the causal removal footprint.
- `2026-04-23`: Extended `run_sae_source_prototype_patching.py` to compare `projection`, `residual`, and `full_rows` source-row patching under the same donor/recipient pairs, and added the corresponding unit tests. Result: donor-specific source differences live mainly in the prototype-orthogonal residual. Residual-only patching recovers most of the donor-transfer effect on PD31 and essentially all of it on MEG, while projection-only patching remains weak.
- `2026-04-23`: Added `compare_sae_source_prototype_residuals.py` plus tests and compared segment-level prototype-orthogonal residual mean vectors across SAE seeds. Result: the residual is highly seed-robust in both PD31 and MEG, with exact same-path nearest-neighbor recovery across seeds, while PD31 also keeps a weaker but consistent cross-subject same-label structure.
- `2026-04-23`: Extended `run_sae_source_prototype_patching.py` to accept explicit donor/recipient pair CSVs, then ran PD31 source-`3` residual patching on cross-subject same-label and different-label pair sets. Result: same-label pairs transfer more donor-specific effect than different-label pairs in both seeds, despite the chosen different-label pairs being more similar before intervention, so the next control should match pairs by baseline residual similarity.
- `2026-04-23`: Built a similarity-matched PD31 source-`3` pair control (`10` same-label and `10` different-label pairs with nearly identical baseline residual cosine) and reran residual patching on both SAE seeds. Result: the same-label transport advantage survives matching almost exactly (`0.4032-0.4033` vs `0.3436-0.3438` donor-alignment), while mean intervention magnitude stays nearly identical, so the label effect is directional rather than just a larger residual perturbation.
- `2026-05-08`: Paper-readiness review and execution plan in `reports/execution_plan.md`. Trained PD31 + MEG `topk64` SAEs at seeds `2` and `3` to extend the seed `0/1` stability pair to a four-seed cohort, ran source-prototype removal on all four PD31 ep10 SAEs and all four MEG SAEs, and added `summarize_multi_seed_stability.py`. Result: prototype-removal damage is `0.01190 ± 0.00015` on PD31 and `0.01174 ± 0.00062` on MEG; pairwise source-prototype direction cosine stays at `≥ 0.998` across all 6 same-dataset seed pairs, so the source-prototype direction is essentially invariant under SAE seed.
- `2026-05-08`: Added `run_pca_ica_source_prototypes.py` for PCA(64)/FastICA(64) baselines on the same source-conditioned activations the SAE saw, plus `run_real_dataset_analyses.py --variant base` for the BrainOmni encoder replication. Result on PD31 source 3: SAE prototype damage `+0.0117`, best PCA component damage `-0.0013` (no damage despite cos `= 0.72` to SAE prototype), best ICA component damage `+0.0172` (more damage than SAE prototype, cos `= 0.71`, identical top-5 sensors). Result on MEG source 2: SAE prototype damage `+0.0120`, best PCA component damage `+0.0214` (more damage than SAE prototype, cos `= 0.72`), best ICA component damage `+0.0056` (less damage, cos `= -0.19`). PD31 base spatial-temporal mean similarity gain: `0.241` for both halves under `time_reverse` (vs `0.278` for tiny), with the temporal half recovering more time-reversal information at base than tiny.
- `2026-05-08`: Expanded the PD31 subset from 4 to 10 subjects (downloaded `hc4, hc7, hc8, pd6, pd9, pd11` from OpenNeuro `ds002778`) and re-preprocessed to 188 segments. Snapshotted the original info.json at `info_expanded10subj.json`, kept the canonical `info.json` at the 4-subject set so existing seed `0..3` SAEs remain reproducible. Added `--metadata-path` and `--dataset-name-override` flags to the export, intervention, and patching scripts so the expanded set runs without clobbering existing results.
- `2026-05-08`: Added `make_paper_figures.py` producing six PNG+PDF figures at 300 dpi: source dominance, prototype damage by seed, seed-stability cosine matrix, projection-vs-residual-vs-full patching donor alignment, similarity-matched same-vs-different-label transport, SAE-vs-PCA-vs-ICA prototype removal damage. Wrote `reports/multi_seed_baseline_replication.md` summarizing the new results and the paper-ready claims.
- `2026-05-08`: Trained an `expanded10_topk64_ep10` SAE on the 10-subject (188-segment) PD31 export (val MSE `0.0294`, expvar `0.971`). Ran prototype removal on a balanced 40-segment subset (4 segments per subject across all 10 subjects). Result: mean Δ input MSE `0.01164` vs `0.01171` on the original 4-subject 76-segment run, with per-subject ΔMSE ranging `0.0029` (sub-hc4) to `0.0239` (sub-pd5) — both diagnosis groups present at both extremes, so no clean diagnosis confound at this sample size. Residual-vs-projection patching on 10 cross-subject pairs reproduced the original component split: residual donor alignment `0.391` (vs `0.404`), projection donor alignment `0.048` (vs `0.065`). The original PD31 source-3 prototype + residual decomposition story is robust to the 4 → 10 subject expansion.
- `2026-05-08`: Implemented `run_downstream_task_probe.py` and ran a HC vs PD linear classifier on segment-pooled PD31 expanded10 activations under four conditions: baseline (full 256-d), prototype direction removed from source-3 rows, prototype direction *only* (1-d projection of source-3 rows), and a random-unit-direction control. Stratified 5-fold result: baseline accuracy `0.681` / AUC `0.748`; prototype removed `0.676` / `0.741` (Δ `−0.005` / `−0.007`); **prototype only `0.686` / `0.760`** (Δ `+0.005` / `+0.012`); random direction removed `0.681` / `0.749` (Δ `0.000` / `+0.001`). LOSO with `N=10` subjects collapses to below-chance (AUC `< 0.5`) for all settings, as expected. The 1-d source-3 prototype direction is sufficient for diagnostic discrimination (matches/exceeds the full 256-d baseline AUC), is not necessary (residual carries duplicate label info), and the random-direction control rules out a generic "remove any 1-d direction" effect. Added `figure_downstream_probe` to `make_paper_figures.py` and updated `reports/multi_seed_baseline_replication.md` with section E and a fifth headline mechanistic claim.
- `2026-05-08`: Launched the uncropped MEG-MMI tiny analyses (`run_real_dataset_analyses.py --max-segments 20` with no `--clip-seconds`/`--source-limit`/`--block-limit`) to remove the previous laptop-only clipping caveats from the spatial-temporal split and source-ablation results.
- `2026-05-08`: Wrote `download_meg_extra_subjects.py` and `run_meg_expansion_pipeline.sh` for a chained MEG-MMI subject expansion: download `sub-24138, sub-24172, sub-24071, sub-24103` (2 HV + 2 MDD) from `ds003568`, preprocess once with `--force` to regenerate `info.json` for all 6 subjects, delete raw `.ds` directories to keep peak disk use under the 50 GiB external-drive budget, then re-export tokenizer activations to a new `MEG-MMI_expanded6` bundle, train an `expanded6_topk64_ep10` SAE, and run prototype intervention plus the downstream-task probe (HV vs MDD). The pipeline is gated on the uncropped MEG analyses finishing first to avoid CPU contention.
- `2026-05-08`: Completed the MEG-MMI 2 → 6 subject expansion. Final metadata has 159 segments (79 HV + 80 MDD) across `sub-23732, sub-24071, sub-24103, sub-24138, sub-24172, sub-24201`. Trained `expanded6_topk64_ep10` SAE (val MSE `0.0236`, expvar `0.976`, dead-feature fraction `0.151`). Source-2 prototype removal on 40 balanced segments gives ΔMSE `+0.01974` (1.6× the original 2-subject `+0.01204`) — the original underestimated the prototype's causal weight. Downstream-task probe (HV vs MDD): LOSO baseline AUC `0.715`, prototype-only `0.668`, prototype-removed `0.710`, random-removed `0.715` (bit-identical to baseline). Stratified-5-fold baseline AUC `0.961` (subject-confound dominated). MEG's diagnostic signal is partly concentrated in the source-2 prototype direction (1-d projection captures 87.7% of baseline LOSO accuracy, 93% of baseline LOSO AUC), less completely than PD31.
- `2026-05-08`: Completed the uncropped MEG analyses (`run_real_dataset_analyses.py --dataset MEG-MMI --variant tiny --max-segments 8 --block-limit 4` with no `--clip-seconds`/`--source-limit`). Findings: (a) tokenizer codebook becomes near-fully utilized (489/512/509/512 active codes vs 397/430/434/442 clipped); (b) **the most causally important MEG source under ablation is now source 9 (ΔMSE `0.0503`), not source 2 (ΔMSE `0.0396`)** — the original 2-subject focus on source 2 was a small-sample artifact; (c) the dominant ablation-damage sensor cluster shifted from {244, 115, 251, 122, 245} to {185, 237, 244, 108, 156}; (d) spatial-temporal patching at full duration shows `spatial_mismatch` is a stronger perturbation than `time_reverse` on MEG (both-halves gain 0.209 vs 0.183), opposite of PD31; (e) the spatial half dominates the residual split (1.6× the temporal half's similarity gain), an asymmetry that PD31 does not show at `tiny`.
- `2026-05-08`: Re-ran the downstream-task probe on the original 4-subject PD31 SAE seeds 0/1/2/3 as a robustness check. Stratified-5-fold mean ± std across 4 seeds: baseline `0.895 ± 0.000` / `0.931 ± 0.000`, prototype removed `0.891 ± 0.007` / `0.942 ± 0.001`, prototype only `0.852 ± 0.007` / `0.926 ± 0.001`, random direction removed `0.895 ± 0.000` / `0.935 ± 0.000`. All four seeds agree to within `0.001` AUC; the 1-d sufficiency claim is seed-robust. Numbers in `results/multi_seed_stability/downstream_probe_pd31_orig4subj.csv`.
- `2026-05-08`: Launched `run_meg_expansion_patching.sh` to run residual-vs-projection source-2 prototype patching on the `expanded6` MEG SAE (10 cross-subject pairs, 40 max segments) so the MEG expansion analysis matches the PD31 expansion analysis from this morning's session.
- `2026-05-08`: Completed MEG-MMI expanded6 source-2 prototype donor/recipient patching. Residual patching: mean donor alignment cosine `0.168`, mean ΔMSE `+0.033`. Projection patching: cosine `0.008`, ΔMSE `−0.001`. The residual:projection donor-alignment ratio is `21×` (vs `8×` on PD31 expanded10) — the prototype-orthogonal residual is the *cleaner* carrier of donor-specific structure on MEG even though absolute donor alignment is lower than EEG due to the 269-channel MEG cross-subject diversity. The original 2-subject MEG residual cosine of `~0.99` was an artifact of having only one subject pair worth of contrast; the honest 6-subject 10-pair value is `0.168`. Updated `reports/multi_seed_baseline_replication.md` (sections F, G) and `reports/research_plan.md`. Regenerated all 7 figures via `make_paper_figures.py --figures all`.
- `2026-05-08`: Implemented `run_pd31_residual_decomposition.py` and ran the open TODO: decomposing the PD31 source-3 prototype-orthogonal residual into label-shared and subject-specific components. On `expanded10` (188 segments, 10 subjects, 10 cross-label random pairs): full-residual donor-alignment cosine `1.00` (by construction); `label_orthogonal` `0.867`; `subject_specific` `0.776`; `subject_mean` `0.592`; `label_aligned` (1-d projection onto HC↔PD axis) `0.437`. So ~44% of the residual donor delta lives in the 1-d label axis but ~87% lives in the orthogonal complement (mostly subject-specific noise). HC vs PD label-mean cosine is `0.821`, so the label axis is a small correction on top of a large shared residual mean. Refines the "label-structured residual" claim from the original session: the residual carries label info but most of the donor transport is label-orthogonal, subject-specific structure.
- `2026-05-08`: Re-centered MEG mechanistic pipeline on source 9 (the new most-causal source from the uncropped 6-subject ablation). Source-9 prototype removal: ΔMSE `+0.02324` (vs `+0.01974` for src 2). Source-9 residual patching: cosine `−0.004`, ΔMSE `+0.0205`. Source-9 projection patching: cosine `−0.008`, ΔMSE `+0.0003`. Both src-9 patching modes transport essentially *zero* donor-specific structure (vs src 2's residual cosine `0.168`). Source-9 downstream probe (LOSO HV vs MDD): baseline `0.730`/`0.715`, prototype-only `0.723`/`0.701` (ΔAUC −0.013 vs src 2's −0.047). Conclusion: source 9 carries the shared population baseline (high reconstruction-causal weight, no cross-subject transferable structure, weak diagnostic axis), source 2 carries label-aligned subject variation. The BrainOmni tokenizer's source basis decomposes MEG into qualitatively different latents.
- `2026-05-08`: Trained MEG `expanded6_topk64_ep10` SAEs at seeds 1, 2, 3 (joining seed 0). Ran source-2 + source-9 prototype removal across all 4 seeds × 2 sources = 8 interventions. Source 2: ΔMSE `0.01988 ± 0.00054` across all 4 seeds (rock stable, std 2.7% of mean). Source 9: ΔMSE `0.02476 ± 0.00129` across seeds 0–2 (stable), with seed 3 a clean outlier at `0.00078` (prototype direction has same magnitude but is oriented away from the reconstruction error axis). Multi-seed scope captures one mechanistic instability for source 9 that does not appear for source 2.
- `2026-05-08`: Killed three follow-ups during the run to free CPU: PCA/ICA expanded6 (after 60 min with no method_summary, direction summary captured: SAE prototype 1.0, PCA best 0.454, ICA best −0.748), 11-block uncropped MEG spatial-temporal (only 4-block result kept), base-variant MEG (only tokenizer_semantics done, source_ablation + spatial_temporal not). Same/diff-label MEG patching aborted because the matched-pair CSV referenced segment indices up to 34 but `choose_examples --max-segments` round-robin only loads up to (max-segs/6) per subject. PD31 same-vs-diff-label result already documents the analog from the prior session.
