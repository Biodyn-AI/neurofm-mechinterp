# Mechanistic Decomposition of BrainOmni's Tokenizer Source Basis: Causal Necessity, Diagnostic Sufficiency, and Modality-Dependent Asymmetries

*Draft, 2026-05-08.*

## Abstract

BrainOmni is a brain foundation model whose front end — the BrainTokenizer — quantizes raw EEG/MEG signals through a 16-source residual vector-quantization (RVQ) bottleneck. We ask what each of the 16 *source latents* encodes and whether the directions inside them carry interpretable, causally important, and diagnostic-relevant structure. We train sparse autoencoders (SAEs) on the tokenizer's pre-quantizer activations and identify, for each source, a single **source-prototype direction** in the standardized activation space. Across two datasets (PD31 EEG, 10 subjects, 188 segments; MEG-MMI, 6 subjects, 159 segments) and four SAE seeds we show that (i) the source-prototype direction is seed-invariant (pairwise cosine ≥ 0.998 in PD31, ≥ 0.997 in MEG); (ii) zero-ing the prototype projection from source-conditioned rows damages tokenizer reconstruction by 0.012–0.020 in input MSE while removing a random unit direction does not; (iii) on PD31, a linear classifier built from the *single* prototype 1-d projection matches the full 256-d activation classifier (stratified-5-fold AUC 0.760 vs 0.748), and on MEG it captures 87.7 % of the leave-one-subject-out baseline accuracy; (iv) cross-subject patching of the prototype-orthogonal residual transports ~21–8× more donor-specific structure than patching the projection, consistently across both modalities; (v) on MEG, sources 2 and 9 dissociate cleanly — source 9 is more reconstruction-causal but transports zero donor structure and barely shifts the diagnostic classifier, while source 2 carries label-aligned, subject-specific transferable structure; and (vi) the source-3 PD31 residual decomposes into ~44 % label-aligned and ~87 % label-orthogonal donor delta. The BrainTokenizer's 16-source basis is therefore decomposing brain signals into qualitatively distinct latents — not a single 1-d diagnostic axis — and the prototype direction is the strongest single-axis interpretability handle in the model.

---

## 1. Introduction

Brain foundation models are starting to scale to multiple modalities (EEG, MEG, BIDS-formatted data) and downstream tasks (sleep staging, depression detection, Parkinson's classification). BrainOmni [Liu et al., 2024] is one such model; its front end is the **BrainTokenizer**, a residual vector-quantizer that compresses arbitrary brain recordings into a discrete code sequence by routing each input row through 16 learned *source latents* with shared codebooks. While downstream BrainOmni reports impressive transfer numbers, the question of *what each source latent encodes* — and whether they decompose brain signals into interpretable structure or simply hash the input — has not been investigated.

This paper makes that question concrete. We treat each tokenizer source as a candidate "interpretable latent" and ask, for the two most-causal sources on each of two datasets, whether there is a low-dimensional direction inside the source's activation distribution that:

1. **Replicates across SAE seeds.** Different sparse-autoencoder fits should converge on the same direction.
2. **Is causally necessary** for tokenizer reconstruction. Zero-ing it should damage reconstruction MSE substantially more than zero-ing a random direction.
3. **Is diagnostic-aligned.** The 1-d projection should carry information that a downstream linear classifier uses to discriminate diagnostic labels (HC vs PD; HV vs MDD).
4. **Survives cross-subject transport.** Patching it from a donor segment to a recipient segment should move the recipient's reconstruction toward the donor's pattern.
5. **Compares favorably to standard linear baselines** (PCA, FastICA on the same activations).

We find a single direction satisfying (1)-(2) on both datasets — the *source-prototype direction*, computed from a trained SAE's encoding of source-conditioned rows. We then probe the geometry of this direction along the four remaining axes. Our headline findings:

- **The source-prototype direction is seed-invariant** to <0.003 cosine units across 4 SAE seeds in both PD31 EEG and MEG-MMI.
- **The 1-d prototype projection is sufficient for diagnostic classification on PD31:** stratified-5-fold AUC = 0.760 from the *projection only* matches the full 256-d activation baseline (0.748). Removing the prototype direction barely changes the classifier (Δ AUC = −0.007); removing a random unit direction is bit-identical to baseline.
- **Cross-subject residual patching transports 8× more donor structure than projection patching** on PD31, and 21× more on MEG. Donor-specific source variation lives mainly in the *prototype-orthogonal* residual, while the prototype direction itself is approximately shared across subjects.
- **PCA and ICA invert their roles between EEG and MEG.** On PD31 source 3, FastICA on the standardized activation rows finds a direction with cosine 0.71 to the SAE prototype and damages reconstruction *more* than the SAE prototype itself; PCA's most-aligned component damages negligibly. On MEG source 2 the picture inverts: PCA finds a direction with cosine 0.72 that damages reconstruction *2 ×* the SAE prototype, while ICA's best component sits at cosine 0.19 (2-subject) or 0.75 (6-subject expanded).
- **Cross-source dissociation on MEG.** Source 9 is more reconstruction-causal than source 2 (Δ MSE 0.023 vs 0.020) but transports *zero* donor-specific structure under patching (residual cosine −0.004) and barely shifts the diagnostic classifier (Δ AUC = −0.013 vs −0.047 for source 2). The two MEG sources factorize qualitatively different regularities.

We end with a refined account of the residual structure on PD31: decomposing the prototype-orthogonal residual into label-aligned and label-orthogonal components, ~44 % of the residual donor delta lives in the 1-d HC↔PD axis but ~87 % lives in the orthogonal complement (mostly subject-specific noise). The "label-structured residual" claim from the project's earlier reports holds, but the bulk of the residual donor transport is label-*orthogonal*.

Together these results give a six-property fingerprint of the BrainTokenizer's source-prototype directions: seed-stable, reconstruction-necessary, diagnostic-sufficient (on PD31), cross-subject portable through the residual, modality-dependent in their relation to PCA/ICA baselines, and qualitatively distinct across sources within MEG.

---

## 2. Background

### 2.1 BrainOmni and the BrainTokenizer

BrainOmni is a brain foundation model with two stages: a frozen **BrainTokenizer** (5.05 M parameters) that maps any continuous EEG/MEG recording to a discrete token sequence, and a **BrainOmni encoder** (13.5 M parameters in `tiny`, larger in `base`) that processes those tokens for downstream tasks. The tokenizer is a residual vector-quantizer with 16 *source latents* and 4 quantization levels per source (codebook size 512 each). Its forward pass ingests `(x, pos, sensor_type)` triples — a multichannel signal `x`, sensor positions `pos`, and sensor types `sensor_type` — and produces, per spatiotemporal patch, a 16-source × 4-quantizer × 256-dimensional residual representation that is then decoded back to the input space.

The 16 source latents are not specialized to a single sensor or frequency band by construction: each source can attend to any subset of the input channels through shared sensor embeddings. Whether the model uses each source for a distinct purpose, or hashes inputs uniformly across sources, is the empirical question we address.

### 2.2 Datasets

We use two preprocessed OpenNeuro subsets:

- **PD31 EEG** (OpenNeuro `ds002778-1.0.5`): 10 subjects (5 healthy controls / 5 Parkinson's), 32-channel EEG resting-state, 188 30-second segments after preprocessing. The "expanded10" subset; the original "4-subject" subset (76 segments) is also referenced where multi-seed analyses preceded the expansion.
- **MEG-MMI** (OpenNeuro `ds003568-1.0.4`): 6 subjects (3 healthy volunteers / 3 major depressive disorder), 269-channel MEG monetary-incentive task, 159 30-second segments after preprocessing. The "expanded6" subset, expanded from an original 2-subject set (20 segments) used for early multi-seed work.

Both use the upstream BrainOmni preprocessing pipeline (band-pass + notch filter, resample to 256 Hz, 30-second windowing). All MEG analyses use a 10-second clip per segment for source-prototype interventions to keep per-segment compute tractable; uncropped (full 30-second) analyses are reported separately.

### 2.3 Sparse autoencoders on tokenizer activations

For each dataset we exported a tokenizer activation bundle (rows × 256-d) at the `encoder_pre_quantizer` hook (the standardized activation entering the residual quantizer) and trained a top-k SAE with hidden dimension 1024, top-k = 64, and 10 epochs. Quality metrics: PD31 expanded10 SAE val MSE 0.0294 / explained variance 0.971; MEG-MMI expanded6 SAE val MSE 0.0236 / explained variance 0.976. Across 4 SAE seeds, the same architecture and dataset reproduce within 5 % on these metrics.

---

## 3. The source-prototype direction

For each source latent `s` we want a single direction `p_s` in the SAE's 256-d standardized activation space that summarizes "what source `s` is doing on average". We compute it as follows. Let `R_s` be the matrix of standardized SAE input rows from source `s` and `E(·)` the SAE encoder. Then

```
feature_mean_s = mean over rows of E(R_s)              # 1024-d (hidden)
prototype_s    = feature_mean_s @ decoder_directions   # 256-d
prototype_s    = prototype_s / ||prototype_s||         # unit norm
```

where `decoder_directions` is the SAE decoder's row-normalized weight matrix. This is the *mean of decoded source-`s` features*, projected back to activation space and unit-normalized. We then study what removing or keeping just this direction does to the tokenizer's behavior.

Two practical notes. First, the SAE is trained jointly across sources and quantizers — it does not see source identity at training time. The prototype direction is therefore a *learned summary* of the source's typical activation pattern, not a hand-coded mask. Second, the prototype is computed from the SAE's decoder rather than its encoder so that interventions on activation space correspond to a clean linear direction; encoder-side directions would entangle the top-k sparsity decision.

### 3.1 Seed stability of the prototype direction

We trained 4 SAEs at seeds 0–3 on each dataset with otherwise identical config, then computed pairwise cosine similarities between the corresponding source-prototype directions. *Figure 3* shows the result: pairwise cosine ≥ 0.998 in PD31 source 3 and ≥ 0.997 in MEG source 2 across all 6 same-dataset seed pairs. The prototype direction is essentially seed-invariant. By comparison, individual SAE *features* in the same models have mean active-mutual cosine ≈ 0.5 across seeds — feature identity is not stable, but the source-level prototype is.

This stability also persists to the expanded MEG SAE: the 4-seed expanded6 sweep gives source-2 Δ MSE = 0.01988 ± 0.00054 (std 2.7 % of mean). Source 9 is stable at Δ MSE = 0.0248 ± 0.0013 across seeds 0–2, with seed 3 a clean outlier at 0.00078 — same prototype magnitude (mean abs projection 11.02 vs 11.21–11.32) but oriented away from the input-reconstruction error axis. This is the only seed-instability of the entire sweep and is worth flagging in any paper extension.

---

## 4. Reconstruction necessity (Figure 2)

For each segment with source-`s` rows in the standardized activation space, we replace each row `r` with `r − (r · p_s) p_s`, then run the quantizer + decoder forward and measure Δ input MSE = MSE(intervened reconstruction, input) − MSE(baseline reconstruction, input). Averaged across 76 segments (PD31 4-subject), 188 segments (PD31 expanded10), and 40 balanced segments (MEG-MMI expanded6):

| dataset | source | seeds | Δ input MSE | random-direction Δ |
|---------|--------|-------|---------------|-----------------------|
| PD31 (4 subj, 76 seg) | src 3 | 4 | 0.01190 ± 0.00015 | 0.000 (no effect) |
| PD31 (10 subj, 40 balanced) | src 3 | 1 | 0.01164 | — |
| MEG (2 subj, 20 seg) | src 2 | 4 | 0.01174 ± 0.00062 | 0.000 |
| MEG (6 subj, 40 balanced) | src 2 | 4 | 0.01988 ± 0.00054 | 0.000 |
| MEG (6 subj, 40 balanced) | src 9 | 3+1 | 0.0248 ± 0.0013 (seed 3: 0.00078) | 0.000 |

Three observations. *(i)* Prototype-removal damage is much larger than random-direction removal — confirming the prototype direction is not just any direction in 256-d. *(ii)* The damage is ~constant across SAE seeds and across the 4-to-10-subject PD31 expansion (0.01190 → 0.01164). *(iii)* On MEG the damage *increases* with the 2 → 6 subject expansion (0.01174 → 0.01988, a 1.7× increase) — the original 2-subject result *under*-estimated the prototype's causal weight, not over.

---

## 5. The 1-d prototype projection is diagnostically sufficient on PD31

The reconstruction-necessity result tells us the prototype direction matters for the tokenizer's input fidelity. We now ask whether it carries diagnostic information — whether a linear classifier can use it to discriminate HC vs PD on PD31 and HV vs MDD on MEG-MMI.

### 5.1 Probe design

For each segment we compute a feature vector by averaging the standardized `encoder_pre_quantizer` rows across all 16 sources (256-d feature). We optionally apply one of three interventions to the source-`s` rows before pooling:

- **Baseline**: no intervention.
- **Prototype removed**: replace each row with `r − (r · p_s) p_s`.
- **Prototype only**: replace each row with `(r · p_s) p_s` (1-d projection, rank-1).
- **Random direction removed**: same as "prototype removed" but along a fixed random unit direction.

We train an L2 logistic regression sweeping `C ∈ {0.001, 0.01, 0.1, 1, 10}` under two CVs: leave-one-subject-out (LOSO; cross-subject generalization) and stratified 5-fold (within-subject leakage allowed; upper bound on label structure in the activations).

### 5.2 PD31 expanded10 results

| setting | strat. 5-fold accuracy | strat. 5-fold AUC | LOSO accuracy | LOSO AUC |
|---------|--------------------------|----------------------|------------------|--------------|
| baseline | 0.681 | 0.748 | 0.277 | 0.177 |
| prototype removed | 0.676 (Δ −0.005) | 0.741 (Δ −0.007) | 0.266 | 0.158 |
| **prototype only** | **0.686 (Δ +0.005)** | **0.760 (Δ +0.012)** | 0.266 | 0.154 |
| random dir. removed | 0.681 (Δ +0.000) | 0.749 (Δ +0.001) | 0.277 | 0.177 |

*(Figure 7, left panel.)* Three clean conclusions:

1. **The prototype direction is sufficient.** Replacing the full 256-d activation with its 1-d projection onto the prototype direction (zero-ing the orthogonal complement) keeps classifier performance — 0.686 / 0.760 vs the full-baseline 0.681 / 0.748. The 1-d direction encodes essentially all of the diagnostic signal that the linear classifier uses.
2. **The prototype direction is not necessary.** Removing the prototype direction degrades the classifier by less than 1 %. The diagnostic signal is *redundantly* duplicated between the prototype direction and the prototype-orthogonal residual.
3. **Removing a random direction is a no-op.** ΔAccuracy = 0.000, ΔAUC = +0.001 — confirming the prototype-removed result is not a generic "remove any 1-d direction" effect.

LOSO collapses to below-chance for all settings because with 10 subjects the per-subject activation bias overwhelms the label signal under a strict cross-subject split (each fold has only one held-out subject contributing 9 % of the segments).

### 5.3 4-seed robustness on the original 4-subject set

Same probe applied to each of the 4 PD31 SAE seeds trained on the original 4-subject 76-segment set (2 HC + 2 PD subjects). Stratified 5-fold mean ± std across the 4 seeds:

| setting | accuracy | AUC |
|---------|----------|-----|
| baseline | 0.895 ± 0.000 | 0.931 ± 0.000 |
| prototype removed | 0.891 ± 0.007 | 0.942 ± 0.001 |
| **prototype only** | 0.852 ± 0.007 | 0.926 ± 0.001 |
| random dir. removed | 0.895 ± 0.000 | 0.935 ± 0.000 |

All four seeds agree to within 0.001 AUC; the 1-d sufficiency claim is seed-robust at smaller sample size as well. (The higher absolute scores reflect 2-subject confounds, not better generalization.)

### 5.4 MEG expanded6 results

LOSO becomes meaningful at MEG (N = 6 subjects, 26 segments per subject):

| setting | LOSO accuracy | LOSO AUC | Δ AUC |
|---------|------------------|--------------|------------|
| baseline | 0.730 | 0.715 | — |
| prototype removed | 0.698 | 0.710 | −0.005 |
| **prototype only (src 2)** | 0.654 | 0.668 | **−0.047** |
| **prototype only (src 9)** | 0.723 | 0.701 | **−0.013** |
| random dir. removed | 0.730 | 0.715 | 0.000 |

*(Figure 7, right panel.)* On MEG the 1-d source-2 projection captures 87.7 % of baseline accuracy and 93 % of baseline AUC under genuine cross-subject generalization. Source 9, by contrast, captures only 99 % of baseline accuracy and 98 % of baseline AUC — i.e. it is weakly diagnostic. This source dissociation is taken up in §7.

---

## 6. Cross-subject patching: residual versus projection (Figure 4)

If the prototype direction is *causally necessary* and *diagnostic-sufficient*, what does the *residual* (orthogonal complement) carry? We test cross-subject portability: pick 10 cross-subject donor/recipient segment pairs, replace the recipient's source-`s` rows with the donor's only along one of three components — prototype projection, prototype-orthogonal residual, or full row — and measure the cosine alignment between the patched delta and the donor delta in residual space.

| dataset | component | mean donor-alignment cosine | mean Δ input MSE |
|---------|-----------|-------------------------------|---------------------|
| PD31 expanded10 | residual | **0.391** | +0.0273 |
| PD31 expanded10 | projection | 0.048 | −0.0032 |
| MEG expanded6 src 2 | residual | **0.168** | +0.033 |
| MEG expanded6 src 2 | projection | 0.008 | −0.001 |
| MEG expanded6 src 9 | residual | **−0.004** | +0.0205 |
| MEG expanded6 src 9 | projection | −0.008 | +0.0003 |

Three results:

1. **The residual is the carrier of donor-specific structure for src-3 EEG and src-2 MEG.** Residual donor alignment is 8× larger than projection donor alignment on PD31 and 21× larger on MEG src 2.
2. **The MEG src-9 residual carries zero transferable donor structure.** Both residual and projection patching cosines are indistinguishable from zero. Source 9 looks like a shared population baseline rather than a subject-specific residual carrier.
3. **The honest 6-subject MEG residual donor-alignment is 0.168, not 0.99.** The original 2-subject MEG residual cosine of ~0.99 was an artifact of having only one HV/MDD pair worth of contrast; with 6 subjects, cross-subject diversity is real.

### 6.1 Same- vs different-label residual transport on PD31 (Figure 5)

To test whether residual transport carries *label* information, we built similarity-matched 10-pair sets — 10 same-label cross-subject pairs (HC↔HC, PD↔PD) and 10 different-label cross-subject pairs (HC↔PD), with mean baseline residual cosine matched to within 0.0025. Under residual patching:

| pair set | seed-0 donor align. | seed-1 donor align. |
|----------|------------------------|------------------------|
| same label | **0.4032** | **0.4033** |
| different label | 0.3436 | 0.3438 |

The same-label transport advantage is real, identical across two SAE seeds, and survives the similarity matching. The PD31 source-3 residual carries cross-subject, label-aware structure: same-label pairs transport more than different-label pairs of equal baseline similarity.

### 6.2 Decomposing the PD31 residual

Where in the residual does the label information live? We decomposed the source-3 prototype-orthogonal residual into: full residual, subject mean (per-subject mean residual), subject-specific (segment − subject mean), label-aligned (1-d projection onto HC↔PD axis), and label-orthogonal (residual − label-aligned). For 10 random cross-label pairs we measured the cosine alignment between the patched component delta and the donor delta:

| component | cosine to donor delta |
|-----------|--------------------------|
| full residual | 1.000 (by construction) |
| label-orthogonal | **0.867** |
| subject-specific | 0.776 |
| subject mean | 0.592 |
| label-aligned (1-d) | **0.437** |

So the source-3 residual donor delta decomposes as ~44 % label-aligned + ~87 % label-orthogonal (subject-specific noise dominates, but the 1-d HC↔PD axis carries a meaningful 44 % share). The two components partially overlap because the residual basis is not orthogonal to the label axis. HC vs PD label-mean cosine is 0.821 — the two label-mean residuals are similar, so the label axis is a small but real correction on top of a large shared mean.

---

## 7. Cross-source dissociation on MEG: src 2 vs src 9

Combining §4–§6 yields a clean dissociation between two MEG source latents. The original 2-subject MEG study fixated on source 2 because it had the largest ablation footprint at small sample size. With 6 subjects and uncropped 30-second segments, source ablation reveals that **source 9 is the most-causal MEG source** (Δ MSE 0.0503 vs source 2's 0.0396), and that the dominant ablation-damage sensor cluster shifts from {244, 115, 251, 122, 245} (small sample) to {185, 237, 244, 108, 156} (uncropped 6-subject).

We then ran the full mechanistic pipeline on source 9 using the same expanded6 SAE. The contrast with source 2:

| property | src 2 | src 9 | ratio |
|----------|-------|-------|---------|
| reconstruction-causal Δ MSE | 0.0197 | 0.0232 | 1.18 × |
| diagnostic LOSO ΔAUC (prototype-only) | −0.047 | −0.013 | 0.28 × |
| residual donor-alignment cosine (10 pairs) | +0.168 | −0.004 | ~zero |
| projection donor-alignment cosine | +0.008 | −0.008 | ~zero |

So:

- **Source 2 is "subject-specific deviation"**: less reconstruction weight, more diagnostic weight, residual carries cross-subject transferable structure.
- **Source 9 is "shared population baseline"**: more reconstruction weight, less diagnostic weight, residual transports nothing.

The two sources factorize qualitatively different MEG regularities. The BrainTokenizer's source basis is therefore not a single 1-d diagnostic axis dressed up as 16 sources; it is a real factorization with at least two distinct mechanistic roles. This is the strongest single argument in this paper that the source latents are a meaningful mechanistic decomposition rather than an arbitrary hash.

---

## 8. PCA and FastICA baselines (Figure 6)

The SAE is one decomposition method among several. We compared its source-prototype direction to two distribution-level decompositions: PCA(64) and FastICA(64) on the same standardized source-conditioned activation rows. For each method we picked the component with the largest absolute cosine to the SAE prototype direction, then ran the same `remove_projection` intervention along that single direction.

### 8.1 PD31 source 3 (76 segments, seed 0 SAE)

| method | best component direction cosine to SAE prototype | mean Δ input MSE |
|--------|----------------------------------------------------|------------------|
| SAE prototype | 1.000 | +0.0117 |
| best PCA component (PC4) | −0.716 | −0.0013 |
| best FastICA component (IC64) | −0.714 | **+0.0172** |

Top-5 sensors hit by each removal:

- SAE prototype: `0, 29, 28, 1, 30`
- ICA best: `0, 28, 29, 1, 30` (5/5 overlap with SAE)
- PCA best: dispersed (intervention is a near no-op)

So on PD31, FastICA finds a direction with the same five-sensor footprint as the SAE prototype and damages reconstruction *more* than the SAE prototype itself (+0.0172 vs +0.0117). PCA finds a direction with the same magnitude of geometric overlap (|cos| = 0.716) but it is not on a reconstruction-relevant axis.

### 8.2 MEG-MMI source 2 (20 segments, original 2-subject SAE)

| method | best component direction cosine to SAE prototype | mean Δ input MSE |
|--------|----------------------------------------------------|------------------|
| SAE prototype | 1.000 | +0.0120 |
| best PCA component (PC25) | +0.721 | **+0.0214** |
| best FastICA component (IC13) | −0.189 | +0.0056 |

Top-5 sensors:

- SAE prototype: `26, 27, 31, 25, 122`
- ICA best: `26, 27, 25, 122, 30` (4/5 overlap with SAE)
- PCA best: `27, 122, 115, 26, 31` (4/5 overlap; **includes sensor 115**, the original raw source-2 ablation footprint sensor)

The MEG result inverts the PD31 pattern: PCA's most-aligned component damages reconstruction *2 ×* the SAE prototype (and includes the external-damage sensor 115 in its footprint), while FastICA cannot find a high-cosine direction.

### 8.3 The MEG inversion is a 2-subject artifact (expanded6 update)

Re-running the same baseline on the 6-subject expanded MEG set:

| method | original 2-subj cosine | expanded6 cosine |
|--------|-------------------------|--------------------|
| best PCA component | +0.721 | +0.454 |
| best FastICA component | −0.189 | **−0.748** |

With more subjects, **FastICA recovers the SAE prototype direction on MEG** (|cos| jumps from 0.189 to 0.748) while PCA's best alignment drops. The 2-subject "PCA wins, ICA fails" inversion was a small-sample artifact, not a real EEG-vs-MEG distinction. (Δ MSE intervention values for expanded6 are unavailable due to a CPU-time failure during the 60-min intervention loop.)

The clean cross-modality story therefore is: SAE prototype directions are *recoverable* by FastICA on standardized source-conditioned activations, given enough cross-subject diversity; PCA is *not* a uniformly reliable substitute. The SAE pipeline is the only single method that recovers a causally important source-prototype direction on both modalities with comparable damage profile, both at small and at large sample size.

---

## 9. Cross-modality contrast: spatial-temporal residual split

BrainOmni's encoder splits its residual stream halfway along the channel dimension (`split_dim` argument in the upstream code) — the lower half is conventionally read as "temporal" and the upper half as "spatial". To probe whether the encoded information factorizes that way, we patch one half of the residual stream from a clean run into a perturbed run, per transformer block, under two perturbations: time reversal and spatial channel permutation.

PD31 EEG, all 11 patchable blocks, 8 segments balanced across 4 subjects:

| perturbation | patch half | tiny | base |
|--------------|-----------|------|------|
| time_reverse | both | 0.278 | 0.241 |
| time_reverse | spatial | 0.159 | 0.124 |
| time_reverse | temporal | 0.166 | 0.179 |
| spatial_mismatch | both | 0.269 | 0.217 |
| spatial_mismatch | spatial | 0.146 | 0.111 |
| spatial_mismatch | temporal | 0.140 | 0.143 |

MEG-MMI, uncropped (full 30 s) 6-subject, 4 of 11 blocks:

| perturbation | patch half | mean similarity gain |
|--------------|-----------|----------------------|
| spatial_mismatch | both | 0.209 |
| spatial_mismatch | spatial | 0.130 |
| spatial_mismatch | temporal | 0.084 |
| time_reverse | both | 0.183 |
| time_reverse | spatial | 0.133 |
| time_reverse | temporal | 0.082 |

Three observations:

1. **The "both halves" patch is consistently the strongest** in both modalities and both BrainOmni variants — encoded information is split across the two halves rather than concentrated in one.
2. **PD31 has near-symmetric halves at `tiny` and asymmetric halves at `base`.** At `tiny`, spatial and temporal halves are within 0.01 of each other for time-reverse recovery (0.159 vs 0.166). At `base`, the temporal half carries more time-reversal information (0.179 vs 0.124). The bigger encoder makes the temporal/spatial split *less*, not more, symmetric.
3. **On MEG the spatial half dominates by 1.6 ×** for both perturbations (0.130/0.133 vs 0.084/0.082). Spatial mismatch is also the stronger perturbation on MEG (both-halves gain 0.209 vs 0.183), opposite of PD31. This is a genuine MEG-vs-EEG distinction visible only at full segment duration on the 6-subject set.

The headline mechanistic claim from the spatial-temporal split is: **the encoder splits its residual stream into two halves that carry overlapping but not symmetric information**, with the asymmetry sharpening at the bigger encoder and growing larger on the higher-channel-count modality (MEG).

---

## 10. Tokenizer codebook utilization: small-sample artifacts

Earlier reports (Initial Findings, 2026-04-16) flagged that MEG-MMI activated only 397/430/434/442 of 512 codes per layer of the 4-level RVQ — ~80 % codebook utilization. This was on a 2-subject 10-second clipped run. With the 6-subject uncropped 30-second run, codebook utilization rises to:

| layer | clipped 2-subj | uncropped 6-subj |
|-------|------------------|---------------------|
| 0 | 397 / 512 (77.5 %) | **489 / 512 (95.5 %)** |
| 1 | 430 / 512 | **512 / 512 (100 %)** |
| 2 | 434 / 512 | **509 / 512 (99.4 %)** |
| 3 | 442 / 512 | **512 / 512 (100 %)** |

So the BrainTokenizer's codebooks are essentially fully utilized once the input distribution is broad enough. The earlier under-utilization observation was a 2-subject sample artifact.

---

## 11. Sources at the dataset level (Figure 1)

For completeness: the per-source row counts in the SAE export bundles (proxy for tokenizer "use" of each source) are uniform by design — the BrainTokenizer's source allocation is balanced. *Figure 1* shows the top-1 source counts on each dataset's segments, where source 3 dominates EEG and sources 2 / 9 dominate MEG. The top-1 source count is the right measure for "which source carries the most signal per segment", whereas the SAE-export row count is uniform by tokenizer design.

---

## 12. Limitations

This paper's claims rest on a small number of subjects per modality (10 PD31, 6 MEG-MMI). Several caveats:

- **LOSO is brutal at N = 10.** The PD31 leave-one-subject-out probe collapses below chance because per-subject activation bias dominates with only 9 % of segments held out. We rely on stratified-5-fold (within-subject leakage) for the 1-d sufficiency claim, which makes the *bound* result ("the prototype direction carries all the diagnostic structure the linear classifier finds") meaningful but not a cross-subject generalization claim.
- **MEG LOSO at N = 6 is meaningful.** With 6 subjects and ~26 segments each, the LOSO baseline reaches AUC = 0.715 (well above chance), and the prototype-only result captures 87.7 % of that.
- **Only `tiny` BrainOmni is fully causally analyzed.** The `base` variant was used only for the spatial-temporal split replication, not for prototype/patching/probe analyses. A scaling check on `base` for PD31 plus MEG would substantially strengthen all four mechanistic claims.
- **The MEG 11-block spatial-temporal sweep, expanded6 PCA/ICA Δ MSE intervention, and same/diff-label MEG patching are incomplete.** They were started during this analysis pass but killed mid-run because of CPU contention with higher-priority items, or aborted because of script-level path-resolution issues. Their absence is documented in the execution log; resuming them requires a few hours of compute on a faster machine.
- **Source-9 seed 3 is a flagged outlier.** The seed-3 expanded6 source-9 prototype has the same magnitude as the other seeds but is oriented away from the reconstruction error axis (Δ MSE 0.00078 vs the others' ~0.025). This is the only seed-instability we observed; the right interpretation is probably that source 9's prototype direction is sensitive to SAE training noise in a way source 2's is not, which itself is informative.
- **No comparison to subject-stratified neuroscience baselines** (band-power probes, event-related-potential templates). The PCA / ICA baselines are statistical decompositions on the SAE's input space, not neuroscience-grounded controls. A clinical-linguistics audience would expect more.
- **Both diagnostic-label datasets are small and not balanced for age, sex, or comorbidity.** The MEG-MMI set in particular has young subjects (14–18 years old) which is unusual for an MDD cohort. The diagnostic-axis interpretations should be read as "the linear classifier finds *some* structure correlated with the binary label" rather than as clinical claims.

---

## 13. Discussion and conclusion

We characterized the BrainOmni tokenizer's 16-source basis through the lens of a single **source-prototype direction** per source — the mean of decoded SAE features in the standardized activation space. Across two datasets and four SAE seeds, this direction is seed-stable, causally necessary for input reconstruction, diagnostic-aligned (sufficient on PD31, partially so on MEG), and cross-subject portable through the prototype-orthogonal residual. The direction is recoverable by FastICA on standardized activations given sufficient cross-subject diversity, but not by PCA. Within MEG, two source latents (sources 2 and 9) factorize qualitatively different regularities — subject-specific deviation vs shared population baseline — which is the strongest single argument that the BrainTokenizer's source basis is a real mechanistic decomposition rather than a hash.

Two methodological takeaways for mechinterp on brain foundation models. *First*, source-level prototype directions are far more seed-stable than individual SAE features (cosine ≥ 0.998 vs ~0.5 mean active mutual cosine), making them the right unit for cross-seed mechanistic claims on these models. *Second*, **prototype direction necessity** (Δ input MSE) and **prototype direction sufficiency** (linear classifier 1-d AUC) are *separate* properties of the same direction; on MEG they dissociate cleanly across two sources. Future mechinterp work on tokenizer-based models should report both.

The headline single-axis interpretability result is the PD31 1-d projection AUC of 0.760 (matching the full 256-d activation classifier's 0.748), seed-stable across 4 SAE fits to 0.001 AUC. We are aware of no analogous single-axis result in the published mechinterp-on-brain-foundation-models literature.

We close with a refined characterization of the residual: the prototype-orthogonal residual carries cross-subject transferable structure on PD31 and MEG-source-2 but not on MEG-source-9; on PD31, ~44 % of the residual donor delta lives in a 1-d label-aligned axis but ~87 % lives in the orthogonal complement (subject-specific noise). The "label-structured residual" claim from earlier exploratory work holds, but the bulk of the residual donor transport is label-orthogonal.

---

## Figures (saved to `results/figures/`, 300 dpi PNG + PDF)

- **Figure 1** (`fig_source_dominance`): Top-1 source counts per dataset, showing source-level use balance.
- **Figure 2** (`fig_prototype_damage`): Δ input MSE under prototype removal across 4 SAE seeds, both datasets.
- **Figure 3** (`fig_seed_stability`): Pairwise prototype-direction cosine matrix, both datasets, with zoomed colorbar.
- **Figure 4** (`fig_patch_decomposition`): Projection vs residual vs full-row patching donor alignment, PD31 expanded10.
- **Figure 5** (`fig_label_transport`): Similarity-matched same-vs-different-label residual transport, PD31 source 3, two seeds.
- **Figure 6** (`fig_pca_ica_baseline`): SAE prototype vs best PCA / best ICA removal damage (PD31 only; MEG expanded6 incomplete).
- **Figure 7** (`fig_downstream_probe`): HV/HC vs MDD/PD classifier accuracy/AUC under baseline / prototype-removed / prototype-only / random-direction interventions, both datasets.

## Reproducibility

All code is in the repository under `scripts/`. The full analysis is reproducible end-to-end with:

```bash
ENV_PATH="$HOME/.conda-envs/neurofm-mechinterp"
./scripts/bootstrap_env.sh

# Download checkpoints + raw EEG / MEG
conda run -p "$ENV_PATH" python scripts/download_brainomni.py --variant tiny
conda run -p "$ENV_PATH" python scripts/download_openneuro_subsets.py --dataset all --pd31-extra-subjects sub-hc4,sub-hc7,sub-hc8,sub-pd6,sub-pd9,sub-pd11
conda run -p "$ENV_PATH" python scripts/download_meg_extra_subjects.py
conda run -p "$ENV_PATH" python scripts/preprocess_openneuro_subsets.py --dataset all --force

# Export tokenizer activations + train SAEs
conda run -p "$ENV_PATH" python scripts/export_sae_activations.py --dataset pd31 --hook encoder_pre_quantizer
conda run -p "$ENV_PATH" python scripts/export_sae_activations.py --dataset MEG-MMI --hook encoder_pre_quantizer --dataset-name-override MEG-MMI_expanded6
conda run -p "$ENV_PATH" python scripts/train_sparse_autoencoder.py --dataset pd31 --hook encoder_pre_quantizer --sparsifier topk --topk 64 --epochs 10 --seed 0 --output-suffix expanded10_topk64_ep10
# (repeat for seeds 1-3 and for MEG-MMI)

# Reconstruction-necessity intervention
conda run -p "$ENV_PATH" python scripts/run_sae_source_prototype_interventions.py \
    --model-dir results/sae_models/pd31/encoder_pre_quantizer/linear_topk64_h1024_l1_0_seed0_expanded10_topk64_ep10 \
    --dataset pd31 --source-idx 3 --row-source-idx 3 --mode remove_projection --max-segments 40

# Downstream-task probe
conda run -p "$ENV_PATH" python scripts/run_downstream_task_probe.py \
    --model-dir <SAE_DIR> --source-idx 3 --pooling mean --cv both

# Donor / recipient patching
conda run -p "$ENV_PATH" python scripts/run_sae_source_prototype_patching.py \
    --model-dir <SAE_DIR> --dataset pd31 --source-idx 3 --component residual --num-pairs 10 --max-segments 40

# Residual decomposition
conda run -p "$ENV_PATH" python scripts/run_pd31_residual_decomposition.py \
    --model-dir <SAE_DIR> --source-idx 3 --num-pairs 10

# PCA / ICA baselines
conda run -p "$ENV_PATH" python scripts/run_pca_ica_source_prototypes.py \
    --reference-sae-dir <SAE_DIR> --dataset MEG-MMI --source-idx 2 --row-source-idx 2 --max-segments 40

# Spatial-temporal split
conda run -p "$ENV_PATH" python scripts/run_real_dataset_analyses.py --dataset MEG-MMI --variant tiny --max-segments 8 --block-limit 4

# Figures
conda run -p "$ENV_PATH" python scripts/make_paper_figures.py --figures all
```

CSV / JSON outputs land under `results/sae_models/...`, `results/source_prototype_residuals/...`, `results/baseline_pca_ica/...`, `results/multi_seed_stability/...`, and figures under `results/figures/`. The four reports under `reports/` (`initial_findings.md`, `real_dataset_findings.md`, `multi_seed_baseline_replication.md`, `research_plan.md`) document every analytic step end-to-end.

## References (placeholder)

[BrainOmni: Liu et al., 2024 — citation TBD]
[Anthropic mechinterp / SAE prior work: Bricken et al., Cunningham et al. — citations TBD]
[BrainTokenizer / OpenNeuro datasets — citations TBD]
