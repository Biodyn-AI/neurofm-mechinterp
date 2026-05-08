#!/usr/bin/env python
"""PCA + FastICA baselines for the SAE source-prototype direction.

For a given dataset and source index, this script:

1. Loads the same `encoder_pre_quantizer` activation bundle that the SAE was
   trained on, and standardizes activations with the per-feature mean/std
   from a reference SAE checkpoint so the comparison is apples-to-apples.
2. Filters rows by `source_idx`, fits PCA(n) and FastICA(n) on the filtered
   matrix, and finds the principal/independent component whose direction has
   the highest absolute cosine similarity with the SAE source prototype.
3. Runs a `remove_projection` intervention along the SAE prototype, the best
   PCA component, and the best ICA component, scoring reconstruction damage,
   sensor footprint, and overlap with the SAE-prototype damage map.

This mirrors `run_sae_source_prototype_interventions.py` so the SAE prototype
result and the PCA/ICA baseline results can be compared directly.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA, FastICA

from neurofm_mechinterp.analysis.common import save_json
from neurofm_mechinterp.loading import load_braintokenizer, resolve_device
from neurofm_mechinterp.paths import REPO_ROOT, RESULTS_ROOT, ensure_dir
from neurofm_mechinterp.processed_data import choose_examples, load_batch_from_examples, load_metadata
from neurofm_mechinterp.sae import (
    SparseAutoencoder,
    choose_ablation_row_mask,
    compute_tokenizer_activations,
    flatten_source_activations,
    restore_source_activations,
)


DATASET_METADATA = {
    "pd31": REPO_ROOT / "share" / "metadata" / "evaluate" / "pd31" / "BrainOmni" / "info.json",
    "MEG-MMI": REPO_ROOT / "share" / "metadata" / "evaluate" / "MEG-MMI" / "BrainOmni" / "info.json",
}


def default_clip_seconds(dataset: str) -> float | None:
    if dataset == "MEG-MMI":
        return 10.0
    return None


def load_sae_checkpoint(model_dir: Path) -> tuple[SparseAutoencoder, dict, torch.Tensor, torch.Tensor]:
    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=False)
    model = SparseAutoencoder(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        sparsifier=checkpoint["config"].get("sparsifier", "relu"),
        topk=checkpoint["config"].get("topk"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    mean = checkpoint["input_mean"].view(1, -1).float()
    std = checkpoint["input_std"].view(1, -1).float()
    return model, checkpoint, mean, std


def resolve_activation_path(checkpoint: dict) -> Path:
    activation_path_value = checkpoint["config"].get("activation_path")
    if activation_path_value:
        return Path(activation_path_value)
    return RESULTS_ROOT / "sae_exports" / checkpoint["config"]["dataset"] / checkpoint["config"]["hook"] / "activations.pt"


@torch.no_grad()
def compute_sae_prototype(
    model: SparseAutoencoder,
    checkpoint: dict,
    standardized_source_rows: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    feature_sum = torch.zeros(checkpoint["hidden_dim"], dtype=torch.float32)
    for start in range(0, standardized_source_rows.shape[0], batch_size):
        end = min(start + batch_size, standardized_source_rows.shape[0])
        feature_sum += model.encode(standardized_source_rows[start:end]).sum(dim=0)
    feature_mean = feature_sum / float(standardized_source_rows.shape[0])
    decoder_directions = checkpoint["model_state_dict"]["decoder.weight"].float().T
    decoder_directions = decoder_directions / decoder_directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
    prototype = feature_mean @ decoder_directions
    prototype = prototype / prototype.norm().clamp_min(1e-12)
    return prototype


def best_match_direction(components: np.ndarray, target: torch.Tensor) -> tuple[int, float, np.ndarray]:
    """Pick the component whose direction has the largest |cos| with target.

    Returns (component_index, signed_cosine, unit_vector_aligned_with_target).
    """
    target_np = target.cpu().numpy().reshape(-1)
    target_np = target_np / (np.linalg.norm(target_np) + 1e-12)
    components_unit = components / (np.linalg.norm(components, axis=1, keepdims=True) + 1e-12)
    cosines = components_unit @ target_np
    idx = int(np.argmax(np.abs(cosines)))
    cos = float(cosines[idx])
    direction = components_unit[idx]
    if cos < 0:
        direction = -direction
    return idx, cos, direction


def apply_remove_projection(
    standardized_rows: torch.Tensor,
    direction: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    projection = standardized_rows @ direction
    return standardized_rows - scale * projection.unsqueeze(-1) * direction.unsqueeze(0)


def summarize_sensor_counts(segment_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, subset in segment_table.groupby(["method", "source_idx"]):
        counter: Counter = Counter()
        for value in subset["top_delta_sensor_indices"]:
            counter.update(int(index) for index in str(value).split(",") if index)
        method, source_idx = key
        for sensor_idx, count in counter.most_common():
            rows.append(
                {
                    "method": str(method),
                    "source_idx": int(source_idx),
                    "sensor_idx": int(sensor_idx),
                    "count": int(count),
                    "fraction": float(count / len(subset)),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["method", "source_idx", "count", "sensor_idx"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-sae-dir", required=True, help="SAE checkpoint to align baselines against.")
    parser.add_argument("--dataset", choices=["pd31", "MEG-MMI"], required=True)
    parser.add_argument("--source-idx", type=int, required=True)
    parser.add_argument("--row-source-idx", type=int, default=None)
    parser.add_argument("--n-components", type=int, default=64)
    parser.add_argument("--ica-max-iter", type=int, default=500)
    parser.add_argument("--ica-tol", type=float, default=1e-3)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--clip-seconds", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    reference_dir = Path(args.reference_sae_dir)
    sae_model, sae_checkpoint, mean, std = load_sae_checkpoint(reference_dir)
    activation_path = resolve_activation_path(sae_checkpoint)
    bundle = torch.load(activation_path, map_location="cpu", weights_only=False)

    raw_activations = bundle["activations"].float()
    bundle_source_idx = bundle["source_idx"].long()
    standardized_all = (raw_activations - mean) / std
    source_mask = bundle_source_idx == int(args.source_idx)
    if int(source_mask.sum().item()) == 0:
        raise ValueError(f"No activation rows for source {args.source_idx}.")
    standardized_source = standardized_all[source_mask]
    source_matrix_np = standardized_source.cpu().numpy()

    sae_prototype = compute_sae_prototype(
        model=sae_model,
        checkpoint=sae_checkpoint,
        standardized_source_rows=standardized_source,
        batch_size=args.batch_size,
    )

    pca = PCA(n_components=args.n_components, random_state=args.seed)
    pca.fit(source_matrix_np)
    pca_components = pca.components_.astype(np.float64)

    ica = FastICA(
        n_components=args.n_components,
        random_state=args.seed,
        max_iter=args.ica_max_iter,
        tol=args.ica_tol,
        whiten="unit-variance",
    )
    ica.fit(source_matrix_np)
    # FastICA mixing matrix columns are the directions in feature space.
    ica_components = ica.mixing_.T.astype(np.float64)

    pca_idx, pca_cos, pca_direction = best_match_direction(pca_components, sae_prototype)
    ica_idx, ica_cos, ica_direction = best_match_direction(ica_components, sae_prototype)

    pca_direction_t = torch.from_numpy(pca_direction.astype(np.float32))
    ica_direction_t = torch.from_numpy(ica_direction.astype(np.float32))

    direction_summary = pd.DataFrame(
        [
            {"method": "sae_prototype", "component_idx": -1, "cosine_to_sae": 1.0,
             "explained_variance_ratio": float("nan"),
             "norm_squared": float(sae_prototype.pow(2).sum().item())},
            {"method": "pca", "component_idx": pca_idx, "cosine_to_sae": pca_cos,
             "explained_variance_ratio": float(pca.explained_variance_ratio_[pca_idx]),
             "norm_squared": 1.0},
            {"method": "ica", "component_idx": ica_idx, "cosine_to_sae": ica_cos,
             "explained_variance_ratio": float("nan"),
             "norm_squared": 1.0},
        ]
    )

    device = resolve_device(args.device)
    tokenizer = load_braintokenizer(device=str(device))
    mean_dev = mean.to(device)
    std_dev = std.to(device)
    sae_prototype_dev = sae_prototype.to(device)
    pca_direction_dev = pca_direction_t.to(device)
    ica_direction_dev = ica_direction_t.to(device)

    methods = [
        ("sae_prototype", sae_prototype_dev),
        ("pca", pca_direction_dev),
        ("ica", ica_direction_dev),
    ]

    metadata = load_metadata(DATASET_METADATA[args.dataset])
    if args.max_segments is not None:
        metadata = choose_examples(metadata, max_segments=args.max_segments)
    clip_seconds = args.clip_seconds if args.clip_seconds is not None else default_clip_seconds(args.dataset)
    row_source_idx = args.row_source_idx if args.row_source_idx is not None else args.source_idx

    if args.output_dir is None:
        output_dir = ensure_dir(
            RESULTS_ROOT / "baseline_pca_ica" / f"{args.dataset}_source{args.source_idx}_seed{args.seed}"
        )
    else:
        output_dir = ensure_dir(Path(args.output_dir))

    direction_summary.to_csv(output_dir / "direction_summary.csv", index=False)
    np.save(output_dir / "sae_prototype.npy", sae_prototype.cpu().numpy())
    np.save(output_dir / "pca_direction.npy", pca_direction)
    np.save(output_dir / "ica_direction.npy", ica_direction)

    segment_rows = []
    for row in metadata:
        batch = load_batch_from_examples([row]).to(device)
        if clip_seconds is not None:
            clip_samples = int(clip_seconds * 256)
            batch.x = batch.x[..., :clip_samples]

        unfolded = tokenizer.unfold(batch.x)
        sensor_embedding = tokenizer.sensor_embed(batch.pos, batch.sensor_type)
        hook_feature = compute_tokenizer_activations(tokenizer, batch, sae_checkpoint["config"]["hook"])
        baseline_quantized = (
            hook_feature
            if sae_checkpoint["config"]["hook"] == "encoder_post_quantizer"
            else tokenizer.quantizer(hook_feature)[0]
        )
        baseline_reconstruction = tokenizer.decoder(baseline_quantized, sensor_embedding)
        baseline_input_mse = float(((baseline_reconstruction - unfolded) ** 2).mean().item())

        rows = flatten_source_activations(hook_feature)
        standardized_rows = (rows.activations - mean_dev) / std_dev
        row_mask = choose_ablation_row_mask(rows.source_idx, row_source_idx)
        eligible = standardized_rows[row_mask]
        if eligible.shape[0] == 0:
            continue

        for method_name, direction in methods:
            intervened_subset = apply_remove_projection(eligible, direction, args.scale)
            intervened_rows = standardized_rows.clone()
            intervened_rows[row_mask] = intervened_subset
            intervened_hook = restore_source_activations(intervened_rows * std_dev + mean_dev, tuple(hook_feature.shape))
            intervened_quant = (
                intervened_hook
                if sae_checkpoint["config"]["hook"] == "encoder_post_quantizer"
                else tokenizer.quantizer(intervened_hook)[0]
            )
            intervened_reconstruction = tokenizer.decoder(intervened_quant, sensor_embedding)
            reconstruction_delta = intervened_reconstruction - baseline_reconstruction
            sensor_delta_energy = reconstruction_delta.pow(2).mean(dim=(0, 2, 3))
            top_sensors = torch.topk(sensor_delta_energy, k=min(5, sensor_delta_energy.shape[0])).indices.cpu().tolist()
            intervened_input_mse = float(((intervened_reconstruction - unfolded) ** 2).mean().item())
            cosine_to_baseline = float(
                F.cosine_similarity(
                    baseline_reconstruction.reshape(baseline_reconstruction.shape[0], -1),
                    intervened_reconstruction.reshape(intervened_reconstruction.shape[0], -1),
                    dim=-1,
                ).mean().item()
            )
            segment_rows.append(
                {
                    "method": method_name,
                    "dataset": args.dataset,
                    "path": row["path"],
                    "subject": row["subject"],
                    "label": int(row["label"]),
                    "source_idx": int(args.source_idx),
                    "row_source_idx": row_source_idx,
                    "scale": float(args.scale),
                    "baseline_input_mse": baseline_input_mse,
                    "intervened_input_mse": intervened_input_mse,
                    "delta_input_mse": intervened_input_mse - baseline_input_mse,
                    "cosine_to_baseline_reconstruction": cosine_to_baseline,
                    "mean_reconstruction_delta_energy": float(reconstruction_delta.pow(2).mean().item()),
                    "max_reconstruction_delta_energy": float(reconstruction_delta.pow(2).amax().item()),
                    "eligible_row_count": int(eligible.shape[0]),
                    "top_delta_sensor_indices": ",".join(str(idx) for idx in top_sensors),
                }
            )

    segment_table = pd.DataFrame(segment_rows).sort_values(["method", "subject", "path"]).reset_index(drop=True)
    method_summary = (
        segment_table.groupby("method", as_index=False)
        .agg(
            segments=("path", "count"),
            mean_delta_input_mse=("delta_input_mse", "mean"),
            mean_cosine_to_baseline=("cosine_to_baseline_reconstruction", "mean"),
            mean_reconstruction_delta_energy=("mean_reconstruction_delta_energy", "mean"),
        )
        .sort_values("mean_delta_input_mse", ascending=False)
        .reset_index(drop=True)
    )
    sensor_summary = summarize_sensor_counts(segment_table)

    segment_table.to_csv(output_dir / "segment_interventions.csv", index=False)
    method_summary.to_csv(output_dir / "method_summary.csv", index=False)
    sensor_summary.to_csv(output_dir / "sensor_summary.csv", index=False)
    save_json(
        output_dir / "summary.json",
        {
            "dataset": args.dataset,
            "source_idx": int(args.source_idx),
            "row_source_idx": int(row_source_idx),
            "n_components": int(args.n_components),
            "reference_sae_dir": str(reference_dir),
            "activation_path": str(activation_path),
            "pca_best_component_idx": int(pca_idx),
            "pca_best_cosine": float(pca_cos),
            "ica_best_component_idx": int(ica_idx),
            "ica_best_cosine": float(ica_cos),
            "scale": float(args.scale),
            "seed": int(args.seed),
            "num_segments": int(segment_table["path"].nunique()),
        },
    )
    print(output_dir)


if __name__ == "__main__":
    main()
