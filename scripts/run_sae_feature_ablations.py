#!/usr/bin/env python
"""Ablate sparse-autoencoder features and measure tokenizer reconstruction changes."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from neurofm_mechinterp.analysis.common import save_json
from neurofm_mechinterp.loading import load_braintokenizer, resolve_device
from neurofm_mechinterp.paths import RESULTS_ROOT, REPO_ROOT, ensure_dir
from neurofm_mechinterp.processed_data import choose_examples, load_batch_from_examples, load_metadata
from neurofm_mechinterp.sae import (
    SparseAutoencoder,
    choose_ablation_row_mask,
    compute_tokenizer_activations,
    flatten_source_activations,
    resolve_row_source_idx,
    restore_source_activations,
)


DATASET_METADATA = {
    "pd31": REPO_ROOT / "share" / "metadata" / "evaluate" / "pd31" / "BrainOmni" / "info.json",
    "MEG-MMI": REPO_ROOT / "share" / "metadata" / "evaluate" / "MEG-MMI" / "BrainOmni" / "info.json",
}


def default_clip_seconds(dataset: str) -> float | None:
    """Use the same practical MEG crop as the other real-data analyses."""
    if dataset == "MEG-MMI":
        return 10.0
    return None


def load_sae(model_dir: Path) -> tuple[SparseAutoencoder, dict, torch.Tensor, torch.Tensor]:
    """Load a trained sparse autoencoder plus normalization statistics."""
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
    return model, checkpoint["config"], mean, std


def choose_feature_indices(model_dir: Path, explicit: list[int] | None, source_idx: int | None, top_features: int) -> list[int]:
    """Resolve feature indices either explicitly or from a source-top-features table."""
    if explicit:
        return sorted(dict.fromkeys(int(index) for index in explicit))
    if source_idx is None:
        raise ValueError("Provide either `--feature-idx` or `--source-idx`.")

    table = pd.read_csv(model_dir / "source_top_features.csv")
    subset = table[table["source_idx"] == int(source_idx)].head(top_features)
    if subset.empty:
        raise ValueError(f"No top features found for source {source_idx}.")
    return subset["feature_idx"].astype(int).tolist()


def summarize_sensor_counts(segment_table: pd.DataFrame) -> pd.DataFrame:
    """Count how often each sensor appears in each feature's top delta-sensor set."""
    rows = []
    for feature_idx, subset in segment_table.groupby("feature_idx"):
        counter = Counter()
        for value in subset["top_delta_sensor_indices"]:
            counter.update(int(index) for index in str(value).split(",") if index)
        for sensor_idx, count in counter.most_common():
            rows.append(
                {
                    "feature_idx": int(feature_idx),
                    "sensor_idx": int(sensor_idx),
                    "count": int(count),
                    "fraction": float(count / len(subset)),
                }
            )
    return pd.DataFrame(rows).sort_values(["feature_idx", "count", "sensor_idx"], ascending=[True, False, True]).reset_index(drop=True)


def build_feature_summary(segment_table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one row per feature across all profiled segments."""
    rows = []
    for feature_idx, subset in segment_table.groupby("feature_idx"):
        top_source_counts = subset["top_active_source_idx"].value_counts().sort_values(ascending=False)
        top_source_idx = int(top_source_counts.index[0])
        rows.append(
            {
                "feature_idx": int(feature_idx),
                "segments_profiled": int(len(subset)),
                "dominant_top_active_source_idx": top_source_idx,
                "dominant_top_active_source_fraction": float((subset["top_active_source_idx"] == top_source_idx).mean()),
                "mean_delta_input_mse": float(subset["delta_input_mse"].mean()),
                "mean_cosine_to_baseline_reconstruction": float(subset["cosine_to_baseline_reconstruction"].mean()),
                "mean_reconstruction_delta_energy": float(subset["mean_reconstruction_delta_energy"].mean()),
                "mean_global_active_row_fraction": float(subset["global_active_row_fraction"].mean()),
                "mean_conditioned_active_row_fraction": float(subset["conditioned_active_row_fraction"].mean()),
                "mean_conditioned_row_fraction": float(subset["conditioned_row_fraction"].mean()),
                "mean_ablated_row_count": float(subset["ablated_row_count"].mean()),
                "mean_top_active_source_fraction": float(subset["top_active_source_fraction"].mean()),
                "row_source_idx": int(subset["row_source_idx"].iloc[0]) if pd.notna(subset["row_source_idx"].iloc[0]) else None,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_delta_input_mse", "mean_reconstruction_delta_energy"],
        ascending=[False, False],
    ).reset_index(drop=True)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset", choices=["pd31", "MEG-MMI"], required=True)
    parser.add_argument("--feature-idx", type=int, nargs="*", default=None)
    parser.add_argument("--source-idx", type=int, default=None)
    parser.add_argument(
        "--row-source-idx",
        type=int,
        default=None,
        help="Only ablate rows from this tokenizer source. Defaults to --source-idx when provided.",
    )
    parser.add_argument("--top-features", type=int, default=3)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=None,
        help="Optionally truncate segments. Defaults to 10 seconds for MEG ablations.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-suffix", default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    sae, config, mean, std = load_sae(model_dir)
    hook = config["hook"]
    feature_indices = choose_feature_indices(model_dir, args.feature_idx, args.source_idx, args.top_features)
    row_source_idx = resolve_row_source_idx(args.source_idx, args.row_source_idx)

    device = resolve_device(args.device)
    tokenizer = load_braintokenizer(device=str(device))
    sae = sae.to(device)
    mean = mean.to(device)
    std = std.to(device)

    metadata = load_metadata(DATASET_METADATA[args.dataset])
    if args.max_segments is not None:
        metadata = choose_examples(metadata, max_segments=args.max_segments)

    run_name = "feature_ablation"
    if args.source_idx is not None:
        run_name = f"{run_name}_source_{args.source_idx}"
    if row_source_idx is not None:
        run_name = f"{run_name}_rows_{row_source_idx}"
    if args.output_suffix:
        run_name = f"{run_name}_{args.output_suffix}"
    output_dir = ensure_dir(model_dir / run_name)

    segment_rows = []
    for row in metadata:
        batch = load_batch_from_examples([row]).to(device)
        clip_seconds = args.clip_seconds if args.clip_seconds is not None else default_clip_seconds(args.dataset)
        if clip_seconds is not None:
            clip_samples = int(clip_seconds * 256)
            batch.x = batch.x[..., :clip_samples]

        unfolded = tokenizer.unfold(batch.x)
        sensor_embedding = tokenizer.sensor_embed(batch.pos, batch.sensor_type)
        encoder_feature = tokenizer.encoder(unfolded, sensor_embedding)
        quantized_feature, _, _ = tokenizer.quantizer(encoder_feature)
        hook_feature = compute_tokenizer_activations(tokenizer, batch, hook)
        baseline_reconstruction = tokenizer.decoder(quantized_feature, sensor_embedding)
        baseline_input_mse = float(((baseline_reconstruction - unfolded) ** 2).mean().item())

        rows = flatten_source_activations(hook_feature)
        standardized_rows = (rows.activations - mean) / std
        sae_features = sae.encode(standardized_rows)

        for feature_idx in feature_indices:
            ablated_sae_features = sae_features.clone()
            global_activation_mask = ablated_sae_features[:, feature_idx] > 0
            ablation_row_mask = choose_ablation_row_mask(rows.source_idx, row_source_idx)
            activation_mask = global_activation_mask & ablation_row_mask
            ablated_sae_features[activation_mask, feature_idx] = 0
            ablated_rows = sae.decode(ablated_sae_features) * std + mean
            ablated_hook_feature = restore_source_activations(ablated_rows, tuple(hook_feature.shape))

            if hook == "encoder_pre_quantizer":
                ablated_quantized_feature, _, _ = tokenizer.quantizer(ablated_hook_feature)
                ablated_reconstruction = tokenizer.decoder(ablated_quantized_feature, sensor_embedding)
            elif hook == "encoder_post_quantizer":
                ablated_reconstruction = tokenizer.decoder(ablated_hook_feature, sensor_embedding)
            else:
                raise ValueError(f"Unsupported hook for ablations: {hook}")

            reconstruction_delta = ablated_reconstruction - baseline_reconstruction
            sensor_delta_energy = reconstruction_delta.pow(2).mean(dim=(0, 2, 3))
            top_sensors = torch.topk(sensor_delta_energy, k=min(5, sensor_delta_energy.shape[0])).indices.cpu().tolist()

            ablated_input_mse = float(((ablated_reconstruction - unfolded) ** 2).mean().item())
            cosine_to_baseline = float(
                F.cosine_similarity(
                    baseline_reconstruction.reshape(baseline_reconstruction.shape[0], -1),
                    ablated_reconstruction.reshape(ablated_reconstruction.shape[0], -1),
                    dim=-1,
                ).mean().item()
            )

            source_active_fraction = (
                pd.DataFrame(
                    {
                        "source_idx": rows.source_idx.cpu().tolist(),
                        "active": global_activation_mask.cpu().tolist(),
                    }
                )
                .groupby("source_idx", as_index=False)["active"]
                .mean()
            )
            top_source_row = source_active_fraction.sort_values(["active", "source_idx"], ascending=[False, True]).iloc[0]
            conditioned_row_count = int(ablation_row_mask.sum().item())
            ablated_row_count = int(activation_mask.sum().item())
            conditioned_active_fraction = 0.0
            if conditioned_row_count > 0:
                conditioned_active_fraction = float(ablated_row_count / conditioned_row_count)

            segment_rows.append(
                {
                    "dataset": args.dataset,
                    "path": row["path"],
                    "subject": row["subject"],
                    "label": int(row["label"]),
                    "feature_idx": int(feature_idx),
                    "hook": hook,
                    "baseline_input_mse": baseline_input_mse,
                    "ablated_input_mse": ablated_input_mse,
                    "delta_input_mse": ablated_input_mse - baseline_input_mse,
                    "cosine_to_baseline_reconstruction": cosine_to_baseline,
                    "mean_reconstruction_delta_energy": float(reconstruction_delta.pow(2).mean().item()),
                    "max_reconstruction_delta_energy": float(reconstruction_delta.pow(2).amax().item()),
                    "global_active_row_fraction": float(global_activation_mask.float().mean().item()),
                    "conditioned_active_row_fraction": conditioned_active_fraction,
                    "conditioned_row_fraction": float(ablation_row_mask.float().mean().item()),
                    "ablated_row_count": ablated_row_count,
                    "eligible_row_count": conditioned_row_count,
                    "row_source_idx": row_source_idx,
                    "top_active_source_idx": int(top_source_row["source_idx"]),
                    "top_active_source_fraction": float(top_source_row["active"]),
                    "top_delta_sensor_indices": ",".join(str(index) for index in top_sensors),
                }
            )

    segment_table = pd.DataFrame(segment_rows).sort_values(["feature_idx", "subject", "path"]).reset_index(drop=True)
    feature_summary = build_feature_summary(segment_table)
    sensor_summary = summarize_sensor_counts(segment_table)

    segment_table.to_csv(output_dir / "segment_feature_ablations.csv", index=False)
    feature_summary.to_csv(output_dir / "feature_summary.csv", index=False)
    sensor_summary.to_csv(output_dir / "sensor_summary.csv", index=False)
    save_json(
        output_dir / "summary.json",
        {
            "dataset": args.dataset,
            "hook": hook,
            "feature_indices": feature_indices,
            "num_segments": int(len(metadata)),
            "source_idx": args.source_idx,
            "row_source_idx": row_source_idx,
            "model_dir": str(model_dir),
        },
    )
    print(output_dir)


if __name__ == "__main__":
    main()
