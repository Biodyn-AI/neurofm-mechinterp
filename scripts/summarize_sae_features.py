#!/usr/bin/env python
"""Summarize trained SAE features against tokenizer source metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from neurofm_mechinterp.paths import RESULTS_ROOT
from neurofm_mechinterp.sae import SparseAutoencoder


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Directory containing `model.pt`.")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=False)
    activation_path_value = checkpoint["config"].get("activation_path")
    if activation_path_value is None:
        activation_path = RESULTS_ROOT / "sae_exports" / checkpoint["config"]["dataset"] / checkpoint["config"]["hook"] / "activations.pt"
    else:
        activation_path = Path(activation_path_value)
    bundle = torch.load(activation_path, map_location="cpu", weights_only=False)

    model = SparseAutoencoder(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        sparsifier=checkpoint["config"].get("sparsifier", "relu"),
        topk=checkpoint["config"].get("topk"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x = bundle["activations"].float()
    source_idx = bundle["source_idx"].long()
    mean = checkpoint["input_mean"].view(1, -1).float()
    std = checkpoint["input_std"].view(1, -1).float()
    x = (x - mean) / std

    n_source = int(source_idx.max().item()) + 1
    hidden_dim = checkpoint["hidden_dim"]
    total_rows = x.shape[0]

    feature_sum = torch.zeros(hidden_dim)
    feature_active = torch.zeros(hidden_dim)
    source_feature_sum = torch.zeros(n_source, hidden_dim)
    source_feature_active = torch.zeros(n_source, hidden_dim)
    source_counts = torch.zeros(n_source)

    for start in range(0, total_rows, args.batch_size):
        end = min(start + args.batch_size, total_rows)
        batch_x = x[start:end]
        batch_source = source_idx[start:end]
        features = model.encode(batch_x)

        feature_sum += features.sum(dim=0)
        feature_active += (features > 0).float().sum(dim=0)

        for source in batch_source.unique(sorted=True).tolist():
            mask = batch_source == source
            source_counts[source] += float(mask.sum().item())
            masked_features = features[mask]
            source_feature_sum[source] += masked_features.sum(dim=0)
            source_feature_active[source] += (masked_features > 0).float().sum(dim=0)

    feature_mean = feature_sum / total_rows
    feature_active_fraction = feature_active / total_rows
    source_mean = source_feature_sum / source_counts.clamp_min(1.0).unsqueeze(-1)
    source_active_fraction = source_feature_active / source_counts.clamp_min(1.0).unsqueeze(-1)

    source_distribution = source_mean.clamp_min(0)
    source_distribution = source_distribution / source_distribution.sum(dim=0, keepdim=True).clamp_min(1e-12)
    source_entropy = -(source_distribution * source_distribution.clamp_min(1e-12).log()).sum(dim=0)

    top_source_idx = source_mean.argmax(dim=0)
    top_source_mean = source_mean.gather(0, top_source_idx.unsqueeze(0)).squeeze(0)
    top_source_active_fraction = source_active_fraction.gather(0, top_source_idx.unsqueeze(0)).squeeze(0)

    feature_table = pd.DataFrame(
        {
            "feature_idx": list(range(hidden_dim)),
            "mean_activation": feature_mean.tolist(),
            "active_fraction": feature_active_fraction.tolist(),
            "top_source_idx": top_source_idx.tolist(),
            "top_source_mean_activation": top_source_mean.tolist(),
            "top_source_active_fraction": top_source_active_fraction.tolist(),
            "source_entropy": source_entropy.tolist(),
            "dead_feature": (feature_active_fraction == 0).tolist(),
        }
    ).sort_values(
        ["mean_activation", "active_fraction"],
        ascending=[False, False],
    ).reset_index(drop=True)

    source_rows = []
    for source in range(n_source):
        ranked = torch.argsort(source_mean[source], descending=True)
        for rank, feature_idx in enumerate(ranked[:16].tolist(), start=1):
            source_rows.append(
                {
                    "source_idx": source,
                    "rank": rank,
                    "feature_idx": feature_idx,
                    "mean_activation": float(source_mean[source, feature_idx].item()),
                    "active_fraction": float(source_active_fraction[source, feature_idx].item()),
                }
            )
    source_table = pd.DataFrame(source_rows).sort_values(["source_idx", "rank"]).reset_index(drop=True)

    feature_table.to_csv(model_dir / "feature_source_summary.csv", index=False)
    source_table.to_csv(model_dir / "source_top_features.csv", index=False)
    print(model_dir)


if __name__ == "__main__":
    main()
