#!/usr/bin/env python
"""Compare source-level SAE structure across runs without relying on feature identity."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from neurofm_mechinterp.analysis.common import save_json
from neurofm_mechinterp.paths import RESULTS_ROOT, ensure_dir
from neurofm_mechinterp.sae import SparseAutoencoder


def load_model_and_features(model_dir: Path) -> tuple[SparseAutoencoder, dict]:
    """Load one trained SAE checkpoint."""
    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=False)
    model = SparseAutoencoder(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        sparsifier=checkpoint["config"].get("sparsifier", "relu"),
        topk=checkpoint["config"].get("topk"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def resolve_activation_path(checkpoint: dict) -> Path:
    """Resolve the activation export path for a checkpoint, including older runs."""
    activation_path_value = checkpoint["config"].get("activation_path")
    if activation_path_value:
        return Path(activation_path_value)
    return RESULTS_ROOT / "sae_exports" / checkpoint["config"]["dataset"] / checkpoint["config"]["hook"] / "activations.pt"


def source_feature_statistics(model_dir: Path, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-source feature mean activations and active fractions."""
    model, checkpoint = load_model_and_features(model_dir)
    activation_path = resolve_activation_path(checkpoint)
    bundle = torch.load(activation_path, map_location="cpu", weights_only=False)

    x = bundle["activations"].float()
    source_idx = bundle["source_idx"].long()
    mean = checkpoint["input_mean"].view(1, -1).float()
    std = checkpoint["input_std"].view(1, -1).float()
    x = (x - mean) / std

    hidden_dim = checkpoint["hidden_dim"]
    n_source = int(source_idx.max().item()) + 1
    source_feature_sum = torch.zeros(n_source, hidden_dim)
    source_feature_active = torch.zeros(n_source, hidden_dim)
    source_counts = torch.zeros(n_source)

    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        batch_x = x[start:end]
        batch_source = source_idx[start:end]
        features = model.encode(batch_x)
        for source in batch_source.unique(sorted=True).tolist():
            mask = batch_source == source
            source_counts[source] += float(mask.sum().item())
            masked_features = features[mask]
            source_feature_sum[source] += masked_features.sum(dim=0)
            source_feature_active[source] += (masked_features > 0).float().sum(dim=0)

    source_mean = source_feature_sum / source_counts.clamp_min(1.0).unsqueeze(-1)
    source_active_fraction = source_feature_active / source_counts.clamp_min(1.0).unsqueeze(-1)
    decoder_directions = checkpoint["model_state_dict"]["decoder.weight"].float().T
    decoder_directions = decoder_directions / decoder_directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return source_mean, source_active_fraction, decoder_directions


def source_prototypes(source_mean: torch.Tensor, decoder_directions: torch.Tensor) -> torch.Tensor:
    """Aggregate decoder directions into one normalized prototype per source."""
    prototypes = source_mean @ decoder_directions
    return prototypes / prototypes.norm(dim=1, keepdim=True).clamp_min(1e-12)


def top_source_feature_subspace(
    source_mean: torch.Tensor,
    decoder_directions: torch.Tensor,
    source_idx: int,
    top_features: int,
) -> torch.Tensor:
    """Return an orthonormal basis for the top source-feature decoder span."""
    feature_rank = torch.argsort(source_mean[source_idx], descending=True)
    selected = feature_rank[:top_features]
    directions = decoder_directions[selected].T
    basis = torch.linalg.qr(directions, mode="reduced")[0]
    return basis


def mean_canonical_correlation(left_basis: torch.Tensor, right_basis: torch.Tensor) -> tuple[float, float]:
    """Return mean and max canonical correlation between two bases."""
    singular_values = torch.linalg.svdvals(left_basis.T @ right_basis)
    return float(singular_values.mean().item()), float(singular_values.max().item())


def compare_sources(
    left_name: str,
    right_name: str,
    left_source_mean: torch.Tensor,
    left_active_fraction: torch.Tensor,
    left_decoder: torch.Tensor,
    right_source_mean: torch.Tensor,
    right_active_fraction: torch.Tensor,
    right_decoder: torch.Tensor,
    top_features: int,
) -> tuple[pd.DataFrame, torch.Tensor]:
    """Compare source prototypes and top-feature spans across all source pairs."""
    left_proto = source_prototypes(left_source_mean, left_decoder)
    right_proto = source_prototypes(right_source_mean, right_decoder)
    prototype_cosine = left_proto @ right_proto.T

    rows = []
    for left_source_idx in range(left_source_mean.shape[0]):
        left_basis = top_source_feature_subspace(left_source_mean, left_decoder, left_source_idx, top_features=top_features)
        left_top_feature_mass = float(left_source_mean[left_source_idx].topk(k=min(top_features, left_source_mean.shape[1])).values.sum().item())
        for right_source_idx in range(right_source_mean.shape[0]):
            right_basis = top_source_feature_subspace(right_source_mean, right_decoder, right_source_idx, top_features=top_features)
            mean_cc, max_cc = mean_canonical_correlation(left_basis, right_basis)
            rows.append(
                {
                    f"{left_name}_source_idx": int(left_source_idx),
                    f"{right_name}_source_idx": int(right_source_idx),
                    "same_source_idx": bool(left_source_idx == right_source_idx),
                    "prototype_cosine": float(prototype_cosine[left_source_idx, right_source_idx].item()),
                    "mean_canonical_correlation": mean_cc,
                    "max_canonical_correlation": max_cc,
                    f"{left_name}_top_feature_mass": left_top_feature_mass,
                    f"{right_name}_top_feature_mass": float(right_source_mean[right_source_idx].topk(k=min(top_features, right_source_mean.shape[1])).values.sum().item()),
                    f"{left_name}_mean_active_fraction": float(left_active_fraction[left_source_idx].mean().item()),
                    f"{right_name}_mean_active_fraction": float(right_active_fraction[right_source_idx].mean().item()),
                }
            )
    table = pd.DataFrame(rows).sort_values(
        ["prototype_cosine", "mean_canonical_correlation", f"{left_name}_source_idx", f"{right_name}_source_idx"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    return table, prototype_cosine


def same_source_summary(
    left_name: str,
    right_name: str,
    comparison_table: pd.DataFrame,
) -> pd.DataFrame:
    """Keep one row per source for the matched source index pair."""
    mask = comparison_table[f"{left_name}_source_idx"] == comparison_table[f"{right_name}_source_idx"]
    return comparison_table[mask].copy().sort_values(f"{left_name}_source_idx").reset_index(drop=True)


def best_match_summary(left_name: str, right_name: str, comparison_table: pd.DataFrame) -> pd.DataFrame:
    """Keep the best right-side source for each left-side source."""
    rows = []
    for left_source_idx, subset in comparison_table.groupby(f"{left_name}_source_idx"):
        top = subset.sort_values(
            ["prototype_cosine", "mean_canonical_correlation", f"{right_name}_source_idx"],
            ascending=[False, False, True],
        ).iloc[0]
        rows.append(top.to_dict())
    return pd.DataFrame(rows).sort_values(f"{left_name}_source_idx").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-model-dir", required=True)
    parser.add_argument("--right-model-dir", required=True)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--top-features", type=int, default=16)
    args = parser.parse_args()

    left_model_dir = Path(args.left_model_dir)
    right_model_dir = Path(args.right_model_dir)
    if args.output_dir is None:
        output_dir = ensure_dir(RESULTS_ROOT / "sae_source_structure" / f"{left_model_dir.name}__vs__{right_model_dir.name}")
    else:
        output_dir = ensure_dir(Path(args.output_dir))

    left_source_mean, left_active_fraction, left_decoder = source_feature_statistics(left_model_dir, batch_size=args.batch_size)
    right_source_mean, right_active_fraction, right_decoder = source_feature_statistics(right_model_dir, batch_size=args.batch_size)

    comparison_table, prototype_cosine = compare_sources(
        left_name=args.left_name,
        right_name=args.right_name,
        left_source_mean=left_source_mean,
        left_active_fraction=left_active_fraction,
        left_decoder=left_decoder,
        right_source_mean=right_source_mean,
        right_active_fraction=right_active_fraction,
        right_decoder=right_decoder,
        top_features=args.top_features,
    )
    same_source = same_source_summary(args.left_name, args.right_name, comparison_table)
    best_match = best_match_summary(args.left_name, args.right_name, comparison_table)

    summary = {
        "left_model_dir": str(left_model_dir),
        "right_model_dir": str(right_model_dir),
        "left_name": args.left_name,
        "right_name": args.right_name,
        "top_features": args.top_features,
        "mean_same_source_prototype_cosine": float(same_source["prototype_cosine"].mean()),
        "min_same_source_prototype_cosine": float(same_source["prototype_cosine"].min()),
        "mean_same_source_mean_canonical_correlation": float(same_source["mean_canonical_correlation"].mean()),
        "min_same_source_mean_canonical_correlation": float(same_source["mean_canonical_correlation"].min()),
        "same_source_best_match_count": int(best_match["same_source_idx"].sum()),
        "prototype_trace_mean": float(prototype_cosine.diag().mean().item()),
    }

    comparison_table.to_csv(output_dir / "source_pair_comparison.csv", index=False)
    same_source.to_csv(output_dir / "same_source_summary.csv", index=False)
    best_match.to_csv(output_dir / "best_match_summary.csv", index=False)
    save_json(output_dir / "summary.json", summary)
    print(output_dir)


if __name__ == "__main__":
    main()
