#!/usr/bin/env python
"""Compare trained SAE feature directions across two datasets or runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from neurofm_mechinterp.analysis.common import save_json
from neurofm_mechinterp.paths import RESULTS_ROOT, ensure_dir


def load_feature_directions(model_dir: Path) -> tuple[torch.Tensor, pd.DataFrame]:
    """Load normalized decoder directions plus feature metadata."""
    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=False)
    decoder_weight = checkpoint["model_state_dict"]["decoder.weight"].float().T
    decoder_weight = decoder_weight / decoder_weight.norm(dim=1, keepdim=True).clamp_min(1e-12)
    feature_table = pd.read_csv(model_dir / "feature_source_summary.csv").sort_values("feature_idx").reset_index(drop=True)
    return decoder_weight, feature_table


def build_match_table(
    left_name: str,
    right_name: str,
    left_directions: torch.Tensor,
    right_directions: torch.Tensor,
    left_features: pd.DataFrame,
    right_features: pd.DataFrame,
) -> pd.DataFrame:
    """Match each left feature to its nearest right feature and mark mutual matches."""
    cosine = left_directions @ right_directions.T
    left_best_score, left_best_idx = cosine.max(dim=1)
    right_best_score, right_best_idx = cosine.max(dim=0)

    rows = []
    for left_feature_idx, right_feature_idx in enumerate(left_best_idx.tolist()):
        left_row = left_features.iloc[left_feature_idx]
        right_row = right_features.iloc[right_feature_idx]
        mutual_best = int(right_best_idx[right_feature_idx].item()) == int(left_feature_idx)
        left_dead_feature = bool(left_row.get("dead_feature", False))
        right_dead_feature = bool(right_row.get("dead_feature", False))
        rows.append(
            {
                f"{left_name}_feature_idx": int(left_feature_idx),
                f"{right_name}_feature_idx": int(right_feature_idx),
                "decoder_cosine": float(left_best_score[left_feature_idx].item()),
                "mutual_best": bool(mutual_best),
                f"{left_name}_top_source_idx": int(left_row["top_source_idx"]),
                f"{right_name}_top_source_idx": int(right_row["top_source_idx"]),
                "same_top_source": bool(int(left_row["top_source_idx"]) == int(right_row["top_source_idx"])),
                f"{left_name}_dead_feature": left_dead_feature,
                f"{right_name}_dead_feature": right_dead_feature,
                "active_in_both": bool((not left_dead_feature) and (not right_dead_feature)),
                f"{left_name}_mean_activation": float(left_row["mean_activation"]),
                f"{right_name}_mean_activation": float(right_row["mean_activation"]),
                f"{left_name}_active_fraction": float(left_row["active_fraction"]),
                f"{right_name}_active_fraction": float(right_row["active_fraction"]),
                f"{left_name}_source_entropy": float(left_row["source_entropy"]),
                f"{right_name}_source_entropy": float(right_row["source_entropy"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["decoder_cosine", f"{left_name}_feature_idx"], ascending=[False, True]).reset_index(drop=True)


def threshold_key(threshold: float) -> str:
    """Render a stable threshold suffix for JSON keys."""
    return str(threshold).replace(".", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-model-dir", required=True)
    parser.add_argument("--right-model-dir", required=True)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--thresholds", type=float, nargs="*", default=[0.6, 0.7, 0.8, 0.9])
    args = parser.parse_args()

    left_model_dir = Path(args.left_model_dir)
    right_model_dir = Path(args.right_model_dir)
    if args.output_dir is None:
        output_dir = ensure_dir(RESULTS_ROOT / "sae_cross_dataset" / f"{left_model_dir.name}__vs__{right_model_dir.name}")
    else:
        output_dir = ensure_dir(Path(args.output_dir))

    left_directions, left_features = load_feature_directions(left_model_dir)
    right_directions, right_features = load_feature_directions(right_model_dir)
    match_table = build_match_table(
        left_name=args.left_name,
        right_name=args.right_name,
        left_directions=left_directions,
        right_directions=right_directions,
        left_features=left_features,
        right_features=right_features,
    )
    mutual_table = match_table[match_table["mutual_best"]].copy().sort_values("decoder_cosine", ascending=False).reset_index(drop=True)
    active_mutual_table = mutual_table[mutual_table["active_in_both"]].copy().reset_index(drop=True)

    source_summary = (
        mutual_table.groupby(f"{args.left_name}_top_source_idx", as_index=False)
        .agg(
            mutual_matches=("mutual_best", "count"),
            same_top_source_matches=("same_top_source", "sum"),
            mean_decoder_cosine=("decoder_cosine", "mean"),
        )
        .rename(columns={f"{args.left_name}_top_source_idx": "top_source_idx"})
        .sort_values(["same_top_source_matches", "mean_decoder_cosine", "top_source_idx"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    active_source_summary = (
        active_mutual_table.groupby(f"{args.left_name}_top_source_idx", as_index=False)
        .agg(
            mutual_matches=("mutual_best", "count"),
            same_top_source_matches=("same_top_source", "sum"),
            mean_decoder_cosine=("decoder_cosine", "mean"),
        )
        .rename(columns={f"{args.left_name}_top_source_idx": "top_source_idx"})
        .sort_values(["same_top_source_matches", "mean_decoder_cosine", "top_source_idx"], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    summary = {
        "left_model_dir": str(left_model_dir),
        "right_model_dir": str(right_model_dir),
        "left_name": args.left_name,
        "right_name": args.right_name,
        "left_num_features": int(len(left_features)),
        "right_num_features": int(len(right_features)),
        "mutual_match_count": int(len(mutual_table)),
        "same_top_source_mutual_match_count": int(mutual_table["same_top_source"].sum()),
        "mean_mutual_decoder_cosine": float(mutual_table["decoder_cosine"].mean()) if len(mutual_table) else 0.0,
        "mean_same_source_mutual_decoder_cosine": float(mutual_table[mutual_table["same_top_source"]]["decoder_cosine"].mean()) if len(mutual_table[mutual_table["same_top_source"]]) else 0.0,
        "active_mutual_match_count": int(len(active_mutual_table)),
        "active_same_top_source_mutual_match_count": int(active_mutual_table["same_top_source"].sum()),
        "mean_active_mutual_decoder_cosine": float(active_mutual_table["decoder_cosine"].mean()) if len(active_mutual_table) else 0.0,
        "mean_active_same_source_mutual_decoder_cosine": float(active_mutual_table[active_mutual_table["same_top_source"]]["decoder_cosine"].mean()) if len(active_mutual_table[active_mutual_table["same_top_source"]]) else 0.0,
    }
    for threshold in args.thresholds:
        threshold_table = mutual_table[mutual_table["decoder_cosine"] >= threshold]
        active_threshold_table = active_mutual_table[active_mutual_table["decoder_cosine"] >= threshold]
        key = threshold_key(threshold)
        summary[f"mutual_matches_ge_{key}"] = int(len(threshold_table))
        summary[f"same_top_source_mutual_matches_ge_{key}"] = int(threshold_table["same_top_source"].sum())
        summary[f"active_mutual_matches_ge_{key}"] = int(len(active_threshold_table))
        summary[f"active_same_top_source_mutual_matches_ge_{key}"] = int(active_threshold_table["same_top_source"].sum())

    match_table.to_csv(output_dir / "left_to_right_best_matches.csv", index=False)
    mutual_table.to_csv(output_dir / "mutual_feature_matches.csv", index=False)
    active_mutual_table.to_csv(output_dir / "active_mutual_feature_matches.csv", index=False)
    source_summary.to_csv(output_dir / "source_match_summary.csv", index=False)
    active_source_summary.to_csv(output_dir / "active_source_match_summary.csv", index=False)
    save_json(output_dir / "summary.json", summary)
    print(output_dir)


if __name__ == "__main__":
    main()
