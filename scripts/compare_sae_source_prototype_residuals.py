#!/usr/bin/env python
"""Compare source-prototype residual structure across SAE runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from neurofm_mechinterp.analysis.common import save_json
from neurofm_mechinterp.paths import RESULTS_ROOT, ensure_dir

try:
    from scripts.run_sae_source_prototype_interventions import (
        load_sae,
        load_source_prototype_bundle,
        resolve_activation_path,
    )
    from scripts.run_sae_source_prototype_patching import decompose_source_prototype_rows
except ModuleNotFoundError:
    from run_sae_source_prototype_interventions import (  # type: ignore[no-redef]
        load_sae,
        load_source_prototype_bundle,
        resolve_activation_path,
    )
    from run_sae_source_prototype_patching import decompose_source_prototype_rows  # type: ignore[no-redef]


def normalize_rows(rows: torch.Tensor) -> torch.Tensor:
    """Normalize row vectors for cosine similarity."""
    return rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-12)


def top_sensor_tuple(vector: torch.Tensor, top_k: int) -> tuple[int, ...]:
    """Return the top absolute-magnitude sensor indices for one vector."""
    values = vector.abs()
    indices = torch.topk(values, k=min(top_k, values.numel())).indices.tolist()
    return tuple(int(index) for index in indices)


def overlap_count(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    """Count overlapping elements between two tuples."""
    return len(set(left) & set(right))


def jaccard(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    """Compute Jaccard overlap between two tuples."""
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


@torch.no_grad()
def segment_residual_summary(
    model_dir: Path,
    source_idx: int,
    *,
    top_k: int,
    batch_size: int,
) -> tuple[pd.DataFrame, torch.Tensor]:
    """Summarize one residual mean vector per segment for a chosen source."""
    _, checkpoint, mean, std = load_sae(model_dir)
    activation_path = resolve_activation_path(checkpoint)
    bundle = torch.load(activation_path, map_location="cpu", weights_only=False)
    prototype_bundle = load_source_prototype_bundle(
        model_dir=model_dir,
        source_idx=source_idx,
        batch_size=batch_size,
    )
    prototype = prototype_bundle["prototype"].cpu()

    row_mask = bundle["source_idx"].long() == int(source_idx)
    selected = row_mask.nonzero(as_tuple=False).squeeze(-1)
    if selected.numel() == 0:
        raise ValueError(f"No activation-export rows found for source {source_idx}.")

    activations = bundle["activations"].index_select(0, selected).float()
    segment_idx = bundle["segment_idx"].index_select(0, selected).long()
    label = bundle["label"].index_select(0, selected).long()
    subjects = [bundle["subject"][row_idx] for row_idx in selected.tolist()]
    paths = [bundle["path"][row_idx] for row_idx in selected.tolist()]

    standardized_rows = (activations - mean) / std
    projection, _, residual = decompose_source_prototype_rows(standardized_rows, prototype)

    rows = []
    residual_vectors = []
    for current_segment_idx in segment_idx.unique(sorted=True).tolist():
        current_mask = segment_idx == int(current_segment_idx)
        current_projection = projection[current_mask]
        current_residual = residual[current_mask]
        current_mean_residual = current_residual.mean(dim=0)
        current_top_sensors = top_sensor_tuple(current_mean_residual, top_k=top_k)
        first_index = int(current_mask.nonzero(as_tuple=False)[0].item())

        rows.append(
            {
                "segment_idx": int(current_segment_idx),
                "path": paths[first_index],
                "subject": subjects[first_index],
                "label": int(label[current_mask][0].item()),
                "rows": int(current_mask.sum().item()),
                "mean_projection": float(current_projection.mean().item()),
                "mean_abs_projection": float(current_projection.abs().mean().item()),
                "mean_row_residual_norm": float(current_residual.norm(dim=1).mean().item()),
                "residual_mean_norm": float(current_mean_residual.norm().item()),
                "top_residual_sensors": ",".join(str(sensor_idx) for sensor_idx in current_top_sensors),
            }
        )
        residual_vectors.append(current_mean_residual)

    table = pd.DataFrame(rows).sort_values(["segment_idx", "path"]).reset_index(drop=True)
    return table, torch.stack(residual_vectors)


def pairwise_segment_similarity(name: str, table: pd.DataFrame, vectors: torch.Tensor) -> pd.DataFrame:
    """Build pairwise cosine summary within one run."""
    cosine = normalize_rows(vectors) @ normalize_rows(vectors).T
    rows = []
    for left_idx in range(len(table)):
        for right_idx in range(left_idx + 1, len(table)):
            left_row = table.iloc[left_idx]
            right_row = table.iloc[right_idx]
            rows.append(
                {
                    f"{name}_left_path": str(left_row["path"]),
                    f"{name}_right_path": str(right_row["path"]),
                    f"{name}_left_subject": str(left_row["subject"]),
                    f"{name}_right_subject": str(right_row["subject"]),
                    f"{name}_left_label": int(left_row["label"]),
                    f"{name}_right_label": int(right_row["label"]),
                    "same_subject": bool(left_row["subject"] == right_row["subject"]),
                    "same_label": bool(int(left_row["label"]) == int(right_row["label"])),
                    "cosine": float(cosine[left_idx, right_idx].item()),
                }
            )
    return pd.DataFrame(rows)


def within_run_nearest_neighbor(name: str, table: pd.DataFrame, vectors: torch.Tensor) -> pd.DataFrame:
    """Find the nearest segment neighbor inside one run."""
    cosine = normalize_rows(vectors) @ normalize_rows(vectors).T
    cosine.fill_diagonal_(-float("inf"))
    nearest = cosine.argmax(dim=1)

    rows = []
    for left_idx, right_idx in enumerate(nearest.tolist()):
        left_row = table.iloc[left_idx]
        right_row = table.iloc[right_idx]
        rows.append(
            {
                f"{name}_path": str(left_row["path"]),
                f"{name}_nearest_path": str(right_row["path"]),
                "same_subject": bool(left_row["subject"] == right_row["subject"]),
                "same_label": bool(int(left_row["label"]) == int(right_row["label"])),
                "cosine": float(cosine[left_idx, right_idx].item()),
            }
        )
    return pd.DataFrame(rows)


def cross_run_same_path_summary(
    left_name: str,
    right_name: str,
    left_table: pd.DataFrame,
    left_vectors: torch.Tensor,
    right_table: pd.DataFrame,
    right_vectors: torch.Tensor,
) -> pd.DataFrame:
    """Compare matched segments across runs by path."""
    left_norm = normalize_rows(left_vectors)
    right_norm = normalize_rows(right_vectors)
    similarity = left_norm @ right_norm.T

    right_lookup = {str(path): idx for idx, path in enumerate(right_table["path"].tolist())}
    rows = []
    for left_idx, left_row in left_table.iterrows():
        left_path = str(left_row["path"])
        if left_path not in right_lookup:
            continue
        right_idx = right_lookup[left_path]
        right_row = right_table.iloc[right_idx]

        left_sensors = tuple(int(value) for value in str(left_row["top_residual_sensors"]).split(",") if value)
        right_sensors = tuple(int(value) for value in str(right_row["top_residual_sensors"]).split(",") if value)
        rows.append(
            {
                f"{left_name}_path": left_path,
                f"{right_name}_path": str(right_row["path"]),
                "subject": str(left_row["subject"]),
                "label": int(left_row["label"]),
                "cosine": float(similarity[left_idx, right_idx].item()),
                "top_sensor_overlap_count": overlap_count(left_sensors, right_sensors),
                "top_sensor_jaccard": jaccard(left_sensors, right_sensors),
                f"{left_name}_top_residual_sensors": str(left_row["top_residual_sensors"]),
                f"{right_name}_top_residual_sensors": str(right_row["top_residual_sensors"]),
            }
        )
    return pd.DataFrame(rows).sort_values(f"{left_name}_path").reset_index(drop=True)


def cross_run_nearest_neighbor(
    left_name: str,
    right_name: str,
    left_table: pd.DataFrame,
    left_vectors: torch.Tensor,
    right_table: pd.DataFrame,
    right_vectors: torch.Tensor,
) -> pd.DataFrame:
    """Find the nearest segment in the right run for each left-run segment."""
    similarity = normalize_rows(left_vectors) @ normalize_rows(right_vectors).T
    nearest = similarity.argmax(dim=1)

    rows = []
    for left_idx, right_idx in enumerate(nearest.tolist()):
        left_row = left_table.iloc[left_idx]
        right_row = right_table.iloc[right_idx]
        rows.append(
            {
                f"{left_name}_path": str(left_row["path"]),
                f"{right_name}_path": str(right_row["path"]),
                f"{left_name}_subject": str(left_row["subject"]),
                f"{right_name}_subject": str(right_row["subject"]),
                f"{left_name}_label": int(left_row["label"]),
                f"{right_name}_label": int(right_row["label"]),
                "same_path": bool(left_row["path"] == right_row["path"]),
                "same_subject": bool(left_row["subject"] == right_row["subject"]),
                "same_label": bool(int(left_row["label"]) == int(right_row["label"])),
                "cosine": float(similarity[left_idx, right_idx].item()),
            }
        )
    return pd.DataFrame(rows).sort_values(f"{left_name}_path").reset_index(drop=True)


def mean_or_nan(table: pd.DataFrame, mask: pd.Series, column: str) -> float:
    """Return one masked mean or NaN when the mask is empty."""
    subset = table[mask]
    if subset.empty:
        return float("nan")
    return float(subset[column].mean())


def build_summary(
    left_pairwise: pd.DataFrame,
    right_pairwise: pd.DataFrame,
    left_nearest: pd.DataFrame,
    right_nearest: pd.DataFrame,
    same_path: pd.DataFrame,
    cross_nearest: pd.DataFrame,
) -> dict[str, float | int]:
    """Aggregate the main residual-structure comparison metrics."""
    return {
        "left_num_segments": int(left_nearest.shape[0]),
        "right_num_segments": int(right_nearest.shape[0]),
        "left_within_subject_mean_cosine": mean_or_nan(left_pairwise, left_pairwise["same_subject"], "cosine"),
        "left_cross_subject_mean_cosine": mean_or_nan(left_pairwise, ~left_pairwise["same_subject"], "cosine"),
        "right_within_subject_mean_cosine": mean_or_nan(right_pairwise, right_pairwise["same_subject"], "cosine"),
        "right_cross_subject_mean_cosine": mean_or_nan(right_pairwise, ~right_pairwise["same_subject"], "cosine"),
        "left_within_label_mean_cosine": mean_or_nan(left_pairwise, left_pairwise["same_label"], "cosine"),
        "left_cross_label_mean_cosine": mean_or_nan(left_pairwise, ~left_pairwise["same_label"], "cosine"),
        "right_within_label_mean_cosine": mean_or_nan(right_pairwise, right_pairwise["same_label"], "cosine"),
        "right_cross_label_mean_cosine": mean_or_nan(right_pairwise, ~right_pairwise["same_label"], "cosine"),
        "left_nearest_same_subject_fraction": float(left_nearest["same_subject"].mean()),
        "left_nearest_same_label_fraction": float(left_nearest["same_label"].mean()),
        "right_nearest_same_subject_fraction": float(right_nearest["same_subject"].mean()),
        "right_nearest_same_label_fraction": float(right_nearest["same_label"].mean()),
        "same_path_mean_cosine": float(same_path["cosine"].mean()),
        "same_path_min_cosine": float(same_path["cosine"].min()),
        "same_path_top_sensor_overlap_mean": float(same_path["top_sensor_overlap_count"].mean()),
        "same_path_top_sensor_jaccard_mean": float(same_path["top_sensor_jaccard"].mean()),
        "cross_seed_nearest_same_path_fraction": float(cross_nearest["same_path"].mean()),
        "cross_seed_nearest_same_subject_fraction": float(cross_nearest["same_subject"].mean()),
        "cross_seed_nearest_same_label_fraction": float(cross_nearest["same_label"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-model-dir", required=True)
    parser.add_argument("--right-model-dir", required=True)
    parser.add_argument("--source-idx", type=int, required=True)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    left_model_dir = Path(args.left_model_dir)
    right_model_dir = Path(args.right_model_dir)
    if args.output_dir is None:
        output_dir = ensure_dir(
            RESULTS_ROOT
            / "source_prototype_residuals"
            / f"{left_model_dir.name}__vs__{right_model_dir.name}__source_{args.source_idx}"
        )
    else:
        output_dir = ensure_dir(Path(args.output_dir))

    left_table, left_vectors = segment_residual_summary(
        left_model_dir,
        source_idx=args.source_idx,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )
    right_table, right_vectors = segment_residual_summary(
        right_model_dir,
        source_idx=args.source_idx,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )

    left_pairwise = pairwise_segment_similarity(args.left_name, left_table, left_vectors)
    right_pairwise = pairwise_segment_similarity(args.right_name, right_table, right_vectors)
    left_nearest = within_run_nearest_neighbor(args.left_name, left_table, left_vectors)
    right_nearest = within_run_nearest_neighbor(args.right_name, right_table, right_vectors)
    same_path = cross_run_same_path_summary(
        args.left_name,
        args.right_name,
        left_table,
        left_vectors,
        right_table,
        right_vectors,
    )
    cross_nearest = cross_run_nearest_neighbor(
        args.left_name,
        args.right_name,
        left_table,
        left_vectors,
        right_table,
        right_vectors,
    )
    summary = build_summary(
        left_pairwise=left_pairwise,
        right_pairwise=right_pairwise,
        left_nearest=left_nearest,
        right_nearest=right_nearest,
        same_path=same_path,
        cross_nearest=cross_nearest,
    )
    summary.update(
        {
            "left_model_dir": str(left_model_dir),
            "right_model_dir": str(right_model_dir),
            "source_idx": int(args.source_idx),
            "top_k": int(args.top_k),
            "left_name": args.left_name,
            "right_name": args.right_name,
        }
    )

    left_table.to_csv(output_dir / "left_segment_summary.csv", index=False)
    right_table.to_csv(output_dir / "right_segment_summary.csv", index=False)
    left_pairwise.to_csv(output_dir / "left_pairwise_similarity.csv", index=False)
    right_pairwise.to_csv(output_dir / "right_pairwise_similarity.csv", index=False)
    left_nearest.to_csv(output_dir / "left_nearest_neighbor.csv", index=False)
    right_nearest.to_csv(output_dir / "right_nearest_neighbor.csv", index=False)
    same_path.to_csv(output_dir / "same_path_comparison.csv", index=False)
    cross_nearest.to_csv(output_dir / "cross_seed_nearest_neighbor.csv", index=False)
    save_json(output_dir / "summary.json", summary)
    print(output_dir)


if __name__ == "__main__":
    main()
