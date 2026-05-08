#!/usr/bin/env python
"""Summarize head-level stability for source-prototype routing interventions."""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

from neurofm_mechinterp.analysis.common import save_json
from neurofm_mechinterp.paths import RESULTS_ROOT, ensure_dir


def compute_head_profile(head_summary: pd.DataFrame) -> dict[str, float | int]:
    """Summarize one head-routing run from its per-head averages."""
    ordered = head_summary.sort_values("head_idx").reset_index(drop=True)
    head_idx = ordered["head_idx"].astype(int).to_numpy()
    mean_abs = ordered["mean_abs_sensor_delta"].astype(float).to_numpy()
    total = float(mean_abs.sum())
    normalized = mean_abs / total if total > 0 else np.zeros_like(mean_abs)

    rank = np.argsort(-mean_abs, kind="stable")
    dominant = int(head_idx[rank[0]])
    second = int(head_idx[rank[1]]) if len(rank) > 1 else dominant
    margin_abs = float(mean_abs[rank[0]] - mean_abs[rank[1]]) if len(rank) > 1 else float(mean_abs[rank[0]])
    margin_fraction = float(margin_abs / total) if total > 0 else 0.0

    entropy = 0.0
    positive = normalized[normalized > 0]
    if positive.size:
        entropy = float(-(positive * np.log(positive)).sum() / math.log(len(normalized)))

    row: dict[str, float | int] = {
        "num_heads": int(len(head_idx)),
        "total_mean_abs_sensor_delta": total,
        "dominant_head_idx": dominant,
        "second_head_idx": second,
        "dominant_head_margin_abs": margin_abs,
        "dominant_head_margin_fraction": margin_fraction,
        "normalized_head_entropy": entropy,
    }
    for idx, value in zip(head_idx.tolist(), mean_abs.tolist()):
        row[f"head_{idx}_mean_abs_sensor_delta"] = float(value)
    for idx, value in zip(head_idx.tolist(), normalized.tolist()):
        row[f"head_{idx}_normalized_mass"] = float(value)
    return row


def segment_head_matrix(segment_table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Convert one segment-level routing table into `segments x heads` matrix form."""
    pivot = (
        segment_table.pivot_table(index="path", columns="head_idx", values="mean_abs_sensor_delta", aggfunc="mean")
        .sort_index(axis=1)
        .sort_index()
    )
    return pivot.to_numpy(dtype=float), pivot.columns.to_numpy(dtype=int)


def bootstrap_dominant_head_frequencies(
    head_matrix: np.ndarray,
    head_idx: np.ndarray,
    num_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap dominant-head attribution over segments."""
    if head_matrix.ndim != 2 or head_matrix.shape[0] == 0 or head_matrix.shape[1] == 0:
        raise ValueError("Head matrix must be non-empty and two-dimensional.")

    rng = np.random.default_rng(seed)
    win_counts = np.zeros(head_matrix.shape[1], dtype=int)
    for _ in range(num_bootstrap):
        sample = rng.integers(0, head_matrix.shape[0], size=head_matrix.shape[0])
        aggregated = head_matrix[sample].mean(axis=0)
        dominant = int(np.argmax(aggregated))
        win_counts[dominant] += 1

    rows = []
    for column_idx, current_head_idx in enumerate(head_idx.tolist()):
        rows.append(
            {
                "head_idx": int(current_head_idx),
                "bootstrap_win_count": int(win_counts[column_idx]),
                "bootstrap_win_fraction": float(win_counts[column_idx] / num_bootstrap),
            }
        )
    return pd.DataFrame(rows).sort_values("head_idx").reset_index(drop=True)


def bootstrap_distribution(bootstrap_table: pd.DataFrame) -> np.ndarray:
    """Extract a stable probability vector from a bootstrap summary table."""
    ordered = bootstrap_table.sort_values("head_idx").reset_index(drop=True)
    return ordered["bootstrap_win_fraction"].astype(float).to_numpy()


def jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence for two discrete distributions."""
    midpoint = 0.5 * (left + right)

    def kl_divergence(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        if not np.any(mask):
            return 0.0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return float(0.5 * kl_divergence(left, midpoint) + 0.5 * kl_divergence(right, midpoint))


def summarize_run(run_dir: Path, run_name: str, num_bootstrap: int, seed: int) -> tuple[dict[str, object], pd.DataFrame]:
    """Summarize one source-prototype head-routing run."""
    head_summary = pd.read_csv(run_dir / "head_summary.csv")
    segment_table = pd.read_csv(run_dir / "segment_head_routing_interventions.csv")

    profile = compute_head_profile(head_summary)
    head_matrix, head_idx = segment_head_matrix(segment_table)
    bootstrap_table = bootstrap_dominant_head_frequencies(
        head_matrix=head_matrix,
        head_idx=head_idx,
        num_bootstrap=num_bootstrap,
        seed=seed,
    )
    dominant_bootstrap = bootstrap_table.sort_values(["bootstrap_win_fraction", "head_idx"], ascending=[False, True]).iloc[0]

    profile_row = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "num_segments": int(head_matrix.shape[0]),
        "bootstrap_num_samples": int(num_bootstrap),
        "bootstrap_seed": int(seed),
        "bootstrap_dominant_head_idx": int(dominant_bootstrap["head_idx"]),
        "bootstrap_dominant_head_fraction": float(dominant_bootstrap["bootstrap_win_fraction"]),
        **profile,
    }
    bootstrap_table = bootstrap_table.copy()
    bootstrap_table.insert(0, "run_name", run_name)
    return profile_row, bootstrap_table


def pairwise_comparisons(run_summary: pd.DataFrame, bootstrap_summary: pd.DataFrame) -> pd.DataFrame:
    """Compare head profiles across runs."""
    rows = []
    by_name = {row["run_name"]: row for _, row in run_summary.iterrows()}
    bootstrap_by_name = {
        run_name: bootstrap_distribution(subset)
        for run_name, subset in bootstrap_summary.groupby("run_name")
    }
    for left_name, right_name in itertools.combinations(run_summary["run_name"].tolist(), 2):
        left = by_name[left_name]
        right = by_name[right_name]
        left_mass = np.array([left[f"head_{idx}_normalized_mass"] for idx in range(int(left["num_heads"]))], dtype=float)
        right_mass = np.array([right[f"head_{idx}_normalized_mass"] for idx in range(int(right["num_heads"]))], dtype=float)
        left_bootstrap = bootstrap_by_name[left_name]
        right_bootstrap = bootstrap_by_name[right_name]

        rows.append(
            {
                "left_run_name": left_name,
                "right_run_name": right_name,
                "same_dominant_head": bool(int(left["dominant_head_idx"]) == int(right["dominant_head_idx"])),
                "same_bootstrap_dominant_head": bool(
                    int(left["bootstrap_dominant_head_idx"]) == int(right["bootstrap_dominant_head_idx"])
                ),
                "normalized_mass_l1_distance": float(np.abs(left_mass - right_mass).sum()),
                "normalized_mass_js_divergence": jensen_shannon(left_mass, right_mass),
                "bootstrap_distribution_l1_distance": float(np.abs(left_bootstrap - right_bootstrap).sum()),
                "bootstrap_distribution_js_divergence": jensen_shannon(left_bootstrap, right_bootstrap),
                "total_mean_abs_sensor_delta_ratio": float(
                    min(float(left["total_mean_abs_sensor_delta"]), float(right["total_mean_abs_sensor_delta"]))
                    / max(float(left["total_mean_abs_sensor_delta"]), float(right["total_mean_abs_sensor_delta"]))
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", nargs="+", required=True)
    parser.add_argument("--run-name", nargs="+", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dirs = [Path(value) for value in args.run_dir]
    run_names = list(args.run_name)
    if len(run_dirs) != len(run_names):
        raise ValueError("Provide the same number of `--run-dir` and `--run-name` values.")

    if args.output_dir is None:
        output_dir = ensure_dir(RESULTS_ROOT / "source_prototype_head_stability" / "manual_compare")
    else:
        output_dir = ensure_dir(Path(args.output_dir))

    run_rows = []
    bootstrap_rows = []
    for run_dir, run_name in zip(run_dirs, run_names):
        run_row, bootstrap_table = summarize_run(
            run_dir=run_dir,
            run_name=run_name,
            num_bootstrap=args.num_bootstrap,
            seed=args.seed,
        )
        run_rows.append(run_row)
        bootstrap_rows.append(bootstrap_table)

    run_summary = pd.DataFrame(run_rows).sort_values("run_name").reset_index(drop=True)
    bootstrap_summary = pd.concat(bootstrap_rows, ignore_index=True).sort_values(
        ["run_name", "head_idx"]
    ).reset_index(drop=True)
    pairwise = pairwise_comparisons(run_summary, bootstrap_summary)

    run_summary.to_csv(output_dir / "run_summary.csv", index=False)
    bootstrap_summary.to_csv(output_dir / "bootstrap_head_summary.csv", index=False)
    pairwise.to_csv(output_dir / "pairwise_comparison.csv", index=False)
    save_json(
        output_dir / "summary.json",
        {
            "run_names": run_names,
            "num_runs": int(len(run_names)),
            "num_bootstrap": int(args.num_bootstrap),
            "seed": int(args.seed),
        },
    )
    print(output_dir)


if __name__ == "__main__":
    main()
