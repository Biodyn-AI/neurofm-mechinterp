#!/usr/bin/env python
"""Post-hoc summaries for SAE ablation and patching result directories."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def relation_label(left_value: str, right_value: str, within_label: str, cross_label: str) -> str:
    """Return a stable within-vs-cross label for two identifiers."""
    if str(left_value) == str(right_value):
        return within_label
    return cross_label


def summarize_ablation(run_dir: Path) -> None:
    """Build subject and label summaries for one ablation run."""
    table = pd.read_csv(run_dir / "segment_feature_ablations.csv")

    subject_summary = (
        table.groupby(["feature_idx", "subject"], as_index=False)
        .agg(
            segments_profiled=("path", "count"),
            mean_delta_input_mse=("delta_input_mse", "mean"),
            mean_cosine_to_baseline_reconstruction=("cosine_to_baseline_reconstruction", "mean"),
            mean_reconstruction_delta_energy=("mean_reconstruction_delta_energy", "mean"),
            mean_global_active_row_fraction=("global_active_row_fraction", "mean"),
            mean_conditioned_active_row_fraction=("conditioned_active_row_fraction", "mean"),
        )
        .sort_values(["feature_idx", "mean_delta_input_mse", "subject"], ascending=[True, False, True])
        .reset_index(drop=True)
    )

    label_summary = (
        table.groupby(["feature_idx", "label"], as_index=False)
        .agg(
            segments_profiled=("path", "count"),
            mean_delta_input_mse=("delta_input_mse", "mean"),
            mean_cosine_to_baseline_reconstruction=("cosine_to_baseline_reconstruction", "mean"),
            mean_reconstruction_delta_energy=("mean_reconstruction_delta_energy", "mean"),
            mean_global_active_row_fraction=("global_active_row_fraction", "mean"),
            mean_conditioned_active_row_fraction=("conditioned_active_row_fraction", "mean"),
        )
        .sort_values(["feature_idx", "mean_delta_input_mse", "label"], ascending=[True, False, True])
        .reset_index(drop=True)
    )

    subject_summary.to_csv(run_dir / "subject_feature_summary.csv", index=False)
    label_summary.to_csv(run_dir / "label_feature_summary.csv", index=False)


def summarize_patch(run_dir: Path) -> None:
    """Build donor/recipient relation summaries for one patching run."""
    table = pd.read_csv(run_dir / "pair_patch_results.csv").copy()
    table["pair_subject_relation"] = [
        relation_label(donor, recipient, "within_subject", "cross_subject")
        for donor, recipient in zip(table["donor_subject"], table["recipient_subject"])
    ]
    table["pair_label_relation"] = [
        relation_label(str(donor), str(recipient), "within_label", "cross_label")
        for donor, recipient in zip(table["donor_label"], table["recipient_label"])
    ]

    relation_summary = (
        table.groupby(["feature_idx", "pair_subject_relation"], as_index=False)
        .agg(
            pairs_profiled=("pair_idx", "count"),
            mean_donor_conditioned_feature_mean=("donor_conditioned_feature_mean", "mean"),
            mean_recipient_conditioned_feature_mean=("recipient_conditioned_feature_mean", "mean"),
            mean_delta_input_mse=("delta_input_mse", "mean"),
            mean_cosine_to_recipient_baseline=("cosine_to_recipient_baseline", "mean"),
            mean_cosine_patch_delta_to_donor_delta=("cosine_patch_delta_to_donor_delta", "mean"),
            mean_patch_delta_energy=("mean_patch_delta_energy", "mean"),
        )
        .sort_values(["feature_idx", "pair_subject_relation"])
        .reset_index(drop=True)
    )

    subject_pair_summary = (
        table.groupby(["feature_idx", "donor_subject", "recipient_subject"], as_index=False)
        .agg(
            pairs_profiled=("pair_idx", "count"),
            mean_donor_conditioned_feature_mean=("donor_conditioned_feature_mean", "mean"),
            mean_recipient_conditioned_feature_mean=("recipient_conditioned_feature_mean", "mean"),
            mean_delta_input_mse=("delta_input_mse", "mean"),
            mean_cosine_to_recipient_baseline=("cosine_to_recipient_baseline", "mean"),
            mean_cosine_patch_delta_to_donor_delta=("cosine_patch_delta_to_donor_delta", "mean"),
            mean_patch_delta_energy=("mean_patch_delta_energy", "mean"),
        )
        .sort_values(["feature_idx", "mean_delta_input_mse", "donor_subject", "recipient_subject"], ascending=[True, True, True, True])
        .reset_index(drop=True)
    )

    label_relation_summary = (
        table.groupby(["feature_idx", "pair_label_relation"], as_index=False)
        .agg(
            pairs_profiled=("pair_idx", "count"),
            mean_delta_input_mse=("delta_input_mse", "mean"),
            mean_cosine_to_recipient_baseline=("cosine_to_recipient_baseline", "mean"),
            mean_cosine_patch_delta_to_donor_delta=("cosine_patch_delta_to_donor_delta", "mean"),
            mean_patch_delta_energy=("mean_patch_delta_energy", "mean"),
        )
        .sort_values(["feature_idx", "pair_label_relation"])
        .reset_index(drop=True)
    )

    table.to_csv(run_dir / "pair_patch_results.csv", index=False)
    relation_summary.to_csv(run_dir / "pair_subject_relation_summary.csv", index=False)
    subject_pair_summary.to_csv(run_dir / "pair_subject_summary.csv", index=False)
    label_relation_summary.to_csv(run_dir / "pair_label_relation_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--mode", choices=["ablation", "patch"], required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if args.mode == "ablation":
        summarize_ablation(run_dir)
    else:
        summarize_patch(run_dir)
    print(run_dir)


if __name__ == "__main__":
    main()
