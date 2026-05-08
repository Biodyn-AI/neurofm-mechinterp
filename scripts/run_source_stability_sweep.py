#!/usr/bin/env python
"""Sweep tokenizer source ablations across many real-data segments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from neurofm_mechinterp.analysis.common import mean_sensor_relative_bandpower, save_json
from neurofm_mechinterp.analysis.source_ablation import compute_source_ablation_table
from neurofm_mechinterp.loading import load_braintokenizer, resolve_device
from neurofm_mechinterp.paths import RESULTS_ROOT, REPO_ROOT, ensure_dir
from neurofm_mechinterp.processed_data import choose_examples, load_batch_from_examples, load_metadata


DATASET_METADATA = {
    "pd31": REPO_ROOT / "share" / "metadata" / "evaluate" / "pd31" / "BrainOmni" / "info.json",
    "MEG-MMI": REPO_ROOT / "share" / "metadata" / "evaluate" / "MEG-MMI" / "BrainOmni" / "info.json",
}


def default_clip_seconds(dataset: str) -> float | None:
    """Use a shorter default clip for MEG so CPU sweeps stay tractable."""
    if dataset == "MEG-MMI":
        return 10.0
    return None


def default_source_limit(dataset: str) -> int | None:
    """Use exhaustive EEG ablations and reduced MEG ablations by default."""
    if dataset == "MEG-MMI":
        return 4
    return None


def summarize_segment_bandpower(batch, sample_rate: float) -> dict[str, float]:
    """Compute relative bandpower averaged across sensors for one segment."""
    bandpower = mean_sensor_relative_bandpower(
        batch.x.detach().cpu().numpy(),
        sample_rate=sample_rate,
        sensor_axis=1,
    )
    return {f"{band_name}_relative_power": float(values.mean()) for band_name, values in bandpower.items()}


def analyze_dataset(
    dataset: str,
    tokenizer,
    device,
    max_segments: int | None,
    clip_seconds: float | None,
    source_limit: int | None,
) -> Path:
    """Run per-segment source-ablation analysis for one dataset."""
    metadata = load_metadata(DATASET_METADATA[dataset])
    if max_segments is not None:
        metadata = choose_examples(metadata, max_segments=max_segments)

    output_dir = ensure_dir(RESULTS_ROOT / "source_stability" / dataset)
    per_source_rows = []
    per_segment_rows = []

    for segment_index, row in enumerate(metadata):
        batch = load_batch_from_examples([row]).to(device)
        if clip_seconds is not None:
            clip_samples = int(clip_seconds * 256)
            batch.x = batch.x[..., :clip_samples]

        table, summary = compute_source_ablation_table(
            tokenizer,
            batch,
            max_sources=source_limit,
        )
        bandpower_summary = summarize_segment_bandpower(batch, sample_rate=256.0)
        top_row = table.iloc[0].to_dict()
        second_best_delta = float(table.iloc[1]["mse_delta"]) if len(table) > 1 else 0.0

        per_segment_rows.append(
            {
                "dataset": dataset,
                "segment_index": int(segment_index),
                "subject": row["subject"],
                "label": int(row["label"]),
                "path": row["path"],
                "channels": int(row["channels"]),
                "clip_seconds": clip_seconds,
                "num_sources_tested": int(summary["num_sources_tested"]),
                "tested_source_indices": ",".join(str(index) for index in summary["tested_source_indices"]),
                "top_source_idx": int(top_row["source_idx"]),
                "top_source_delta_mse": float(top_row["mse_delta"]),
                "top_source_code_change_rate": float(top_row["code_change_rate"]),
                "top_source_mean_feature_energy": float(top_row["mean_feature_energy"]),
                "top_source_top_sensor_indices": str(top_row["top_sensor_indices"]),
                "top_source_margin_over_second": float(top_row["mse_delta"] - second_best_delta),
                **bandpower_summary,
            }
        )

        for rank, source_row in enumerate(table.itertuples(index=False), start=1):
            per_source_rows.append(
                {
                    "dataset": dataset,
                    "segment_index": int(segment_index),
                    "subject": row["subject"],
                    "label": int(row["label"]),
                    "path": row["path"],
                    "source_rank": int(rank),
                    "source_idx": int(source_row.source_idx),
                    "mse_delta": float(source_row.mse_delta),
                    "code_change_rate": float(source_row.code_change_rate),
                    "mean_feature_energy": float(source_row.mean_feature_energy),
                    "top_sensor_indices": str(source_row.top_sensor_indices),
                    "is_top_source": int(rank == 1),
                }
            )

    per_segment = pd.DataFrame(per_segment_rows).sort_values(["segment_index"]).reset_index(drop=True)
    per_source = pd.DataFrame(per_source_rows).sort_values(["segment_index", "source_rank"]).reset_index(drop=True)

    source_summary = (
        per_source.groupby("source_idx", as_index=False)
        .agg(
            segments_tested=("segment_index", "count"),
            times_ranked_first=("is_top_source", "sum"),
            mean_mse_delta=("mse_delta", "mean"),
            mean_code_change_rate=("code_change_rate", "mean"),
            mean_feature_energy=("mean_feature_energy", "mean"),
        )
        .sort_values(["times_ranked_first", "mean_mse_delta"], ascending=[False, False])
        .reset_index(drop=True)
    )
    source_summary["top_source_fraction"] = (
        source_summary["times_ranked_first"] / max(len(per_segment), 1)
    )

    top_source_bandpower = (
        per_segment.groupby("top_source_idx", as_index=False)
        .agg(
            segments=("segment_index", "count"),
            mean_top_source_delta_mse=("top_source_delta_mse", "mean"),
            mean_margin_over_second=("top_source_margin_over_second", "mean"),
            delta_relative_power=("delta_relative_power", "mean"),
            theta_relative_power=("theta_relative_power", "mean"),
            alpha_relative_power=("alpha_relative_power", "mean"),
            beta_relative_power=("beta_relative_power", "mean"),
            gamma_relative_power=("gamma_relative_power", "mean"),
        )
        .sort_values(["segments", "mean_top_source_delta_mse"], ascending=[False, False])
        .reset_index(drop=True)
    )

    per_segment.to_csv(output_dir / "segment_top_sources.csv", index=False)
    per_source.to_csv(output_dir / "segment_source_rankings.csv", index=False)
    source_summary.to_csv(output_dir / "source_summary.csv", index=False)
    top_source_bandpower.to_csv(output_dir / "top_source_bandpower.csv", index=False)

    save_json(
        output_dir / "summary.json",
        {
            "dataset": dataset,
            "num_segments_analyzed": int(len(per_segment)),
            "clip_seconds": clip_seconds,
            "source_limit": source_limit,
            "most_frequent_top_source": int(source_summary.iloc[0]["source_idx"]),
            "most_frequent_top_source_count": int(source_summary.iloc[0]["times_ranked_first"]),
            "mean_top_source_delta_mse": float(per_segment["top_source_delta_mse"].mean()),
            "mean_top_source_margin_over_second": float(per_segment["top_source_margin_over_second"].mean()),
        },
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["pd31", "MEG-MMI", "all"], default="all")
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=None,
        help="Optionally truncate each segment before analysis. Defaults to 10 seconds for MEG sweeps.",
    )
    parser.add_argument(
        "--source-limit",
        type=int,
        default=None,
        help="Optionally test only the top-energy tokenizer sources. Defaults to 4 for MEG sweeps.",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    datasets = ["pd31", "MEG-MMI"] if args.dataset == "all" else [args.dataset]
    device = resolve_device(args.device)
    tokenizer = load_braintokenizer(device=str(device))

    for dataset in datasets:
        clip_seconds = args.clip_seconds if args.clip_seconds is not None else default_clip_seconds(dataset)
        source_limit = args.source_limit if args.source_limit is not None else default_source_limit(dataset)
        output_dir = analyze_dataset(
            dataset=dataset,
            tokenizer=tokenizer,
            device=device,
            max_segments=args.max_segments,
            clip_seconds=clip_seconds,
            source_limit=source_limit,
        )
        print(f"{dataset} -> {output_dir}")


if __name__ == "__main__":
    main()
