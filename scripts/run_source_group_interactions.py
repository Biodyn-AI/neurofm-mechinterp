#!/usr/bin/env python
"""Profile reconstruction behavior for small groups of tokenizer sources."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from neurofm_mechinterp.analysis.common import flatten_windowed_time, save_json
from neurofm_mechinterp.loading import load_braintokenizer, resolve_device
from neurofm_mechinterp.paths import RESULTS_ROOT, REPO_ROOT, ensure_dir
from neurofm_mechinterp.processed_data import choose_examples, load_batch_from_examples, load_metadata


DATASET_METADATA = {
    "pd31": REPO_ROOT / "share" / "metadata" / "evaluate" / "pd31" / "BrainOmni" / "info.json",
    "MEG-MMI": REPO_ROOT / "share" / "metadata" / "evaluate" / "MEG-MMI" / "BrainOmni" / "info.json",
}

DEFAULT_SOURCE_GROUPS = {
    "pd31": [(3, 2, 8), (3, 2, 4)],
}


def default_clip_seconds(dataset: str) -> float | None:
    """Use the same practical MEG crop as the other real-data analyses."""
    if dataset == "MEG-MMI":
        return 10.0
    return None


def mean_cosine_similarity(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """Compute mean cosine similarity after flattening each batch item."""
    ref_flat = reference.reshape(reference.shape[0], -1)
    cand_flat = candidate.reshape(candidate.shape[0], -1)
    return float(F.cosine_similarity(ref_flat, cand_flat, dim=-1).mean().item())


def parse_source_groups(raw_groups: list[str] | None) -> list[tuple[int, ...]] | None:
    """Parse repeated `a,b,c` CLI values into normalized source groups."""
    if raw_groups is None:
        return None

    normalized = []
    for raw_group in raw_groups:
        group = tuple(int(value) for value in raw_group.split(",") if value)
        if len(group) < 2:
            raise ValueError("Source groups must contain at least two indices.")
        if len(set(group)) != len(group):
            raise ValueError("Source groups must not repeat the same source index.")
        if group not in normalized:
            normalized.append(group)
    if not normalized:
        raise ValueError("At least one source group must be provided.")
    return normalized


def dataset_groups(dataset: str, source_groups: list[tuple[int, ...]] | None) -> list[tuple[int, ...]]:
    """Resolve the profiled source groups for one dataset."""
    if source_groups is not None:
        return source_groups
    if dataset not in DEFAULT_SOURCE_GROUPS:
        raise ValueError(f"No default source groups are configured for dataset '{dataset}'.")
    return DEFAULT_SOURCE_GROUPS[dataset]


def reconstruct_from_sources(tokenizer, feature: torch.Tensor, sensor_embedding: torch.Tensor, source_indices: tuple[int, ...]):
    """Keep only the requested source slots, re-quantize, and decode them."""
    isolated_feature = torch.zeros_like(feature)
    for source_idx in source_indices:
        isolated_feature[:, source_idx] = feature[:, source_idx]
    quantized, _, _ = tokenizer.quantizer(isolated_feature)
    return tokenizer.decoder(quantized, sensor_embedding)


@torch.no_grad()
def profile_dataset(
    dataset: str,
    tokenizer,
    device,
    source_groups: list[tuple[int, ...]],
    max_segments: int | None,
    clip_seconds: float | None,
    output_suffix: str | None,
) -> Path:
    """Profile source-group reconstructions across one dataset."""
    metadata = load_metadata(DATASET_METADATA[dataset])
    if max_segments is not None:
        metadata = choose_examples(metadata, max_segments=max_segments)

    output_name = dataset if output_suffix is None else f"{dataset}_{output_suffix}"
    output_dir = ensure_dir(RESULTS_ROOT / "source_group_interactions" / output_name)
    segment_rows = []

    unique_sources = sorted({index for group in source_groups for index in group})
    for row in metadata:
        batch = load_batch_from_examples([row]).to(device)
        if clip_seconds is not None:
            clip_samples = int(clip_seconds * 256)
            batch.x = batch.x[..., :clip_samples]

        unfolded = tokenizer.unfold(batch.x)
        sensor_embedding = tokenizer.sensor_embed(batch.pos, batch.sensor_type)
        feature = tokenizer.encoder(unfolded, sensor_embedding)

        baseline_quantized, _, _ = tokenizer.quantizer(feature)
        baseline_reconstruction = tokenizer.decoder(baseline_quantized, sensor_embedding)
        baseline_flat = flatten_windowed_time(baseline_reconstruction)
        unfolded_flat = flatten_windowed_time(unfolded)
        baseline_energy = baseline_reconstruction.pow(2).sum() + 1e-12

        single_reconstructions = {}
        single_energy_fraction = {}
        single_cosine_to_input = {}
        single_cosine_to_baseline = {}
        for source_idx in unique_sources:
            source_reconstruction = reconstruct_from_sources(
                tokenizer,
                feature,
                sensor_embedding,
                (source_idx,),
            )
            source_flat = flatten_windowed_time(source_reconstruction)
            single_reconstructions[source_idx] = source_reconstruction
            single_energy_fraction[source_idx] = float((source_reconstruction.pow(2).sum() / baseline_energy).item())
            single_cosine_to_input[source_idx] = mean_cosine_similarity(source_flat, unfolded_flat)
            single_cosine_to_baseline[source_idx] = mean_cosine_similarity(source_flat, baseline_flat)

        for group in source_groups:
            group_reconstruction = reconstruct_from_sources(
                tokenizer,
                feature,
                sensor_embedding,
                group,
            )
            group_flat = flatten_windowed_time(group_reconstruction)
            group_energy_fraction = float((group_reconstruction.pow(2).sum() / baseline_energy).item())
            group_cosine_to_input = mean_cosine_similarity(group_flat, unfolded_flat)
            group_cosine_to_baseline = mean_cosine_similarity(group_flat, baseline_flat)

            sum_reconstruction = sum(single_reconstructions[source_idx] for source_idx in group)
            sum_flat = flatten_windowed_time(sum_reconstruction)
            sum_cosine_to_group = mean_cosine_similarity(sum_flat, group_flat)
            group_minus_sum_mse = float(((group_reconstruction - sum_reconstruction) ** 2).mean().item())

            best_single_input = max(single_cosine_to_input[source_idx] for source_idx in group)
            best_single_baseline = max(single_cosine_to_baseline[source_idx] for source_idx in group)
            summed_single_energy = sum(single_energy_fraction[source_idx] for source_idx in group)

            group_sensor_energy = group_reconstruction.pow(2).mean(dim=(0, 2, 3))
            group_top_sensors = torch.topk(
                group_sensor_energy,
                k=min(5, group_sensor_energy.shape[0]),
            ).indices.cpu().tolist()

            segment_rows.append(
                {
                    "dataset": dataset,
                    "subject": row["subject"],
                    "label": int(row["label"]),
                    "path": row["path"],
                    "channels": int(row["channels"]),
                    "clip_seconds": clip_seconds,
                    "group_label": ",".join(str(source_idx) for source_idx in group),
                    "group_size": int(len(group)),
                    "group_energy_fraction": group_energy_fraction,
                    "superadditive_energy_fraction": group_energy_fraction - summed_single_energy,
                    "group_cosine_to_input": group_cosine_to_input,
                    "group_gain_over_best_single_input": group_cosine_to_input - best_single_input,
                    "group_cosine_to_baseline": group_cosine_to_baseline,
                    "group_gain_over_best_single_baseline": group_cosine_to_baseline - best_single_baseline,
                    "sum_cosine_to_group": sum_cosine_to_group,
                    "group_minus_sum_mse": group_minus_sum_mse,
                    "group_top_sensor_indices": ",".join(str(index) for index in group_top_sensors),
                }
            )

    segment_table = pd.DataFrame(segment_rows).sort_values(["group_label", "subject", "path"]).reset_index(drop=True)

    group_summary = (
        segment_table.groupby(["group_label", "group_size"], as_index=False)
        .agg(
            segments_profiled=("path", "count"),
            mean_group_energy_fraction=("group_energy_fraction", "mean"),
            mean_superadditive_energy_fraction=("superadditive_energy_fraction", "mean"),
            mean_group_cosine_to_input=("group_cosine_to_input", "mean"),
            mean_group_gain_over_best_single_input=("group_gain_over_best_single_input", "mean"),
            mean_group_cosine_to_baseline=("group_cosine_to_baseline", "mean"),
            mean_group_gain_over_best_single_baseline=("group_gain_over_best_single_baseline", "mean"),
            mean_sum_cosine_to_group=("sum_cosine_to_group", "mean"),
            mean_group_minus_sum_mse=("group_minus_sum_mse", "mean"),
        )
        .sort_values(["mean_group_cosine_to_baseline", "mean_group_cosine_to_input"], ascending=[False, False])
        .reset_index(drop=True)
    )

    subject_summary = (
        segment_table.groupby(["group_label", "subject"], as_index=False)
        .agg(
            segments=("path", "count"),
            mean_group_energy_fraction=("group_energy_fraction", "mean"),
            mean_group_cosine_to_input=("group_cosine_to_input", "mean"),
            mean_group_cosine_to_baseline=("group_cosine_to_baseline", "mean"),
            mean_group_gain_over_best_single_baseline=("group_gain_over_best_single_baseline", "mean"),
            mean_sum_cosine_to_group=("sum_cosine_to_group", "mean"),
            mean_group_minus_sum_mse=("group_minus_sum_mse", "mean"),
        )
        .sort_values(["group_label", "subject"])
        .reset_index(drop=True)
    )

    label_summary = (
        segment_table.groupby(["group_label", "label"], as_index=False)
        .agg(
            segments=("path", "count"),
            mean_group_energy_fraction=("group_energy_fraction", "mean"),
            mean_group_cosine_to_input=("group_cosine_to_input", "mean"),
            mean_group_cosine_to_baseline=("group_cosine_to_baseline", "mean"),
            mean_group_gain_over_best_single_baseline=("group_gain_over_best_single_baseline", "mean"),
            mean_sum_cosine_to_group=("sum_cosine_to_group", "mean"),
            mean_group_minus_sum_mse=("group_minus_sum_mse", "mean"),
        )
        .sort_values(["group_label", "label"])
        .reset_index(drop=True)
    )

    sensor_rows = []
    for group_label, subset in segment_table.groupby("group_label"):
        counter = Counter()
        for value in subset["group_top_sensor_indices"]:
            counter.update(int(index) for index in str(value).split(",") if index)
        for sensor_idx, count in counter.most_common():
            sensor_rows.append(
                {
                    "group_label": group_label,
                    "sensor_idx": int(sensor_idx),
                    "count": int(count),
                    "fraction": float(count / len(subset)),
                }
            )
    sensor_summary = pd.DataFrame(sensor_rows).sort_values(
        ["group_label", "count", "sensor_idx"],
        ascending=[True, False, True],
    )

    segment_table.to_csv(output_dir / "group_segment_profiles.csv", index=False)
    group_summary.to_csv(output_dir / "group_summary.csv", index=False)
    subject_summary.to_csv(output_dir / "group_subject_summary.csv", index=False)
    label_summary.to_csv(output_dir / "group_label_summary.csv", index=False)
    sensor_summary.to_csv(output_dir / "group_sensor_summary.csv", index=False)

    save_json(
        output_dir / "summary.json",
        {
            "dataset": dataset,
            "num_segments_profiled": int(len(segment_table["path"].unique())),
            "source_groups": [group_label for group_label in segment_table["group_label"].drop_duplicates().tolist()],
            "clip_seconds": clip_seconds,
        },
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["pd31", "MEG-MMI", "all"], default="pd31")
    parser.add_argument(
        "--source-group",
        action="append",
        dest="source_groups",
        default=None,
        help="Repeatable source group in the form `a,b,c`.",
    )
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=None,
        help="Optionally truncate segments. Defaults to 10 seconds for MEG profiles.",
    )
    parser.add_argument(
        "--output-suffix",
        default=None,
        help="Optional suffix for the output directory, e.g. `fourth_source`.",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    datasets = ["pd31", "MEG-MMI"] if args.dataset == "all" else [args.dataset]
    source_groups = parse_source_groups(args.source_groups)
    device = resolve_device(args.device)
    tokenizer = load_braintokenizer(device=str(device))

    for dataset in datasets:
        output_dir = profile_dataset(
            dataset=dataset,
            tokenizer=tokenizer,
            device=device,
            source_groups=dataset_groups(dataset, source_groups),
            max_segments=args.max_segments,
            clip_seconds=args.clip_seconds if args.clip_seconds is not None else default_clip_seconds(dataset),
            output_suffix=args.output_suffix,
        )
        print(f"{dataset} -> {output_dir}")


if __name__ == "__main__":
    main()
