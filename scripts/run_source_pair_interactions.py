#!/usr/bin/env python
"""Profile additive vs interactive behavior for selected tokenizer source pairs."""

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

DEFAULT_SOURCE_PAIRS = {
    "pd31": [(3, 2)],
    "MEG-MMI": [(2, 11)],
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


def parse_source_pairs(raw_pairs: list[str] | None) -> list[tuple[int, int]] | None:
    """Parse repeated `a,b` CLI values into normalized source-index pairs."""
    if raw_pairs is None:
        return None

    normalized = []
    for raw_pair in raw_pairs:
        left, right = raw_pair.split(",", maxsplit=1)
        pair = (int(left), int(right))
        if pair[0] == pair[1]:
            raise ValueError("Source pairs must contain two distinct indices.")
        if pair not in normalized:
            normalized.append(pair)
    if not normalized:
        raise ValueError("At least one source pair must be provided.")
    return normalized


def dataset_pairs(dataset: str, source_pairs: list[tuple[int, int]] | None) -> list[tuple[int, int]]:
    """Resolve the profiled source pairs for one dataset."""
    if source_pairs is not None:
        return source_pairs
    return DEFAULT_SOURCE_PAIRS[dataset]


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
    source_pairs: list[tuple[int, int]],
    max_segments: int | None,
    clip_seconds: float | None,
) -> Path:
    """Profile source-pair reconstructions across one dataset."""
    metadata = load_metadata(DATASET_METADATA[dataset])
    if max_segments is not None:
        metadata = choose_examples(metadata, max_segments=max_segments)

    output_dir = ensure_dir(RESULTS_ROOT / "source_pair_interactions" / dataset)
    segment_rows = []

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
        for source_idx in sorted({index for pair in source_pairs for index in pair}):
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

        for source_a, source_b in source_pairs:
            pair_reconstruction = reconstruct_from_sources(
                tokenizer,
                feature,
                sensor_embedding,
                (source_a, source_b),
            )
            pair_flat = flatten_windowed_time(pair_reconstruction)
            pair_energy_fraction = float((pair_reconstruction.pow(2).sum() / baseline_energy).item())
            pair_cosine_to_input = mean_cosine_similarity(pair_flat, unfolded_flat)
            pair_cosine_to_baseline = mean_cosine_similarity(pair_flat, baseline_flat)

            sum_reconstruction = single_reconstructions[source_a] + single_reconstructions[source_b]
            sum_flat = flatten_windowed_time(sum_reconstruction)
            sum_cosine_to_pair = mean_cosine_similarity(sum_flat, pair_flat)
            pair_minus_sum_mse = float(((pair_reconstruction - sum_reconstruction) ** 2).mean().item())

            pair_sensor_energy = pair_reconstruction.pow(2).mean(dim=(0, 2, 3))
            pair_top_sensors = torch.topk(
                pair_sensor_energy,
                k=min(5, pair_sensor_energy.shape[0]),
            ).indices.cpu().tolist()

            segment_rows.append(
                {
                    "dataset": dataset,
                    "subject": row["subject"],
                    "label": int(row["label"]),
                    "path": row["path"],
                    "channels": int(row["channels"]),
                    "clip_seconds": clip_seconds,
                    "source_a": int(source_a),
                    "source_b": int(source_b),
                    "pair_label": f"{source_a},{source_b}",
                    "single_a_energy_fraction": single_energy_fraction[source_a],
                    "single_b_energy_fraction": single_energy_fraction[source_b],
                    "pair_energy_fraction": pair_energy_fraction,
                    "superadditive_energy_fraction": pair_energy_fraction
                    - single_energy_fraction[source_a]
                    - single_energy_fraction[source_b],
                    "single_a_cosine_to_input": single_cosine_to_input[source_a],
                    "single_b_cosine_to_input": single_cosine_to_input[source_b],
                    "pair_cosine_to_input": pair_cosine_to_input,
                    "pair_gain_over_best_single_input": pair_cosine_to_input
                    - max(single_cosine_to_input[source_a], single_cosine_to_input[source_b]),
                    "single_a_cosine_to_baseline": single_cosine_to_baseline[source_a],
                    "single_b_cosine_to_baseline": single_cosine_to_baseline[source_b],
                    "pair_cosine_to_baseline": pair_cosine_to_baseline,
                    "pair_gain_over_best_single_baseline": pair_cosine_to_baseline
                    - max(single_cosine_to_baseline[source_a], single_cosine_to_baseline[source_b]),
                    "sum_cosine_to_pair": sum_cosine_to_pair,
                    "pair_minus_sum_mse": pair_minus_sum_mse,
                    "pair_top_sensor_indices": ",".join(str(index) for index in pair_top_sensors),
                }
            )

    segment_table = pd.DataFrame(segment_rows).sort_values(["pair_label", "subject", "path"]).reset_index(drop=True)

    pair_summary = (
        segment_table.groupby(["source_a", "source_b", "pair_label"], as_index=False)
        .agg(
            segments_profiled=("path", "count"),
            mean_single_a_energy_fraction=("single_a_energy_fraction", "mean"),
            mean_single_b_energy_fraction=("single_b_energy_fraction", "mean"),
            mean_pair_energy_fraction=("pair_energy_fraction", "mean"),
            mean_superadditive_energy_fraction=("superadditive_energy_fraction", "mean"),
            mean_single_a_cosine_to_input=("single_a_cosine_to_input", "mean"),
            mean_single_b_cosine_to_input=("single_b_cosine_to_input", "mean"),
            mean_pair_cosine_to_input=("pair_cosine_to_input", "mean"),
            mean_pair_gain_over_best_single_input=("pair_gain_over_best_single_input", "mean"),
            mean_single_a_cosine_to_baseline=("single_a_cosine_to_baseline", "mean"),
            mean_single_b_cosine_to_baseline=("single_b_cosine_to_baseline", "mean"),
            mean_pair_cosine_to_baseline=("pair_cosine_to_baseline", "mean"),
            mean_pair_gain_over_best_single_baseline=("pair_gain_over_best_single_baseline", "mean"),
            mean_sum_cosine_to_pair=("sum_cosine_to_pair", "mean"),
            mean_pair_minus_sum_mse=("pair_minus_sum_mse", "mean"),
        )
        .sort_values(["mean_pair_cosine_to_baseline", "mean_pair_cosine_to_input"], ascending=[False, False])
        .reset_index(drop=True)
    )

    subject_summary = (
        segment_table.groupby(["source_a", "source_b", "pair_label", "subject"], as_index=False)
        .agg(
            segments=("path", "count"),
            mean_pair_energy_fraction=("pair_energy_fraction", "mean"),
            mean_pair_cosine_to_input=("pair_cosine_to_input", "mean"),
            mean_pair_cosine_to_baseline=("pair_cosine_to_baseline", "mean"),
            mean_pair_gain_over_best_single_baseline=("pair_gain_over_best_single_baseline", "mean"),
            mean_sum_cosine_to_pair=("sum_cosine_to_pair", "mean"),
            mean_pair_minus_sum_mse=("pair_minus_sum_mse", "mean"),
        )
        .sort_values(["pair_label", "subject"])
        .reset_index(drop=True)
    )

    label_summary = (
        segment_table.groupby(["source_a", "source_b", "pair_label", "label"], as_index=False)
        .agg(
            segments=("path", "count"),
            mean_pair_energy_fraction=("pair_energy_fraction", "mean"),
            mean_pair_cosine_to_input=("pair_cosine_to_input", "mean"),
            mean_pair_cosine_to_baseline=("pair_cosine_to_baseline", "mean"),
            mean_pair_gain_over_best_single_baseline=("pair_gain_over_best_single_baseline", "mean"),
            mean_sum_cosine_to_pair=("sum_cosine_to_pair", "mean"),
            mean_pair_minus_sum_mse=("pair_minus_sum_mse", "mean"),
        )
        .sort_values(["pair_label", "label"])
        .reset_index(drop=True)
    )

    sensor_rows = []
    for pair_label, subset in segment_table.groupby("pair_label"):
        counter = Counter()
        for value in subset["pair_top_sensor_indices"]:
            counter.update(int(index) for index in str(value).split(",") if index)
        for sensor_idx, count in counter.most_common():
            sensor_rows.append(
                {
                    "pair_label": pair_label,
                    "sensor_idx": int(sensor_idx),
                    "count": int(count),
                    "fraction": float(count / len(subset)),
                }
            )
    sensor_summary = pd.DataFrame(sensor_rows).sort_values(
        ["pair_label", "count", "sensor_idx"],
        ascending=[True, False, True],
    )

    segment_table.to_csv(output_dir / "pair_segment_profiles.csv", index=False)
    pair_summary.to_csv(output_dir / "pair_summary.csv", index=False)
    subject_summary.to_csv(output_dir / "pair_subject_summary.csv", index=False)
    label_summary.to_csv(output_dir / "pair_label_summary.csv", index=False)
    sensor_summary.to_csv(output_dir / "pair_sensor_summary.csv", index=False)

    save_json(
        output_dir / "summary.json",
        {
            "dataset": dataset,
            "num_segments_profiled": int(len(segment_table["path"].unique())),
            "source_pairs": [pair_label for pair_label in segment_table["pair_label"].drop_duplicates().tolist()],
            "clip_seconds": clip_seconds,
        },
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["pd31", "MEG-MMI", "all"], default="all")
    parser.add_argument(
        "--source-pair",
        action="append",
        dest="source_pairs",
        default=None,
        help="Repeatable source pair in the form `a,b`.",
    )
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=None,
        help="Optionally truncate segments. Defaults to 10 seconds for MEG profiles.",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    datasets = ["pd31", "MEG-MMI"] if args.dataset == "all" else [args.dataset]
    source_pairs = parse_source_pairs(args.source_pairs)
    device = resolve_device(args.device)
    tokenizer = load_braintokenizer(device=str(device))

    for dataset in datasets:
        output_dir = profile_dataset(
            dataset=dataset,
            tokenizer=tokenizer,
            device=device,
            source_pairs=dataset_pairs(dataset, source_pairs),
            max_segments=args.max_segments,
            clip_seconds=args.clip_seconds if args.clip_seconds is not None else default_clip_seconds(dataset),
        )
        print(f"{dataset} -> {output_dir}")


if __name__ == "__main__":
    main()
