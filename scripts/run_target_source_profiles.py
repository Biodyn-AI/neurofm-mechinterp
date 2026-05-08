#!/usr/bin/env python
"""Profile a small set of tokenizer sources across real datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from neurofm_mechinterp.analysis.common import (
    flatten_windowed_time,
    mean_sensor_relative_bandpower,
    save_json,
)
from neurofm_mechinterp.analysis.source_ablation import compute_source_ablation_table
from neurofm_mechinterp.loading import load_braintokenizer, resolve_device
from neurofm_mechinterp.paths import RESULTS_ROOT, REPO_ROOT, ensure_dir
from neurofm_mechinterp.processed_data import choose_examples, load_batch_from_examples, load_metadata


DATASET_METADATA = {
    "pd31": REPO_ROOT / "share" / "metadata" / "evaluate" / "pd31" / "BrainOmni" / "info.json",
    "MEG-MMI": REPO_ROOT / "share" / "metadata" / "evaluate" / "MEG-MMI" / "BrainOmni" / "info.json",
}

DEFAULT_TARGET_SOURCES = {
    "pd31": [3, 2],
    "MEG-MMI": [2],
}


def default_clip_seconds(dataset: str) -> float | None:
    """Use the same practical MEG crop as the real-data analysis script."""
    if dataset == "MEG-MMI":
        return 10.0
    return None


def load_rank_lookup(dataset: str) -> dict[tuple[str, int], int]:
    """Load full-source ranking data when an earlier stability sweep is available."""
    path = RESULTS_ROOT / "source_stability" / dataset / "segment_source_rankings.csv"
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    return {
        (str(row.path), int(row.source_idx)): int(row.source_rank)
        for row in table.itertuples(index=False)
    }


def summarize_segment_bandpower(batch, sample_rate: float) -> dict[str, float]:
    """Compute corrected segment-level bandpower averaged across sensors."""
    bandpower = mean_sensor_relative_bandpower(
        batch.x.detach().cpu().numpy(),
        sample_rate=sample_rate,
        sensor_axis=1,
    )
    return {f"{band_name}_relative_power": float(values.mean()) for band_name, values in bandpower.items()}


def mean_cosine_similarity(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """Compute mean cosine similarity after flattening each batch item."""
    ref_flat = reference.reshape(reference.shape[0], -1)
    cand_flat = candidate.reshape(candidate.shape[0], -1)
    return float(F.cosine_similarity(ref_flat, cand_flat, dim=-1).mean().item())


def dataset_targets(dataset: str, sources: list[int] | None) -> list[int]:
    """Resolve the profiled sources for one dataset."""
    if sources:
        return [int(source) for source in sources]
    return DEFAULT_TARGET_SOURCES[dataset]


@torch.no_grad()
def profile_dataset(
    dataset: str,
    tokenizer,
    device,
    source_indices: list[int],
    max_segments: int | None,
    clip_seconds: float | None,
) -> Path:
    """Profile a fixed set of tokenizer sources across one dataset."""
    metadata = load_metadata(DATASET_METADATA[dataset])
    if max_segments is not None:
        metadata = choose_examples(metadata, max_segments=max_segments)

    output_dir = ensure_dir(RESULTS_ROOT / "source_profiles" / dataset)
    rank_lookup = load_rank_lookup(dataset)

    segment_rows = []
    code_rows = []
    codebook_size = int(tokenizer.quantizer.rvq.codebook_size)

    for row in metadata:
        batch = load_batch_from_examples([row]).to(device)
        if clip_seconds is not None:
            clip_samples = int(clip_seconds * 256)
            batch.x = batch.x[..., :clip_samples]

        ablation_table, _ = compute_source_ablation_table(
            tokenizer,
            batch,
            source_indices=source_indices,
        )

        unfolded = tokenizer.unfold(batch.x)
        sensor_embedding = tokenizer.sensor_embed(batch.pos, batch.sensor_type)
        feature = tokenizer.encoder(unfolded, sensor_embedding)
        baseline_quantized, indices, _ = tokenizer.quantizer(feature)
        indices_np = indices.detach().cpu().numpy()
        baseline_reconstruction = tokenizer.decoder(baseline_quantized, sensor_embedding)
        baseline_flat = flatten_windowed_time(baseline_reconstruction)
        unfolded_flat = flatten_windowed_time(unfolded)
        baseline_energy = baseline_reconstruction.pow(2).sum() + 1e-12
        bandpower_summary = summarize_segment_bandpower(batch, sample_rate=256.0)

        for source_idx in source_indices:
            source_row = ablation_table[ablation_table["source_idx"] == source_idx].iloc[0]
            source_feature = feature[:, source_idx]
            isolated_feature = torch.zeros_like(feature)
            isolated_feature[:, source_idx] = feature[:, source_idx]
            isolated_quantized, _, _ = tokenizer.quantizer(isolated_feature)
            isolated_reconstruction = tokenizer.decoder(isolated_quantized, sensor_embedding)
            isolated_flat = flatten_windowed_time(isolated_reconstruction)
            isolated_bandpower = mean_sensor_relative_bandpower(
                isolated_flat.detach().cpu().numpy(),
                sample_rate=256.0,
                sensor_axis=1,
            )
            isolated_sensor_energy = isolated_reconstruction.pow(2).mean(dim=(0, 2, 3))
            isolated_top_sensors = torch.topk(
                isolated_sensor_energy,
                k=min(5, isolated_sensor_energy.shape[0]),
            ).indices.cpu().tolist()
            segment_row = {
                "dataset": dataset,
                "subject": row["subject"],
                "label": int(row["label"]),
                "path": row["path"],
                "channels": int(row["channels"]),
                "clip_seconds": clip_seconds,
                "source_idx": int(source_idx),
                "full_rank": rank_lookup.get((str(row["path"]), int(source_idx))),
                "mse_delta": float(source_row["mse_delta"]),
                "code_change_rate": float(source_row["code_change_rate"]),
                "mean_feature_energy": float(source_row["mean_feature_energy"]),
                "top_sensor_indices": str(source_row["top_sensor_indices"]),
                "feature_std": float(source_feature.std().item()),
                "source_only_energy_fraction": float((isolated_reconstruction.pow(2).sum() / baseline_energy).item()),
                "source_only_mse_to_input": float(((isolated_reconstruction - unfolded) ** 2).mean().item()),
                "source_only_cosine_to_input": mean_cosine_similarity(isolated_flat, unfolded_flat),
                "source_only_cosine_to_baseline": mean_cosine_similarity(isolated_flat, baseline_flat),
                "source_only_top_sensor_indices": ",".join(str(index) for index in isolated_top_sensors),
                **bandpower_summary,
            }
            for band_name, values in isolated_bandpower.items():
                segment_row[f"source_only_{band_name}_relative_power"] = float(values.mean())

            source_codes = indices_np[0, source_idx].reshape(-1, indices_np.shape[-1])
            for layer_idx in range(source_codes.shape[-1]):
                layer_codes = source_codes[:, layer_idx]
                counts = np.bincount(layer_codes, minlength=codebook_size)
                total = int(counts.sum())
                top_code = int(np.argmax(counts))
                top_fraction = float(counts[top_code] / total) if total else 0.0
                probabilities = counts[counts > 0] / max(total, 1)
                entropy = float(-(probabilities * np.log2(probabilities)).sum()) if len(probabilities) else 0.0
                segment_row[f"layer_{layer_idx}_top_code"] = top_code
                segment_row[f"layer_{layer_idx}_top_code_fraction"] = top_fraction
                segment_row[f"layer_{layer_idx}_entropy"] = entropy

                nonzero_codes = np.nonzero(counts)[0]
                for code in nonzero_codes.tolist():
                    code_rows.append(
                        {
                            "dataset": dataset,
                            "subject": row["subject"],
                            "label": int(row["label"]),
                            "path": row["path"],
                            "source_idx": int(source_idx),
                            "layer": int(layer_idx),
                            "code": int(code),
                            "count": int(counts[code]),
                            "fraction": float(counts[code] / total) if total else 0.0,
                        }
                    )

            segment_rows.append(segment_row)

    segment_table = pd.DataFrame(segment_rows).sort_values(["source_idx", "subject", "path"]).reset_index(drop=True)
    code_table = pd.DataFrame(code_rows).sort_values(["source_idx", "layer", "path", "code"]).reset_index(drop=True)

    source_summary = (
        segment_table.groupby("source_idx", as_index=False)
        .agg(
            segments_profiled=("path", "count"),
            mean_full_rank=("full_rank", "mean"),
            top_rank_count=("full_rank", lambda values: int((pd.Series(values) == 1).sum())),
            mean_mse_delta=("mse_delta", "mean"),
            mean_code_change_rate=("code_change_rate", "mean"),
            mean_feature_energy=("mean_feature_energy", "mean"),
            mean_feature_std=("feature_std", "mean"),
            mean_source_only_energy_fraction=("source_only_energy_fraction", "mean"),
            mean_source_only_mse_to_input=("source_only_mse_to_input", "mean"),
            mean_source_only_cosine_to_input=("source_only_cosine_to_input", "mean"),
            mean_source_only_cosine_to_baseline=("source_only_cosine_to_baseline", "mean"),
            delta_relative_power=("delta_relative_power", "mean"),
            theta_relative_power=("theta_relative_power", "mean"),
            alpha_relative_power=("alpha_relative_power", "mean"),
            beta_relative_power=("beta_relative_power", "mean"),
            gamma_relative_power=("gamma_relative_power", "mean"),
            source_only_delta_relative_power=("source_only_delta_relative_power", "mean"),
            source_only_theta_relative_power=("source_only_theta_relative_power", "mean"),
            source_only_alpha_relative_power=("source_only_alpha_relative_power", "mean"),
            source_only_beta_relative_power=("source_only_beta_relative_power", "mean"),
            source_only_gamma_relative_power=("source_only_gamma_relative_power", "mean"),
        )
        .sort_values(["top_rank_count", "mean_mse_delta"], ascending=[False, False])
        .reset_index(drop=True)
    )

    subject_summary = (
        segment_table.groupby(["source_idx", "subject"], as_index=False)
        .agg(
            segments=("path", "count"),
            mean_full_rank=("full_rank", "mean"),
            mean_mse_delta=("mse_delta", "mean"),
            mean_feature_energy=("mean_feature_energy", "mean"),
            mean_feature_std=("feature_std", "mean"),
            mean_source_only_energy_fraction=("source_only_energy_fraction", "mean"),
            mean_source_only_cosine_to_input=("source_only_cosine_to_input", "mean"),
        )
        .sort_values(["source_idx", "subject"])
        .reset_index(drop=True)
    )

    label_summary = (
        segment_table.groupby(["source_idx", "label"], as_index=False)
        .agg(
            segments=("path", "count"),
            mean_full_rank=("full_rank", "mean"),
            mean_mse_delta=("mse_delta", "mean"),
            mean_feature_energy=("mean_feature_energy", "mean"),
            mean_feature_std=("feature_std", "mean"),
            mean_source_only_energy_fraction=("source_only_energy_fraction", "mean"),
            mean_source_only_cosine_to_input=("source_only_cosine_to_input", "mean"),
        )
        .sort_values(["source_idx", "label"])
        .reset_index(drop=True)
    )

    code_summary = (
        code_table.groupby(["source_idx", "layer", "code"], as_index=False)
        .agg(
            total_count=("count", "sum"),
            mean_fraction=("fraction", "mean"),
            segments_present=("path", "count"),
        )
        .sort_values(["source_idx", "layer", "total_count"], ascending=[True, True, False])
        .reset_index(drop=True)
    )

    sensor_rows = []
    source_only_sensor_rows = []
    for source_idx, subset in segment_table.groupby("source_idx"):
        counter = Counter()
        source_only_counter = Counter()
        for value in subset["top_sensor_indices"]:
            counter.update(int(index) for index in str(value).split(",") if index)
        for value in subset["source_only_top_sensor_indices"]:
            source_only_counter.update(int(index) for index in str(value).split(",") if index)
        for sensor_idx, count in counter.most_common():
            sensor_rows.append(
                {
                    "source_idx": int(source_idx),
                    "sensor_idx": int(sensor_idx),
                    "count": int(count),
                    "fraction": float(count / len(subset)),
                }
            )
        for sensor_idx, count in source_only_counter.most_common():
            source_only_sensor_rows.append(
                {
                    "source_idx": int(source_idx),
                    "sensor_idx": int(sensor_idx),
                    "count": int(count),
                    "fraction": float(count / len(subset)),
                }
            )
    sensor_summary = pd.DataFrame(sensor_rows).sort_values(
        ["source_idx", "count", "sensor_idx"],
        ascending=[True, False, True],
    )
    source_only_sensor_summary = pd.DataFrame(source_only_sensor_rows).sort_values(
        ["source_idx", "count", "sensor_idx"],
        ascending=[True, False, True],
    )

    segment_table.to_csv(output_dir / "segment_profiles.csv", index=False)
    source_summary.to_csv(output_dir / "source_summary.csv", index=False)
    subject_summary.to_csv(output_dir / "subject_summary.csv", index=False)
    label_summary.to_csv(output_dir / "label_summary.csv", index=False)
    code_summary.to_csv(output_dir / "code_summary.csv", index=False)
    sensor_summary.to_csv(output_dir / "sensor_summary.csv", index=False)
    source_only_sensor_summary.to_csv(output_dir / "source_only_sensor_summary.csv", index=False)

    save_json(
        output_dir / "summary.json",
        {
            "dataset": dataset,
            "num_segments_profiled": int(len(segment_table["path"].unique())),
            "source_indices": [int(source) for source in source_indices],
            "clip_seconds": clip_seconds,
            "top_rank_counts": {
                str(int(row["source_idx"])): int(row["top_rank_count"])
                for _, row in source_summary.iterrows()
            },
        },
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["pd31", "MEG-MMI", "all"], default="all")
    parser.add_argument("--sources", type=int, nargs="*", default=None)
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
    device = resolve_device(args.device)
    tokenizer = load_braintokenizer(device=str(device))

    for dataset in datasets:
        output_dir = profile_dataset(
            dataset=dataset,
            tokenizer=tokenizer,
            device=device,
            source_indices=dataset_targets(dataset, args.sources),
            max_segments=args.max_segments,
            clip_seconds=args.clip_seconds if args.clip_seconds is not None else default_clip_seconds(dataset),
        )
        print(f"{dataset} -> {output_dir}")


if __name__ == "__main__":
    main()
