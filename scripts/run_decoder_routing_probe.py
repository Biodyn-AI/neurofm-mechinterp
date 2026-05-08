#!/usr/bin/env python
"""Probe decoder-side sensor-to-source routing in the BrainTokenizer."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from einops import rearrange

from neurofm_mechinterp.loading import load_braintokenizer, resolve_device
from neurofm_mechinterp.paths import RESULTS_ROOT, REPO_ROOT, ensure_dir
from neurofm_mechinterp.processed_data import choose_examples, load_batch_from_examples, load_metadata
from neurofm_mechinterp.analysis.common import save_json


DATASET_METADATA = {
    "pd31": REPO_ROOT / "share" / "metadata" / "evaluate" / "pd31" / "BrainOmni" / "info.json",
    "MEG-MMI": REPO_ROOT / "share" / "metadata" / "evaluate" / "MEG-MMI" / "BrainOmni" / "info.json",
}

DEFAULT_TARGET_SOURCES = {
    "pd31": [3, 2],
    "MEG-MMI": [2, 11],
}


def default_clip_seconds(dataset: str) -> float | None:
    """Use the same practical MEG crop as the other real-data analyses."""
    if dataset == "MEG-MMI":
        return 10.0
    return None


def dataset_targets(dataset: str, sources: list[int] | None) -> list[int]:
    """Resolve the probed sources for one dataset."""
    if sources:
        return [int(source) for source in sources]
    return DEFAULT_TARGET_SOURCES[dataset]


def external_top_sensor_sets(dataset: str, filename: str, top_k: int = 5) -> dict[int, set[int]]:
    """Load the top aggregated sensors from an existing profile summary if available."""
    path = RESULTS_ROOT / "source_profiles" / dataset / filename
    if not path.exists():
        return {}

    table = pd.read_csv(path)
    result = {}
    for source_idx, subset in table.groupby("source_idx"):
        top_sensors = subset.head(top_k)["sensor_idx"].astype(int).tolist()
        result[int(source_idx)] = set(top_sensors)
    return result


def summarize_sensor_counts(
    segment_table: pd.DataFrame,
    column: str,
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Count how often each sensor appears in a top-sensor list."""
    group_columns = group_columns or ["source_idx"]
    if segment_table.empty:
        return pd.DataFrame(columns=[*group_columns, "sensor_idx", "count", "fraction"])
    rows = []
    for group_key, subset in segment_table.groupby(group_columns):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        counter = Counter()
        for value in subset[column]:
            counter.update(int(index) for index in str(value).split(",") if index)
        for sensor_idx, count in counter.most_common():
            row = {
                column_name: group_value
                for column_name, group_value in zip(group_columns, group_key)
            }
            row.update(
                {
                    "sensor_idx": int(sensor_idx),
                    "count": int(count),
                    "fraction": float(count / len(subset)),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        [*group_columns, "count", "sensor_idx"],
        ascending=[*[True] * len(group_columns), False, True],
    )


def top_sensor_set(
    sensor_summary: pd.DataFrame,
    top_k: int = 5,
    **filters,
) -> set[int]:
    """Return the top aggregated sensors from a summary table under explicit filters."""
    subset = sensor_summary
    for column_name, value in filters.items():
        subset = subset[subset[column_name] == value]
    if subset.empty:
        return set()
    return set(subset.head(top_k)["sensor_idx"].astype(int).tolist())


def compute_decoder_routing(
    tokenizer,
    quantized_feature: torch.Tensor,
    sensor_embedding: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute aggregate and per-head decoder routing maps per sensor/source."""
    forwardsolution = tokenizer.decoder.forwardsolution
    _, _, _, _, embedding_dim = quantized_feature.shape
    head_dim = embedding_dim // forwardsolution.n_head

    neurons = rearrange(quantized_feature, "B S N T D -> (B N T) S D")
    repeated_sensor_embedding = rearrange(
        sensor_embedding.view(sensor_embedding.shape[0], sensor_embedding.shape[1], 1, 1, embedding_dim).repeat(
            1,
            1,
            quantized_feature.shape[2],
            quantized_feature.shape[3],
            1,
        ),
        "B C N T D -> (B N T) C D",
    )

    kv = forwardsolution.kv(neurons)
    keys, values = torch.split(kv, split_size_or_sections=embedding_dim, dim=-1)
    queries = rearrange(repeated_sensor_embedding, "B C (H D) -> B H C D", H=forwardsolution.n_head)
    keys = rearrange(keys, "B S (H D) -> B H S D", H=forwardsolution.n_head)
    values = rearrange(values, "B S (H D) -> B H S D", H=forwardsolution.n_head)

    logits = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(head_dim)
    attention = logits.softmax(dim=-1)
    contribution = attention.unsqueeze(-1) * values.unsqueeze(2)
    contribution_norm = contribution.norm(dim=-1)

    mean_attention = attention.mean(dim=(0, 1))
    mean_contribution_norm = contribution_norm.mean(dim=(0, 1))
    head_attention = attention.mean(dim=0)
    head_contribution_norm = contribution_norm.mean(dim=0)
    return mean_attention, mean_contribution_norm, head_attention, head_contribution_norm


@torch.no_grad()
def probe_dataset(
    dataset: str,
    tokenizer,
    device,
    source_indices: list[int],
    max_segments: int | None,
    clip_seconds: float | None,
) -> Path:
    """Run the decoder routing probe across one dataset."""
    metadata = load_metadata(DATASET_METADATA[dataset])
    if max_segments is not None:
        metadata = choose_examples(metadata, max_segments=max_segments)

    output_dir = ensure_dir(RESULTS_ROOT / "decoder_routing" / dataset)
    segment_rows = []
    head_segment_rows = []

    for row in metadata:
        batch = load_batch_from_examples([row]).to(device)
        if clip_seconds is not None:
            clip_samples = int(clip_seconds * 256)
            batch.x = batch.x[..., :clip_samples]

        unfolded = tokenizer.unfold(batch.x)
        sensor_embedding = tokenizer.sensor_embed(batch.pos, batch.sensor_type)
        feature = tokenizer.encoder(unfolded, sensor_embedding)
        quantized_feature, _, _ = tokenizer.quantizer(feature)

        (
            attention_by_sensor_source,
            contribution_by_sensor_source,
            head_attention_by_sensor_source,
            head_contribution_by_sensor_source,
        ) = compute_decoder_routing(
            tokenizer,
            quantized_feature,
            sensor_embedding,
        )

        for source_idx in source_indices:
            attention_values = attention_by_sensor_source[:, source_idx]
            contribution_values = contribution_by_sensor_source[:, source_idx]
            top_attention_sensors = torch.topk(
                attention_values,
                k=min(5, attention_values.shape[0]),
            ).indices.cpu().tolist()
            top_contribution_sensors = torch.topk(
                contribution_values,
                k=min(5, contribution_values.shape[0]),
            ).indices.cpu().tolist()

            segment_rows.append(
                {
                    "dataset": dataset,
                    "subject": row["subject"],
                    "label": int(row["label"]),
                    "path": row["path"],
                    "channels": int(row["channels"]),
                    "clip_seconds": clip_seconds,
                    "source_idx": int(source_idx),
                    "mean_attention_weight": float(attention_values.mean().item()),
                    "max_attention_weight": float(attention_values.max().item()),
                    "mean_contribution_norm": float(contribution_values.mean().item()),
                    "max_contribution_norm": float(contribution_values.max().item()),
                    "top_attention_sensor_indices": ",".join(str(index) for index in top_attention_sensors),
                    "top_contribution_sensor_indices": ",".join(str(index) for index in top_contribution_sensors),
                }
            )

            for head_idx in range(head_attention_by_sensor_source.shape[0]):
                head_attention_values = head_attention_by_sensor_source[head_idx, :, source_idx]
                head_contribution_values = head_contribution_by_sensor_source[head_idx, :, source_idx]
                head_top_attention_sensors = torch.topk(
                    head_attention_values,
                    k=min(5, head_attention_values.shape[0]),
                ).indices.cpu().tolist()
                head_top_contribution_sensors = torch.topk(
                    head_contribution_values,
                    k=min(5, head_contribution_values.shape[0]),
                ).indices.cpu().tolist()

                head_segment_rows.append(
                    {
                        "dataset": dataset,
                        "subject": row["subject"],
                        "label": int(row["label"]),
                        "path": row["path"],
                        "channels": int(row["channels"]),
                        "clip_seconds": clip_seconds,
                        "source_idx": int(source_idx),
                        "head_idx": int(head_idx),
                        "mean_attention_weight": float(head_attention_values.mean().item()),
                        "max_attention_weight": float(head_attention_values.max().item()),
                        "mean_contribution_norm": float(head_contribution_values.mean().item()),
                        "max_contribution_norm": float(head_contribution_values.max().item()),
                        "top_attention_sensor_indices": ",".join(str(index) for index in head_top_attention_sensors),
                        "top_contribution_sensor_indices": ",".join(str(index) for index in head_top_contribution_sensors),
                    }
                )

    segment_table = pd.DataFrame(segment_rows).sort_values(["source_idx", "subject", "path"]).reset_index(drop=True)
    head_segment_table = pd.DataFrame(head_segment_rows).sort_values(
        ["source_idx", "head_idx", "subject", "path"]
    ).reset_index(drop=True)

    source_summary = (
        segment_table.groupby("source_idx", as_index=False)
        .agg(
            segments_profiled=("path", "count"),
            mean_attention_weight=("mean_attention_weight", "mean"),
            mean_max_attention_weight=("max_attention_weight", "mean"),
            mean_contribution_norm=("mean_contribution_norm", "mean"),
            mean_max_contribution_norm=("max_contribution_norm", "mean"),
        )
        .sort_values(["mean_contribution_norm", "mean_attention_weight"], ascending=[False, False])
        .reset_index(drop=True)
    )

    subject_summary = (
        segment_table.groupby(["source_idx", "subject"], as_index=False)
        .agg(
            segments=("path", "count"),
            mean_attention_weight=("mean_attention_weight", "mean"),
            mean_contribution_norm=("mean_contribution_norm", "mean"),
        )
        .sort_values(["source_idx", "subject"])
        .reset_index(drop=True)
    )

    label_summary = (
        segment_table.groupby(["source_idx", "label"], as_index=False)
        .agg(
            segments=("path", "count"),
            mean_attention_weight=("mean_attention_weight", "mean"),
            mean_contribution_norm=("mean_contribution_norm", "mean"),
        )
        .sort_values(["source_idx", "label"])
        .reset_index(drop=True)
    )

    attention_sensor_summary = summarize_sensor_counts(segment_table, "top_attention_sensor_indices")
    contribution_sensor_summary = summarize_sensor_counts(segment_table, "top_contribution_sensor_indices")

    head_summary = (
        head_segment_table.groupby(["source_idx", "head_idx"], as_index=False)
        .agg(
            segments_profiled=("path", "count"),
            mean_attention_weight=("mean_attention_weight", "mean"),
            mean_max_attention_weight=("max_attention_weight", "mean"),
            mean_contribution_norm=("mean_contribution_norm", "mean"),
            mean_max_contribution_norm=("max_contribution_norm", "mean"),
        )
        .sort_values(["source_idx", "mean_contribution_norm", "mean_attention_weight"], ascending=[True, False, False])
        .reset_index(drop=True)
    )
    head_attention_sensor_summary = summarize_sensor_counts(
        head_segment_table,
        "top_attention_sensor_indices",
        group_columns=["source_idx", "head_idx"],
    )
    head_contribution_sensor_summary = summarize_sensor_counts(
        head_segment_table,
        "top_contribution_sensor_indices",
        group_columns=["source_idx", "head_idx"],
    )

    damage_sets = external_top_sensor_sets(dataset, "sensor_summary.csv")
    source_only_sets = external_top_sensor_sets(dataset, "source_only_sensor_summary.csv")
    overlap_rows = []
    for source_idx in source_summary["source_idx"].astype(int).tolist():
        attention_set = top_sensor_set(attention_sensor_summary, source_idx=source_idx)
        contribution_set = top_sensor_set(contribution_sensor_summary, source_idx=source_idx)
        damage_set = damage_sets.get(source_idx, set())
        source_only_set = source_only_sets.get(source_idx, set())
        overlap_rows.append(
            {
                "source_idx": int(source_idx),
                "attention_top_sensors": ",".join(str(sensor) for sensor in sorted(attention_set)),
                "contribution_top_sensors": ",".join(str(sensor) for sensor in sorted(contribution_set)),
                "damage_top_sensors": ",".join(str(sensor) for sensor in sorted(damage_set)),
                "source_only_top_sensors": ",".join(str(sensor) for sensor in sorted(source_only_set)),
                "attention_damage_overlap": int(len(attention_set & damage_set)),
                "attention_source_only_overlap": int(len(attention_set & source_only_set)),
                "contribution_damage_overlap": int(len(contribution_set & damage_set)),
                "contribution_source_only_overlap": int(len(contribution_set & source_only_set)),
            }
        )
    overlap_summary = pd.DataFrame(overlap_rows).sort_values("source_idx").reset_index(drop=True)

    head_overlap_rows = []
    for row in head_summary.itertuples(index=False):
        source_idx = int(row.source_idx)
        head_idx = int(row.head_idx)
        attention_set = top_sensor_set(
            head_attention_sensor_summary,
            source_idx=source_idx,
            head_idx=head_idx,
        )
        contribution_set = top_sensor_set(
            head_contribution_sensor_summary,
            source_idx=source_idx,
            head_idx=head_idx,
        )
        damage_set = damage_sets.get(source_idx, set())
        source_only_set = source_only_sets.get(source_idx, set())
        head_overlap_rows.append(
            {
                "source_idx": source_idx,
                "head_idx": head_idx,
                "attention_top_sensors": ",".join(str(sensor) for sensor in sorted(attention_set)),
                "contribution_top_sensors": ",".join(str(sensor) for sensor in sorted(contribution_set)),
                "damage_top_sensors": ",".join(str(sensor) for sensor in sorted(damage_set)),
                "source_only_top_sensors": ",".join(str(sensor) for sensor in sorted(source_only_set)),
                "attention_damage_overlap": int(len(attention_set & damage_set)),
                "attention_source_only_overlap": int(len(attention_set & source_only_set)),
                "contribution_damage_overlap": int(len(contribution_set & damage_set)),
                "contribution_source_only_overlap": int(len(contribution_set & source_only_set)),
            }
        )
    head_overlap_summary = (
        pd.DataFrame(head_overlap_rows)
        .sort_values(["source_idx", "head_idx"])
        .reset_index(drop=True)
    )

    segment_table.to_csv(output_dir / "segment_routing_profiles.csv", index=False)
    source_summary.to_csv(output_dir / "source_summary.csv", index=False)
    subject_summary.to_csv(output_dir / "subject_summary.csv", index=False)
    label_summary.to_csv(output_dir / "label_summary.csv", index=False)
    attention_sensor_summary.to_csv(output_dir / "attention_sensor_summary.csv", index=False)
    contribution_sensor_summary.to_csv(output_dir / "contribution_sensor_summary.csv", index=False)
    overlap_summary.to_csv(output_dir / "routing_overlap_summary.csv", index=False)
    head_segment_table.to_csv(output_dir / "head_segment_routing_profiles.csv", index=False)
    head_summary.to_csv(output_dir / "head_summary.csv", index=False)
    head_attention_sensor_summary.to_csv(output_dir / "head_attention_sensor_summary.csv", index=False)
    head_contribution_sensor_summary.to_csv(output_dir / "head_contribution_sensor_summary.csv", index=False)
    head_overlap_summary.to_csv(output_dir / "head_routing_overlap_summary.csv", index=False)

    save_json(
        output_dir / "summary.json",
        {
            "dataset": dataset,
            "num_segments_profiled": int(len(segment_table["path"].unique())),
            "source_indices": [int(source) for source in source_indices],
            "clip_seconds": clip_seconds,
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
        help="Optionally truncate segments. Defaults to 10 seconds for MEG probes.",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    datasets = ["pd31", "MEG-MMI"] if args.dataset == "all" else [args.dataset]
    device = resolve_device(args.device)
    tokenizer = load_braintokenizer(device=str(device))

    for dataset in datasets:
        output_dir = probe_dataset(
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
