#!/usr/bin/env python
"""Compare source-prototype intervention sensor shifts against decoder routing footprints."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from neurofm_mechinterp.paths import RESULTS_ROOT


def parse_sensor_set(value: str) -> tuple[int, ...]:
    """Parse a comma-separated sensor list into a stable tuple."""
    if pd.isna(value):
        return ()
    return tuple(int(sensor_idx) for sensor_idx in str(value).split(",") if sensor_idx)


def overlap_count(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    """Count set overlap between two sensor tuples."""
    return len(set(left) & set(right))


def jaccard(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    """Compute Jaccard overlap for two sensor tuples."""
    union = set(left) | set(right)
    if not union:
        return 0.0
    return float(len(set(left) & set(right)) / len(union))


def choose_source_idx(sensor_summary: pd.DataFrame, explicit_source_idx: int | None) -> int:
    """Resolve which tokenizer source should be compared against routing footprints."""
    if explicit_source_idx is not None:
        return int(explicit_source_idx)
    non_null = sensor_summary["source_idx"].dropna().unique().tolist()
    if len(non_null) != 1:
        raise ValueError("Provide `--source-idx` when multiple or no `source_idx` values are present.")
    return int(non_null[0])


def prototype_top_sensor_rows(sensor_summary: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Keep the top-k sensors for each source-prototype intervention condition."""
    rows = []
    group_columns = ["source_idx", "mode", "scale"]
    for group_key, subset in sensor_summary.groupby(group_columns):
        top = subset.sort_values(["count", "sensor_idx"], ascending=[False, True]).head(top_k)
        sensors = tuple(int(sensor_idx) for sensor_idx in top["sensor_idx"].tolist())
        source_idx, mode, scale = group_key
        rows.append(
            {
                "source_idx": int(source_idx),
                "mode": str(mode),
                "scale": float(scale),
                "prototype_top_sensors": ",".join(str(sensor_idx) for sensor_idx in sensors),
                "prototype_sensor_tuple": sensors,
            }
        )
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset", choices=["pd31", "MEG-MMI"], required=True)
    parser.add_argument("--source-idx", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    sensor_summary = pd.read_csv(run_dir / "sensor_summary.csv")
    source_idx = choose_source_idx(sensor_summary, args.source_idx)

    routing_dir = RESULTS_ROOT / "decoder_routing" / args.dataset
    aggregate_routing = pd.read_csv(routing_dir / "routing_overlap_summary.csv")
    head_routing = pd.read_csv(routing_dir / "head_routing_overlap_summary.csv")

    aggregate_row = aggregate_routing[aggregate_routing["source_idx"] == source_idx]
    if aggregate_row.empty:
        raise ValueError(f"No aggregate routing summary found for source {source_idx}.")
    aggregate_row = aggregate_row.iloc[0]

    head_rows = head_routing[head_routing["source_idx"] == source_idx].copy()
    if head_rows.empty:
        raise ValueError(f"No head routing summary found for source {source_idx}.")

    aggregate_attention = parse_sensor_set(aggregate_row["attention_top_sensors"])
    aggregate_contribution = parse_sensor_set(aggregate_row["contribution_top_sensors"])
    aggregate_damage = parse_sensor_set(aggregate_row.get("damage_top_sensors"))
    aggregate_source_only = parse_sensor_set(aggregate_row.get("source_only_top_sensors"))

    prototype_rows = prototype_top_sensor_rows(sensor_summary, top_k=args.top_k)

    per_head_rows = []
    per_condition_rows = []
    for _, prototype_row in prototype_rows.iterrows():
        prototype_tuple = tuple(prototype_row["prototype_sensor_tuple"])
        best_attention = None
        best_contribution = None

        for _, head_row in head_rows.iterrows():
            attention_tuple = parse_sensor_set(head_row["attention_top_sensors"])
            contribution_tuple = parse_sensor_set(head_row["contribution_top_sensors"])
            current = {
                "source_idx": int(prototype_row["source_idx"]),
                "mode": str(prototype_row["mode"]),
                "scale": float(prototype_row["scale"]),
                "head_idx": int(head_row["head_idx"]),
                "prototype_top_sensors": prototype_row["prototype_top_sensors"],
                "attention_top_sensors": ",".join(str(sensor_idx) for sensor_idx in attention_tuple),
                "contribution_top_sensors": ",".join(str(sensor_idx) for sensor_idx in contribution_tuple),
                "attention_overlap_count": overlap_count(prototype_tuple, attention_tuple),
                "attention_jaccard": jaccard(prototype_tuple, attention_tuple),
                "contribution_overlap_count": overlap_count(prototype_tuple, contribution_tuple),
                "contribution_jaccard": jaccard(prototype_tuple, contribution_tuple),
            }
            per_head_rows.append(current)

            attention_key = (current["attention_overlap_count"], current["attention_jaccard"], -current["head_idx"])
            contribution_key = (current["contribution_overlap_count"], current["contribution_jaccard"], -current["head_idx"])
            if best_attention is None or attention_key > best_attention[0]:
                best_attention = (attention_key, current)
            if best_contribution is None or contribution_key > best_contribution[0]:
                best_contribution = (contribution_key, current)

        per_condition_rows.append(
            {
                "source_idx": int(prototype_row["source_idx"]),
                "mode": str(prototype_row["mode"]),
                "scale": float(prototype_row["scale"]),
                "prototype_top_sensors": prototype_row["prototype_top_sensors"],
                "aggregate_attention_top_sensors": ",".join(str(sensor_idx) for sensor_idx in aggregate_attention),
                "aggregate_contribution_top_sensors": ",".join(str(sensor_idx) for sensor_idx in aggregate_contribution),
                "aggregate_damage_top_sensors": ",".join(str(sensor_idx) for sensor_idx in aggregate_damage),
                "aggregate_source_only_top_sensors": ",".join(str(sensor_idx) for sensor_idx in aggregate_source_only),
                "aggregate_attention_overlap_count": overlap_count(prototype_tuple, aggregate_attention),
                "aggregate_attention_jaccard": jaccard(prototype_tuple, aggregate_attention),
                "aggregate_contribution_overlap_count": overlap_count(prototype_tuple, aggregate_contribution),
                "aggregate_contribution_jaccard": jaccard(prototype_tuple, aggregate_contribution),
                "aggregate_damage_overlap_count": overlap_count(prototype_tuple, aggregate_damage),
                "aggregate_damage_jaccard": jaccard(prototype_tuple, aggregate_damage),
                "aggregate_source_only_overlap_count": overlap_count(prototype_tuple, aggregate_source_only),
                "aggregate_source_only_jaccard": jaccard(prototype_tuple, aggregate_source_only),
                "best_attention_head_idx": int(best_attention[1]["head_idx"]),
                "best_attention_head_overlap_count": int(best_attention[1]["attention_overlap_count"]),
                "best_attention_head_jaccard": float(best_attention[1]["attention_jaccard"]),
                "best_contribution_head_idx": int(best_contribution[1]["head_idx"]),
                "best_contribution_head_overlap_count": int(best_contribution[1]["contribution_overlap_count"]),
                "best_contribution_head_jaccard": float(best_contribution[1]["contribution_jaccard"]),
            }
        )

    pd.DataFrame(per_condition_rows).sort_values(
        ["aggregate_source_only_overlap_count", "aggregate_damage_overlap_count", "mode", "scale"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True).to_csv(run_dir / "routing_overlap_summary.csv", index=False)
    pd.DataFrame(per_head_rows).sort_values(
        ["mode", "scale", "contribution_overlap_count", "head_idx"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True).to_csv(run_dir / "head_routing_overlap_summary.csv", index=False)
    print(run_dir)


if __name__ == "__main__":
    main()
