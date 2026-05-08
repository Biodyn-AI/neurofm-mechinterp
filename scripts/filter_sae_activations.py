#!/usr/bin/env python
"""Filter an exported SAE activation bundle down to one tokenizer source."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from neurofm_mechinterp.analysis.common import save_json
from neurofm_mechinterp.paths import RESULTS_ROOT, ensure_dir
from neurofm_mechinterp.sae import subset_activation_bundle


def default_output_dir(activation_path: Path, source_idx: int) -> Path:
    """Place filtered exports next to the original activation bundle."""
    return activation_path.parent / f"source_{source_idx}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-path", required=True, help="Path to an exported `activations.pt` bundle.")
    parser.add_argument("--source-idx", type=int, required=True, help="Original tokenizer source index to keep.")
    parser.add_argument("--output-dir", default=None, help="Directory to write `activations.pt` plus summaries.")
    parser.add_argument(
        "--keep-original-source-idx",
        action="store_true",
        help="Keep the original source labels instead of reindexing the kept source to `0`.",
    )
    args = parser.parse_args()

    activation_path = Path(args.activation_path)
    output_dir = ensure_dir(Path(args.output_dir) if args.output_dir else default_output_dir(activation_path, args.source_idx))
    bundle = torch.load(activation_path, map_location="cpu", weights_only=False)

    row_mask = bundle["source_idx"].long() == int(args.source_idx)
    if not bool(row_mask.any()):
        raise ValueError(f"No rows found for source {args.source_idx} in {activation_path}.")

    filtered = subset_activation_bundle(
        bundle,
        row_mask,
        reindex_source_idx=not args.keep_original_source_idx,
    )
    torch.save(filtered, output_dir / "activations.pt")

    norm = filtered["activations"].norm(dim=-1)
    segment_table = (
        pd.DataFrame(
            {
                "segment_idx": filtered["segment_idx"].tolist(),
                "subject": filtered["subject"],
                "label": filtered["label"].tolist(),
                "path": filtered["path"],
                "rows_exported": 1,
                "mean_activation_norm": norm.tolist(),
            }
        )
        .groupby(["segment_idx", "subject", "label", "path"], as_index=False)
        .agg(
            rows_exported=("rows_exported", "sum"),
            mean_activation_norm=("mean_activation_norm", "mean"),
            max_activation_norm=("mean_activation_norm", "max"),
        )
        .sort_values(["segment_idx"])
        .reset_index(drop=True)
    )
    source_summary = (
        pd.DataFrame(
            {
                "source_idx": filtered["source_idx"].tolist(),
                "norm": norm.tolist(),
            }
        )
        .groupby("source_idx", as_index=False)
        .agg(
            rows=("source_idx", "count"),
            mean_activation_norm=("norm", "mean"),
            max_activation_norm=("norm", "max"),
        )
        .sort_values(["rows", "source_idx"], ascending=[False, True])
        .reset_index(drop=True)
    )

    segment_table.to_csv(output_dir / "segment_summary.csv", index=False)
    source_summary.to_csv(output_dir / "source_summary.csv", index=False)
    save_json(
        output_dir / "summary.json",
        {
            "activation_path": str(activation_path),
            "output_dir": str(output_dir),
            "selected_source_idx": int(args.source_idx),
            "reindexed_source_idx": not bool(args.keep_original_source_idx),
            "num_rows": int(filtered["activations"].shape[0]),
            "num_segments": int(segment_table.shape[0]),
            "activation_dim": int(filtered["activations"].shape[1]),
        },
    )
    print(output_dir)


if __name__ == "__main__":
    main()
