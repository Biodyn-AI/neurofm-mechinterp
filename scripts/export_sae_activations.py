#!/usr/bin/env python
"""Export BrainTokenizer activations for sparse-autoencoder training."""

from __future__ import annotations

import argparse

import pandas as pd
import torch

from neurofm_mechinterp.analysis.common import save_json
from neurofm_mechinterp.loading import load_braintokenizer, resolve_device
from neurofm_mechinterp.paths import RESULTS_ROOT, REPO_ROOT, ensure_dir
from neurofm_mechinterp.processed_data import choose_examples, load_batch_from_examples, load_metadata
from neurofm_mechinterp.sae import compute_tokenizer_activations, flatten_source_activations, list_activation_hooks, select_row_indices


DATASET_METADATA = {
    "pd31": REPO_ROOT / "share" / "metadata" / "evaluate" / "pd31" / "BrainOmni" / "info.json",
    "MEG-MMI": REPO_ROOT / "share" / "metadata" / "evaluate" / "MEG-MMI" / "BrainOmni" / "info.json",
}


def default_clip_seconds(dataset: str) -> float | None:
    """Use the same practical MEG crop as the other real-data analyses."""
    if dataset == "MEG-MMI":
        return 10.0
    return None


@torch.no_grad()
def export_dataset(
    dataset: str,
    tokenizer,
    device,
    hook: str,
    max_segments: int | None,
    clip_seconds: float | None,
    max_rows_per_segment: int | None,
    seed: int,
    metadata_path: str | None = None,
    dataset_name_override: str | None = None,
) -> None:
    """Export one dataset/hook activation matrix and metadata bundle."""
    from pathlib import Path
    if metadata_path is None:
        metadata = load_metadata(DATASET_METADATA[dataset])
    else:
        metadata = load_metadata(Path(metadata_path))
    if max_segments is not None:
        metadata = choose_examples(metadata, max_segments=max_segments)

    output_name = dataset_name_override or dataset
    output_dir = ensure_dir(RESULTS_ROOT / "sae_exports" / output_name / hook)
    activation_chunks: list[torch.Tensor] = []
    source_chunks: list[torch.Tensor] = []
    patch_chunks: list[torch.Tensor] = []
    frame_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    segment_chunks: list[torch.Tensor] = []
    subject_rows: list[str] = []
    path_rows: list[str] = []
    segment_rows: list[dict] = []

    for segment_idx, row in enumerate(metadata):
        batch = load_batch_from_examples([row]).to(device)
        if clip_seconds is not None:
            clip_samples = int(clip_seconds * 256)
            batch.x = batch.x[..., :clip_samples]

        activations = compute_tokenizer_activations(tokenizer, batch, hook)
        rows = flatten_source_activations(activations)
        selected = select_row_indices(
            num_rows=rows.activations.shape[0],
            max_rows=max_rows_per_segment,
            seed=seed + segment_idx,
        )

        activations_cpu = rows.activations.index_select(0, selected).cpu()
        source_cpu = rows.source_idx.index_select(0, selected).cpu()
        patch_cpu = rows.patch_idx.index_select(0, selected).cpu()
        frame_cpu = rows.frame_idx.index_select(0, selected).cpu()
        label_cpu = torch.full((selected.numel(),), int(row["label"]), dtype=torch.long)
        segment_cpu = torch.full((selected.numel(),), segment_idx, dtype=torch.long)

        activation_chunks.append(activations_cpu)
        source_chunks.append(source_cpu)
        patch_chunks.append(patch_cpu)
        frame_chunks.append(frame_cpu)
        label_chunks.append(label_cpu)
        segment_chunks.append(segment_cpu)
        subject_rows.extend([row["subject"]] * selected.numel())
        path_rows.extend([row["path"]] * selected.numel())

        source_norm = activations_cpu.norm(dim=-1)
        segment_rows.append(
            {
                "dataset": dataset,
                "hook": hook,
                "segment_idx": segment_idx,
                "subject": row["subject"],
                "label": int(row["label"]),
                "path": row["path"],
                "rows_exported": int(selected.numel()),
                "activation_dim": int(activations_cpu.shape[-1]),
                "mean_activation_norm": float(source_norm.mean().item()),
                "max_activation_norm": float(source_norm.max().item()),
            }
        )

    if not activation_chunks:
        raise ValueError(f"No activations were exported for dataset '{dataset}'.")

    bundle = {
        "activations": torch.cat(activation_chunks, dim=0),
        "source_idx": torch.cat(source_chunks, dim=0),
        "patch_idx": torch.cat(patch_chunks, dim=0),
        "frame_idx": torch.cat(frame_chunks, dim=0),
        "label": torch.cat(label_chunks, dim=0),
        "segment_idx": torch.cat(segment_chunks, dim=0),
        "subject": subject_rows,
        "path": path_rows,
    }
    torch.save(bundle, output_dir / "activations.pt")

    segment_table = pd.DataFrame(segment_rows).sort_values(["segment_idx"]).reset_index(drop=True)
    source_summary = (
        pd.DataFrame(
            {
                "source_idx": bundle["source_idx"].tolist(),
                "norm": bundle["activations"].norm(dim=-1).tolist(),
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
            "dataset": dataset,
            "hook": hook,
            "num_segments": int(len(metadata)),
            "num_rows": int(bundle["activations"].shape[0]),
            "activation_dim": int(bundle["activations"].shape[1]),
            "clip_seconds": clip_seconds,
            "max_rows_per_segment": max_rows_per_segment,
            "seed": seed,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["pd31", "MEG-MMI", "all"], default="all")
    parser.add_argument("--hook", choices=list_activation_hooks(), default="encoder_pre_quantizer")
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=None,
        help="Optionally truncate segments. Defaults to 10 seconds for MEG exports.",
    )
    parser.add_argument("--max-rows-per-segment", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--metadata-path", default=None, help="Override info.json path.")
    parser.add_argument("--dataset-name-override", default=None, help="Output folder under sae_exports.")
    args = parser.parse_args()

    datasets = ["pd31", "MEG-MMI"] if args.dataset == "all" else [args.dataset]
    device = resolve_device(args.device)
    tokenizer = load_braintokenizer(device=str(device))

    for dataset in datasets:
        export_dataset(
            dataset=dataset,
            tokenizer=tokenizer,
            device=device,
            hook=args.hook,
            max_segments=args.max_segments,
            clip_seconds=args.clip_seconds if args.clip_seconds is not None else default_clip_seconds(dataset),
            max_rows_per_segment=args.max_rows_per_segment,
            seed=args.seed,
            metadata_path=args.metadata_path,
            dataset_name_override=args.dataset_name_override,
        )
        out_name = args.dataset_name_override or dataset
        print(RESULTS_ROOT / "sae_exports" / out_name / args.hook)


if __name__ == "__main__":
    main()
