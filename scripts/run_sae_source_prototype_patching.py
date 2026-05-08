#!/usr/bin/env python
"""Patch source-prototype projections between segments and measure reconstruction shifts."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from neurofm_mechinterp.analysis.common import save_json
from neurofm_mechinterp.loading import load_braintokenizer, resolve_device
from neurofm_mechinterp.paths import REPO_ROOT, ensure_dir
from neurofm_mechinterp.processed_data import choose_examples, load_batch_from_examples, load_metadata
from neurofm_mechinterp.sae import (
    choose_ablation_row_mask,
    compute_tokenizer_activations,
    flatten_source_activations,
    resolve_row_source_idx,
    restore_source_activations,
)
try:
    from scripts.run_sae_source_prototype_interventions import (
        default_clip_seconds,
        load_sae,
        load_source_prototype_bundle,
    )
except ModuleNotFoundError:
    from run_sae_source_prototype_interventions import (  # type: ignore[no-redef]
        default_clip_seconds,
        load_sae,
        load_source_prototype_bundle,
    )


DATASET_METADATA = {
    "pd31": REPO_ROOT / "share" / "metadata" / "evaluate" / "pd31" / "BrainOmni" / "info.json",
    "MEG-MMI": REPO_ROOT / "share" / "metadata" / "evaluate" / "MEG-MMI" / "BrainOmni" / "info.json",
}


@dataclass
class SegmentCache:
    """Minimal per-segment state needed for source-prototype patching."""

    row: dict
    standardized_rows: torch.Tensor
    row_source_idx: torch.Tensor
    hook_shape: tuple[int, int, int, int, int]
    baseline_reconstruction: torch.Tensor
    baseline_input_mse: float
    projection: torch.Tensor
    mean_projection: float
    mean_abs_projection: float


def choose_patch_pairs(score_table: pd.DataFrame, num_pairs: int) -> list[tuple[str, str]]:
    """Pair high-projection donors with low-projection recipients."""
    donors = score_table.sort_values(["mean_projection", "path"], ascending=[False, True])
    recipients = score_table.sort_values(["mean_projection", "path"], ascending=[True, True])

    pairs: list[tuple[str, str]] = []
    used_recipients: set[str] = set()
    for _, donor_row in donors.iterrows():
        donor_path = str(donor_row["path"])
        for _, recipient_row in recipients.iterrows():
            recipient_path = str(recipient_row["path"])
            if recipient_path == donor_path or recipient_path in used_recipients:
                continue
            pairs.append((donor_path, recipient_path))
            used_recipients.add(recipient_path)
            break
        if len(pairs) >= num_pairs:
            break
    return pairs


def load_explicit_pairs(path: Path) -> list[tuple[str, str]]:
    """Load an explicit donor/recipient patch list from CSV."""
    table = pd.read_csv(path)
    required = {"donor_path", "recipient_path"}
    if not required.issubset(table.columns):
        raise ValueError(f"Explicit pair CSV must contain columns {sorted(required)}.")
    return [(str(row["donor_path"]), str(row["recipient_path"])) for _, row in table.iterrows()]


def apply_source_prototype_patch(
    standardized_rows: torch.Tensor,
    donor_projection: torch.Tensor,
    prototype: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Move each row's prototype projection toward the donor row's projection."""
    recipient_projection = standardized_rows @ prototype
    projection_delta = scale * (donor_projection - recipient_projection)
    return standardized_rows + projection_delta.unsqueeze(-1) * prototype.unsqueeze(0)


def decompose_source_prototype_rows(
    standardized_rows: torch.Tensor,
    prototype: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split source rows into prototype-parallel and prototype-orthogonal parts."""
    projection = standardized_rows @ prototype
    parallel = projection.unsqueeze(-1) * prototype.unsqueeze(0)
    residual = standardized_rows - parallel
    return projection, parallel, residual


def apply_source_component_patch(
    recipient_rows: torch.Tensor,
    donor_rows: torch.Tensor,
    prototype: torch.Tensor,
    component: str,
    scale: float,
) -> torch.Tensor:
    """Patch one source-row component from donor to recipient."""
    donor_projection, donor_parallel, donor_residual = decompose_source_prototype_rows(donor_rows, prototype)
    _, recipient_parallel, recipient_residual = decompose_source_prototype_rows(recipient_rows, prototype)

    if component == "projection":
        return apply_source_prototype_patch(
            standardized_rows=recipient_rows,
            donor_projection=donor_projection,
            prototype=prototype,
            scale=scale,
        )
    if component == "residual":
        return recipient_parallel + (1.0 - scale) * recipient_residual + scale * donor_residual
    if component == "full_rows":
        return (1.0 - scale) * recipient_rows + scale * donor_rows
    raise ValueError(f"Unsupported patch component: {component}")


def summarize_sensor_counts(pair_table: pd.DataFrame) -> pd.DataFrame:
    """Count how often each sensor appears in the top patch-delta sensor set."""
    counter = Counter()
    for value in pair_table["top_delta_sensor_indices"]:
        counter.update(int(index) for index in str(value).split(",") if index)

    rows = []
    for sensor_idx, count in counter.most_common():
        rows.append(
            {
                "sensor_idx": int(sensor_idx),
                "count": int(count),
                "fraction": float(count / len(pair_table)),
            }
        )
    return pd.DataFrame(rows)


def build_run_summary(pair_table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one row summarizing the prototype patch run."""
    return pd.DataFrame(
        [
            {
                "pairs_profiled": int(len(pair_table)),
                "mean_donor_projection": float(pair_table["donor_mean_projection"].mean()),
                "mean_recipient_projection": float(pair_table["recipient_mean_projection"].mean()),
                "mean_patched_projection": float(pair_table["patched_mean_projection"].mean()),
                "mean_projection_gap": float((pair_table["donor_mean_projection"] - pair_table["recipient_mean_projection"]).mean()),
                "mean_parallel_delta_energy": float(pair_table["mean_parallel_delta_energy"].mean()),
                "mean_residual_delta_energy": float(pair_table["mean_residual_delta_energy"].mean()),
                "mean_residual_delta_fraction": float(pair_table["residual_delta_fraction"].mean()),
                "mean_delta_input_mse": float(pair_table["delta_input_mse"].mean()),
                "mean_cosine_to_recipient_baseline": float(pair_table["cosine_to_recipient_baseline"].mean()),
                "mean_cosine_patch_delta_to_donor_delta": float(pair_table["cosine_patch_delta_to_donor_delta"].mean()),
                "same_subject_fraction": float(pair_table["same_subject"].mean()),
            }
        ]
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset", choices=["pd31", "MEG-MMI"], required=True)
    parser.add_argument("--source-idx", type=int, required=True)
    parser.add_argument("--row-source-idx", type=int, default=None)
    parser.add_argument("--component", choices=["projection", "residual", "full_rows"], default="projection")
    parser.add_argument("--pair-csv", default=None, help="Optional CSV with donor_path and recipient_path columns.")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--num-pairs", type=int, default=5)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--clip-seconds", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-suffix", default=None)
    parser.add_argument("--metadata-path", default=None, help="Override info.json path.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    _, checkpoint, mean, std = load_sae(model_dir)
    hook = checkpoint["config"]["hook"]
    row_source_idx = resolve_row_source_idx(args.source_idx, args.row_source_idx)

    prototype_bundle = load_source_prototype_bundle(
        model_dir=model_dir,
        source_idx=args.source_idx,
        batch_size=args.batch_size,
    )
    prototype = prototype_bundle["prototype"]

    device = resolve_device(args.device)
    tokenizer = load_braintokenizer(device=str(device))
    mean = mean.to(device)
    std = std.to(device)
    prototype = prototype.to(device)

    metadata_path = Path(args.metadata_path) if args.metadata_path else DATASET_METADATA[args.dataset]
    metadata = load_metadata(metadata_path)
    if args.max_segments is not None:
        metadata = choose_examples(metadata, max_segments=args.max_segments)

    clip_seconds = args.clip_seconds if args.clip_seconds is not None else default_clip_seconds(args.dataset)

    run_name = f"source_prototype_patch_source_{args.source_idx}"
    if row_source_idx is not None:
        run_name = f"{run_name}_rows_{row_source_idx}"
    if args.component != "projection":
        run_name = f"{run_name}_{args.component}"
    run_name = f"{run_name}_scale_{args.scale:g}"
    if args.output_suffix:
        run_name = f"{run_name}_{args.output_suffix}"
    output_dir = ensure_dir(model_dir / run_name)

    segment_cache: dict[str, SegmentCache] = {}
    score_rows = []
    for row in metadata:
        batch = load_batch_from_examples([row]).to(device)
        if clip_seconds is not None:
            clip_samples = int(clip_seconds * 256)
            batch.x = batch.x[..., :clip_samples]

        unfolded = tokenizer.unfold(batch.x)
        sensor_embedding = tokenizer.sensor_embed(batch.pos, batch.sensor_type)
        hook_feature = compute_tokenizer_activations(tokenizer, batch, hook)
        baseline_quantized_feature = hook_feature if hook == "encoder_post_quantizer" else tokenizer.quantizer(hook_feature)[0]
        baseline_reconstruction = tokenizer.decoder(baseline_quantized_feature, sensor_embedding)
        baseline_input_mse = float(((baseline_reconstruction - unfolded) ** 2).mean().item())

        rows = flatten_source_activations(hook_feature)
        standardized_rows = (rows.activations - mean) / std
        row_mask = choose_ablation_row_mask(rows.source_idx, row_source_idx)
        eligible_rows = standardized_rows[row_mask]
        if eligible_rows.shape[0] == 0:
            continue

        projection = eligible_rows @ prototype
        mean_projection = float(projection.mean().item())
        mean_abs_projection = float(projection.abs().mean().item())
        segment_cache[row["path"]] = SegmentCache(
            row=row,
            standardized_rows=standardized_rows.cpu(),
            row_source_idx=rows.source_idx.cpu(),
            hook_shape=tuple(hook_feature.shape),
            baseline_reconstruction=baseline_reconstruction.cpu(),
            baseline_input_mse=baseline_input_mse,
            projection=projection.cpu(),
            mean_projection=mean_projection,
            mean_abs_projection=mean_abs_projection,
        )
        score_rows.append(
            {
                "dataset": args.dataset,
                "path": row["path"],
                "subject": row["subject"],
                "label": int(row["label"]),
                "source_idx": int(args.source_idx),
                "row_source_idx": row_source_idx,
                "mean_projection": mean_projection,
                "mean_abs_projection": mean_abs_projection,
            }
        )

    score_table = pd.DataFrame(score_rows).sort_values(["mean_projection", "path"], ascending=[False, True]).reset_index(drop=True)
    score_table.to_csv(output_dir / "segment_projection_scores.csv", index=False)

    if args.pair_csv is not None:
        patch_pairs = load_explicit_pairs(Path(args.pair_csv))
    else:
        patch_pairs = choose_patch_pairs(score_table, args.num_pairs)

    pair_rows = []
    for pair_idx, (donor_path, recipient_path) in enumerate(patch_pairs, start=1):
        donor = segment_cache[donor_path]
        recipient = segment_cache[recipient_path]
        if donor.hook_shape != recipient.hook_shape:
            raise ValueError("Donor and recipient hook activations must have the same shape for source-prototype patching.")
        if not torch.equal(donor.row_source_idx, recipient.row_source_idx):
            raise ValueError("Donor and recipient row source indices must align for source-prototype patching.")
        if donor.projection.shape != recipient.projection.shape:
            raise ValueError("Donor and recipient source-row counts must match for source-prototype patching.")

        recipient_batch = load_batch_from_examples([recipient.row]).to(device)
        if clip_seconds is not None:
            clip_samples = int(clip_seconds * 256)
            recipient_batch.x = recipient_batch.x[..., :clip_samples]

        unfolded = tokenizer.unfold(recipient_batch.x)
        sensor_embedding = tokenizer.sensor_embed(recipient_batch.pos, recipient_batch.sensor_type)

        recipient_rows = recipient.standardized_rows.to(device)
        donor_rows = donor.standardized_rows.to(device)
        row_mask = choose_ablation_row_mask(recipient.row_source_idx.to(device), row_source_idx)
        donor_subset = donor_rows[row_mask]
        recipient_subset = recipient_rows[row_mask]
        donor_projection, donor_parallel, donor_residual = decompose_source_prototype_rows(donor_subset, prototype)
        recipient_projection, recipient_parallel, recipient_residual = decompose_source_prototype_rows(recipient_subset, prototype)

        patched_subset = apply_source_component_patch(
            recipient_rows=recipient_subset,
            donor_rows=donor_subset,
            prototype=prototype,
            component=args.component,
            scale=args.scale,
        )
        patched_projection, _, patched_residual = decompose_source_prototype_rows(patched_subset, prototype)

        parallel_delta_energy = float((donor_parallel - recipient_parallel).pow(2).mean().item())
        residual_delta_energy = float((donor_residual - recipient_residual).pow(2).mean().item())
        total_component_delta_energy = parallel_delta_energy + residual_delta_energy
        residual_delta_fraction = residual_delta_energy / max(total_component_delta_energy, 1e-12)

        patched_rows = recipient_rows.clone()
        patched_rows[row_mask] = patched_subset
        patched_hook_feature = restore_source_activations(patched_rows * std + mean, recipient.hook_shape)
        patched_quantized_feature = patched_hook_feature if hook == "encoder_post_quantizer" else tokenizer.quantizer(patched_hook_feature)[0]
        patched_reconstruction = tokenizer.decoder(patched_quantized_feature, sensor_embedding)

        recipient_baseline = recipient.baseline_reconstruction.to(device)
        donor_baseline = donor.baseline_reconstruction.to(device)
        patch_delta = patched_reconstruction - recipient_baseline
        donor_delta = donor_baseline - recipient_baseline
        sensor_delta_energy = patch_delta.pow(2).mean(dim=(0, 2, 3))
        top_sensors = torch.topk(sensor_delta_energy, k=min(5, sensor_delta_energy.shape[0])).indices.cpu().tolist()

        patched_input_mse = float(((patched_reconstruction - unfolded) ** 2).mean().item())
        cosine_to_recipient_baseline = float(
            F.cosine_similarity(
                patched_reconstruction.reshape(patched_reconstruction.shape[0], -1),
                recipient_baseline.reshape(recipient_baseline.shape[0], -1),
            ).mean().item()
        )
        cosine_patch_delta_to_donor_delta = float(
            F.cosine_similarity(
                patch_delta.reshape(patch_delta.shape[0], -1),
                donor_delta.reshape(donor_delta.shape[0], -1),
            ).mean().item()
        )

        pair_rows.append(
            {
                "pair_idx": int(pair_idx),
                "dataset": args.dataset,
                "source_idx": int(args.source_idx),
                "row_source_idx": row_source_idx,
                "component": args.component,
                "scale": float(args.scale),
                "donor_path": donor_path,
                "recipient_path": recipient_path,
                "donor_subject": donor.row["subject"],
                "recipient_subject": recipient.row["subject"],
                "same_subject": bool(donor.row["subject"] == recipient.row["subject"]),
                "donor_label": int(donor.row["label"]),
                "recipient_label": int(recipient.row["label"]),
                "donor_mean_projection": donor.mean_projection,
                "recipient_mean_projection": recipient.mean_projection,
                "patched_mean_projection": float(patched_projection.mean().item()),
                "donor_mean_abs_projection": donor.mean_abs_projection,
                "recipient_mean_abs_projection": recipient.mean_abs_projection,
                "patched_mean_abs_projection": float(patched_projection.abs().mean().item()),
                "donor_mean_residual_norm": float(donor_residual.norm(dim=-1).mean().item()),
                "recipient_mean_residual_norm": float(recipient_residual.norm(dim=-1).mean().item()),
                "patched_mean_residual_norm": float(patched_residual.norm(dim=-1).mean().item()),
                "mean_parallel_delta_energy": parallel_delta_energy,
                "mean_residual_delta_energy": residual_delta_energy,
                "residual_delta_fraction": residual_delta_fraction,
                "recipient_baseline_input_mse": recipient.baseline_input_mse,
                "patched_input_mse": patched_input_mse,
                "delta_input_mse": float(patched_input_mse - recipient.baseline_input_mse),
                "cosine_to_recipient_baseline": cosine_to_recipient_baseline,
                "cosine_patch_delta_to_donor_delta": cosine_patch_delta_to_donor_delta,
                "mean_patch_delta_energy": float(patch_delta.pow(2).mean().item()),
                "top_delta_sensor_indices": ",".join(str(index) for index in top_sensors),
            }
        )

    pair_table = pd.DataFrame(pair_rows).sort_values(["donor_mean_projection", "recipient_mean_projection"], ascending=[False, True]).reset_index(drop=True)
    sensor_table = summarize_sensor_counts(pair_table)
    run_summary = build_run_summary(pair_table)

    pair_table.to_csv(output_dir / "pair_summary.csv", index=False)
    sensor_table.to_csv(output_dir / "sensor_summary.csv", index=False)
    run_summary.to_csv(output_dir / "run_summary.csv", index=False)
    save_json(
        output_dir / "summary.json",
        {
            "dataset": args.dataset,
            "model_dir": str(model_dir),
            "source_idx": int(args.source_idx),
            "row_source_idx": row_source_idx,
            "component": args.component,
            "scale": float(args.scale),
            "num_pairs": int(pair_table.shape[0]),
            "prototype_activation_path": prototype_bundle["activation_path"],
            "prototype_num_rows": int(prototype_bundle["num_rows"]),
            "prototype_mean_projection": float(prototype_bundle["mean_projection"]),
            "prototype_mean_abs_projection": float(prototype_bundle["mean_abs_projection"]),
            "mean_parallel_delta_energy": float(pair_table["mean_parallel_delta_energy"].mean()),
            "mean_residual_delta_energy": float(pair_table["mean_residual_delta_energy"].mean()),
            "mean_residual_delta_fraction": float(pair_table["residual_delta_fraction"].mean()),
            "mean_delta_input_mse": float(pair_table["delta_input_mse"].mean()),
            "mean_cosine_patch_delta_to_donor_delta": float(pair_table["cosine_patch_delta_to_donor_delta"].mean()),
        },
    )
    print(output_dir)


if __name__ == "__main__":
    main()
