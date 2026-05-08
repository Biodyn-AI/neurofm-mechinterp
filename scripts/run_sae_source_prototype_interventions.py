#!/usr/bin/env python
"""Intervene on source rows along an SAE-derived source prototype direction."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from neurofm_mechinterp.analysis.common import save_json
from neurofm_mechinterp.loading import load_braintokenizer, resolve_device
from neurofm_mechinterp.paths import REPO_ROOT, RESULTS_ROOT, ensure_dir
from neurofm_mechinterp.processed_data import choose_examples, load_batch_from_examples, load_metadata
from neurofm_mechinterp.sae import (
    SparseAutoencoder,
    choose_ablation_row_mask,
    compute_tokenizer_activations,
    flatten_source_activations,
    resolve_row_source_idx,
    restore_source_activations,
)


DATASET_METADATA = {
    "pd31": REPO_ROOT / "share" / "metadata" / "evaluate" / "pd31" / "BrainOmni" / "info.json",
    "MEG-MMI": REPO_ROOT / "share" / "metadata" / "evaluate" / "MEG-MMI" / "BrainOmni" / "info.json",
}


def default_clip_seconds(dataset: str) -> float | None:
    """Use the same practical MEG crop as the other real-data analyses."""
    if dataset == "MEG-MMI":
        return 10.0
    return None


def load_sae(model_dir: Path) -> tuple[SparseAutoencoder, dict, torch.Tensor, torch.Tensor]:
    """Load one trained sparse autoencoder plus its normalization statistics."""
    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=False)
    model = SparseAutoencoder(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        sparsifier=checkpoint["config"].get("sparsifier", "relu"),
        topk=checkpoint["config"].get("topk"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    mean = checkpoint["input_mean"].view(1, -1).float()
    std = checkpoint["input_std"].view(1, -1).float()
    return model, checkpoint, mean, std


def resolve_activation_path(checkpoint: dict) -> Path:
    """Resolve the activation export path, including older checkpoints."""
    activation_path_value = checkpoint["config"].get("activation_path")
    if activation_path_value:
        return Path(activation_path_value)
    return RESULTS_ROOT / "sae_exports" / checkpoint["config"]["dataset"] / checkpoint["config"]["hook"] / "activations.pt"


@torch.no_grad()
def load_source_prototype_bundle(
    model_dir: Path,
    source_idx: int,
    batch_size: int,
) -> dict[str, object]:
    """Build one normalized source prototype direction from exported activations."""
    model, checkpoint, mean, std = load_sae(model_dir)
    activation_path = resolve_activation_path(checkpoint)
    bundle = torch.load(activation_path, map_location="cpu", weights_only=False)

    source_rows = bundle["source_idx"].long() == int(source_idx)
    if int(source_rows.sum().item()) == 0:
        raise ValueError(f"No activation-export rows found for source {source_idx}.")

    standardized_rows = ((bundle["activations"][source_rows].float()) - mean) / std
    source_feature_sum = torch.zeros(checkpoint["hidden_dim"], dtype=torch.float32)
    for start in range(0, standardized_rows.shape[0], batch_size):
        end = min(start + batch_size, standardized_rows.shape[0])
        source_feature_sum += model.encode(standardized_rows[start:end]).sum(dim=0)

    source_feature_mean = source_feature_sum / float(standardized_rows.shape[0])
    decoder_directions = checkpoint["model_state_dict"]["decoder.weight"].float().T
    decoder_directions = decoder_directions / decoder_directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
    prototype = source_feature_mean @ decoder_directions
    prototype = prototype / prototype.norm().clamp_min(1e-12)
    source_projection = standardized_rows @ prototype

    return {
        "prototype": prototype,
        "mean_projection": float(source_projection.mean().item()),
        "mean_abs_projection": float(source_projection.abs().mean().item()),
        "num_rows": int(standardized_rows.shape[0]),
        "activation_path": str(activation_path),
    }


def apply_source_prototype_intervention(
    standardized_rows: torch.Tensor,
    prototype: torch.Tensor,
    mode: str,
    scale: float,
    target_projection: float,
) -> torch.Tensor:
    """Modify only the source-prototype component of a set of activation rows."""
    projection = standardized_rows @ prototype
    if mode == "remove_projection":
        return standardized_rows - scale * projection.unsqueeze(-1) * prototype.unsqueeze(0)
    if mode == "set_mean_projection":
        projection_delta = scale * (target_projection - projection)
        return standardized_rows + projection_delta.unsqueeze(-1) * prototype.unsqueeze(0)
    raise ValueError(f"Unsupported intervention mode: {mode}")


def summarize_sensor_counts(segment_table: pd.DataFrame) -> pd.DataFrame:
    """Count how often each sensor appears in the top reconstruction-shift set."""
    rows = []
    for dataset_source_key, subset in segment_table.groupby(["source_idx", "mode", "scale"]):
        counter = Counter()
        for value in subset["top_delta_sensor_indices"]:
            counter.update(int(index) for index in str(value).split(",") if index)
        source_idx, mode, scale = dataset_source_key
        for sensor_idx, count in counter.most_common():
            rows.append(
                {
                    "source_idx": int(source_idx),
                    "mode": str(mode),
                    "scale": float(scale),
                    "sensor_idx": int(sensor_idx),
                    "count": int(count),
                    "fraction": float(count / len(subset)),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["source_idx", "mode", "scale", "count", "sensor_idx"],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)


def build_run_summary(segment_table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one summary row for the intervention run."""
    return pd.DataFrame(
        [
            {
                "segments_profiled": int(len(segment_table)),
                "mean_delta_input_mse": float(segment_table["delta_input_mse"].mean()),
                "mean_cosine_to_baseline_reconstruction": float(segment_table["cosine_to_baseline_reconstruction"].mean()),
                "mean_reconstruction_delta_energy": float(segment_table["mean_reconstruction_delta_energy"].mean()),
                "mean_baseline_projection": float(segment_table["baseline_mean_projection"].mean()),
                "mean_intervened_projection": float(segment_table["intervened_mean_projection"].mean()),
                "mean_delta_projection": float(segment_table["delta_mean_projection"].mean()),
                "mean_baseline_abs_projection": float(segment_table["baseline_mean_abs_projection"].mean()),
                "mean_intervened_abs_projection": float(segment_table["intervened_mean_abs_projection"].mean()),
                "mean_eligible_row_count": float(segment_table["eligible_row_count"].mean()),
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
    parser.add_argument("--mode", choices=["remove_projection", "set_mean_projection"], default="remove_projection")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--clip-seconds", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-suffix", default=None)
    parser.add_argument(
        "--metadata-path",
        default=None,
        help="Override the dataset info.json path (e.g. for the expanded PD31 subset).",
    )
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
    target_projection = float(prototype_bundle["mean_projection"])

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

    run_name = f"source_prototype_intervention_source_{args.source_idx}"
    if row_source_idx is not None:
        run_name = f"{run_name}_rows_{row_source_idx}"
    run_name = f"{run_name}_{args.mode}_scale_{args.scale:g}"
    if args.output_suffix:
        run_name = f"{run_name}_{args.output_suffix}"
    output_dir = ensure_dir(model_dir / run_name)

    segment_rows = []
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

        baseline_projection = eligible_rows @ prototype
        intervened_subset = apply_source_prototype_intervention(
            standardized_rows=eligible_rows,
            prototype=prototype,
            mode=args.mode,
            scale=args.scale,
            target_projection=target_projection,
        )
        intervened_projection = intervened_subset @ prototype

        intervened_rows = standardized_rows.clone()
        intervened_rows[row_mask] = intervened_subset
        intervened_hook_feature = restore_source_activations(intervened_rows * std + mean, tuple(hook_feature.shape))
        intervened_quantized_feature = (
            intervened_hook_feature if hook == "encoder_post_quantizer" else tokenizer.quantizer(intervened_hook_feature)[0]
        )
        intervened_reconstruction = tokenizer.decoder(intervened_quantized_feature, sensor_embedding)

        reconstruction_delta = intervened_reconstruction - baseline_reconstruction
        sensor_delta_energy = reconstruction_delta.pow(2).mean(dim=(0, 2, 3))
        top_sensors = torch.topk(sensor_delta_energy, k=min(5, sensor_delta_energy.shape[0])).indices.cpu().tolist()

        intervened_input_mse = float(((intervened_reconstruction - unfolded) ** 2).mean().item())
        cosine_to_baseline = float(
            F.cosine_similarity(
                baseline_reconstruction.reshape(baseline_reconstruction.shape[0], -1),
                intervened_reconstruction.reshape(intervened_reconstruction.shape[0], -1),
                dim=-1,
            ).mean().item()
        )

        segment_rows.append(
            {
                "dataset": args.dataset,
                "path": row["path"],
                "subject": row["subject"],
                "label": int(row["label"]),
                "source_idx": int(args.source_idx),
                "row_source_idx": row_source_idx,
                "mode": args.mode,
                "scale": float(args.scale),
                "baseline_input_mse": baseline_input_mse,
                "intervened_input_mse": intervened_input_mse,
                "delta_input_mse": intervened_input_mse - baseline_input_mse,
                "cosine_to_baseline_reconstruction": cosine_to_baseline,
                "mean_reconstruction_delta_energy": float(reconstruction_delta.pow(2).mean().item()),
                "max_reconstruction_delta_energy": float(reconstruction_delta.pow(2).amax().item()),
                "eligible_row_count": int(eligible_rows.shape[0]),
                "baseline_mean_projection": float(baseline_projection.mean().item()),
                "intervened_mean_projection": float(intervened_projection.mean().item()),
                "delta_mean_projection": float((intervened_projection.mean() - baseline_projection.mean()).item()),
                "baseline_mean_abs_projection": float(baseline_projection.abs().mean().item()),
                "intervened_mean_abs_projection": float(intervened_projection.abs().mean().item()),
                "top_delta_sensor_indices": ",".join(str(index) for index in top_sensors),
            }
        )

    segment_table = pd.DataFrame(segment_rows).sort_values(["subject", "path"]).reset_index(drop=True)
    run_summary = build_run_summary(segment_table)
    subject_summary = (
        segment_table.groupby("subject", as_index=False)
        .agg(
            segments_profiled=("path", "count"),
            mean_delta_input_mse=("delta_input_mse", "mean"),
            mean_cosine_to_baseline_reconstruction=("cosine_to_baseline_reconstruction", "mean"),
            mean_delta_projection=("delta_mean_projection", "mean"),
        )
        .sort_values(["mean_delta_input_mse", "subject"], ascending=[False, True])
        .reset_index(drop=True)
    )
    label_summary = (
        segment_table.groupby("label", as_index=False)
        .agg(
            segments_profiled=("path", "count"),
            mean_delta_input_mse=("delta_input_mse", "mean"),
            mean_cosine_to_baseline_reconstruction=("cosine_to_baseline_reconstruction", "mean"),
            mean_delta_projection=("delta_mean_projection", "mean"),
        )
        .sort_values(["mean_delta_input_mse", "label"], ascending=[False, True])
        .reset_index(drop=True)
    )
    sensor_summary = summarize_sensor_counts(segment_table)

    segment_table.to_csv(output_dir / "segment_source_prototype_interventions.csv", index=False)
    run_summary.to_csv(output_dir / "run_summary.csv", index=False)
    subject_summary.to_csv(output_dir / "subject_summary.csv", index=False)
    label_summary.to_csv(output_dir / "label_summary.csv", index=False)
    sensor_summary.to_csv(output_dir / "sensor_summary.csv", index=False)
    save_json(
        output_dir / "summary.json",
        {
            "dataset": args.dataset,
            "hook": hook,
            "source_idx": int(args.source_idx),
            "row_source_idx": row_source_idx,
            "mode": args.mode,
            "scale": float(args.scale),
            "num_segments": int(len(segment_table)),
            "model_dir": str(model_dir),
            "activation_path": prototype_bundle["activation_path"],
            "prototype_mean_projection": float(prototype_bundle["mean_projection"]),
            "prototype_mean_abs_projection": float(prototype_bundle["mean_abs_projection"]),
            "prototype_export_rows": int(prototype_bundle["num_rows"]),
        },
    )
    print(output_dir)


if __name__ == "__main__":
    main()
