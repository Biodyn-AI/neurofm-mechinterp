#!/usr/bin/env python
"""Measure decoder head-routing shifts after source-prototype interventions."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from einops import rearrange

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
    """Load one trained sparse autoencoder plus normalization statistics."""
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
def load_source_prototype_bundle(model_dir: Path, source_idx: int, batch_size: int) -> dict[str, object]:
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


def compute_decoder_routing(
    tokenizer,
    quantized_feature: torch.Tensor,
    sensor_embedding: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-head attention and contribution maps per sensor/source."""
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
    return attention.mean(dim=0), contribution_norm.mean(dim=0)


def summarize_head_sensor_counts(segment_table: pd.DataFrame) -> pd.DataFrame:
    """Count how often each head-level top-delta sensor appears."""
    rows = []
    for head_idx, subset in segment_table.groupby("head_idx"):
        counter = Counter()
        for value in subset["top_abs_delta_sensor_indices"]:
            counter.update(int(index) for index in str(value).split(",") if index)
        for sensor_idx, count in counter.most_common():
            rows.append(
                {
                    "head_idx": int(head_idx),
                    "sensor_idx": int(sensor_idx),
                    "count": int(count),
                    "fraction": float(count / len(subset)),
                }
            )
    return pd.DataFrame(rows).sort_values(["head_idx", "count", "sensor_idx"], ascending=[True, False, True]).reset_index(drop=True)


def build_head_summary(segment_table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one row per head."""
    return (
        segment_table.groupby("head_idx", as_index=False)
        .agg(
            segments_profiled=("path", "count"),
            mean_baseline_mean_contribution=("baseline_mean_contribution", "mean"),
            mean_intervened_mean_contribution=("intervened_mean_contribution", "mean"),
            mean_delta_mean_contribution=("delta_mean_contribution", "mean"),
            mean_abs_sensor_delta=("mean_abs_sensor_delta", "mean"),
        )
        .sort_values(["mean_abs_sensor_delta", "head_idx"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_condition_summary(head_summary: pd.DataFrame) -> pd.DataFrame:
    """Pick the dominant affected head for the intervention condition."""
    top = head_summary.sort_values(["mean_abs_sensor_delta", "head_idx"], ascending=[False, True]).iloc[0]
    return pd.DataFrame(
        [
            {
                "dominant_head_idx": int(top["head_idx"]),
                "dominant_head_mean_abs_sensor_delta": float(top["mean_abs_sensor_delta"]),
                "dominant_head_mean_delta_mean_contribution": float(top["mean_delta_mean_contribution"]),
            }
        ]
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset", choices=["pd31", "MEG-MMI"], required=True)
    parser.add_argument("--source-idx", type=int, required=True)
    parser.add_argument("--decoder-source-idx", type=int, default=None)
    parser.add_argument("--row-source-idx", type=int, default=None)
    parser.add_argument("--mode", choices=["remove_projection", "set_mean_projection"], default="remove_projection")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--clip-seconds", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-suffix", default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    _, checkpoint, mean, std = load_sae(model_dir)
    hook = checkpoint["config"]["hook"]
    row_source_idx = resolve_row_source_idx(args.source_idx, args.row_source_idx)
    decoder_source_idx = args.decoder_source_idx if args.decoder_source_idx is not None else row_source_idx
    if decoder_source_idx is None:
        raise ValueError("Provide `--decoder-source-idx` or a source-conditioned intervention.")

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

    metadata = load_metadata(DATASET_METADATA[args.dataset])
    if args.max_segments is not None:
        metadata = choose_examples(metadata, max_segments=args.max_segments)

    clip_seconds = args.clip_seconds if args.clip_seconds is not None else default_clip_seconds(args.dataset)

    run_name = f"source_prototype_head_routing_source_{args.source_idx}"
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

        sensor_embedding = tokenizer.sensor_embed(batch.pos, batch.sensor_type)
        hook_feature = compute_tokenizer_activations(tokenizer, batch, hook)
        baseline_quantized_feature = hook_feature if hook == "encoder_post_quantizer" else tokenizer.quantizer(hook_feature)[0]
        _, baseline_head_contribution = compute_decoder_routing(
            tokenizer,
            baseline_quantized_feature,
            sensor_embedding,
        )

        rows = flatten_source_activations(hook_feature)
        standardized_rows = (rows.activations - mean) / std
        row_mask = choose_ablation_row_mask(rows.source_idx, row_source_idx)
        eligible_rows = standardized_rows[row_mask]
        if eligible_rows.shape[0] == 0:
            continue

        intervened_subset = apply_source_prototype_intervention(
            standardized_rows=eligible_rows,
            prototype=prototype,
            mode=args.mode,
            scale=args.scale,
            target_projection=target_projection,
        )
        intervened_rows = standardized_rows.clone()
        intervened_rows[row_mask] = intervened_subset
        intervened_hook_feature = restore_source_activations(intervened_rows * std + mean, tuple(hook_feature.shape))
        intervened_quantized_feature = (
            intervened_hook_feature if hook == "encoder_post_quantizer" else tokenizer.quantizer(intervened_hook_feature)[0]
        )
        _, intervened_head_contribution = compute_decoder_routing(
            tokenizer,
            intervened_quantized_feature,
            sensor_embedding,
        )

        for head_idx in range(baseline_head_contribution.shape[0]):
            baseline_values = baseline_head_contribution[head_idx, :, decoder_source_idx]
            intervened_values = intervened_head_contribution[head_idx, :, decoder_source_idx]
            sensor_delta = intervened_values - baseline_values
            top_sensors = torch.topk(sensor_delta.abs(), k=min(5, sensor_delta.shape[0])).indices.cpu().tolist()

            segment_rows.append(
                {
                    "dataset": args.dataset,
                    "path": row["path"],
                    "subject": row["subject"],
                    "label": int(row["label"]),
                    "source_idx": int(args.source_idx),
                    "row_source_idx": row_source_idx,
                    "decoder_source_idx": int(decoder_source_idx),
                    "mode": args.mode,
                    "scale": float(args.scale),
                    "head_idx": int(head_idx),
                    "eligible_row_count": int(eligible_rows.shape[0]),
                    "baseline_mean_contribution": float(baseline_values.mean().item()),
                    "intervened_mean_contribution": float(intervened_values.mean().item()),
                    "delta_mean_contribution": float((intervened_values.mean() - baseline_values.mean()).item()),
                    "mean_abs_sensor_delta": float(sensor_delta.abs().mean().item()),
                    "top_abs_delta_sensor_indices": ",".join(str(index) for index in top_sensors),
                }
            )

    segment_table = pd.DataFrame(segment_rows).sort_values(["head_idx", "subject", "path"]).reset_index(drop=True)
    head_summary = build_head_summary(segment_table)
    condition_summary = build_condition_summary(head_summary)
    sensor_summary = summarize_head_sensor_counts(segment_table)

    segment_table.to_csv(output_dir / "segment_head_routing_interventions.csv", index=False)
    head_summary.to_csv(output_dir / "head_summary.csv", index=False)
    condition_summary.to_csv(output_dir / "condition_summary.csv", index=False)
    sensor_summary.to_csv(output_dir / "sensor_summary.csv", index=False)
    save_json(
        output_dir / "summary.json",
        {
            "dataset": args.dataset,
            "hook": hook,
            "source_idx": int(args.source_idx),
            "row_source_idx": row_source_idx,
            "decoder_source_idx": int(decoder_source_idx),
            "mode": args.mode,
            "scale": float(args.scale),
            "num_segments": int(len(metadata)),
            "model_dir": str(model_dir),
            "activation_path": prototype_bundle["activation_path"],
            "prototype_export_rows": int(prototype_bundle["num_rows"]),
        },
    )
    print(output_dir)


if __name__ == "__main__":
    main()
