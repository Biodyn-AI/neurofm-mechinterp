#!/usr/bin/env python
"""Measure decoder head-routing shifts after source-conditioned SAE feature ablations."""

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
from neurofm_mechinterp.paths import REPO_ROOT, ensure_dir
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
    """Load a trained sparse autoencoder plus normalization statistics."""
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
    return model, checkpoint["config"], mean, std


def choose_feature_indices(model_dir: Path, explicit: list[int] | None, source_idx: int | None, top_features: int) -> list[int]:
    """Resolve feature indices either explicitly or from a source-top-features table."""
    if explicit:
        return sorted(dict.fromkeys(int(index) for index in explicit))
    if source_idx is None:
        raise ValueError("Provide either `--feature-idx` or `--source-idx`.")

    table = pd.read_csv(model_dir / "source_top_features.csv")
    subset = table[table["source_idx"] == int(source_idx)].head(top_features)
    if subset.empty:
        raise ValueError(f"No top features found for source {source_idx}.")
    return subset["feature_idx"].astype(int).tolist()


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
    for (feature_idx, head_idx), subset in segment_table.groupby(["feature_idx", "head_idx"]):
        counter = Counter()
        for value in subset["top_abs_delta_sensor_indices"]:
            counter.update(int(index) for index in str(value).split(",") if index)
        for sensor_idx, count in counter.most_common():
            rows.append(
                {
                    "feature_idx": int(feature_idx),
                    "head_idx": int(head_idx),
                    "sensor_idx": int(sensor_idx),
                    "count": int(count),
                    "fraction": float(count / len(subset)),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["feature_idx", "head_idx", "count", "sensor_idx"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)


def build_head_summary(segment_table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one row per feature/head pair."""
    return (
        segment_table.groupby(["feature_idx", "head_idx"], as_index=False)
        .agg(
            segments_profiled=("path", "count"),
            mean_baseline_mean_contribution=("baseline_mean_contribution", "mean"),
            mean_ablated_mean_contribution=("ablated_mean_contribution", "mean"),
            mean_delta_mean_contribution=("delta_mean_contribution", "mean"),
            mean_abs_sensor_delta=("mean_abs_sensor_delta", "mean"),
        )
        .sort_values(["feature_idx", "mean_abs_sensor_delta", "head_idx"], ascending=[True, False, True])
        .reset_index(drop=True)
    )


def build_feature_summary(head_summary: pd.DataFrame) -> pd.DataFrame:
    """Pick the dominant affected head for each feature."""
    rows = []
    for feature_idx, subset in head_summary.groupby("feature_idx"):
        top = subset.sort_values(["mean_abs_sensor_delta", "head_idx"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "feature_idx": int(feature_idx),
                "dominant_head_idx": int(top["head_idx"]),
                "dominant_head_mean_abs_sensor_delta": float(top["mean_abs_sensor_delta"]),
                "dominant_head_mean_delta_mean_contribution": float(top["mean_delta_mean_contribution"]),
            }
        )
    return pd.DataFrame(rows).sort_values("feature_idx").reset_index(drop=True)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset", choices=["pd31", "MEG-MMI"], required=True)
    parser.add_argument("--feature-idx", type=int, nargs="*", default=None)
    parser.add_argument("--source-idx", type=int, default=None)
    parser.add_argument("--decoder-source-idx", type=int, default=None)
    parser.add_argument("--row-source-idx", type=int, default=None)
    parser.add_argument("--top-features", type=int, default=1)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--clip-seconds", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-suffix", default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    sae, config, mean, std = load_sae(model_dir)
    hook = config["hook"]
    feature_indices = choose_feature_indices(model_dir, args.feature_idx, args.source_idx, args.top_features)
    row_source_idx = resolve_row_source_idx(args.source_idx, args.row_source_idx)
    decoder_source_idx = args.decoder_source_idx if args.decoder_source_idx is not None else row_source_idx
    if decoder_source_idx is None:
        raise ValueError("Provide `--decoder-source-idx` or a source-conditioned feature selection.")

    device = resolve_device(args.device)
    tokenizer = load_braintokenizer(device=str(device))
    sae = sae.to(device)
    mean = mean.to(device)
    std = std.to(device)

    metadata = load_metadata(DATASET_METADATA[args.dataset])
    if args.max_segments is not None:
        metadata = choose_examples(metadata, max_segments=args.max_segments)

    run_name = "head_routing_intervention"
    if args.source_idx is not None:
        run_name = f"{run_name}_source_{args.source_idx}"
    if row_source_idx is not None:
        run_name = f"{run_name}_rows_{row_source_idx}"
    if args.output_suffix:
        run_name = f"{run_name}_{args.output_suffix}"
    output_dir = ensure_dir(model_dir / run_name)

    segment_rows = []
    clip_seconds = args.clip_seconds if args.clip_seconds is not None else default_clip_seconds(args.dataset)
    for row in metadata:
        batch = load_batch_from_examples([row]).to(device)
        if clip_seconds is not None:
            clip_samples = int(clip_seconds * 256)
            batch.x = batch.x[..., :clip_samples]

        unfolded = tokenizer.unfold(batch.x)
        sensor_embedding = tokenizer.sensor_embed(batch.pos, batch.sensor_type)
        hook_feature = compute_tokenizer_activations(tokenizer, batch, hook)
        baseline_quantized_feature = hook_feature if hook == "encoder_post_quantizer" else tokenizer.quantizer(hook_feature)[0]
        baseline_head_attention, baseline_head_contribution = compute_decoder_routing(
            tokenizer,
            baseline_quantized_feature,
            sensor_embedding,
        )

        rows = flatten_source_activations(hook_feature)
        standardized_rows = (rows.activations - mean) / std
        sae_features = sae.encode(standardized_rows)
        ablation_row_mask = choose_ablation_row_mask(rows.source_idx, row_source_idx)

        for feature_idx in feature_indices:
            ablated_sae_features = sae_features.clone()
            feature_activation_mask = ablated_sae_features[:, feature_idx] > 0
            applied_mask = feature_activation_mask & ablation_row_mask
            ablated_sae_features[applied_mask, feature_idx] = 0
            ablated_rows = sae.decode(ablated_sae_features) * std + mean
            ablated_hook_feature = restore_source_activations(ablated_rows, tuple(hook_feature.shape))
            ablated_quantized_feature = ablated_hook_feature if hook == "encoder_post_quantizer" else tokenizer.quantizer(ablated_hook_feature)[0]
            ablated_head_attention, ablated_head_contribution = compute_decoder_routing(
                tokenizer,
                ablated_quantized_feature,
                sensor_embedding,
            )

            for head_idx in range(baseline_head_contribution.shape[0]):
                baseline_values = baseline_head_contribution[head_idx, :, decoder_source_idx]
                ablated_values = ablated_head_contribution[head_idx, :, decoder_source_idx]
                sensor_delta = ablated_values - baseline_values
                top_sensors = torch.topk(sensor_delta.abs(), k=min(5, sensor_delta.shape[0])).indices.cpu().tolist()

                segment_rows.append(
                    {
                        "dataset": args.dataset,
                        "path": row["path"],
                        "subject": row["subject"],
                        "label": int(row["label"]),
                        "feature_idx": int(feature_idx),
                        "head_idx": int(head_idx),
                        "row_source_idx": row_source_idx,
                        "decoder_source_idx": int(decoder_source_idx),
                        "ablated_row_count": int(applied_mask.sum().item()),
                        "baseline_mean_contribution": float(baseline_values.mean().item()),
                        "ablated_mean_contribution": float(ablated_values.mean().item()),
                        "delta_mean_contribution": float((ablated_values.mean() - baseline_values.mean()).item()),
                        "mean_abs_sensor_delta": float(sensor_delta.abs().mean().item()),
                        "top_abs_delta_sensor_indices": ",".join(str(index) for index in top_sensors),
                    }
                )

    segment_table = pd.DataFrame(segment_rows).sort_values(["feature_idx", "head_idx", "subject", "path"]).reset_index(drop=True)
    head_summary = build_head_summary(segment_table)
    feature_summary = build_feature_summary(head_summary)
    sensor_summary = summarize_head_sensor_counts(segment_table)

    segment_table.to_csv(output_dir / "segment_head_routing_interventions.csv", index=False)
    head_summary.to_csv(output_dir / "head_summary.csv", index=False)
    feature_summary.to_csv(output_dir / "feature_summary.csv", index=False)
    sensor_summary.to_csv(output_dir / "sensor_summary.csv", index=False)
    save_json(
        output_dir / "summary.json",
        {
            "dataset": args.dataset,
            "hook": hook,
            "feature_indices": feature_indices,
            "num_segments": int(len(metadata)),
            "source_idx": args.source_idx,
            "row_source_idx": row_source_idx,
            "decoder_source_idx": int(decoder_source_idx),
            "model_dir": str(model_dir),
        },
    )
    print(output_dir)


if __name__ == "__main__":
    main()
