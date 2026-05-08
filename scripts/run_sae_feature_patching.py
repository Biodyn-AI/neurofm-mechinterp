#!/usr/bin/env python
"""Patch sparse-autoencoder features between segments and measure reconstruction shifts."""

from __future__ import annotations

import argparse
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


@dataclass
class SegmentCache:
    """Minimal per-segment state needed for feature patching."""

    row: dict
    sae_features: torch.Tensor
    row_source_idx: torch.Tensor
    hook_shape: tuple[int, int, int, int, int]
    baseline_reconstruction: torch.Tensor
    baseline_input_mse: float


def choose_patch_pairs(score_table: pd.DataFrame, feature_idx: int, num_pairs: int) -> list[tuple[str, str]]:
    """Pair high-activation donors with low-activation recipients for one feature."""
    subset = score_table[score_table["feature_idx"] == int(feature_idx)].copy()
    donors = subset.sort_values(["conditioned_feature_mean", "path"], ascending=[False, True])
    recipients = subset.sort_values(["conditioned_feature_mean", "path"], ascending=[True, True])

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


def summarize_sensor_counts(pair_table: pd.DataFrame) -> pd.DataFrame:
    """Count how often each sensor appears in each feature's top patch-delta set."""
    rows = []
    for feature_idx, subset in pair_table.groupby("feature_idx"):
        counts = {}
        for value in subset["top_delta_sensor_indices"]:
            for sensor_idx in str(value).split(","):
                if not sensor_idx:
                    continue
                sensor_int = int(sensor_idx)
                counts[sensor_int] = counts.get(sensor_int, 0) + 1
        for sensor_idx, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            rows.append(
                {
                    "feature_idx": int(feature_idx),
                    "sensor_idx": int(sensor_idx),
                    "count": int(count),
                    "fraction": float(count / len(subset)),
                }
            )
    return pd.DataFrame(rows)


def build_patch_summary(pair_table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one row per patched feature."""
    rows = []
    for feature_idx, subset in pair_table.groupby("feature_idx"):
        rows.append(
            {
                "feature_idx": int(feature_idx),
                "pairs_profiled": int(len(subset)),
                "mean_donor_conditioned_feature_mean": float(subset["donor_conditioned_feature_mean"].mean()),
                "mean_recipient_conditioned_feature_mean": float(subset["recipient_conditioned_feature_mean"].mean()),
                "mean_feature_mean_gap": float((subset["donor_conditioned_feature_mean"] - subset["recipient_conditioned_feature_mean"]).mean()),
                "mean_delta_input_mse": float(subset["delta_input_mse"].mean()),
                "mean_cosine_to_recipient_baseline": float(subset["cosine_to_recipient_baseline"].mean()),
                "mean_cosine_patch_delta_to_donor_delta": float(subset["cosine_patch_delta_to_donor_delta"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_feature_mean_gap", ascending=False).reset_index(drop=True)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset", choices=["pd31", "MEG-MMI"], required=True)
    parser.add_argument("--feature-idx", type=int, nargs="*", default=None)
    parser.add_argument("--source-idx", type=int, default=None)
    parser.add_argument(
        "--row-source-idx",
        type=int,
        default=None,
        help="Only patch rows from this tokenizer source. Defaults to --source-idx when provided.",
    )
    parser.add_argument("--top-features", type=int, default=1)
    parser.add_argument("--num-pairs", type=int, default=5)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=None,
        help="Optionally truncate segments. Defaults to 10 seconds for MEG patching.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-suffix", default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    sae, config, mean, std = load_sae(model_dir)
    hook = config["hook"]
    feature_indices = choose_feature_indices(model_dir, args.feature_idx, args.source_idx, args.top_features)
    row_source_idx = resolve_row_source_idx(args.source_idx, args.row_source_idx)

    device = resolve_device(args.device)
    tokenizer = load_braintokenizer(device=str(device))
    sae = sae.to(device)
    mean = mean.to(device)
    std = std.to(device)

    metadata = load_metadata(DATASET_METADATA[args.dataset])
    if args.max_segments is not None:
        metadata = choose_examples(metadata, max_segments=args.max_segments)

    run_name = "feature_patch"
    if args.source_idx is not None:
        run_name = f"{run_name}_source_{args.source_idx}"
    if row_source_idx is not None:
        run_name = f"{run_name}_rows_{row_source_idx}"
    if args.output_suffix:
        run_name = f"{run_name}_{args.output_suffix}"
    output_dir = ensure_dir(model_dir / run_name)

    clip_seconds = args.clip_seconds if args.clip_seconds is not None else default_clip_seconds(args.dataset)
    segment_cache: dict[str, SegmentCache] = {}
    score_rows = []

    # Precompute baseline SAE features once so patching can be cheap to sweep later.
    for row in metadata:
        batch = load_batch_from_examples([row]).to(device)
        if clip_seconds is not None:
            clip_samples = int(clip_seconds * 256)
            batch.x = batch.x[..., :clip_samples]

        unfolded = tokenizer.unfold(batch.x)
        sensor_embedding = tokenizer.sensor_embed(batch.pos, batch.sensor_type)
        hook_feature = compute_tokenizer_activations(tokenizer, batch, hook)
        rows = flatten_source_activations(hook_feature)
        standardized_rows = (rows.activations - mean) / std
        sae_features = sae.encode(standardized_rows)

        if hook == "encoder_pre_quantizer":
            quantized_feature, _, _ = tokenizer.quantizer(hook_feature)
            baseline_reconstruction = tokenizer.decoder(quantized_feature, sensor_embedding)
        elif hook == "encoder_post_quantizer":
            baseline_reconstruction = tokenizer.decoder(hook_feature, sensor_embedding)
        else:
            raise ValueError(f"Unsupported hook for patching: {hook}")

        baseline_input_mse = float(((baseline_reconstruction - unfolded) ** 2).mean().item())
        row_mask = choose_ablation_row_mask(rows.source_idx, row_source_idx)

        segment_cache[row["path"]] = SegmentCache(
            row=row,
            sae_features=sae_features.cpu(),
            row_source_idx=rows.source_idx.cpu(),
            hook_shape=tuple(hook_feature.shape),
            baseline_reconstruction=baseline_reconstruction.cpu(),
            baseline_input_mse=baseline_input_mse,
        )

        for feature_idx in feature_indices:
            conditioned_mean = 0.0
            if int(row_mask.sum().item()) > 0:
                conditioned_mean = float(sae_features[row_mask, feature_idx].mean().item())
            score_rows.append(
                {
                    "dataset": args.dataset,
                    "path": row["path"],
                    "subject": row["subject"],
                    "label": int(row["label"]),
                    "feature_idx": int(feature_idx),
                    "row_source_idx": row_source_idx,
                    "conditioned_feature_mean": conditioned_mean,
                    "global_feature_mean": float(sae_features[:, feature_idx].mean().item()),
                }
            )

    score_table = pd.DataFrame(score_rows).sort_values(["feature_idx", "conditioned_feature_mean", "path"], ascending=[True, False, True]).reset_index(drop=True)
    score_table.to_csv(output_dir / "segment_feature_scores.csv", index=False)

    pair_rows = []
    for feature_idx in feature_indices:
        pairs = choose_patch_pairs(score_table, feature_idx=feature_idx, num_pairs=args.num_pairs)
        for pair_idx, (donor_path, recipient_path) in enumerate(pairs, start=1):
            donor = segment_cache[donor_path]
            recipient = segment_cache[recipient_path]
            if donor.hook_shape != recipient.hook_shape:
                raise ValueError("Donor and recipient hook activations must have the same shape for patching.")
            if not torch.equal(donor.row_source_idx, recipient.row_source_idx):
                raise ValueError("Donor and recipient row source indices must align for patching.")

            recipient_batch = load_batch_from_examples([recipient.row]).to(device)
            if clip_seconds is not None:
                clip_samples = int(clip_seconds * 256)
                recipient_batch.x = recipient_batch.x[..., :clip_samples]

            unfolded = tokenizer.unfold(recipient_batch.x)
            sensor_embedding = tokenizer.sensor_embed(recipient_batch.pos, recipient_batch.sensor_type)

            donor_features = donor.sae_features.to(device)
            patched_features = recipient.sae_features.to(device).clone()
            row_mask = choose_ablation_row_mask(recipient.row_source_idx.to(device), row_source_idx)
            patched_features[row_mask, feature_idx] = donor_features[row_mask, feature_idx]

            patched_rows = sae.decode(patched_features) * std + mean
            patched_hook_feature = restore_source_activations(patched_rows, recipient.hook_shape)

            if hook == "encoder_pre_quantizer":
                patched_quantized_feature, _, _ = tokenizer.quantizer(patched_hook_feature)
                patched_reconstruction = tokenizer.decoder(patched_quantized_feature, sensor_embedding)
            elif hook == "encoder_post_quantizer":
                patched_reconstruction = tokenizer.decoder(patched_hook_feature, sensor_embedding)
            else:
                raise ValueError(f"Unsupported hook for patching: {hook}")

            recipient_baseline = recipient.baseline_reconstruction.to(device)
            donor_baseline = donor.baseline_reconstruction.to(device)
            patch_delta = patched_reconstruction - recipient_baseline
            donor_delta = donor_baseline - recipient_baseline
            sensor_delta_energy = patch_delta.pow(2).mean(dim=(0, 2, 3))
            top_sensors = torch.topk(sensor_delta_energy, k=min(5, sensor_delta_energy.shape[0])).indices.cpu().tolist()

            patched_input_mse = float(((patched_reconstruction - unfolded) ** 2).mean().item())
            cosine_to_recipient_baseline = float(
                F.cosine_similarity(
                    recipient_baseline.reshape(recipient_baseline.shape[0], -1),
                    patched_reconstruction.reshape(patched_reconstruction.shape[0], -1),
                    dim=-1,
                ).mean().item()
            )
            cosine_patch_delta_to_donor_delta = float(
                F.cosine_similarity(
                    patch_delta.reshape(patch_delta.shape[0], -1),
                    donor_delta.reshape(donor_delta.shape[0], -1),
                    dim=-1,
                ).mean().item()
            )

            donor_score = score_table[(score_table["path"] == donor_path) & (score_table["feature_idx"] == int(feature_idx))].iloc[0]
            recipient_score = score_table[(score_table["path"] == recipient_path) & (score_table["feature_idx"] == int(feature_idx))].iloc[0]
            pair_rows.append(
                {
                    "dataset": args.dataset,
                    "feature_idx": int(feature_idx),
                    "pair_idx": int(pair_idx),
                    "row_source_idx": row_source_idx,
                    "donor_path": donor_path,
                    "donor_subject": donor.row["subject"],
                    "donor_label": int(donor.row["label"]),
                    "recipient_path": recipient_path,
                    "recipient_subject": recipient.row["subject"],
                    "recipient_label": int(recipient.row["label"]),
                    "donor_conditioned_feature_mean": float(donor_score["conditioned_feature_mean"]),
                    "recipient_conditioned_feature_mean": float(recipient_score["conditioned_feature_mean"]),
                    "recipient_baseline_input_mse": recipient.baseline_input_mse,
                    "patched_input_mse": patched_input_mse,
                    "delta_input_mse": patched_input_mse - recipient.baseline_input_mse,
                    "cosine_to_recipient_baseline": cosine_to_recipient_baseline,
                    "cosine_patch_delta_to_donor_delta": cosine_patch_delta_to_donor_delta,
                    "mean_patch_delta_energy": float(patch_delta.pow(2).mean().item()),
                    "top_delta_sensor_indices": ",".join(str(index) for index in top_sensors),
                }
            )

    pair_table = pd.DataFrame(pair_rows).sort_values(["feature_idx", "pair_idx"]).reset_index(drop=True)
    patch_summary = build_patch_summary(pair_table)
    sensor_summary = summarize_sensor_counts(pair_table)

    pair_table.to_csv(output_dir / "pair_patch_results.csv", index=False)
    patch_summary.to_csv(output_dir / "feature_patch_summary.csv", index=False)
    sensor_summary.to_csv(output_dir / "sensor_summary.csv", index=False)
    save_json(
        output_dir / "summary.json",
        {
            "dataset": args.dataset,
            "hook": hook,
            "feature_indices": feature_indices,
            "num_segments": int(len(metadata)),
            "num_pairs": int(args.num_pairs),
            "source_idx": args.source_idx,
            "row_source_idx": row_source_idx,
            "model_dir": str(model_dir),
        },
    )
    print(output_dir)


if __name__ == "__main__":
    main()
