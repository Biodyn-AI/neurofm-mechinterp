#!/usr/bin/env python
"""Decompose the PD31 source-3 prototype-orthogonal residual into label-shared
structure and subject-specific remainder, then measure how much of the
donor-recipient residual transport survives each component.

For each segment with source-3 rows R, prototype p:
    projection_i = (R_i @ p) * p
    residual_i   = R_i - projection_i

Per-subject mean residual: subj_mean(s) = mean over rows in subject s
Overall HC and PD means:    label_mean(0), label_mean(1)
Label axis (1-d):           label_axis = label_mean(0) - label_mean(1) (unit norm)
For each row:
    label_aligned_i  = ((residual_i - subj_mean(s_i)) @ label_axis) * label_axis  # within-subject component along label axis
    label_shared_i   = (subj_mean(s_i) @ label_axis) * label_axis                # subject's average label-axis position
    subject_specific_i = residual_i - subj_mean(s_i)                              # within-subject deviation
    subject_mean_i     = subj_mean(s_i)                                          # subject's average residual

We patch each component donor->recipient and measure cosine alignment with
the donor delta in residual space.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neurofm_mechinterp.paths import RESULTS_ROOT, ensure_dir
from neurofm_mechinterp.sae import SparseAutoencoder


def load_sae_and_residuals(model_dir: Path, source_idx: int):
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

    activation_path = checkpoint["config"].get("activation_path") or (
        RESULTS_ROOT / "sae_exports" / checkpoint["config"]["dataset"] / checkpoint["config"]["hook"] / "activations.pt"
    )
    bundle = torch.load(activation_path, map_location="cpu", weights_only=False)

    src_mask = bundle["source_idx"].long() == int(source_idx)
    src_rows = ((bundle["activations"][src_mask].float()) - mean) / std

    feature_sum = torch.zeros(checkpoint["hidden_dim"], dtype=torch.float32)
    bs = 1024
    with torch.no_grad():
        for i in range(0, src_rows.shape[0], bs):
            feature_sum += model.encode(src_rows[i:i + bs]).sum(dim=0)
    feature_mean = feature_sum / float(src_rows.shape[0])
    decoder_directions = checkpoint["model_state_dict"]["decoder.weight"].float().T
    decoder_directions = decoder_directions / decoder_directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
    prototype = feature_mean @ decoder_directions
    prototype = prototype / prototype.norm().clamp_min(1e-12)

    projection = src_rows @ prototype
    residuals = src_rows - projection.unsqueeze(-1) * prototype.unsqueeze(0)
    residuals_np = residuals.numpy()

    paths = [bundle["path"][i] for i, m in enumerate(src_mask.tolist()) if m]
    subjects = [bundle["subject"][i] for i, m in enumerate(src_mask.tolist()) if m]
    labels = bundle["label"][src_mask].long().numpy()
    seg_idx = bundle["segment_idx"][src_mask].long().numpy()

    rows_df = pd.DataFrame({
        "path": paths, "subject": subjects, "label": labels, "segment_idx": seg_idx,
    })
    return rows_df, residuals_np, prototype.numpy()


def per_segment_residuals(rows_df: pd.DataFrame, residuals: np.ndarray):
    """Return one mean residual per segment."""
    segments = rows_df["segment_idx"].drop_duplicates().sort_values().to_numpy()
    seg_res = []
    seg_meta = []
    for s in segments:
        m = (rows_df["segment_idx"] == s).to_numpy()
        first = rows_df[m].iloc[0]
        seg_res.append(residuals[m].mean(axis=0))
        seg_meta.append({"segment_idx": int(s), "subject": first["subject"],
                         "label": int(first["label"]), "path": first["path"]})
    return pd.DataFrame(seg_meta), np.stack(seg_res)


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def decompose(seg_meta: pd.DataFrame, seg_res: np.ndarray):
    """Return label_axis, subj_means dict, label_mean dict."""
    subjects = sorted(seg_meta["subject"].unique())
    subj_means = {}
    for s in subjects:
        mask = (seg_meta["subject"] == s).to_numpy()
        subj_means[s] = seg_res[mask].mean(axis=0)
    label_means = {}
    for label in [0, 1]:
        mask = (seg_meta["label"] == label).to_numpy()
        if mask.sum():
            label_means[label] = seg_res[mask].mean(axis=0)
    label_axis = label_means[0] - label_means[1]
    label_axis = label_axis / (np.linalg.norm(label_axis) + 1e-12)
    return label_axis, subj_means, label_means


def patch_component(seg_meta, seg_res, label_axis, subj_means,
                    component: str, num_pairs: int = 10) -> dict:
    """Greedy donor-recipient pairs across opposite labels, similarity-matched."""
    n = len(seg_meta)
    cosines_donor = []
    cosines_patch = []
    pairs_used = []
    by_label = {0: [], 1: []}
    for i in range(n):
        by_label[int(seg_meta.iloc[i]["label"])].append(i)
    rng = np.random.default_rng(0)

    # Randomly pair across labels
    label0 = list(by_label[0])
    label1 = list(by_label[1])
    rng.shuffle(label0)
    rng.shuffle(label1)
    pairs = list(zip(label0, label1))[:num_pairs]

    for donor_i, recipient_i in pairs:
        donor = seg_res[donor_i]
        recipient = seg_res[recipient_i]
        donor_subj = seg_meta.iloc[donor_i]["subject"]
        recipient_subj = seg_meta.iloc[recipient_i]["subject"]
        if donor_subj == recipient_subj:
            continue

        # Component selection
        if component == "subject_mean":
            patch_donor = subj_means[donor_subj]
            patch_recipient = subj_means[recipient_subj]
        elif component == "subject_specific":
            patch_donor = donor - subj_means[donor_subj]
            patch_recipient = recipient - subj_means[recipient_subj]
        elif component == "label_aligned":
            patch_donor = (donor @ label_axis) * label_axis
            patch_recipient = (recipient @ label_axis) * label_axis
        elif component == "label_orthogonal":
            patch_donor = donor - (donor @ label_axis) * label_axis
            patch_recipient = recipient - (recipient @ label_axis) * label_axis
        elif component == "full_residual":
            patch_donor = donor
            patch_recipient = recipient
        else:
            raise ValueError(f"Unknown component: {component}")

        donor_delta = donor - recipient  # full donor delta in residual space
        patch_delta = patch_donor - patch_recipient
        cosines_donor.append(cosine(donor_delta, donor_delta))  # = 1
        cosines_patch.append(cosine(patch_delta, donor_delta))
        pairs_used.append({
            "donor_idx": donor_i, "recipient_idx": recipient_i,
            "donor_subject": donor_subj, "recipient_subject": recipient_subj,
            "donor_label": int(seg_meta.iloc[donor_i]["label"]),
            "recipient_label": int(seg_meta.iloc[recipient_i]["label"]),
            "patch_delta_norm": float(np.linalg.norm(patch_delta)),
            "donor_delta_norm": float(np.linalg.norm(donor_delta)),
            "cosine_patch_to_donor_delta": float(cosine(patch_delta, donor_delta)),
        })
    return {
        "component": component,
        "num_pairs": len(pairs_used),
        "mean_cosine_patch_to_donor_delta": float(np.mean(cosines_patch)) if cosines_patch else 0.0,
        "std_cosine_patch_to_donor_delta": float(np.std(cosines_patch)) if cosines_patch else 0.0,
        "pairs": pairs_used,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--source-idx", type=int, default=3)
    parser.add_argument("--num-pairs", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir) if args.output_dir else ensure_dir(
        model_dir / f"residual_decomposition_source_{args.source_idx}"
    )
    ensure_dir(out_dir)

    rows_df, residuals, prototype = load_sae_and_residuals(model_dir, args.source_idx)
    seg_meta, seg_res = per_segment_residuals(rows_df, residuals)
    print(f"[decomp] segments={len(seg_meta)} subjects={seg_meta['subject'].nunique()}")
    label_axis, subj_means, label_means = decompose(seg_meta, seg_res)
    print(f"[decomp] label_axis cosine to itself: {cosine(label_axis, label_axis):.3f}")
    print(f"[decomp] HC vs PD label_mean cosine: {cosine(label_means[0], label_means[1]):.3f}")

    components = ["full_residual", "subject_mean", "subject_specific", "label_aligned", "label_orthogonal"]
    summaries = {}
    pair_rows = []
    for c in components:
        res = patch_component(seg_meta, seg_res, label_axis, subj_means, c, args.num_pairs)
        summaries[c] = {k: v for k, v in res.items() if k != "pairs"}
        for p in res["pairs"]:
            pair_rows.append({"component": c, **p})
    pd.DataFrame(pair_rows).to_csv(out_dir / "patch_pair_alignments.csv", index=False)
    summary_table = pd.DataFrame([
        {"component": c, **v} for c, v in summaries.items()
    ])
    summary_table.to_csv(out_dir / "component_alignment_summary.csv", index=False)

    out_summary = {
        "model_dir": str(model_dir),
        "source_idx": args.source_idx,
        "num_segments": int(len(seg_meta)),
        "num_subjects": int(seg_meta["subject"].nunique()),
        "num_pairs_per_component": args.num_pairs,
        "label_mean_cosine_HC_vs_PD": cosine(label_means[0], label_means[1]),
        "components": summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(out_summary, indent=2))
    print(f"[decomp] wrote {out_dir}/summary.json")
    for c in components:
        print(f"  {c:<20s}: cosine = {summaries[c]['mean_cosine_patch_to_donor_delta']:.4f} "
              f"(std {summaries[c]['std_cosine_patch_to_donor_delta']:.4f})")


if __name__ == "__main__":
    main()
