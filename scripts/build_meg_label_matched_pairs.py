#!/usr/bin/env python
"""Build similarity-matched same-label / different-label cross-subject pair
CSVs for MEG expanded6. Mirrors the PD31 matched-pair construction used in
the multi-seed report's section D, but on the 6-subject MEG set.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neurofm_mechinterp.paths import RESULTS_ROOT
from neurofm_mechinterp.sae import SparseAutoencoder


def load_residuals(model_dir: Path, source_idx: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Compute per-segment mean residual (relative to source prototype)."""
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

    # Compute prototype direction
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

    paths = [bundle["path"][i] for i, m in enumerate(src_mask.tolist()) if m]
    subjects = [bundle["subject"][i] for i, m in enumerate(src_mask.tolist()) if m]
    labels = bundle["label"][src_mask].tolist()
    seg_idx = bundle["segment_idx"][src_mask].tolist()

    df = pd.DataFrame({
        "path": paths, "subject": subjects, "label": labels, "segment_idx": seg_idx,
    })
    # Per-segment mean residual
    seg_residuals = []
    seg_meta_rows = []
    seen = set()
    for s in df.itertuples(index=False):
        if s.segment_idx in seen:
            continue
        seen.add(s.segment_idx)
        seg_meta_rows.append({"path": s.path, "subject": s.subject, "label": int(s.label), "segment_idx": int(s.segment_idx)})
    seg_meta = pd.DataFrame(seg_meta_rows).sort_values("segment_idx").reset_index(drop=True)

    seg_resid_mat = np.zeros((len(seg_meta), residuals.shape[1]), dtype=np.float32)
    for pos, row in seg_meta.iterrows():
        mask = (df["segment_idx"] == row["segment_idx"]).to_numpy()
        seg_resid_mat[pos] = residuals[torch.from_numpy(mask)].mean(dim=0).numpy()
    return seg_meta, seg_resid_mat


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def build_pairs(seg_meta: pd.DataFrame, residuals: np.ndarray, num_pairs: int, tol: float):
    """Return matched same-label and different-label pair dataframes."""
    rows = []
    for i in range(len(seg_meta)):
        for j in range(i + 1, len(seg_meta)):
            if seg_meta.iloc[i]["subject"] == seg_meta.iloc[j]["subject"]:
                continue
            c = cosine(residuals[i], residuals[j])
            same_label = seg_meta.iloc[i]["label"] == seg_meta.iloc[j]["label"]
            rows.append({"i": i, "j": j, "cosine": c, "same_label": bool(same_label)})
    df = pd.DataFrame(rows)
    same = df[df["same_label"]].sort_values("cosine", ascending=False).reset_index(drop=True)
    diff = df[~df["same_label"]].sort_values("cosine", ascending=False).reset_index(drop=True)

    # Greedy similarity-matching: for each top same-label pair, find a diff-label
    # pair with cosine within `tol` that hasn't been used yet.
    used_diff = set()
    matched_same, matched_diff = [], []
    for _, sr in same.iterrows():
        if len(matched_same) >= num_pairs:
            break
        # Find best unused diff-label pair within tolerance
        for d_idx, dr in diff.iterrows():
            if d_idx in used_diff:
                continue
            if abs(dr["cosine"] - sr["cosine"]) <= tol:
                matched_same.append(sr)
                matched_diff.append(dr)
                used_diff.add(d_idx)
                break
    return pd.DataFrame(matched_same), pd.DataFrame(matched_diff)


def to_pair_csv(rows: pd.DataFrame, seg_meta: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, r in rows.iterrows():
        i, j = int(r["i"]), int(r["j"])
        donor = seg_meta.iloc[i]
        recipient = seg_meta.iloc[j]
        out.append({
            "donor_path": donor["path"],
            "recipient_path": recipient["path"],
            "donor_subject": donor["subject"],
            "recipient_subject": recipient["subject"],
            "donor_label": int(donor["label"]),
            "recipient_label": int(recipient["label"]),
            "cosine": float(r["cosine"]),
        })
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--source-idx", type=int, default=2)
    parser.add_argument("--num-pairs", type=int, default=10)
    parser.add_argument("--tol", type=float, default=0.04)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_meta, residuals = load_residuals(Path(args.model_dir), args.source_idx)
    print(f"[pairs] segments={len(seg_meta)}, residual_dim={residuals.shape[1]}")
    same_rows, diff_rows = build_pairs(seg_meta, residuals, args.num_pairs, args.tol)
    print(f"[pairs] matched {len(same_rows)} pairs (tol={args.tol})")
    same_csv = to_pair_csv(same_rows, seg_meta)
    diff_csv = to_pair_csv(diff_rows, seg_meta)
    same_path = out_dir / f"meg_matched{args.num_pairs}_same_label_pairs_src{args.source_idx}.csv"
    diff_path = out_dir / f"meg_matched{args.num_pairs}_different_label_pairs_src{args.source_idx}.csv"
    same_csv.to_csv(same_path, index=False)
    diff_csv.to_csv(diff_path, index=False)
    print(f"[pairs] wrote {same_path}")
    print(f"[pairs] wrote {diff_path}")
    print(f"[pairs] same mean cosine: {same_csv['cosine'].mean():.4f}, diff mean cosine: {diff_csv['cosine'].mean():.4f}")


if __name__ == "__main__":
    main()
