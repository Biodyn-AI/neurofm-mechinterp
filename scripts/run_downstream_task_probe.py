#!/usr/bin/env python
"""Downstream-task probe: HC vs PD classifier on tokenizer activations.

Tests whether the source-prototype direction identified by the SAE carries
information that a simple linear classifier uses to discriminate diagnostic
labels. Compares baseline activations against activations with the prototype
projection removed from source rows, under leave-one-subject-out CV.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from neurofm_mechinterp.paths import RESULTS_ROOT, ensure_dir
from neurofm_mechinterp.sae import SparseAutoencoder


def load_sae_and_prototype(model_dir: Path, source_idx: int, batch_size: int = 1024):
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

    activation_path = checkpoint["config"].get("activation_path")
    if activation_path is None:
        activation_path = (
            RESULTS_ROOT
            / "sae_exports"
            / checkpoint["config"]["dataset"]
            / checkpoint["config"]["hook"]
            / "activations.pt"
        )
    bundle = torch.load(activation_path, map_location="cpu", weights_only=False)

    source_rows = bundle["source_idx"].long() == int(source_idx)
    standardized_rows = ((bundle["activations"][source_rows].float()) - mean) / std

    feature_sum = torch.zeros(checkpoint["hidden_dim"], dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, standardized_rows.shape[0], batch_size):
            end = min(start + batch_size, standardized_rows.shape[0])
            feature_sum += model.encode(standardized_rows[start:end]).sum(dim=0)
    feature_mean = feature_sum / float(standardized_rows.shape[0])
    decoder_directions = checkpoint["model_state_dict"]["decoder.weight"].float().T
    decoder_directions = decoder_directions / decoder_directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
    prototype = feature_mean @ decoder_directions
    prototype = prototype / prototype.norm().clamp_min(1e-12)
    return prototype, mean, std, bundle, checkpoint


def remove_projection(rows: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
    projection = rows @ prototype
    return rows - projection.unsqueeze(-1) * prototype.unsqueeze(0)


def keep_only_projection(rows: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
    """Replace each row with its component along the prototype direction only."""
    projection = rows @ prototype
    return projection.unsqueeze(-1) * prototype.unsqueeze(0)


def random_unit_direction(dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm().clamp_min(1e-12)


def pool_segments(
    bundle: dict,
    mean: torch.Tensor,
    std: torch.Tensor,
    prototype: torch.Tensor | None,
    row_source_idx: int,
    pooling: str,
    intervention: str = "remove",
):
    """Return one feature vector per segment."""
    activations = ((bundle["activations"].float()) - mean) / std
    if prototype is not None:
        target_mask = bundle["source_idx"].long() == int(row_source_idx)
        if int(target_mask.sum()) > 0:
            target_rows = activations[target_mask]
            activations = activations.clone()
            if intervention == "remove":
                activations[target_mask] = remove_projection(target_rows, prototype)
            elif intervention == "keep_only":
                activations[target_mask] = keep_only_projection(target_rows, prototype)
            else:
                raise ValueError(f"Unknown intervention: {intervention}")

    segment_idx = bundle["segment_idx"].long().numpy()
    source_idx_arr = bundle["source_idx"].long().numpy()
    subjects = list(bundle["subject"])
    labels = bundle["label"].long().numpy()
    n_sources = int(source_idx_arr.max()) + 1

    df = pd.DataFrame(
        {
            "segment": segment_idx,
            "source": source_idx_arr,
        }
    )
    seg_meta = []
    seen = set()
    for i, seg in enumerate(segment_idx):
        if seg in seen:
            continue
        seen.add(seg)
        seg_meta.append((int(seg), subjects[i], int(labels[i])))
    seg_meta.sort()
    seg_ids = [s for s, _, _ in seg_meta]
    seg_subj = [s for _, s, _ in seg_meta]
    seg_label = np.array([l for _, _, l in seg_meta], dtype=np.int64)

    feat_dim = activations.shape[1]
    if pooling == "mean":
        # Pool all rows in segment regardless of source
        feats = np.zeros((len(seg_ids), feat_dim), dtype=np.float32)
        seg_to_pos = {s: i for i, s in enumerate(seg_ids)}
        for s, pos in seg_to_pos.items():
            mask = segment_idx == s
            feats[pos] = activations[torch.from_numpy(mask)].mean(dim=0).numpy()
    elif pooling == "per_source":
        # Mean per source, concat across sources
        feats = np.zeros((len(seg_ids), n_sources * feat_dim), dtype=np.float32)
        seg_to_pos = {s: i for i, s in enumerate(seg_ids)}
        for s, pos in seg_to_pos.items():
            for src in range(n_sources):
                mask = (segment_idx == s) & (source_idx_arr == src)
                if mask.any():
                    feats[pos, src * feat_dim : (src + 1) * feat_dim] = (
                        activations[torch.from_numpy(mask)].mean(dim=0).numpy()
                    )
    else:
        raise ValueError(f"Unknown pooling: {pooling}")

    return feats, seg_label, seg_subj, seg_ids


def cv_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    groups: list,
    splitter,
    c_values=(0.001, 0.01, 0.1, 1.0, 10.0),
):
    """Run logistic regression with the supplied CV splitter, sweep C, return best."""
    fold_rows = []
    cached_preds = {}
    for c in c_values:
        preds = np.zeros_like(labels)
        proba = np.zeros_like(labels, dtype=np.float64)
        for train_idx, test_idx in splitter.split(features, labels, groups):
            clf = LogisticRegression(C=c, solver="liblinear", max_iter=2000)
            clf.fit(features[train_idx], labels[train_idx])
            preds[test_idx] = clf.predict(features[test_idx])
            proba[test_idx] = clf.predict_proba(features[test_idx])[:, 1]
        acc = accuracy_score(labels, preds)
        try:
            auc = roc_auc_score(labels, proba)
        except ValueError:
            auc = float("nan")
        fold_rows.append({"C": c, "accuracy": acc, "auc": auc})
        cached_preds[c] = (preds.copy(), proba.copy())
    fold_df = pd.DataFrame(fold_rows)
    best_c = fold_df.loc[fold_df["accuracy"].idxmax(), "C"]
    best_acc = float(fold_df.loc[fold_df["accuracy"].idxmax(), "accuracy"])
    best_auc = float(fold_df.loc[fold_df["accuracy"].idxmax(), "auc"])
    best_preds, best_proba = cached_preds[best_c]
    return fold_df, best_preds, best_proba, best_acc, best_auc, float(best_c)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--source-idx", type=int, default=3)
    parser.add_argument("--row-source-idx", type=int, default=None)
    parser.add_argument("--pooling", choices=["mean", "per_source"], default="mean")
    parser.add_argument("--cv", choices=["loso", "stratified5", "both"], default="both")
    parser.add_argument("--output-suffix", default="downstream_task")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    row_source_idx = args.row_source_idx if args.row_source_idx is not None else args.source_idx

    print(f"[probe] Loading SAE + prototype from {model_dir}", flush=True)
    prototype, mean, std, bundle, checkpoint = load_sae_and_prototype(
        model_dir=model_dir, source_idx=args.source_idx
    )
    dataset = checkpoint["config"]["dataset"]
    hook = checkpoint["config"]["hook"]

    print(f"[probe] Pooling segments (pooling={args.pooling})...", flush=True)
    base_feats, labels, subjects, seg_ids = pool_segments(
        bundle=bundle, mean=mean, std=std, prototype=None,
        row_source_idx=row_source_idx, pooling=args.pooling,
    )
    abl_feats, _, _, _ = pool_segments(
        bundle=bundle, mean=mean, std=std, prototype=prototype,
        row_source_idx=row_source_idx, pooling=args.pooling, intervention="remove",
    )
    keep_feats, _, _, _ = pool_segments(
        bundle=bundle, mean=mean, std=std, prototype=prototype,
        row_source_idx=row_source_idx, pooling=args.pooling, intervention="keep_only",
    )
    rand_dir = random_unit_direction(prototype.shape[0], seed=12345)
    rand_feats, _, _, _ = pool_segments(
        bundle=bundle, mean=mean, std=std, prototype=rand_dir,
        row_source_idx=row_source_idx, pooling=args.pooling, intervention="remove",
    )
    print(f"[probe] features={base_feats.shape}, labels={labels.shape}, subjects={len(set(subjects))}", flush=True)

    splitters = []
    if args.cv in ("loso", "both"):
        splitters.append(("loso", LeaveOneGroupOut()))
    if args.cv in ("stratified5", "both"):
        splitters.append(("stratified5", StratifiedKFold(n_splits=5, shuffle=True, random_state=0)))

    out_dir = ensure_dir(model_dir / f"downstream_{args.output_suffix}")
    summary = {
        "model_dir": str(model_dir),
        "dataset": dataset,
        "hook": hook,
        "source_idx": args.source_idx,
        "row_source_idx": row_source_idx,
        "pooling": args.pooling,
        "n_segments": int(len(labels)),
        "n_subjects": int(len(set(subjects))),
        "n_features": int(base_feats.shape[1]),
        "results": {},
    }

    rows = []
    settings = [
        ("baseline", base_feats),
        ("prototype_removed", abl_feats),
        ("prototype_only", keep_feats),
        ("random_dir_removed", rand_feats),
    ]
    for cv_name, splitter in splitters:
        cv_summary = {}
        cv_setting_preds = {}
        for setting_name, feats in settings:
            print(f"[probe] CV={cv_name} setting={setting_name}...", flush=True)
            folds, preds, proba, acc, auc, c = cv_logistic(feats, labels, subjects, splitter)
            print(
                f"[probe]   acc={acc:.3f} auc={auc:.3f} (C={c})",
                flush=True,
            )
            folds.to_csv(out_dir / f"{cv_name}_{setting_name}_folds.csv", index=False)
            cv_summary[setting_name] = {
                "best_accuracy": acc, "best_auc": auc, "best_C": c,
            }
            cv_setting_preds[setting_name] = (preds, proba)
        baseline_acc = cv_summary["baseline"]["best_accuracy"]
        baseline_auc = cv_summary["baseline"]["best_auc"]
        for setting_name in ("prototype_removed", "prototype_only", "random_dir_removed"):
            cv_summary[setting_name]["delta_accuracy"] = float(
                cv_summary[setting_name]["best_accuracy"] - baseline_acc
            )
            cv_summary[setting_name]["delta_auc"] = float(
                cv_summary[setting_name]["best_auc"] - baseline_auc
            )
        for setting_name, (preds, proba) in cv_setting_preds.items():
            for subj in sorted(set(subjects)):
                mask = np.array([s == subj for s in subjects])
                acc = accuracy_score(labels[mask], preds[mask])
                rows.append({
                    "cv": cv_name,
                    "setting": setting_name,
                    "subject": subj,
                    "label": int(labels[mask][0]),
                    "n_segments": int(mask.sum()),
                    "accuracy": float(acc),
                    "mean_pred_proba_pd": float(proba[mask].mean()),
                })
        summary["results"][cv_name] = cv_summary

    pd.DataFrame(rows).to_csv(out_dir / "per_subject_accuracy.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[probe] Wrote {out_dir}/summary.json", flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
