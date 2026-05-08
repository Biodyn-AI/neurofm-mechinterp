#!/usr/bin/env python
"""Train a simple sparse autoencoder on exported BrainTokenizer activations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from neurofm_mechinterp.analysis.common import save_json
from neurofm_mechinterp.paths import RESULTS_ROOT, ensure_dir
from neurofm_mechinterp.sae import SparseAutoencoder, train_val_split_indices


def default_activation_path(dataset: str, hook: str) -> Path:
    """Resolve the default exported activation bundle path."""
    return RESULTS_ROOT / "sae_exports" / dataset / hook / "activations.pt"


def load_activation_matrix(path: Path) -> dict:
    """Load an activation bundle produced by `export_sae_activations.py`."""
    return torch.load(path, map_location="cpu", weights_only=False)


def standardize(train_x: torch.Tensor, other_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardize activations using train-set statistics."""
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_x - mean) / std, (other_x - mean) / std, mean.squeeze(0)


@torch.no_grad()
def evaluate_model(model: SparseAutoencoder, x: torch.Tensor, batch_size: int) -> dict[str, float]:
    """Evaluate reconstruction and sparsity metrics on one activation matrix."""
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
    total_rows = 0
    total_mse = 0.0
    total_l1 = 0.0
    total_active = 0.0
    total_variance = 0.0
    feature_activity = None
    feature_max = None

    for (batch_x,) in loader:
        reconstruction, features = model(batch_x)
        mse_per_row = (reconstruction - batch_x).pow(2).mean(dim=-1)
        total_rows += batch_x.shape[0]
        total_mse += float(mse_per_row.sum().item())
        total_l1 += float(features.abs().sum().item())
        total_active += float((features > 0).sum(dim=-1).sum().item())
        total_variance += float(batch_x.pow(2).sum().item())

        batch_sum = features.sum(dim=0)
        batch_max = features.max(dim=0).values
        feature_activity = batch_sum if feature_activity is None else feature_activity + batch_sum
        feature_max = batch_max if feature_max is None else torch.maximum(feature_max, batch_max)

    mean_mse = total_mse / total_rows
    mean_l1 = total_l1 / total_rows / model.decoder.in_features
    mean_active = total_active / total_rows
    explained_variance = 1.0 - (total_mse * x.shape[1]) / max(total_variance, 1e-12)
    feature_activity = feature_activity / total_rows
    dead_feature_fraction = float((feature_max <= 1e-8).float().mean().item())
    return {
        "mean_mse": mean_mse,
        "mean_l1": mean_l1,
        "mean_active_features": mean_active,
        "explained_variance": explained_variance,
        "dead_feature_fraction": dead_feature_fraction,
        "feature_mean_activation": feature_activity,
        "feature_max_activation": feature_max,
    }


def train_model(
    x: torch.Tensor,
    hidden_dim: int,
    l1_coef: float,
    sparsifier: str,
    topk: int | None,
    normalize_decoder_features: bool,
    epochs: int,
    batch_size: int,
    lr: float,
    val_fraction: float,
    seed: int,
    device: torch.device,
) -> tuple[SparseAutoencoder, pd.DataFrame, dict, dict, torch.Tensor, torch.Tensor]:
    """Train the sparse autoencoder and return metrics plus normalization stats."""
    train_index, val_index = train_val_split_indices(x.shape[0], val_fraction=val_fraction, seed=seed)
    train_x_raw = x.index_select(0, train_index)
    val_x_raw = x.index_select(0, val_index)
    train_x, val_x, mean = standardize(train_x_raw, val_x_raw)
    std = train_x_raw.std(dim=0).clamp_min(1e-6)

    model = SparseAutoencoder(
        input_dim=x.shape[1],
        hidden_dim=hidden_dim,
        sparsifier=sparsifier,
        topk=topk,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(train_x),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    history_rows = []
    train_x_device = train_x.to(device)
    val_x_device = val_x.to(device)
    for epoch_idx in range(epochs):
        model.train()
        total_rows = 0
        total_loss = 0.0
        total_mse = 0.0
        total_l1 = 0.0
        total_active = 0.0

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            reconstruction, features = model(batch_x)
            mse = (reconstruction - batch_x).pow(2).mean()
            l1 = features.abs().mean()
            loss = mse if sparsifier == "topk" else mse + l1_coef * l1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if normalize_decoder_features:
                model.normalize_decoder_features()

            rows = batch_x.shape[0]
            total_rows += rows
            total_loss += float(loss.item()) * rows
            total_mse += float(mse.item()) * rows
            total_l1 += float(l1.item()) * rows
            total_active += float((features > 0).sum(dim=-1).float().sum().item())

        model.eval()
        train_eval = evaluate_model(model, train_x_device, batch_size=batch_size)
        val_eval = evaluate_model(model, val_x_device, batch_size=batch_size)
        history_rows.append(
            {
                "epoch": epoch_idx + 1,
                "train_loss": total_loss / total_rows,
                "train_mse_step": total_mse / total_rows,
                "train_l1_step": total_l1 / total_rows,
                "train_mean_active_step": total_active / total_rows,
                "train_mse_eval": train_eval["mean_mse"],
                "train_mean_active_eval": train_eval["mean_active_features"],
                "train_explained_variance": train_eval["explained_variance"],
                "val_mse": val_eval["mean_mse"],
                "val_mean_active_features": val_eval["mean_active_features"],
                "val_explained_variance": val_eval["explained_variance"],
                "val_dead_feature_fraction": val_eval["dead_feature_fraction"],
            }
        )

    history = pd.DataFrame(history_rows)
    return model, history, train_eval, val_eval, mean.cpu(), std.cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["pd31", "MEG-MMI"], required=True)
    parser.add_argument("--hook", choices=["encoder_pre_quantizer", "encoder_post_quantizer"], required=True)
    parser.add_argument("--activation-path", default=None)
    parser.add_argument("--hidden-multiplier", type=int, default=4)
    parser.add_argument("--sparsifier", choices=["relu", "topk"], default="relu")
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--normalize-decoder-features", action="store_true")
    parser.add_argument("--l1-coef", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-suffix", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cpu")

    activation_path = Path(args.activation_path) if args.activation_path else default_activation_path(args.dataset, args.hook)
    bundle = load_activation_matrix(activation_path)
    x = bundle["activations"].float()
    hidden_dim = x.shape[1] * args.hidden_multiplier

    model, history, train_eval, val_eval, mean, std = train_model(
        x=x,
        hidden_dim=hidden_dim,
        l1_coef=args.l1_coef,
        sparsifier=args.sparsifier,
        topk=args.topk,
        normalize_decoder_features=args.normalize_decoder_features,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=device,
    )

    sparsifier_label = args.sparsifier if args.sparsifier == "relu" else f"topk{args.topk}"
    run_name = f"linear_{sparsifier_label}_h{hidden_dim}_l1_{args.l1_coef:g}_seed{args.seed}"
    if args.normalize_decoder_features:
        run_name = f"{run_name}_decnorm"
    if args.output_suffix:
        run_name = f"{run_name}_{args.output_suffix}"
    output_dir = ensure_dir(RESULTS_ROOT / "sae_models" / args.dataset / args.hook / run_name)

    feature_summary = pd.DataFrame(
        {
            "feature_idx": list(range(model.decoder.in_features)),
            "mean_activation": train_eval["feature_mean_activation"].cpu().tolist(),
            "max_activation": train_eval["feature_max_activation"].cpu().tolist(),
        }
    )
    feature_summary["active"] = feature_summary["max_activation"] > 1e-8
    feature_summary = feature_summary.sort_values(["mean_activation", "max_activation"], ascending=[False, False]).reset_index(drop=True)

    history.to_csv(output_dir / "history.csv", index=False)
    feature_summary.to_csv(output_dir / "feature_summary.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "input_mean": mean,
            "input_std": std,
            "config": {**vars(args), "activation_path": str(activation_path)},
            "input_dim": int(x.shape[1]),
            "hidden_dim": int(hidden_dim),
        },
        output_dir / "model.pt",
    )
    save_json(
        output_dir / "summary.json",
        {
            "dataset": args.dataset,
            "hook": args.hook,
            "activation_path": str(activation_path),
            "input_dim": int(x.shape[1]),
            "hidden_dim": int(hidden_dim),
            "hidden_multiplier": args.hidden_multiplier,
            "sparsifier": args.sparsifier,
            "topk": args.topk,
            "normalize_decoder_features": bool(args.normalize_decoder_features),
            "l1_coef": args.l1_coef,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "val_fraction": args.val_fraction,
            "seed": args.seed,
            "train_mean_mse": train_eval["mean_mse"],
            "train_explained_variance": train_eval["explained_variance"],
            "train_mean_active_features": train_eval["mean_active_features"],
            "val_mean_mse": val_eval["mean_mse"],
            "val_explained_variance": val_eval["explained_variance"],
            "val_mean_active_features": val_eval["mean_active_features"],
            "val_dead_feature_fraction": val_eval["dead_feature_fraction"],
            "run_name": run_name,
        },
    )
    print(output_dir)


if __name__ == "__main__":
    main()
