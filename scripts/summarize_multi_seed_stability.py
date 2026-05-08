#!/usr/bin/env python
"""Aggregate per-seed source-prototype intervention runs into one stability table.

Reads existing prototype-removal `run_summary.csv` files for the four PD31 and
four MEG SAE seeds, plus same-source prototype cosines from
`compare_sae_source_structure.py` outputs, and writes:

- results/multi_seed_stability/seed_damage.csv
- results/multi_seed_stability/seed_pair_prototype_cosine.csv
- results/multi_seed_stability/summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from neurofm_mechinterp.paths import RESULTS_ROOT, ensure_dir
from neurofm_mechinterp.sae import SparseAutoencoder


PD31_SEED_DIRS = {
    0: "pd31/encoder_pre_quantizer/linear_topk64_h1024_l1_0_seed0_pilot_topk64_ep10",
    1: "pd31/encoder_pre_quantizer/linear_topk64_h1024_l1_0_seed1_pilot_topk64_ep10",
    2: "pd31/encoder_pre_quantizer/linear_topk64_h1024_l1_0_seed2_pilot_topk64_ep10",
    3: "pd31/encoder_pre_quantizer/linear_topk64_h1024_l1_0_seed3_pilot_topk64_ep10",
}

MEG_SEED_DIRS = {
    0: "MEG-MMI/encoder_pre_quantizer/linear_topk64_h1024_l1_0_seed0_pilot_topk64",
    1: "MEG-MMI/encoder_pre_quantizer/linear_topk64_h1024_l1_0_seed1",
    2: "MEG-MMI/encoder_pre_quantizer/linear_topk64_h1024_l1_0_seed2",
    3: "MEG-MMI/encoder_pre_quantizer/linear_topk64_h1024_l1_0_seed3",
}

PD31_INTERVENTION_DIR = "source_prototype_intervention_source_3_rows_3_remove_projection_scale_1"
MEG_INTERVENTION_DIR = "source_prototype_intervention_source_2_rows_2_remove_projection_scale_1"


def load_run_summary(model_dir: Path, intervention_dir: str) -> pd.DataFrame:
    csv_path = model_dir / intervention_dir / "run_summary.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def load_sae_prototype(model_dir: Path, source_idx: int) -> torch.Tensor:
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

    activation_path_value = checkpoint["config"].get("activation_path")
    activation_path = (
        Path(activation_path_value)
        if activation_path_value
        else RESULTS_ROOT / "sae_exports" / checkpoint["config"]["dataset"] / checkpoint["config"]["hook"] / "activations.pt"
    )
    bundle = torch.load(activation_path, map_location="cpu", weights_only=False)
    source_rows = bundle["source_idx"].long() == int(source_idx)
    standardized = (bundle["activations"][source_rows].float() - mean) / std

    feature_sum = torch.zeros(checkpoint["hidden_dim"], dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, standardized.shape[0], 1024):
            end = min(start + 1024, standardized.shape[0])
            feature_sum += model.encode(standardized[start:end]).sum(dim=0)
    feature_mean = feature_sum / float(standardized.shape[0])
    decoder_directions = checkpoint["model_state_dict"]["decoder.weight"].float().T
    decoder_directions = decoder_directions / decoder_directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
    prototype = feature_mean @ decoder_directions
    return prototype / prototype.norm().clamp_min(1e-12)


def collect_damage_rows(seed_dirs: dict[int, str], dataset_name: str, intervention_dir: str) -> pd.DataFrame:
    rows = []
    for seed, rel in sorted(seed_dirs.items()):
        model_dir = RESULTS_ROOT / "sae_models" / rel
        summary = load_run_summary(model_dir, intervention_dir)
        if summary.empty:
            continue
        record = summary.iloc[0].to_dict()
        record["seed"] = int(seed)
        record["dataset"] = dataset_name
        record["model_dir"] = str(model_dir)
        rows.append(record)
    return pd.DataFrame(rows)


def collect_pairwise_cosine(seed_dirs: dict[int, str], dataset_name: str, source_idx: int) -> pd.DataFrame:
    prototypes = {seed: load_sae_prototype(RESULTS_ROOT / "sae_models" / rel, source_idx) for seed, rel in sorted(seed_dirs.items())}
    rows = []
    seeds = sorted(prototypes)
    for left in seeds:
        for right in seeds:
            cos = float((prototypes[left] @ prototypes[right]).item())
            rows.append({"dataset": dataset_name, "left_seed": int(left), "right_seed": int(right), "prototype_cosine": cos})
    return pd.DataFrame(rows)


def main() -> None:
    output_dir = ensure_dir(RESULTS_ROOT / "multi_seed_stability")

    pd31_damage = collect_damage_rows(PD31_SEED_DIRS, "pd31", PD31_INTERVENTION_DIR)
    meg_damage = collect_damage_rows(MEG_SEED_DIRS, "MEG-MMI", MEG_INTERVENTION_DIR)
    damage = pd.concat([pd31_damage, meg_damage], ignore_index=True)
    damage.to_csv(output_dir / "seed_damage.csv", index=False)

    pd31_cosine = collect_pairwise_cosine(PD31_SEED_DIRS, "pd31", source_idx=3)
    meg_cosine = collect_pairwise_cosine(MEG_SEED_DIRS, "MEG-MMI", source_idx=2)
    cosine = pd.concat([pd31_cosine, meg_cosine], ignore_index=True)
    cosine.to_csv(output_dir / "seed_pair_prototype_cosine.csv", index=False)

    summary = {
        "pd31": {
            "seeds": sorted(PD31_SEED_DIRS),
            "mean_delta_input_mse_per_seed": pd31_damage.set_index("seed")["mean_delta_input_mse"].to_dict() if not pd31_damage.empty else {},
            "mean_delta_input_mse_across_seeds": float(pd31_damage["mean_delta_input_mse"].mean()) if not pd31_damage.empty else None,
            "std_delta_input_mse_across_seeds": float(pd31_damage["mean_delta_input_mse"].std(ddof=0)) if not pd31_damage.empty else None,
            "min_offdiag_prototype_cosine": float(pd31_cosine[pd31_cosine["left_seed"] != pd31_cosine["right_seed"]]["prototype_cosine"].abs().min()),
            "mean_offdiag_prototype_cosine": float(pd31_cosine[pd31_cosine["left_seed"] != pd31_cosine["right_seed"]]["prototype_cosine"].abs().mean()),
        },
        "MEG-MMI": {
            "seeds": sorted(MEG_SEED_DIRS),
            "mean_delta_input_mse_per_seed": meg_damage.set_index("seed")["mean_delta_input_mse"].to_dict() if not meg_damage.empty else {},
            "mean_delta_input_mse_across_seeds": float(meg_damage["mean_delta_input_mse"].mean()) if not meg_damage.empty else None,
            "std_delta_input_mse_across_seeds": float(meg_damage["mean_delta_input_mse"].std(ddof=0)) if not meg_damage.empty else None,
            "min_offdiag_prototype_cosine": float(meg_cosine[meg_cosine["left_seed"] != meg_cosine["right_seed"]]["prototype_cosine"].abs().min()),
            "mean_offdiag_prototype_cosine": float(meg_cosine[meg_cosine["left_seed"] != meg_cosine["right_seed"]]["prototype_cosine"].abs().mean()),
        },
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(output_dir)


if __name__ == "__main__":
    main()
