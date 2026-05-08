"""Latent-source ablation analysis for BrainTokenizer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from .common import save_bar_plot, save_json


def _select_source_indices(
    feature: torch.Tensor,
    max_sources: int | None = None,
    source_indices: list[int] | None = None,
) -> list[int]:
    """Choose which tokenizer sources to ablate."""
    num_sources = int(feature.shape[1])

    if source_indices is not None:
        normalized = []
        for index in source_indices:
            source_index = int(index)
            if source_index < 0 or source_index >= num_sources:
                raise ValueError(f"Source index {source_index} is out of range for {num_sources} sources.")
            if source_index not in normalized:
                normalized.append(source_index)
        if not normalized:
            raise ValueError("At least one source index must be provided.")
        return normalized

    if max_sources is None or max_sources >= num_sources:
        return list(range(num_sources))
    if max_sources <= 0:
        raise ValueError("max_sources must be positive when provided.")

    source_energy = feature.pow(2).mean(dim=(0, 2, 3, 4))
    top_indices = torch.topk(source_energy, k=max_sources).indices.cpu().tolist()
    return [int(index) for index in top_indices]


@torch.no_grad()
def compute_source_ablation_table(
    tokenizer: torch.nn.Module,
    batch,
    max_sources: int | None = None,
    source_indices: list[int] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Ablate each learned source variable in the tokenizer encoder and measure damage.

    The current metrics are:
    - increase in reconstruction MSE relative to the raw input
    - fraction of residual-quantizer codes that change
    - sensors with the largest reconstruction damage
    """
    unfolded = tokenizer.unfold(batch.x)
    sensor_embedding = tokenizer.sensor_embed(batch.pos, batch.sensor_type)
    feature = tokenizer.encoder(unfolded, sensor_embedding)

    baseline_quantized, baseline_indices, _ = tokenizer.quantizer(feature)
    baseline_reconstruction = tokenizer.decoder(baseline_quantized, sensor_embedding)
    baseline_mse = ((baseline_reconstruction - unfolded) ** 2).mean()

    num_sources = feature.shape[1]
    candidate_sources = _select_source_indices(
        feature,
        max_sources=max_sources,
        source_indices=source_indices,
    )
    rows = []
    top_sensor_summary = {}

    for source_idx in candidate_sources:
        ablated_feature = feature.clone()
        ablated_feature[:, source_idx] = 0.0

        ablated_quantized, ablated_indices, _ = tokenizer.quantizer(ablated_feature)
        ablated_reconstruction = tokenizer.decoder(ablated_quantized, sensor_embedding)

        mse_delta = ((ablated_reconstruction - unfolded) ** 2).mean() - baseline_mse
        sensor_mse_delta = ((ablated_reconstruction - unfolded) ** 2).mean(dim=(0, 2, 3))
        code_change_rate = (ablated_indices != baseline_indices).float().mean()

        top_sensor_indices = torch.topk(sensor_mse_delta, k=min(5, sensor_mse_delta.shape[0])).indices.cpu().tolist()
        top_sensor_summary[str(source_idx)] = top_sensor_indices
        rows.append(
            {
                "source_idx": int(source_idx),
                "mse_delta": float(mse_delta.item()),
                "code_change_rate": float(code_change_rate.item()),
                "mean_feature_energy": float(feature[:, source_idx].pow(2).mean().item()),
                "top_sensor_indices": ",".join(str(index) for index in top_sensor_indices),
            }
        )

    table = pd.DataFrame(rows).sort_values("mse_delta", ascending=False).reset_index(drop=True)
    summary = {
        "baseline_mse": float(baseline_mse.item()),
        "num_sources": int(num_sources),
        "num_sources_tested": int(len(candidate_sources)),
        "tested_source_indices": [int(index) for index in candidate_sources],
        "most_important_source": int(table.iloc[0]["source_idx"]),
        "top_sensor_indices_per_source": top_sensor_summary,
    }
    return table, summary


@torch.no_grad()
def run_source_ablation(
    tokenizer: torch.nn.Module,
    batch,
    output_dir: Path,
    max_sources: int | None = None,
    source_indices: list[int] | None = None,
) -> dict:
    """Run source ablation and write the tabular and plot outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    table, summary = compute_source_ablation_table(
        tokenizer,
        batch,
        max_sources=max_sources,
        source_indices=source_indices,
    )
    table.to_csv(output_dir / "source_ablation.csv", index=False)
    save_json(
        output_dir / "summary.json",
        {
            "baseline_mse": summary["baseline_mse"],
            "num_sources_total": summary["num_sources"],
            "num_sources_tested": summary["num_sources_tested"],
            "tested_source_indices": summary["tested_source_indices"],
            "top_sensor_indices_per_source": summary["top_sensor_indices_per_source"],
        },
    )
    save_bar_plot(
        values=table["mse_delta"].to_numpy(),
        labels=[str(index) for index in table["source_idx"].tolist()],
        title="Reconstruction damage by source ablation",
        ylabel="Delta MSE",
        path=output_dir / "source_ablation_mse.png",
    )
    save_bar_plot(
        values=table["code_change_rate"].to_numpy(),
        labels=[str(index) for index in table["source_idx"].tolist()],
        title="Code change rate by source ablation",
        ylabel="Fraction of changed codes",
        path=output_dir / "source_ablation_code_change.png",
    )
    return {
        "baseline_mse": summary["baseline_mse"],
        "num_sources": summary["num_sources"],
        "num_sources_tested": summary["num_sources_tested"],
        "tested_source_indices": summary["tested_source_indices"],
        "most_important_source": summary["most_important_source"],
    }
