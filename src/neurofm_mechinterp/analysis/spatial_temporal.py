"""Interventions on BrainOmni's split temporal/spatial residual stream."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from .common import cosine_similarity_to_reference, save_json


def _select_block_indices(
    num_blocks: int,
    max_blocks: int | None = None,
    block_indices: list[int] | None = None,
) -> list[int]:
    """Choose which transformer blocks to patch."""
    if block_indices is not None:
        normalized = []
        for index in block_indices:
            block_index = int(index)
            if block_index < 0 or block_index >= num_blocks:
                raise ValueError(f"Block index {block_index} is out of range for {num_blocks} patchable blocks.")
            if block_index not in normalized:
                normalized.append(block_index)
        if not normalized:
            raise ValueError("At least one block index must be provided.")
        return normalized

    if max_blocks is None or max_blocks >= num_blocks:
        return list(range(num_blocks))
    if max_blocks <= 0:
        raise ValueError("max_blocks must be positive when provided.")
    return list(range(max_blocks))


def _prepare_hidden(model: torch.nn.Module, batch) -> torch.Tensor:
    """Run the tokenizer and projection stack up to the first transformer block."""
    hidden, _ = model.tokenizer.tokenize(
        batch.x,
        batch.pos,
        batch.sensor_type,
        model.overlap_ratio,
    )
    _, num_sources, _, _ = hidden.shape
    neuro = model.tokenizer.encoder.neuros.type_as(hidden).detach().view(1, num_sources, 1, -1)
    hidden = hidden + neuro
    return model.projection(hidden)


def _encode_from_hidden(
    model: torch.nn.Module,
    hidden: torch.Tensor,
    donor_hidden: torch.Tensor | None = None,
    patch_block: int | None = None,
    patch_half: str | None = None,
) -> torch.Tensor:
    """Mirror BrainOmni.encode while optionally patching one half of the residual stream."""
    donor = donor_hidden
    current = hidden
    blocks = model.blocks[:-1]
    half_dim = current.shape[-1] // 2

    for block_index, block in enumerate(blocks):
        if donor is not None and patch_block == block_index and patch_half is not None:
            patched = current.clone()
            if patch_half == "temporal":
                patched[..., :half_dim] = donor[..., :half_dim]
            elif patch_half == "spatial":
                patched[..., half_dim:] = donor[..., half_dim:]
            elif patch_half == "both":
                patched = donor.clone()
            else:
                raise ValueError(f"Unknown patch half '{patch_half}'.")
            current = patched
        current = block(current)
        if donor is not None:
            donor = block(donor)

    return F.normalize(current, p=2.0, dim=-1, eps=1e-6)


@torch.no_grad()
def run_spatial_temporal_interventions(
    model: torch.nn.Module,
    clean_batch,
    time_perturbed_batch,
    spatial_perturbed_batch,
    output_dir: Path,
    max_blocks: int | None = None,
    block_indices: list[int] | None = None,
) -> dict:
    """
    Compare temporal and spatial residual-stream patching under two perturbations.

    - `time_perturbed_batch` changes the time structure but leaves sensor geometry intact.
    - `spatial_perturbed_batch` keeps the geometry fixed while mismatching the channel signals.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_hidden = _prepare_hidden(model, clean_batch)
    clean_embedding = _encode_from_hidden(model, clean_hidden)

    rows = []
    perturbations = {
        "time_reverse": _prepare_hidden(model, time_perturbed_batch),
        "spatial_mismatch": _prepare_hidden(model, spatial_perturbed_batch),
    }
    num_patchable_blocks = len(model.blocks) - 1
    candidate_blocks = _select_block_indices(
        num_patchable_blocks,
        max_blocks=max_blocks,
        block_indices=block_indices,
    )

    for perturbation_name, perturbed_hidden in perturbations.items():
        baseline_embedding = _encode_from_hidden(model, perturbed_hidden)
        baseline_similarity = cosine_similarity_to_reference(clean_embedding, baseline_embedding)
        for block_idx in candidate_blocks:
            for patch_half in ("temporal", "spatial", "both"):
                patched_embedding = _encode_from_hidden(
                    model,
                    perturbed_hidden.clone(),
                    donor_hidden=clean_hidden.clone(),
                    patch_block=block_idx,
                    patch_half=patch_half,
                )
                patched_similarity = cosine_similarity_to_reference(clean_embedding, patched_embedding)
                rows.append(
                    {
                        "perturbation": perturbation_name,
                        "block_idx": int(block_idx),
                        "patch_half": patch_half,
                        "baseline_similarity": baseline_similarity,
                        "patched_similarity": patched_similarity,
                        "similarity_gain": patched_similarity - baseline_similarity,
                    }
                )

    table = pd.DataFrame(rows).sort_values(["perturbation", "block_idx", "patch_half"])
    table.to_csv(output_dir / "spatial_temporal_interventions.csv", index=False)
    mean_gain_table = (
        table.groupby(["perturbation", "patch_half"], as_index=False)["similarity_gain"]
        .mean()
        .sort_values(["perturbation", "patch_half"])
    )
    mean_gain_table.to_csv(output_dir / "mean_similarity_gain.csv", index=False)

    best_single_half_rows = []
    for perturbation_name in sorted(table["perturbation"].unique()):
        subset = table[table["perturbation"] == perturbation_name]
        for patch_half in ("temporal", "spatial"):
            best_row = subset[subset["patch_half"] == patch_half].sort_values(
                "similarity_gain",
                ascending=False,
            ).iloc[0]
            best_single_half_rows.append(best_row.to_dict())

    summary = {
        "num_blocks_total": int(num_patchable_blocks),
        "num_blocks_tested": int(len(candidate_blocks)),
        "tested_block_indices": [int(index) for index in candidate_blocks],
        "mean_similarity_gain": mean_gain_table.to_dict(orient="records"),
        "best_single_half_rows": best_single_half_rows,
        "best_rows": table.sort_values("similarity_gain", ascending=False).head(6).to_dict(orient="records"),
    }
    save_json(output_dir / "summary.json", summary)
    return summary
