"""Sparse-autoencoder preparation utilities for tokenizer activations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import BrainBatch


ActivationHook = Literal["encoder_pre_quantizer", "encoder_post_quantizer"]


@dataclass
class ActivationRows:
    """One flattened activation matrix plus row-level metadata."""

    activations: torch.Tensor
    source_idx: torch.Tensor
    patch_idx: torch.Tensor
    frame_idx: torch.Tensor


def list_activation_hooks() -> tuple[str, ...]:
    """Return the supported activation hook names."""
    return ("encoder_pre_quantizer", "encoder_post_quantizer")


@torch.no_grad()
def compute_tokenizer_activations(
    tokenizer,
    batch: BrainBatch,
    hook: ActivationHook,
) -> torch.Tensor:
    """Extract one activation tensor from the BrainTokenizer for SAE training."""
    unfolded = tokenizer.unfold(batch.x)
    sensor_embedding = tokenizer.sensor_embed(batch.pos, batch.sensor_type)
    encoder_feature = tokenizer.encoder(unfolded, sensor_embedding)

    if hook == "encoder_pre_quantizer":
        return encoder_feature
    if hook == "encoder_post_quantizer":
        quantized_feature, _, _ = tokenizer.quantizer(encoder_feature)
        return quantized_feature
    raise ValueError(f"Unsupported activation hook: {hook}")


def flatten_source_activations(activations: torch.Tensor) -> ActivationRows:
    """Flatten `B x source x patch x frame x dim` activations into row form."""
    if activations.ndim != 5:
        raise ValueError("Expected activations shaped as `B x source x patch x frame x dim`.")

    batch_size, n_source, n_patch, n_frame, dim = activations.shape
    flat_activations = activations.reshape(batch_size * n_source * n_patch * n_frame, dim)

    source_idx = torch.arange(n_source, device=activations.device).view(1, n_source, 1, 1)
    patch_idx = torch.arange(n_patch, device=activations.device).view(1, 1, n_patch, 1)
    frame_idx = torch.arange(n_frame, device=activations.device).view(1, 1, 1, n_frame)

    source_idx = source_idx.expand(batch_size, n_source, n_patch, n_frame).reshape(-1)
    patch_idx = patch_idx.expand(batch_size, n_source, n_patch, n_frame).reshape(-1)
    frame_idx = frame_idx.expand(batch_size, n_source, n_patch, n_frame).reshape(-1)

    return ActivationRows(
        activations=flat_activations,
        source_idx=source_idx,
        patch_idx=patch_idx,
        frame_idx=frame_idx,
    )


def restore_source_activations(flat_activations: torch.Tensor, shape: tuple[int, int, int, int, int]) -> torch.Tensor:
    """Restore flattened source activations to `B x source x patch x frame x dim` shape."""
    expected_rows = shape[0] * shape[1] * shape[2] * shape[3]
    if flat_activations.ndim != 2:
        raise ValueError("Expected flattened activations shaped as `rows x dim`.")
    if flat_activations.shape[0] != expected_rows or flat_activations.shape[1] != shape[4]:
        raise ValueError("Flattened activations do not match the requested target shape.")
    return flat_activations.reshape(shape)


def resolve_row_source_idx(feature_source_idx: int | None, row_source_idx: int | None) -> int | None:
    """Default row conditioning to the feature-selection source when available."""
    if row_source_idx is not None:
        return int(row_source_idx)
    if feature_source_idx is not None:
        return int(feature_source_idx)
    return None


def choose_ablation_row_mask(source_idx: torch.Tensor, row_source_idx: int | None) -> torch.Tensor:
    """Select the source rows eligible for a feature intervention."""
    if row_source_idx is None:
        return torch.ones_like(source_idx, dtype=torch.bool)
    return source_idx == int(row_source_idx)


def select_row_indices(num_rows: int, max_rows: int | None, seed: int) -> torch.Tensor:
    """Choose row indices, optionally subsampling deterministically."""
    if max_rows is None or num_rows <= max_rows:
        return torch.arange(num_rows)

    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randperm(num_rows, generator=generator)[:max_rows].sort().values


def subset_activation_bundle(
    bundle: dict,
    row_mask: torch.Tensor,
    *,
    reindex_source_idx: bool = False,
) -> dict:
    """Select rows from an activation-export bundle while preserving metadata alignment."""
    if row_mask.ndim != 1:
        raise ValueError("Expected `row_mask` to be one-dimensional.")
    num_rows = int(bundle["activations"].shape[0])
    if row_mask.shape[0] != num_rows:
        raise ValueError("`row_mask` length must match the number of activation rows.")

    selected = row_mask.nonzero(as_tuple=False).squeeze(-1)
    result = {}
    for key, value in bundle.items():
        if torch.is_tensor(value):
            result[key] = value.index_select(0, selected)
        elif isinstance(value, list):
            selected_rows = selected.tolist()
            result[key] = [value[row_idx] for row_idx in selected_rows]
        else:
            result[key] = value

    if reindex_source_idx:
        result["original_source_idx"] = result["source_idx"].clone()
        result["source_idx"] = torch.zeros_like(result["source_idx"])
    return result


def train_val_split_indices(num_rows: int, val_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split row indices into deterministic train and validation subsets."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("Validation fraction must lie strictly between 0 and 1.")
    if num_rows < 2:
        raise ValueError("Need at least two rows to create a train/validation split.")

    generator = torch.Generator()
    generator.manual_seed(seed)
    permutation = torch.randperm(num_rows, generator=generator)
    n_val = max(1, int(round(num_rows * val_fraction)))
    n_val = min(n_val, num_rows - 1)
    val_index = permutation[:n_val].sort().values
    train_index = permutation[n_val:].sort().values
    return train_index, val_index


class SparseAutoencoder(nn.Module):
    """A simple linear-ReLU sparse autoencoder baseline."""

    def __init__(self, input_dim: int, hidden_dim: int, sparsifier: str = "relu", topk: int | None = None):
        super().__init__()
        if sparsifier not in {"relu", "topk"}:
            raise ValueError("Sparsifier must be 'relu' or 'topk'.")
        if sparsifier == "topk" and (topk is None or topk < 1 or topk > hidden_dim):
            raise ValueError("Top-k sparsifier requires `1 <= topk <= hidden_dim`.")
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.sparsifier = sparsifier
        self.topk = topk

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_activation = self.encoder(x)
        if self.sparsifier == "relu":
            return F.relu(pre_activation)

        positive = F.relu(pre_activation)
        values, indices = torch.topk(positive, k=self.topk, dim=-1)
        features = torch.zeros_like(positive)
        features.scatter_(dim=-1, index=indices, src=values)
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)

    @torch.no_grad()
    def normalize_decoder_features(self) -> None:
        """Project decoder feature directions to unit norm."""
        weight = self.decoder.weight
        norms = weight.norm(dim=0, keepdim=True).clamp_min(1e-12)
        weight.div_(norms)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features
