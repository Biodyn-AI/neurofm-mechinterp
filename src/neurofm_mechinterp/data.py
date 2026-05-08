"""Small data helpers for the shipped BrainOmni demo sample and perturbations."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .paths import UPSTREAM_ROOT


@dataclass
class BrainBatch:
    """Container for one batch of brain signals plus sensor metadata."""

    x: torch.Tensor
    pos: torch.Tensor
    sensor_type: torch.Tensor

    def to(self, device: torch.device) -> "BrainBatch":
        return BrainBatch(
            x=self.x.to(device),
            pos=self.pos.to(device),
            sensor_type=self.sensor_type.to(device),
        )

    def clone(self) -> "BrainBatch":
        return BrainBatch(
            x=self.x.clone(),
            pos=self.pos.clone(),
            sensor_type=self.sensor_type.clone(),
        )


def load_example_batch(device: torch.device | None = None) -> BrainBatch:
    """Load the upstream demo sample and add a batch dimension."""
    example_path = UPSTREAM_ROOT / "assets" / "example_data.pt"
    payload = torch.load(example_path, map_location="cpu", weights_only=True)
    batch = BrainBatch(
        x=payload["x"].unsqueeze(0),
        pos=payload["pos"].unsqueeze(0),
        sensor_type=payload["sensor_type"].unsqueeze(0),
    )
    return batch if device is None else batch.to(device)


def make_time_reversed_batch(batch: BrainBatch) -> BrainBatch:
    """Reverse the temporal axis while leaving the sensor geometry intact."""
    perturbed = batch.clone()
    perturbed.x = torch.flip(perturbed.x, dims=(-1,))
    return perturbed


def make_spatial_mismatch_batch(batch: BrainBatch) -> BrainBatch:
    """
    Permute the observed channel signals without permuting sensor metadata.

    This keeps the geometry fixed and creates a deliberately inconsistent spatial input.
    """
    perturbed = batch.clone()
    permutation = torch.randperm(perturbed.x.shape[1], device=perturbed.x.device)
    perturbed.x = perturbed.x[:, permutation]
    return perturbed

