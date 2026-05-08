"""Shared helpers for the first-pass BrainOmni analyses."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import welch


BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def save_json(path: Path, payload: dict) -> None:
    """Write JSON with deterministic indentation."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def flatten_embedding(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten a normalized embedding tensor into a single vector for comparison."""
    return tensor.reshape(tensor.shape[0], -1)


def cosine_similarity_to_reference(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """Compute the mean cosine similarity between two batched embedding tensors."""
    ref_flat = flatten_embedding(reference)
    cand_flat = flatten_embedding(candidate)
    return torch.nn.functional.cosine_similarity(ref_flat, cand_flat, dim=-1).mean().item()


def relative_bandpower(data: np.ndarray, sample_rate: float) -> dict[str, np.ndarray]:
    """
    Compute relative bandpower for the last axis of a time-series array.

    The output arrays keep every leading dimension of the input.
    """
    frequencies, spectrum = welch(
        data,
        fs=sample_rate,
        axis=-1,
        nperseg=min(int(sample_rate), data.shape[-1]),
    )
    total_power = spectrum.sum(axis=-1) + 1e-12
    result = {}
    for band_name, (low, high) in BANDS.items():
        mask = (frequencies >= low) & (frequencies < high)
        result[band_name] = spectrum[..., mask].sum(axis=-1) / total_power
    return result


def mean_sensor_relative_bandpower(
    data: np.ndarray,
    sample_rate: float,
    sensor_axis: int,
) -> dict[str, np.ndarray]:
    """
    Compute relative bandpower per sensor and average over sensors.

    Averaging raw signals across sensors before taking the spectrum can cancel
    oscillatory structure when sensors are out of phase. This helper keeps the
    spectral transform local to each sensor and only averages afterwards.
    """
    bandpower = relative_bandpower(data, sample_rate)
    return {band_name: values.mean(axis=sensor_axis) for band_name, values in bandpower.items()}


def flatten_windowed_time(data):
    """Merge the trailing `(num_windows, window_length)` axes into one time axis."""
    return data.reshape(*data.shape[:-2], data.shape[-2] * data.shape[-1])


def save_bar_plot(values: np.ndarray, labels: list[str], title: str, ylabel: str, path: Path) -> None:
    """Save a simple bar plot with rotated x labels."""
    plt.figure(figsize=(12, 4))
    plt.bar(np.arange(len(values)), values)
    plt.xticks(np.arange(len(values)), labels, rotation=60, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
