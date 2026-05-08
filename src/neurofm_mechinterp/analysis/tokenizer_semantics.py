"""First-pass codebook semantics analysis for the BrainTokenizer."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .common import mean_sensor_relative_bandpower, save_json


@torch.no_grad()
def run_tokenizer_semantics(
    tokenizer: torch.nn.Module,
    batch,
    output_dir: Path,
    sample_rate: float = 256.0,
    overlap_ratio: float = 0.0,
    top_k_codes: int = 12,
) -> dict:
    """
    Summarize how the tokenizer codebooks are used on a concrete sample.

    The semantics are intentionally coarse for the first pass: usage frequency,
    source-latent usage, segment-level bandpower, and reconstruction error.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    viz = tokenizer.visualize(batch.x, batch.pos, batch.sensor_type)
    _, indices = tokenizer.tokenize(batch.x, batch.pos, batch.sensor_type, overlap_ratio=overlap_ratio)

    raw_segments = viz["x"].detach().cpu().numpy()
    reconstructed_segments = viz["x_rec"].detach().cpu().numpy()
    indices_np = indices.detach().cpu().numpy()

    # Token indices are emitted per learned source variable, not per observed sensor.
    # We therefore summarize each token position against segment-level signal statistics
    # averaged across per-sensor spectra, then track which source variables use each code.
    segment_mse = ((reconstructed_segments - raw_segments) ** 2).mean(axis=(1, 3))
    bandpower = mean_sensor_relative_bandpower(raw_segments, sample_rate=sample_rate, sensor_axis=1)

    batch_size, num_sources, num_positions, num_quantizers = indices_np.shape
    _, _, num_segments, _ = raw_segments.shape
    tokens_per_segment = num_positions // num_segments

    expanded_segment_mse = np.repeat(segment_mse[:, None, :, None], num_sources, axis=1)
    expanded_segment_mse = np.repeat(expanded_segment_mse, tokens_per_segment, axis=3)
    expanded_segment_mse = expanded_segment_mse.reshape(batch_size, num_sources, num_positions)
    expanded_bandpower = {}
    for band_name, values in bandpower.items():
        expanded = np.repeat(values[:, None, :, None], num_sources, axis=1)
        expanded = np.repeat(expanded, tokens_per_segment, axis=3)
        expanded_bandpower[band_name] = expanded.reshape(batch_size, num_sources, num_positions)

    codebook_size = tokenizer.quantizer.rvq.codebook_size
    rows = []
    summary = {
        "num_quantizers": int(num_quantizers),
        "codebook_size": int(codebook_size),
        "tokens_per_segment": int(tokens_per_segment),
    }

    for layer_idx in range(num_quantizers):
        layer_codes = indices_np[..., layer_idx]
        counts = np.bincount(layer_codes.reshape(-1), minlength=codebook_size)
        nonzero_codes = int((counts > 0).sum())
        summary[f"layer_{layer_idx}_active_codes"] = nonzero_codes
        top_codes = np.argsort(counts)[::-1][:top_k_codes]

        plt.figure(figsize=(10, 4))
        plt.bar(np.arange(top_k_codes), counts[top_codes])
        plt.xticks(np.arange(top_k_codes), [str(code) for code in top_codes], rotation=45, ha="right")
        plt.title(f"Layer {layer_idx} top code usage")
        plt.ylabel("Token count")
        plt.tight_layout()
        plt.savefig(output_dir / f"layer_{layer_idx}_top_codes.png", dpi=180)
        plt.close()

        for code in top_codes:
            code_mask = layer_codes == code
            if not code_mask.any():
                continue

            row = {
                "layer": int(layer_idx),
                "code": int(code),
                "count": int(code_mask.sum()),
                "fraction": float(code_mask.mean()),
                "mean_segment_mse": float(expanded_segment_mse[code_mask].mean()),
            }
            for band_name, values in expanded_bandpower.items():
                row[f"{band_name}_relative_power"] = float(values[code_mask].mean())
            source_counts = code_mask.sum(axis=(0, 2))
            row["active_sources"] = int((source_counts > 0).sum())
            row["dominant_source"] = int(np.argmax(source_counts))
            row["dominant_source_fraction"] = float(source_counts.max() / source_counts.sum())
            rows.append(row)

    table = pd.DataFrame(rows).sort_values(["layer", "count"], ascending=[True, False])
    table.to_csv(output_dir / "top_codes.csv", index=False)
    save_json(output_dir / "summary.json", summary)
    return summary
