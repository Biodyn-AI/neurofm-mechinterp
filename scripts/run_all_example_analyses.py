#!/usr/bin/env python
"""Run the first-pass analyses on the upstream BrainOmni demo sample."""

from __future__ import annotations

import argparse

from neurofm_mechinterp.analysis.source_ablation import run_source_ablation
from neurofm_mechinterp.analysis.spatial_temporal import run_spatial_temporal_interventions
from neurofm_mechinterp.analysis.tokenizer_semantics import run_tokenizer_semantics
from neurofm_mechinterp.data import (
    load_example_batch,
    make_spatial_mismatch_batch,
    make_time_reversed_batch,
)
from neurofm_mechinterp.loading import load_braintokenizer, load_brainomni, resolve_device
from neurofm_mechinterp.paths import RESULTS_ROOT, ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["tiny", "base"], default="tiny")
    parser.add_argument("--device", default=None, help="Torch device string, for example 'cpu' or 'mps'.")
    args = parser.parse_args()

    device = resolve_device(args.device)
    batch = load_example_batch(device=device)

    tokenizer = load_braintokenizer(device=str(device))
    brainomni = load_brainomni(variant=args.variant, device=str(device))

    root = ensure_dir(RESULTS_ROOT / "example" / args.variant)
    tokenizer_summary = run_tokenizer_semantics(tokenizer, batch, root / "tokenizer_semantics")
    source_summary = run_source_ablation(tokenizer, batch, root / "source_ablation")
    spatial_temporal_summary = run_spatial_temporal_interventions(
        brainomni,
        clean_batch=batch,
        time_perturbed_batch=make_time_reversed_batch(batch),
        spatial_perturbed_batch=make_spatial_mismatch_batch(batch),
        output_dir=root / "spatial_temporal",
    )

    print("tokenizer_semantics", tokenizer_summary)
    print("source_ablation", source_summary)
    print("spatial_temporal", spatial_temporal_summary)


if __name__ == "__main__":
    main()
