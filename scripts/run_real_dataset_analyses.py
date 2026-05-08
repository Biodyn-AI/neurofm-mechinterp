#!/usr/bin/env python
"""Run the first-pass analyses on downloaded and preprocessed real datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from neurofm_mechinterp.analysis.source_ablation import run_source_ablation
from neurofm_mechinterp.analysis.spatial_temporal import run_spatial_temporal_interventions
from neurofm_mechinterp.analysis.tokenizer_semantics import run_tokenizer_semantics
from neurofm_mechinterp.data import make_spatial_mismatch_batch, make_time_reversed_batch
from neurofm_mechinterp.loading import load_braintokenizer, load_brainomni, resolve_device
from neurofm_mechinterp.paths import RESULTS_ROOT, REPO_ROOT, ensure_dir
from neurofm_mechinterp.processed_data import choose_examples, load_batch_from_examples, load_metadata


DATASET_METADATA = {
    "pd31": REPO_ROOT / "share" / "metadata" / "evaluate" / "pd31" / "BrainOmni" / "info.json",
    "MEG-MMI": REPO_ROOT / "share" / "metadata" / "evaluate" / "MEG-MMI" / "BrainOmni" / "info.json",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["pd31", "MEG-MMI"], required=True)
    parser.add_argument("--variant", choices=["tiny", "base"], default="tiny")
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=None,
        help="Optionally truncate each segment to this many seconds after loading.",
    )
    parser.add_argument(
        "--source-limit",
        type=int,
        default=None,
        help="Optionally test only the top-energy tokenizer sources.",
    )
    parser.add_argument(
        "--block-limit",
        type=int,
        default=None,
        help="Optionally test only the first few patchable BrainOmni blocks.",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    metadata_path = DATASET_METADATA[args.dataset]
    metadata = load_metadata(metadata_path)
    if args.max_segments is None:
        args.max_segments = 8 if args.dataset == "pd31" else 2
    examples = choose_examples(metadata, max_segments=args.max_segments)

    device = resolve_device(args.device)
    batch = load_batch_from_examples(examples).to(device)
    if args.clip_seconds is not None:
        clip_samples = int(args.clip_seconds * 256)
        batch.x = batch.x[..., :clip_samples]

    tokenizer = load_braintokenizer(device=str(device))
    brainomni = load_brainomni(variant=args.variant, device=str(device))

    root = ensure_dir(RESULTS_ROOT / "real_data" / args.dataset / args.variant)
    tokenizer_summary = run_tokenizer_semantics(tokenizer, batch, root / "tokenizer_semantics")
    source_summary = run_source_ablation(
        tokenizer,
        batch,
        root / "source_ablation",
        max_sources=args.source_limit,
    )
    spatial_temporal_summary = run_spatial_temporal_interventions(
        brainomni,
        clean_batch=batch,
        time_perturbed_batch=make_time_reversed_batch(batch),
        spatial_perturbed_batch=make_spatial_mismatch_batch(batch),
        output_dir=root / "spatial_temporal",
        max_blocks=args.block_limit,
    )

    print("tokenizer_semantics", tokenizer_summary)
    print("source_ablation", source_summary)
    print("spatial_temporal", spatial_temporal_summary)


if __name__ == "__main__":
    main()
