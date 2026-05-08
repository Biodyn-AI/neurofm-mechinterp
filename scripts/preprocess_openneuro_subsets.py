#!/usr/bin/env python
"""Preprocess the downloaded EEG/MEG subsets with the upstream BrainOmni scripts."""

from __future__ import annotations

import argparse
import sys
import json
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = ROOT / "upstream" / "BrainOmni"

DATASET_TO_SCRIPT = {
    "pd31": UPSTREAM_ROOT / "downstream" / "dataset_maker" / "make_pd31.py",
    "MEG-MMI": UPSTREAM_ROOT / "downstream" / "dataset_maker" / "make_MEG-MMI.py",
}


def load_module(script_path: Path):
    """Import one upstream dataset-maker module by path."""
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_preprocess(module, force: bool) -> None:
    """Mirror the upstream script main function while disabling multiprocessing."""
    accessor = module.DataAccessor(read_only=False)
    logger = module.get_logger()
    module.logger = logger
    info_path = Path(module.EVALUATE_METADATA_PATH) / module.DATASET_NAME / "BrainOmni" / "info.json"
    if force and info_path.exists():
        info_path.unlink()
    if info_path.exists():
        print(f"Skipping existing metadata: {info_path}")
        return
    info_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_list = module.process(accessor, "BrainOmni", logger, multi_process=False)
    metadata_list = sorted(metadata_list, key=lambda row: row["path"])
    with info_path.open("w") as handle:
        json.dump(metadata_list, handle, indent=4, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["pd31", "MEG-MMI", "all"],
        default="all",
        help="Which downloaded subset to preprocess.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate metadata even if the output JSON already exists.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(UPSTREAM_ROOT))
    datasets = ["pd31", "MEG-MMI"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        module = load_module(DATASET_TO_SCRIPT[dataset])
        run_preprocess(module, force=args.force)


if __name__ == "__main__":
    main()
