"""Merge a supplementary necessity-atlas run into a main run.

Used for the MDD-MEG atlas, in which segments of four subjects were omitted and those of five others
processed twice: the supplementary run edits the omitted subjects with the same seed, fold assignment
and decoded sensor subset, duplicate rows (same segment, source, edit and draw) are dropped, and the
merged atlas.csv and baseline.csv replace those of the main run (the originals are kept next to the
supplementary run).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from neurofm_mechinterp.paths import RESULTS_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--main", type=Path, default=RESULTS_ROOT / "necessity_atlas" / "mdd_meg")
    parser.add_argument("--extra", type=Path, default=RESULTS_ROOT / "necessity_atlas_meg_missing")
    args = parser.parse_args()

    keys = {"atlas.csv": ["path", "source", "edit", "rep"], "baseline.csv": ["path"]}
    for name, key in keys.items():
        main_table = pd.read_csv(args.main / name)
        extra_table = pd.read_csv(args.extra / name)
        backup = args.extra / f"original_{name}"
        if not backup.exists():
            shutil.copy(args.main / name, backup)
        merged = pd.concat([main_table.drop_duplicates(subset=key, keep="first"), extra_table], ignore_index=True)
        merged = merged.drop_duplicates(subset=key, keep="first")
        merged.to_csv(args.main / name, index=False)
        print(f"{name}: {len(main_table)} + {len(extra_table)} rows -> {len(merged)} "
              f"({merged['subject'].nunique()} subjects, {merged['path'].nunique()} segments)")
    meta_path = args.main / "atlas_meta.json"
    meta = json.loads(meta_path.read_text())
    baseline = pd.read_csv(args.main / "baseline.csv")
    meta.update(segments=int(baseline["path"].nunique()), subjects=int(baseline["subject"].nunique()),
                merged_from=str(args.extra.relative_to(RESULTS_ROOT.parent)))
    meta_path.write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
