"""Write the analysed subjects and segments as plain-text lists in lists/.

For every cohort: <cohort>_subjects.csv (subject, group, number of segments) and <cohort>_segments.csv (one row per
10-s segment: subject, group, index of the segment within the subject's recording, processed segment file), read from
the export of the released tokenizer (results/source_geometry/<cohort>/export.pt) and the participants table of the
OpenNeuro dataset. tokenizer_training_split.json is the subject split (training / held-out) shared by all tokenizers
trained from scratch.
"""

from __future__ import annotations

import csv
import json
from collections import Counter

import torch

from neurofm_mechinterp.cohorts import get_cohort
from neurofm_mechinterp.paths import REPO_ROOT, RESULTS_ROOT

COHORTS = ("pd_ucsd", "pd_iowa", "dementia", "mdd_meg")
OUT = REPO_ROOT / "lists"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for key in COHORTS:
        export = torch.load(RESULTS_ROOT / "source_geometry" / key / "export.pt", map_location="cpu", weights_only=False)
        groups = get_cohort(key).groups()
        subjects, positions, paths = export["subjects"], export["seg_in_subject"].tolist(), export["paths"]
        with open(OUT / f"{key}_segments.csv", "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["subject", "group", "segment_in_subject", "segment_file"])
            for subject, position, path in zip(subjects, positions, paths):
                writer.writerow([subject, groups.get(subject, ""), int(position), path])
        counts = Counter(subjects)
        with open(OUT / f"{key}_subjects.csv", "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["subject", "group", "segments"])
            for subject in sorted(counts):
                writer.writerow([subject, groups.get(subject, ""), counts[subject]])
        print(f"{key}: {len(counts)} subjects, {len(subjects)} segments")
    splits = {tag: json.loads((RESULTS_ROOT / "tokenizer_replicas" / tag / "split.json").read_text())
              for tag in ("seed1", "seed2", "seed1_srcemb", "seed2_srcemb", "seed1_upstream")}
    first = next(iter(splits.values()))
    if any(split != first for split in splits.values()):
        raise SystemExit("the tokenizers trained from scratch do not share one subject split")
    (OUT / "tokenizer_training_split.json").write_text(json.dumps(first, indent=1) + "\n")


if __name__ == "__main__":
    main()
