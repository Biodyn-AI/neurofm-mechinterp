#!/usr/bin/env python
"""Download a small EEG+MEG subset from OpenNeuro's public S3 bucket."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


OPENNEURO_BUCKET = "s3://openneuro.org"

PD31_DEFAULT_SUBJECTS = ["sub-hc1", "sub-hc2", "sub-pd3", "sub-pd5"]


def pd31_keys(subjects: list[str]) -> list[str]:
    """Build the per-subject BDF S3 keys plus shared metadata keys."""
    shared = [
        "ds002778/README",
        "ds002778/dataset_description.json",
        "ds002778/participants.tsv",
    ]
    per_subject = []
    for subject in subjects:
        session = "ses-hc" if subject.startswith("sub-hc") else "ses-off"
        per_subject.append(f"ds002778/{subject}/{session}/eeg/{subject}_{session}_task-rest_eeg.bdf")
    return shared + per_subject


PD31_FILES = pd31_keys(PD31_DEFAULT_SUBJECTS)

MEG_MMI_FILES = [
    "ds003568/README",
    "ds003568/dataset_description.json",
    "ds003568/participants.tsv",
    "ds003568/sub-24201/meg/sub-24201_task-mmi3_run-1_channels.tsv",
    "ds003568/sub-24201/meg/sub-24201_task-mmi3_run-1_events.tsv",
    "ds003568/sub-24201/meg/sub-24201_task-mmi3_run-1_meg.json",
    "ds003568/sub-23732/meg/sub-23732_task-mmi3_run-2_channels.tsv",
    "ds003568/sub-23732/meg/sub-23732_task-mmi3_run-2_events.tsv",
    "ds003568/sub-23732/meg/sub-23732_task-mmi3_run-2_meg.json",
]

MEG_MMI_DIRS = [
    "ds003568/sub-24201/meg/sub-24201_task-mmi3_run-1_meg.ds",
    "ds003568/sub-23732/meg/sub-23732_task-mmi3_run-2_meg.ds",
]


def aws_path() -> str:
    aws = shutil.which("aws")
    if aws is None:
        raise RuntimeError("The `aws` CLI is required. Install it in the active environment first.")
    return aws


def download_file(key: str, destination_root: Path) -> None:
    destination = destination_root / key
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    subprocess.run(
        [
            aws_path(),
            "s3",
            "cp",
            "--no-sign-request",
            f"{OPENNEURO_BUCKET}/{key}",
            str(destination),
        ],
        check=True,
    )


def download_dir(key: str, destination_root: Path) -> None:
    destination = destination_root / key
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            aws_path(),
            "s3",
            "cp",
            "--no-sign-request",
            "--recursive",
            f"{OPENNEURO_BUCKET}/{key}",
            str(destination),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination-root",
        default="data/evaluate",
        help="Local root directory that will receive the downloaded dataset folders.",
    )
    parser.add_argument(
        "--pd31-extra-subjects",
        default="",
        help=(
            "Comma-separated list of extra PD31 subject IDs (e.g. 'sub-hc4,sub-pd6') to "
            "download in addition to the default four. HC subjects use ses-hc, PD subjects use ses-off."
        ),
    )
    parser.add_argument(
        "--skip-meg",
        action="store_true",
        help="Skip the MEG-MMI download (useful when only expanding the PD31 subset).",
    )
    args = parser.parse_args()

    root = Path(args.destination_root)
    root.mkdir(parents=True, exist_ok=True)

    extra = [s.strip() for s in args.pd31_extra_subjects.split(",") if s.strip()]
    pd31_subjects = list(PD31_DEFAULT_SUBJECTS)
    for subject in extra:
        if subject not in pd31_subjects:
            pd31_subjects.append(subject)
    pd31_files = pd31_keys(pd31_subjects)

    for key in pd31_files:
        local_key = key.replace("ds002778", "ds002778-1.0.5", 1)
        download_file(key, root)
        downloaded = root / key
        final_path = root / local_key
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if downloaded != final_path and downloaded.exists() and not final_path.exists():
            downloaded.rename(final_path)

    if args.skip_meg:
        return

    for key in MEG_MMI_FILES:
        local_key = key.replace("ds003568", "ds003568-1.0.4", 1)
        download_file(key, root)
        downloaded = root / key
        final_path = root / local_key
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if downloaded != final_path and downloaded.exists() and not final_path.exists():
            downloaded.rename(final_path)

    for key in MEG_MMI_DIRS:
        local_key = key.replace("ds003568", "ds003568-1.0.4", 1)
        download_dir(key, root)
        downloaded = root / key
        final_path = root / local_key
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if downloaded != final_path and downloaded.exists() and not final_path.exists():
            downloaded.rename(final_path)


if __name__ == "__main__":
    main()
