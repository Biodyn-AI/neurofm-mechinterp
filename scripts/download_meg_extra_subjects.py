#!/usr/bin/env python
"""Download additional MEG-MMI subjects from OpenNeuro ds003568.

Mirrors the layout produced by `download_openneuro_subsets.py`. After downloading
each subject, optionally call preprocess + delete the raw .ds to keep peak disk
usage low.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


BUCKET = "s3://openneuro.org"
SUBJECT_DEFAULTS = ["sub-24138", "sub-24172", "sub-24071", "sub-24103"]


def aws_path() -> str:
    aws = shutil.which("aws")
    if aws is None:
        raise RuntimeError("`aws` CLI not on PATH. Run via `conda run -p $ENV_PATH ...`.")
    return aws


def download_one(s3_key: str, dest: Path, recursive: bool) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [aws_path(), "s3", "cp", "--no-sign-request"]
    if recursive:
        cmd.append("--recursive")
    cmd.extend([f"{BUCKET}/{s3_key}", str(dest)])
    subprocess.run(cmd, check=True)


def per_subject_keys(subject: str) -> tuple[list[str], list[str]]:
    base = f"ds003568/{subject}/meg/{subject}_task-mmi3_run-1_meg"
    files = [
        f"{base}.json",
        f"ds003568/{subject}/meg/{subject}_task-mmi3_run-1_channels.tsv",
        f"ds003568/{subject}/meg/{subject}_task-mmi3_run-1_events.tsv",
    ]
    dirs = [f"{base}.ds"]
    return files, dirs


def to_local(s3_key: str) -> str:
    return s3_key.replace("ds003568", "ds003568-1.0.4", 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination-root", default="data/evaluate")
    parser.add_argument("--subjects", default=",".join(SUBJECT_DEFAULTS))
    args = parser.parse_args()

    root = Path(args.destination_root)
    root.mkdir(parents=True, exist_ok=True)
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]

    for subject in subjects:
        print(f"[download] {subject}", flush=True)
        files, dirs = per_subject_keys(subject)
        for key in files:
            local_key = to_local(key)
            local_path = root / local_key
            if local_path.exists():
                print(f"  skip: {local_key}")
                continue
            download_one(key, local_path, recursive=False)
            print(f"  ok: {local_key}")
        for key in dirs:
            local_key = to_local(key)
            local_path = root / local_key
            if local_path.exists():
                print(f"  skip: {local_key}/")
                continue
            download_one(key, local_path, recursive=True)
            print(f"  ok: {local_key}/")
        print(f"[download] {subject} done", flush=True)


if __name__ == "__main__":
    main()
