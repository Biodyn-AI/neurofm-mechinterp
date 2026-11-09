"""Download the first N seconds of every resting-state MEG run in OpenNeuro ds003568.

CTF `.meg4` files store data as consecutive trials ("blocks") that each contain
all channels, after an 8-byte header. A byte-range download of the header plus
the first K blocks is therefore a valid, shorter recording (MNE reads it
directly; the samples are identical to the corresponding span of the full file).
All small files of the `.ds` folder and the BIDS sidecars are downloaded in full.

Usage:
    python scripts/download_meg_rest_partial.py --dest ~/neurofm-data/openneuro/ds003568 --seconds 190
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET = "s3://openneuro.org/ds003568"
# the AWS CLI on the PATH, else next to the Python interpreter (scripts/bootstrap_env.sh installs it there)
AWS = shutil.which("aws") or str(Path(sys.executable).parent / "aws")


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def list_rest_files() -> dict[str, list[tuple[str, int]]]:
    """Map subject -> [(key, size)] for every file of its task-rest_run-1 recording."""
    listing = run([AWS, "s3", "ls", "--no-sign-request", "--recursive", BUCKET + "/"])
    per_subject: dict[str, list[tuple[str, int]]] = {}
    for line in listing.splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) != 4:
            continue
        size, key = int(parts[2]), parts[3]
        if "task-rest_run-1" not in key or not key.startswith("ds003568/sub-"):
            continue
        subject = key.split("/")[1]
        per_subject.setdefault(subject, []).append((key, size))
    return per_subject


def block_bytes(ds_dir: Path) -> tuple[int, int, float]:
    """Return (bytes per trial block, samples per block, sampling rate) from the .res4 header."""
    from mne.io.ctf.res4 import _read_res4

    res4 = _read_res4(str(ds_dir))
    return int(res4["nchan"]) * int(res4["nsamp"]) * 4, int(res4["nsamp"]), float(res4["sfreq"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=190.0)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=6, help="subjects downloaded concurrently")
    args = parser.parse_args()
    if not Path(AWS).exists():
        raise SystemExit("The AWS CLI is required: python -m pip install awscli")
    dest = args.dest.expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    for top in ("participants.tsv", "participants.json", "dataset_description.json", "README"):
        if not (dest / top).exists():
            subprocess.run([AWS, "s3", "cp", "--no-sign-request", "--only-show-errors", f"{BUCKET}/{top}", str(dest / top)], check=False)

    per_subject = list_rest_files()
    subjects = sorted(per_subject) if not args.subjects else args.subjects

    def fetch(subject: str) -> tuple[str, dict]:
        files = per_subject[subject]
        big = [(k, s) for k, s in files if k.endswith("_meg.meg4") and "/hz.ds/" not in k]
        if len(big) != 1:
            return subject, {"status": "skipped", "reason": f"{len(big)} main meg4 files"}
        meg4_key, meg4_size = big[0]
        for key, _ in files:
            if key == meg4_key:
                continue
            local = dest / key.split("/", 1)[1]
            if local.exists():
                continue
            local.parent.mkdir(parents=True, exist_ok=True)
            run([AWS, "s3", "cp", "--no-sign-request", "--only-show-errors", f"s3://openneuro.org/{key}", str(local)])
        meg4_local = dest / meg4_key.split("/", 1)[1]
        ds_dir = meg4_local.parent
        nbytes, nsamp, sfreq = block_bytes(ds_dir)
        n_blocks = int(-(-args.seconds * sfreq // nsamp))
        want = min(meg4_size, 8 + n_blocks * nbytes)
        have = meg4_local.stat().st_size if meg4_local.exists() else 0
        if have < want:
            run([
                AWS, "s3api", "get-object", "--no-sign-request", "--bucket", "openneuro.org",
                "--key", meg4_key, "--range", f"bytes=0-{want - 1}", str(meg4_local),
            ])
        entry = {
            "status": "ok",
            "meg4_bytes": meg4_local.stat().st_size,
            "full_meg4_bytes": meg4_size,
            "seconds_requested": args.seconds,
            "block_seconds": nsamp / sfreq,
        }
        print(subject, entry, flush=True)
        return subject, entry

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        report = dict(pool.map(fetch, subjects))
    (dest / "partial_download_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
