"""Check that every partially downloaded resting-state MEG recording has the requested length."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=190.0)
    args = parser.parse_args()
    report, bad = {}, []
    for ds_dir in sorted(args.root.expanduser().glob("sub-*/meg/*_task-rest_run-1_meg.ds")):
        if ds_dir.name.startswith("._"):
            continue
        raw = mne.io.read_raw_ctf(str(ds_dir), preload=False, verbose="ERROR")
        duration = float(raw.times[-1] + 1.0 / raw.info["sfreq"])
        ok = duration >= args.seconds - 1e-6
        report[ds_dir.parent.parent.name] = {"seconds": duration, "ok": ok}
        if not ok:
            bad.append(ds_dir.parent.parent.name)
    print(json.dumps({"n": len(report), "short": bad}, indent=1))
    (args.root.expanduser() / "partial_validation.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
