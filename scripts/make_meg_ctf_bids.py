#!/usr/bin/env python
"""Preprocess CTF MEG recordings of a BIDS dataset (ds003568) into BrainOmni evaluate format,
one recording at a time. Raw data are fetched separately (scripts/download_meg_rest_partial.py).

Signal path = upstream downstream/dataset_maker/make_MEG-MMI.py::preprocess_BrainOmni:
  read_raw_ctf -> pick MEG (no refs, no EEG) minus 3 sensors -> notch 60 -> band-pass 0.1-96
  -> resample 256 -> pos/sensor_type (CTF axial grads -> sensor_type 2) -> normalize_pos
  -> non-overlapping windows (strict '<') -> sensortype_wise_normalize (eps 1e-13 T) -> torch.save.
Differences are flags: --segment-seconds (upstream 30), --meg-pos-norm (default 'pretrain':
sensor positions are normalised with the MEG mask as in tokenizer pretraining, factory/utils.py:288;
'upstream' reproduces the downstream makers, which pass mag_mask and so leave CTF positions in
device-frame metres), --comp-grade (upstream keeps the stored grade), --task (upstream keeps
every task except rest).
Run from the repo root (metadata paths are written relative to the cwd as './data/...').
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch

ROOT = Path(os.environ.get("NEUROFM_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "upstream" / "BrainOmni"))
from constant import HIGH, LOW, SAMPLE_RATE  # noqa: E402
from factory.utils import (  # noqa: E402
    accept_segment, extract_pos_sensor_type, get_sensor_type_mask, normalize_pos, sensortype_wise_normalize,
)

def preprocess(ds_dir: Path, args, label: int, ref_names: list[str] | None):
    raw = mne.io.read_raw_ctf(str(ds_dir), preload=True, verbose=False)
    if args.crop_seconds is not None and raw.times[-1] > args.crop_seconds:
        # same span as a recording fetched with scripts/download_meg_rest_partial.py --seconds
        raw.crop(tmax=args.crop_seconds - 1.0 / raw.info["sfreq"])
    grade_in = raw.compensation_grade
    if args.comp_grade is not None and grade_in != args.comp_grade:
        raw.apply_gradient_compensation(args.comp_grade)  # must happen before the reference channels are dropped
    exclude = [c for c in raw.ch_names if c.split("-")[0] in set(args.exclude_base)]
    raw.pick(mne.pick_types(raw.info, meg=True, eeg=False, ref_meg=False, exclude=exclude))
    if ref_names is not None and raw.ch_names != ref_names:
        missing = sorted(set(ref_names) - set(raw.ch_names)); extra = sorted(set(raw.ch_names) - set(ref_names))
        raise RuntimeError(f"{ds_dir.name}: channel set differs from reference (missing {missing}, extra {extra})")
    raw = raw.notch_filter(freqs=[args.line_freq], verbose=False)
    raw = raw.filter(LOW, HIGH, verbose=False)
    raw = raw.resample(SAMPLE_RATE, verbose=False)
    pos, sensor_type = extract_pos_sensor_type(raw.info)
    eeg_mask, mag_mask, grad_mask, meg_mask = get_sensor_type_mask(sensor_type)
    pos = normalize_pos(pos, eeg_mask, mag_mask if args.meg_pos_norm == "upstream" else meg_mask)
    data = raw.get_data()
    seg_len = int(SAMPLE_RATE * args.segment_seconds)
    stem = ds_dir.name.split(".")[0]
    out_dir = os.path.join(args.processed_root, args.dataset_name, "BrainOmni", stem)
    os.makedirs(out_dir, exist_ok=True)
    rows, start, end = [], 0, seg_len
    while end < data.shape[1]:
        seg = sensortype_wise_normalize(data[:, start:end], eeg_mask, mag_mask, grad_mask)
        assert accept_segment(seg, pos), f"nan in {ds_dir}"
        path = os.path.join(out_dir, f"{len(rows)}.pt")
        torch.save({"x": torch.from_numpy(seg), "pos": torch.from_numpy(pos), "sensor_type": torch.from_numpy(sensor_type)}, path)
        rows.append({"path": path, "subject": stem.split("_")[0], "label": label, "channels": int(seg.shape[0]),
                     "run": stem, "comp_grade_in": int(grade_in), "dataset": args.dataset_name})
        start += seg_len
        end += seg_len
    return rows, raw.ch_names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True, help="BIDS root, e.g. ~/neurofm-data/openneuro/ds003568")
    ap.add_argument("--dataset-name", default="MEG-REST")
    ap.add_argument("--task", default="rest")
    ap.add_argument("--run", default="1")
    ap.add_argument("--subjects", default="all")
    ap.add_argument("--segment-seconds", type=float, default=10.0)
    ap.add_argument("--crop-seconds", type=float, default=None, help="keep only the first N seconds of every recording")
    ap.add_argument("--meg-pos-norm", choices=["upstream", "pretrain"], default="pretrain")
    ap.add_argument("--comp-grade", type=int, default=3, help="target CTF gradient-compensation grade; -1 keeps the stored one")
    ap.add_argument("--exclude-base", nargs="*", default=["MRO11", "MLP33", "MLP54"])
    ap.add_argument("--line-freq", type=float, default=60.0)
    ap.add_argument("--processed-root", default="./data/processed_evaluate")
    ap.add_argument("--metadata-root", default="./share/metadata/evaluate")
    args = ap.parse_args()
    if args.comp_grade == -1:
        args.comp_grade = None

    raw_root = Path(args.raw_root).expanduser()
    part = pd.read_csv(raw_root / "participants.tsv", sep="\t")
    labels = {r.participant_id: int(r.group == "MDD") for r in part.itertuples()}
    subjects = sorted(labels) if args.subjects == "all" else [s.strip() for s in args.subjects.split(",")]
    meta_dir = Path(args.metadata_root) / args.dataset_name / "BrainOmni"
    per_rec = meta_dir / "per_recording"
    per_rec.mkdir(parents=True, exist_ok=True)
    ref_names, log = None, []
    for subject in subjects:
        base = f"{subject}_task-{args.task}_run-{args.run}"
        stem = f"{base}_meg"
        ds_dir = raw_root / subject / "meg" / f"{stem}.ds"
        out_json = per_rec / f"{stem}.json"
        if out_json.exists():
            ref_names = ref_names or json.loads(out_json.read_text()).get("ch_names")
            continue
        if not ds_dir.exists():
            print(f"[meg] {stem}: not found -> skipped", flush=True)
            log.append({"recording": stem, "status": "missing"})
            continue
        t1 = time.time()
        try:
            rows, names = preprocess(ds_dir, args, labels[subject], ref_names)
        except Exception as exc:  # recorded, never silently dropped
            print(f"[meg] {stem}: FAILED {exc}", flush=True)
            log.append({"recording": stem, "status": f"failed: {exc}"})
            continue
        ref_names = ref_names or names
        out_json.write_text(json.dumps({"rows": rows, "ch_names": names}, indent=1))
        print(f"[meg] {stem}: label={labels[subject]} segments={len(rows)} ch={len(names)} "
              f"preprocess={time.time()-t1:.0f}s", flush=True)
        log.append({"recording": stem, "status": "ok", "segments": len(rows)})
    records = sorted(p for p in per_rec.glob("*.json") if not p.name.startswith("._"))  # skip macOS AppleDouble files
    rows = [r for p in records for r in json.loads(p.read_text())["rows"]]
    rows.sort(key=lambda r: r["path"])
    (meta_dir / "info.json").write_text(json.dumps(rows, indent=4))
    (meta_dir / "preprocess_log.json").write_text(json.dumps(log, indent=2))
    print(f"[meg] {len(rows)} rows, {len({r['subject'] for r in rows})} subjects -> {meta_dir / 'info.json'}", flush=True)


if __name__ == "__main__":
    main()
