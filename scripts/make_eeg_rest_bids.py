#!/usr/bin/env python
"""Preprocess a BIDS EEG resting-state dataset (BrainVision .vhdr / EEGLAB .set / BDF / EDF)
into the BrainOmni 'evaluate' format consumed by this repo.

Signal path is identical to upstream downstream/dataset_maker/make_pd31.py:
  pick EEG channels that exist in a standard montage -> set_montage -> notch(line) ->
  band-pass 0.1-96 Hz -> resample 256 Hz -> pos/sensor_type -> normalize_pos ->
  non-overlapping 10-s windows (last partial window dropped, strict '<') ->
  sensortype_wise_normalize (per-time common average + one global std over all EEG
  channels, eps 1e-5 in VOLTS) -> torch.save({'x','pos','sensor_type'}).
Extra guards (not in upstream): case-insensitive channel-name matching, a fixed channel
set/order shared by all recordings, unit sanity check, optional BIDS bad-channel
interpolation and optional annotation-based cropping/segment rejection.

Run from the repo root (upstream constant.py paths are relative: ./data, ./share).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
UPSTREAM = Path(os.environ.get("BRAINOMNI_UPSTREAM", ROOT / "upstream" / "BrainOmni"))
sys.path.insert(0, str(UPSTREAM))
from constant import HIGH, LOW, SAMPLE_RATE  # noqa: E402  (96, 0.1, 256)
from factory.utils import (  # noqa: E402
    accept_segment, extract_pos_sensor_type, get_sensor_type_mask,
    normalize_pos, sensortype_wise_normalize,
)

READERS = {
    ".vhdr": mne.io.read_raw_brainvision,
    ".set": mne.io.read_raw_eeglab,
    ".bdf": mne.io.read_raw_bdf,
    ".edf": mne.io.read_raw_edf,
}
BIDS_RE = re.compile(r"(sub-[A-Za-z0-9]+)(?:_(ses-[A-Za-z0-9]+))?.*?(?:_(task-[A-Za-z0-9]+))?(?:_(acq-[A-Za-z0-9]+))?(?:_(run-[0-9]+))?_eeg\.")


def find_recordings(bids_root: Path, task: str | None, session: str | None, include_regex: str | None = None) -> list[Path]:
    out = []
    for ext in READERS:
        for p in bids_root.rglob(f"*_eeg{ext}"):
            name = p.name
            if p.name.startswith("._"):
                continue  # macOS resource forks on exFAT volumes
            if task and f"_task-{task}_" not in name:
                continue
            if session and f"_ses-{session}_" not in name:
                continue
            if include_regex and not re.search(include_regex, name):
                continue
            out.append(p)
    return sorted(out)


def canonicalize_names(raw: mne.io.BaseRaw, montage, rename: dict[str, str], strip_regex: str | None = None) -> dict[str, str]:
    """Optional regex strip (e.g. '^EEG\\s+|-(REF|A1A2|M1M2|LE)$'), explicit renames, then
    case-insensitive match onto montage names (FP1->Fp1, FPZ->Fpz, CZ->Cz ...)."""
    lut = {n.lower(): n for n in montage.ch_names}
    mapping = {}
    for ch in raw.ch_names:
        new = re.sub(strip_regex, "", ch) if strip_regex else ch
        new = rename.get(ch, rename.get(new, new))
        new = lut.get(new.lower(), new)
        if new != ch:
            mapping[ch] = new
    if mapping:
        raw.rename_channels(mapping)
    return mapping


def bids_bad_channels(rec: Path) -> list[str]:
    tsv = Path(re.sub(r"_eeg\.[a-z]+$", "_channels.tsv", str(rec)))
    if not tsv.exists():
        return []
    df = pd.read_csv(tsv, sep="\t")
    if "status" not in df.columns:
        return []
    return df.loc[df["status"].astype(str).str.lower() == "bad", "name"].astype(str).tolist()


def preprocess_one(rec: Path, args, montage, channel_order: list[str] | None):
    raw = READERS[rec.suffix.lower()](rec, preload=True, verbose=False)
    mapping = canonicalize_names(raw, montage, args.rename, args.strip_regex)
    drop = set(args.drop_channels)
    eeg_names = [c for c in raw.ch_names if c in montage.ch_names and c not in drop]
    # force type 'eeg' for montage channels (some readers type mastoids/EOG-like names oddly)
    raw.set_channel_types({c: "eeg" for c in eeg_names if raw.get_channel_types([c])[0] != "eeg"},
                          on_unit_change="ignore")
    if channel_order is not None:
        missing = [c for c in channel_order if c not in eeg_names]
        if missing:
            raise RuntimeError(f"{rec.name}: missing channels {missing}")
        eeg_names = list(channel_order)
    raw.pick(eeg_names)                       # also fixes channel order
    raw.set_montage(montage, on_missing="raise")  # template positions override file positions
    if args.interpolate_bids_bads:
        bads = [mapping.get(b, b) for b in bids_bad_channels(rec)]   # channels.tsv uses file names
        bads = [b for b in bads if b in raw.ch_names]
        if bads:
            raw.info["bads"] = bads
            raw.interpolate_bads(reset_bads=True, mode="accurate", verbose=False)
    # --- identical to make_pd31.py lines 69-71 (filter at native rate, then resample)
    if args.line_freq > 0:
        raw = raw.notch_filter(freqs=[args.line_freq], verbose=False)
    raw = raw.filter(LOW, HIGH, verbose=False)
    raw = raw.resample(SAMPLE_RATE, verbose=False)
    if args.crop_annotation:  # crop AFTER filtering so the crop edges carry no filter transients
        on = [(o, d) for o, d, desc in zip(raw.annotations.onset, raw.annotations.duration,
                                           raw.annotations.description) if re.search(args.crop_annotation, desc)]
        if not on:
            raise RuntimeError(f"{rec.name}: annotation {args.crop_annotation!r} not found")
        o, d = on[0]
        raw.crop(tmin=o - raw.first_time, tmax=min(o - raw.first_time + (d or args.crop_duration), raw.times[-1]))
    pos, sensor_type = extract_pos_sensor_type(raw.info)
    eeg_mask, mag_mask, grad_mask, _ = get_sensor_type_mask(sensor_type)
    pos = normalize_pos(pos, eeg_mask, mag_mask)  # same call as upstream makers (EEG-only: mask arg irrelevant)
    data = raw.get_data()  # Volts
    med_sd = float(np.median(data.std(axis=1)))
    if not (args.min_sd_volts <= med_sd <= args.max_sd_volts):
        raise RuntimeError(f"{rec.name}: median channel SD {med_sd:.3g} V outside "
                           f"[{args.min_sd_volts},{args.max_sd_volts}] -> unit/scaling problem")
    bad_spans = []
    if args.reject_annotations:
        for o, d, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
            if desc.lower().startswith("bad") or desc.lower() == "boundary":
                s0 = int(round((o - raw.first_time) * SAMPLE_RATE))
                bad_spans.append((s0, s0 + max(1, int(round(d * SAMPLE_RATE)))))
    seg_len = int(SAMPLE_RATE * args.segment_seconds)
    out_dir = Path(args.processed_root) / args.dataset_name / "BrainOmni" / rec.name.split(".")[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    m = BIDS_RE.search(rec.name)
    subject = m.group(1) if m else rec.name.split("_")[0]
    session = (m.group(2) if m else None)
    rows, start, end, k = [], 0, seg_len, 0
    while end < data.shape[1]:                # strict '<' as upstream: last full window is dropped if it ends exactly at n
        if not any(s < end and e > start for s, e in bad_spans):
            seg = sensortype_wise_normalize(data[:, start:end], eeg_mask, mag_mask, grad_mask)
            if accept_segment(seg, pos):
                path = out_dir / f"{k}.pt"
                torch.save({"x": torch.from_numpy(seg), "pos": torch.from_numpy(pos),
                            "sensor_type": torch.from_numpy(sensor_type)}, path)
                rows.append({"path": "./" + os.path.relpath(path, Path.cwd()).replace(os.sep, "/"),
                             "subject": subject, "session": session, "label": None,
                             "channels": int(seg.shape[0]), "dataset": args.dataset_name})
                k += 1
        start += seg_len
        end += seg_len
    return rows, raw.ch_names, med_sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids-root", required=True, help="e.g. data/evaluate/ds00XXXX-1.0.0")
    ap.add_argument("--dataset-name", required=True, help="output name under data/processed_evaluate/")
    ap.add_argument("--montage", default="standard_1005")
    ap.add_argument("--task", default=None)
    ap.add_argument("--session", default=None)
    ap.add_argument("--line-freq", type=float, default=60.0, help="50 (EU/Asia) or 60 (US); 0 disables")
    ap.add_argument("--segment-seconds", type=float, default=10.0)
    ap.add_argument("--rename", type=json.loads, default={}, help='JSON dict, e.g. {"T3":"T7"}')
    ap.add_argument("--strip-regex", default=None, help=r"regex removed from names first, e.g. '^EEG\s+|-REF$'")
    ap.add_argument("--drop-channels", nargs="*", default=[], help="montage-named channels to drop (e.g. TP9 TP10 if used as reference)")
    ap.add_argument("--channel-order", nargs="*", default=None, help="explicit common channel list; default = intersection over recordings")
    ap.add_argument("--include-regex", default=None, help="e.g. 'ses-(hc|off)_' for ds002778 OFF-medication + HC")
    ap.add_argument("--label-regex", default=None, help="label = 1 if regex matches subject ID (ds002778: 'sub-pd'); overrides participants.tsv")
    ap.add_argument("--participants-tsv", default=None)
    ap.add_argument("--label-column", default="group")
    ap.add_argument("--positive-values", nargs="+", default=["PD"])
    ap.add_argument("--interpolate-bids-bads", action="store_true")
    ap.add_argument("--reject-annotations", action="store_true", help="skip windows overlapping BAD*/boundary annotations")
    ap.add_argument("--crop-annotation", default=None, help="regex; keep only the span of the first matching annotation (BrainVision descriptions look like 'Comment/eyes_closed')")
    ap.add_argument("--crop-duration", type=float, default=60.0)
    ap.add_argument("--min-sd-volts", type=float, default=5e-7)
    ap.add_argument("--max-sd-volts", type=float, default=5e-4)
    ap.add_argument("--processed-root", default="./data/processed_evaluate")
    ap.add_argument("--metadata-root", default="./share/metadata/evaluate")
    ap.add_argument("--metadata-name", default="info.json")
    ap.add_argument("--channels-only", action="store_true", help="only write channels.json (processed channel order)")
    args = ap.parse_args()

    montage = mne.channels.make_standard_montage(args.montage)
    recs = find_recordings(Path(args.bids_root).expanduser(), args.task, args.session, args.include_regex)
    if not recs:
        raise SystemExit("no recordings found")
    order = args.channel_order
    if order is None:   # pass 1: common channel set (headers only)
        common = None
        for rec in recs:
            raw = READERS[rec.suffix.lower()](rec, preload=False, verbose=False)
            canonicalize_names(raw, montage, args.rename, args.strip_regex)
            names = [c for c in raw.ch_names if c in montage.ch_names and c not in set(args.drop_channels)]
            common = names if common is None else [c for c in common if c in names]
        order = common
        print(f"[maker] common channel set ({len(order)}): {order}")
    meta_dir = Path(args.metadata_root) / args.dataset_name / "BrainOmni"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "channels.json").write_text(json.dumps(order))  # processed channel order
    if args.channels_only:
        return
    labels = {}
    if args.participants_tsv:
        part = pd.read_csv(args.participants_tsv, sep="\t")
        labels = {r["participant_id"]: int(str(r[args.label_column]) in args.positive_values) for _, r in part.iterrows()}
    rows, failures = [], []
    for rec in recs:
        try:
            r, names, med_sd = preprocess_one(rec, args, montage, order)
        except Exception as exc:  # recorded and reported, never silently dropped
            print(f"[maker] {rec.name}: FAILED {exc}")
            failures.append({"recording": rec.name, "error": str(exc)})
            continue
        for row in r:
            row["label"] = (int(bool(re.search(args.label_regex, row["subject"]))) if args.label_regex
                            else labels.get(row["subject"], -1))
        print(f"[maker] {rec.name}: {len(r)} segments, {len(names)} ch, median SD {med_sd*1e6:.1f} uV")
        rows += r
    rows.sort(key=lambda x: x["path"])
    meta = Path(args.metadata_root) / args.dataset_name / "BrainOmni" / args.metadata_name
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(rows, indent=4, ensure_ascii=False))
    (meta.parent / (meta.stem + "_failures.json")).write_text(json.dumps(failures, indent=2))
    print(f"[maker] wrote {len(rows)} rows ({len({r['subject'] for r in rows})} subjects) -> {meta}; "
          f"{len(failures)} recording(s) failed")


if __name__ == "__main__":
    main()
