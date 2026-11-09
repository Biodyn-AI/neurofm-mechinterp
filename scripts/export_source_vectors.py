"""Export per-source tokenizer vectors on the unit sphere for every segment of a cohort.

For each 10-s segment the frozen BrainTokenizer is run once and the following are stored
(`results/source_geometry/<cohort>/export.pt`):

* ``u_mean``   (n_seg, S, D): mean over the segment's rows of the L2-normalised source vectors;
* ``norm_mean``/``norm_std`` (n_seg, S): statistics of the pre-normalisation vector norms
  (information the quantizer discards);
* ``codes``    (n_seg, S, rows, Q): RVQ indices;
* ``mse``      (n_seg,): reconstruction MSE against the tokenizer's per-window z-scored target;
* ``bandpower`` (n_seg, C, 5): log10 Welch power in delta/theta/alpha/beta/gamma per channel;
* ``rows``     (n_row_seg, S, rows, D) float16 unit vectors for the first
  ``--row-segments-per-subject`` segments of every subject (used for geometry and ablations).

Unless --device is cpu, the device is first compared with the CPU on the first four segments (the unit vectors and,
except for the untrained random_init tokenizers, the reconstruction error and codes); the export falls back to the CPU
if they disagree.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.signal import welch

from neurofm_mechinterp.cohorts import get_cohort
from neurofm_mechinterp.data import BrainBatch
from neurofm_mechinterp.loading import load_tokenizer
from neurofm_mechinterp.paths import REPO_ROOT, RESULTS_ROOT
from neurofm_mechinterp.requantize import TokenizerProbe

BANDS = ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))


def tokenizer_tag(key: str) -> str:
    """Short name of a tokenizer key for output directories."""
    return (Path(key).name if "/" in key else key).replace("@", "_")


def ordered_segments(rows: list[dict], max_per_subject: int | None) -> list[dict]:
    """Group metadata rows by subject and order each subject's segments in time."""
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_subject[row["subject"]].append(row)
    ordered = []
    for subject in sorted(by_subject):
        segs = sorted(by_subject[subject], key=lambda r: (str(Path(r["path"]).parent), int(Path(r["path"]).stem)))
        if max_per_subject is not None:
            segs = segs[:max_per_subject]
        for k, seg in enumerate(segs):
            ordered.append({**seg, "seg_in_subject": k})
    return ordered


def load_batch(rows: list[dict]) -> BrainBatch:
    xs, poses, types = [], [], []
    for row in rows:
        payload = torch.load(REPO_ROOT / row["path"], map_location="cpu", weights_only=True)
        xs.append(payload["x"])
        poses.append(payload["pos"])
        types.append(payload["sensor_type"])
    return BrainBatch(torch.stack(xs), torch.stack(poses), torch.stack(types))


def band_power(x: np.ndarray, fs: float = 256.0) -> np.ndarray:
    freqs, psd = welch(x, fs=fs, nperseg=512, noverlap=256, axis=-1)
    out = np.stack([psd[..., (freqs >= lo) & (freqs < hi)].mean(axis=-1) for lo, hi in BANDS], axis=-1)
    return np.log10(out + 1e-20).astype(np.float32)


@torch.no_grad()
def run_batch(probe: TokenizerProbe, batch: BrainBatch) -> dict[str, torch.Tensor]:
    encoded = probe.encode(batch)
    feature = encoded.feature                                   # B S N T D
    quantized, indices = probe.quantize(feature)
    reconstruction = probe.tokenizer.decoder(quantized, encoded.sensor_embedding).float()
    mse = ((reconstruction - encoded.target.float()) ** 2).mean(dim=(1, 2, 3))
    b, s, n, t, d = feature.shape
    norms = feature.norm(dim=-1).reshape(b, s, n * t)
    u = torch.nn.functional.normalize(feature, dim=-1).reshape(b, s, n * t, d)
    return {
        "u": u.cpu(),
        "u_mean": u.mean(dim=2).cpu(),
        "norm_mean": norms.mean(dim=2).cpu(),
        "norm_std": norms.std(dim=2).cpu(),
        "codes": indices.reshape(b, s, n * t, -1).to(torch.int16).cpu(),
        "mse": mse.cpu(),
    }


def check_device(rows: list[dict], device: str, tokenizer: str = "released") -> dict:
    """Compare MPS against CPU on a few segments; returns max deviations."""
    batch = load_batch(rows)
    cpu = run_batch(TokenizerProbe(load_tokenizer(tokenizer, device="cpu")), batch)
    dev = run_batch(TokenizerProbe(load_tokenizer(tokenizer, device=device)), batch)
    return {
        "u_max_abs_diff": float((cpu["u"] - dev["u"]).abs().max()),
        "mse_max_abs_diff": float((cpu["mse"] - dev["mse"]).abs().max()),
        "code_agreement": float((cpu["codes"] == dev["codes"]).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-segments-per-subject", type=int, default=None)
    parser.add_argument("--row-segments-per-subject", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tokenizer", default="released", help="released | brainomni_tiny | random_init<seed> | replica dir")
    parser.add_argument("--subjects-file", type=Path, default=None, help="optional JSON list restricting the subjects")
    args = parser.parse_args()

    cohort = get_cohort(args.cohort)
    rows = cohort.load_metadata()
    if args.subjects_file is not None:
        keep = set(json.loads(args.subjects_file.read_text()))
        rows = [r for r in rows if r["subject"] in keep]
    rows = ordered_segments(rows, args.max_segments_per_subject)
    channel_counts = {row["channels"] for row in rows}
    if len(channel_counts) != 1:
        raise SystemExit(f"{args.cohort}: mixed channel counts {channel_counts}")
    run_name = args.cohort if args.tokenizer == "released" else f"{args.cohort}__{tokenizer_tag(args.tokenizer)}"
    out_dir = args.output_dir or RESULTS_ROOT / "source_geometry" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    check = {}
    if device != "cpu":
        check = check_device(rows[: min(4, len(rows))], device, args.tokenizer)
        print(f"[export] {device} vs cpu: {check}", flush=True)
        # an untrained tokenizer initialises its codebooks by k-means on the first batch, so its codes
        # (and reconstruction) differ between devices by construction; only the vectors are compared
        geometry_only = args.tokenizer.startswith("random_init")
        if check["u_max_abs_diff"] > 1e-3 or (not geometry_only and (check["mse_max_abs_diff"] > 1e-3 or check["code_agreement"] < 0.999)):
            print("[export] device mismatch -> falling back to cpu", flush=True)
            device = "cpu"
    probe = TokenizerProbe(load_tokenizer(args.tokenizer, device=device))

    keep_rows = [r["seg_in_subject"] < args.row_segments_per_subject for r in rows]
    collected: dict[str, list] = defaultdict(list)
    start = time.time()
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i : i + args.batch_size]
        batch = load_batch(chunk)
        out = run_batch(probe, batch)
        for key in ("u_mean", "norm_mean", "norm_std", "codes", "mse"):
            collected[key].append(out[key])
        mask = torch.tensor(keep_rows[i : i + args.batch_size])
        if mask.any():
            collected["rows"].append(out["u"][mask].to(torch.float16))
        collected["bandpower"].append(torch.from_numpy(band_power(batch.x.numpy())))
        if (i // args.batch_size) % 20 == 0:
            print(f"[export] {args.cohort}: {i + len(chunk)}/{len(rows)} segments, {time.time() - start:.0f}s", flush=True)

    export = {key: torch.cat(value) for key, value in collected.items()}
    export.update(
        subjects=[r["subject"] for r in rows],
        metadata_labels=[r.get("label") for r in rows],
        paths=[r["path"] for r in rows],
        seg_in_subject=torch.tensor([r["seg_in_subject"] for r in rows]),
        row_segment_index=torch.tensor([i for i, keep in enumerate(keep_rows) if keep]),
    )
    torch.save(export, out_dir / "export.pt")
    summary = {
        "cohort": args.cohort,
        "tokenizer": args.tokenizer,
        "accession": cohort.accession,
        "segments": len(rows),
        "subjects": len({r["subject"] for r in rows}),
        "channels": channel_counts.pop(),
        "row_segments": int(export["row_segment_index"].numel()),
        "device": device,
        "device_check": check,
        "mse_mean": float(export["mse"].mean()),
        "seconds": time.time() - start,
    }
    (out_dir / "export_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[export] done: {summary}", flush=True)


if __name__ == "__main__":
    main()
