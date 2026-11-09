"""Necessity atlas: reconstruction damage of sphere-preserving edits of one source at a time.

For every source of every selected segment, the unit vectors of that source are edited (see
neurofm_mechinterp.anchors: every edit maps the unit sphere onto itself), the frozen quantizer and
decoder are re-run, and the change in reconstruction error against the tokenizer's own target is
recorded (MSE, mean absolute error, and 1 - mean Pearson r), together with the mean geodesic
displacement, the mean change of colatitude and the mean change of tangent direction.

Anchors, mean anchor coordinates and tangent principal axes are cross-fitted: subjects are split
into two folds and every segment is edited with statistics estimated on the other fold (rows of the
second to fourth segment of each recording, results/source_geometry/<run>/export.pt).

Edits per source (p anchor, a anchor coordinate, s tangent direction):
  location (keep s): zero_anchor, half_anchor, mean_anchor, resample_anchor (within segment),
                     colat_{m20,m10,p10,p20} (colatitude shifted by -20..+20 degrees), to_pole (u -> p)
  content (keep a):  tpc_resample_{1,4,16,64} (resample ablation of the leading tangent principal
                     axes with the tangent directions of another subject's segment), tpc_mean_{1,4}
                     (mean ablation of the leading axes), rand_support_{1,4,16} (resample ablation of
                     random subspaces inside the span of the 64 leading tangent axes), rand_off_16
                     (random subspace of the whole tangent space), tangent_resample (the whole
                     tangent direction of another subject's segment; reference for content shares),
                     rot_matched_off / rot_matched_on (tangent rotation with zero_anchor's displacement,
                     towards a random direction / a direction in the support), rot_on_{30,60,90}
  identity:          swap (rotation taking p onto the anchor of another source; --swap-partners draws)
  whole source:      source_resample (vectors of another subject's segment)
  routing:           zero_anchor_keys / zero_anchor_values and swap_keys / swap_values (first partner): the
                     edit is applied only to the input of the decoder's attention keys or values.
With --edit-set routing, only zero_anchor, swap (first partner), tangent_resample and tpc_resample_4 are
applied, each also to the keys only and to the values only (results/necessity_routing/<run>/).
Every batch contains segments of at least two subjects, so that donor segments come from another subject.
Output: results/necessity_atlas/<run>/atlas.csv and baseline.csv.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neurofm_mechinterp.anchors import (
    anchor_geometry,
    decompose,
    geodesic_angle,
    matched_tangent_angle,
    mean_anchor,
    random_orthonormal,
    resample_anchor,
    rotate_anchor_to,
    scale_anchor,
    set_anchor_coordinate,
    set_tangent_direction,
    shift_colatitude,
    tangent_rotation,
    tangent_subspace_mean,
    tangent_subspace_resample,
    unit_rows,
    zero_anchor,
)
from neurofm_mechinterp.cohorts import get_cohort
from neurofm_mechinterp.loading import load_tokenizer
from neurofm_mechinterp.paths import RESULTS_ROOT
from neurofm_mechinterp.requantize import TokenizerProbe, decode_split, reconstruction_errors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_source_vectors import check_device, load_batch, ordered_segments, tokenizer_tag  # noqa: E402

N_COMPONENTS = 64


def select_segments(rows: list[dict], per_subject: int, positions: tuple[int, ...]) -> list[dict]:
    by_subject: dict[str, list[dict]] = {}
    for row in rows:
        by_subject.setdefault(row["subject"], []).append(row)
    chosen = []
    for subject in sorted(by_subject):
        segs = by_subject[subject]
        picks = [segs[p] for p in positions if p < len(segs)]
        if len(picks) < per_subject:
            picks += [s for s in segs if s not in picks][: per_subject - len(picks)]
        chosen += picks[:per_subject]
    return chosen


def derangement_across_subjects(subjects: list[str], rng: np.random.Generator) -> np.ndarray:
    n = len(subjects)
    for _ in range(1000):
        perm = rng.permutation(n)
        if all(subjects[i] != subjects[perm[i]] for i in range(n)):
            return perm
    return np.roll(np.arange(n), 1)


def fold_statistics(export: dict, subjects: list[str], n_sources: int) -> dict:
    """Per-source anchor statistics from the geometry rows (segments 2-4) of the given subjects."""
    seg_subjects = np.array(export["subjects"])
    row_index = export["row_segment_index"].numpy()
    seg_pos = export["seg_in_subject"].numpy()[row_index]
    mask = np.isin(seg_subjects[row_index], subjects) & (seg_pos >= 1)
    if mask.sum() == 0:
        mask = np.isin(seg_subjects[row_index], subjects)
    rows = export["rows"][torch.from_numpy(mask)]
    stats = {"anchor": [], "coord_mean": [], "tangent_mean": [], "tangent_pcs": [], "tangent_eigvals": []}
    for s in range(n_sources):
        g = anchor_geometry(rows[:, s].reshape(-1, rows.shape[-1]).to(torch.float64), n_components=N_COMPONENTS)
        stats["anchor"].append(g.anchor)
        stats["coord_mean"].append(g.coord_mean)
        stats["tangent_mean"].append(g.tangent_mean)
        stats["tangent_pcs"].append(g.tangent_pcs)
        stats["tangent_eigvals"].append(g.tangent_eigvals)
    return {
        "anchor": torch.stack(stats["anchor"]),
        "coord_mean": torch.tensor(stats["coord_mean"]),
        "tangent_mean": torch.stack(stats["tangent_mean"]),
        "tangent_pcs": torch.stack(stats["tangent_pcs"]),
        "tangent_eigvals": torch.stack(stats["tangent_eigvals"]),
    }


def build_edits(unit: torch.Tensor, groups: torch.Tensor, perm_rows: torch.Tensor, s: int, st: dict,
                gen: torch.Generator, n_random: int, partners: list[int]) -> tuple[dict, dict]:
    """Return {name: edited unit rows} and {name: extra info} for one source (float64, CPU)."""
    p = st["anchor"][s]
    support = st["tangent_pcs"][s][:N_COMPONENTS]
    t_mean = st["tangent_mean"][s]
    edits, info = {}, {}
    edits["zero_anchor"] = zero_anchor(unit, p)
    edits["half_anchor"] = scale_anchor(unit, p, 0.5)
    edits["mean_anchor"] = mean_anchor(unit, p, float(st["coord_mean"][s]))
    edits["resample_anchor"] = resample_anchor(unit, p, groups=groups, generator=gen)
    for deg in (-20, -10, 10, 20):
        edits[f"colat_{'m' if deg < 0 else 'p'}{abs(deg)}"] = shift_colatitude(unit, p, math.radians(deg))
    edits["to_pole"] = p.expand_as(unit).clone()
    _, s_dir = decompose(unit, p)
    donor = s_dir[perm_rows]
    for k in (1, 4, 16, 64):
        edits[f"tpc_resample_{k}"] = tangent_subspace_resample(unit, p, support[:k], donor)
    edits["tpc_mean_1"] = tangent_subspace_mean(unit, p, support[:1], t_mean)
    edits["tpc_mean_4"] = tangent_subspace_mean(unit, p, support[:4], t_mean)
    for r in range(n_random):
        for k in (1, 4, 16):
            basis = random_orthonormal(k, unit.shape[1], generator=gen, within=support, orthogonal_to=p)
            edits[f"rand_support_{k}#{r}"] = tangent_subspace_resample(unit, p, basis, donor)
    basis = random_orthonormal(16, unit.shape[1], generator=gen, orthogonal_to=p)
    edits["rand_off_16"] = tangent_subspace_resample(unit, p, basis, donor)
    edits["tangent_resample"] = set_tangent_direction(unit, p, donor)
    phi, _ = matched_tangent_angle(unit, p)
    edits["rot_matched_off"] = tangent_rotation(unit, p, phi, generator=gen)
    edits["rot_matched_on"] = tangent_rotation(unit, p, phi, generator=gen, within=support)
    for deg in (30, 60, 90):
        edits[f"rot_on_{deg}"] = tangent_rotation(unit, p, math.radians(deg), generator=gen, within=support)
    for j, other in enumerate(partners):
        edits[f"swap#{j}"] = rotate_anchor_to(unit, p, st["anchor"][other])
        info[f"swap#{j}"] = {"partner": int(other), "partner_cos": float(p @ st["anchor"][other])}
    edits["source_resample"] = unit[perm_rows].clone()
    return edits, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--segments-per-subject", type=int, default=2)
    parser.add_argument("--positions", type=int, nargs="+", default=[2, 6])
    parser.add_argument("--decode-channels", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-random", type=int, default=2)
    parser.add_argument("--swap-partners", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokenizer", default="released")
    parser.add_argument("--subjects-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--edit-set", choices=["full", "routing"], default="full",
                        help="routing: only zero_anchor, swap (first partner), tangent_resample and tpc_resample_4, each "
                             "also applied to the decoder's keys only and values only")
    args = parser.parse_args()

    run_name = args.cohort if args.tokenizer == "released" else f"{args.cohort}__{tokenizer_tag(args.tokenizer)}"
    cohort = get_cohort(args.cohort)
    meta_rows = cohort.load_metadata()
    if args.subjects_file is not None:
        keep = set(json.loads(args.subjects_file.read_text()))
        meta_rows = [r for r in meta_rows if r["subject"] in keep]
    segments = select_segments(ordered_segments(meta_rows, None), args.segments_per_subject, tuple(args.positions))
    export = torch.load(RESULTS_ROOT / "source_geometry" / run_name / "export.pt", weights_only=False)
    n_sources = export["rows"].shape[1]

    rng = np.random.default_rng(args.seed)
    all_subjects = sorted(set(export["subjects"]))
    fold_of = {s: i % 2 for i, s in enumerate(rng.permutation(all_subjects))}
    stats = {f: fold_statistics(export, [s for s in all_subjects if fold_of[s] != f], n_sources) for f in (0, 1)}
    segments.sort(key=lambda r: (fold_of.get(r["subject"], 0), r["subject"]))

    device = args.device
    check = {}
    if device != "cpu":
        check = check_device(segments[:4], device, args.tokenizer)
        print(f"[atlas] {device} vs cpu: {check}", flush=True)
        geometry_only = args.tokenizer.endswith("@init") or args.tokenizer.startswith("random_init")
        if check["u_max_abs_diff"] > 1e-3 or (not geometry_only and check["mse_max_abs_diff"] > 1e-3):
            device = "cpu"
    probe = TokenizerProbe(load_tokenizer(args.tokenizer, device=device))
    tokenizer = probe.tokenizer

    n_channels = segments[0]["channels"]
    channels = None
    if n_channels > args.decode_channels:
        channels = torch.from_numpy(np.sort(rng.choice(n_channels, size=args.decode_channels, replace=False))).to(probe.device)

    out_dir = args.output_dir or RESULTS_ROOT / ("necessity_atlas" if args.edit_set == "full" else "necessity_routing") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(args.seed)
    records, baseline_rows = [], []
    start = time.time()
    batches = []
    for f in (0, 1):
        fold_segments = [r for r in segments if fold_of.get(r["subject"], 0) == f]
        chunks = [fold_segments[i : i + args.batch_size] for i in range(0, len(fold_segments), args.batch_size)]
        while len(chunks) > 1 and len({r["subject"] for r in chunks[-1]}) < 2:   # donors must come from another subject
            last = chunks.pop()
            chunks[-1] = chunks[-1] + last
        batches += [(f, chunk) for chunk in chunks]

    for b_idx, (fold, chunk) in enumerate(batches):
        st = stats[fold]
        encoded = probe.encode(load_batch(chunk))
        feature = encoded.feature
        bsz, _, n_patch, n_frame, dim = feature.shape
        q_base, _ = probe.quantize(feature)
        rec_base, att_base = decode_split(tokenizer, q_base, q_base, encoded.sensor_embedding, channels, return_attention=True)
        target = encoded.target if channels is None else encoded.target[:, channels]
        base = {k: v.cpu().numpy() for k, v in reconstruction_errors(rec_base, target).items()}
        att_base = att_base.reshape(bsz, n_patch * n_frame, -1, n_sources).mean(dim=(1, 2)).cpu().numpy()
        for i, row in enumerate(chunk):
            baseline_rows.append({"subject": row["subject"], "label": row.get("label"), "path": row["path"], "fold": fold,
                                  **{k: float(v[i]) for k, v in base.items()}})
        rows_per_seg = n_patch * n_frame
        groups = torch.arange(bsz).repeat_interleave(rows_per_seg)
        seg_perm = derangement_across_subjects([r["subject"] for r in chunk], rng)
        perm_rows = (torch.from_numpy(seg_perm).repeat_interleave(rows_per_seg) * rows_per_seg
                     + torch.arange(rows_per_seg).repeat(bsz))
        for s in range(n_sources):
            unit = unit_rows(feature[:, s].reshape(-1, dim).cpu().to(torch.float64))
            others = [o for o in range(n_sources) if o != s]
            partners = list(rng.choice(others, size=args.swap_partners, replace=False))
            edits, info = build_edits(unit, groups, perm_rows, s, st, gen, args.n_random, partners)
            routed = ("zero_anchor", "swap#0")
            if args.edit_set == "routing":
                edits = {k: v for k, v in edits.items() if k in ("zero_anchor", "swap#0", "tangent_resample", "tpc_resample_4")}
                routed = tuple(edits)
            a0, s0 = decompose(unit, st["anchor"][s])
            for name, edited in edits.items():
                new_feature = feature.clone()
                new_feature[:, s] = edited.reshape(bsz, n_patch, n_frame, dim).to(feature)
                q_edit, _ = probe.quantize(new_feature)
                variants = {name: (q_edit, q_edit)}
                if name in routed:
                    stem = name.split("#")[0]
                    variants[f"{stem}_keys"] = (q_edit, q_base)
                    variants[f"{stem}_values"] = (q_base, q_edit)
                a1, s1 = decompose(edited, st["anchor"][s])
                disp = torch.rad2deg(geodesic_angle(unit, edited)).reshape(bsz, -1).mean(1).numpy()
                dtheta = torch.rad2deg(torch.arccos(a1) - torch.arccos(a0)).reshape(bsz, -1).mean(1).numpy()
                dtan = torch.rad2deg(geodesic_angle(s0, s1)).reshape(bsz, -1).mean(1).numpy()
                for vname, (qk, qv) in variants.items():
                    need_att = vname == "zero_anchor"
                    out = decode_split(tokenizer, qk, qv, encoded.sensor_embedding, channels, return_attention=need_att)
                    rec, att = (out if need_att else (out, None))
                    err = {k: v.cpu().numpy() for k, v in reconstruction_errors(rec, target).items()}
                    att_s = att.reshape(bsz, rows_per_seg, -1, n_sources).mean(dim=(1, 2))[:, s].cpu().numpy() if att is not None else None
                    base_name, _, rep = vname.partition("#")
                    extra = info.get(name, {}) if base_name.startswith("swap") else {}
                    for i, row in enumerate(chunk):
                        records.append({
                            "subject": row["subject"], "label": row.get("label"), "path": row["path"], "fold": fold,
                            "source": s, "edit": base_name, "rep": int(rep) if rep else 0,
                            "d_mse": float(err["mse"][i] - base["mse"][i]),
                            "d_mae": float(err["mae"][i] - base["mae"][i]),
                            "d_omr": float(err["one_minus_r"][i] - base["one_minus_r"][i]),
                            "base_mse": float(base["mse"][i]),
                            "displacement_deg": float(disp[i]), "dtheta_deg": float(dtheta[i]), "dtangent_deg": float(dtan[i]),
                            "partner": extra.get("partner"), "partner_cos": extra.get("partner_cos"),
                            "att_base": float(att_base[i, s]) if att_s is not None else None,
                            "att_edit": float(att_s[i]) if att_s is not None else None,
                        })
        print(f"[atlas] {run_name}: batch {b_idx + 1}/{len(batches)} ({sum(len(c) for _, c in batches[: b_idx + 1])}/{len(segments)} segments), "
              f"{time.time() - start:.0f}s", flush=True)
        pd.DataFrame(records).to_csv(out_dir / "atlas.csv", index=False)
        pd.DataFrame(baseline_rows).to_csv(out_dir / "baseline.csv", index=False)

    meta = {"run": run_name, "cohort": args.cohort, "tokenizer": args.tokenizer, "segments": len(segments),
            "subjects": len({r["subject"] for r in segments}), "decode_channels": None if channels is None else channels.cpu().tolist(),
            "device": device, "device_check": check, "folds": {k: int(v) for k, v in fold_of.items()},
            "seconds": time.time() - start}
    (out_dir / "atlas_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[atlas] done {meta['segments']} segments in {meta['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
