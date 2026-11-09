"""Hyperspherical geometry of the per-source tokenizer vectors of one or more runs.

Reads `results/source_geometry/<run>/export.pt` (scripts/export_source_vectors.py; a run is a cohort,
optionally with a tokenizer suffix) and writes per run `geometry.json`, `geometry_sources.csv` and
`anchors.pt`. Geometry uses the stored rows of the second to fourth segment of every recording.

Per source s (unit rows u, anchor p, anchor coordinate a, tangent part t, tangent direction s):
* concentration: mean resultant length R, mean angle to the anchor;
* anchor vs principal directions: cos(p, v1) for the leading eigenvector v1 of E[uu^T], the residual
  bound ||Cov(a, t)|| / (E[a^2] - lambda_2(M)), the same with lambda_2(M) replaced by ||Sigma||_op, the
  concentration bound sqrt(Var(a) ||Sigma||_op) / (E[a^2] - lambda_2(M)) and the Yu-Wang-Samworth bound
  2||Sigma|| / R^2; |cos(p, PC_1)| and
  the squared norm of p in the span of the k leading centered PCs (k = 1, 10, 16, 64); the share of
  the centered variance along p and its rank among the centered eigenvalues;
* tangent principal axes (PCA of s, orthogonal to p): participation ratio and variance in the
  16 and 64 leading axes;
* reliability: subject-cluster bootstrap CIs, split halves over subjects, leave-one-subject-out
  agreement of each subject's anchor with the anchor of the remaining subjects;
* the 16 x 16 anchor cosine matrix.
With several plain cohorts, cross_cohort.json compares the anchors of every pair of cohorts; other
combinations of runs are written to cross_<runs>.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from neurofm_mechinterp.anchors import anchor_geometry
from neurofm_mechinterp.paths import RESULTS_ROOT

GEOM_ROOT = RESULTS_ROOT / "source_geometry"
BOOT_KEYS = ["R", "mean_angle_deg", "cos_anchor_uncentered_top", "cos_anchor_cpc1", "anchor_in_cpc16",
             "anchor_variance_fraction", "tangent_participation_ratio"]


def geometry_rows(export: dict) -> tuple[torch.Tensor, np.ndarray]:
    """Rows of the second to fourth segment of every recording, and the subject of each row segment."""
    subjects = np.array(export["subjects"])
    row_index = export["row_segment_index"].numpy()
    seg_pos = export["seg_in_subject"].numpy()[row_index]
    keep = seg_pos >= 1
    if keep.sum() == 0:
        keep = np.ones_like(seg_pos, dtype=bool)
    return export["rows"][torch.from_numpy(keep)], subjects[row_index][keep]


def source_rows(rows: torch.Tensor, source: int, mask: np.ndarray | None = None) -> torch.Tensor:
    block = rows[:, source]
    if mask is not None:
        block = block[torch.from_numpy(mask)]
    return block.reshape(-1, block.shape[-1]).to(torch.float64)


def summary(rows: torch.Tensor) -> tuple[dict, object]:
    g = anchor_geometry(rows, n_components=64)
    cos_top = float(g.uncentered_top @ g.anchor)
    t_eig = g.tangent_eigvals
    t_total = 1.0 - float(g.tangent_mean.norm() ** 2)        # total variance of the unit tangent directions
    return {
        "R": g.mean_resultant_length,
        "mean_angle_deg": g.mean_angle_deg,
        "coord_mean": g.coord_mean,
        "coord_var": g.coord_var,
        "coord_second_moment": g.coord_second_moment,
        "anchor_variance_fraction": g.anchor_variance_fraction,
        "anchor_variance_rank": g.anchor_variance_rank,
        "total_centered_variance": g.total_centered_variance,
        "cos_anchor_uncentered_top": cos_top,
        "sin_anchor_uncentered_top": float(np.sqrt(max(0.0, 1.0 - cos_top**2))),
        "residual_sin_bound": g.residual_sin_bound,
        "davis_kahan_sin_bound": g.davis_kahan_sin_bound,
        "weyl_sin_bound": g.weyl_sin_bound,
        "concentration_sin_bound": g.concentration_sin_bound,
        "cross_covariance_norm": g.cross_covariance_norm,
        "eigengap": g.eigengap,
        "coord_sd": float(np.sqrt(g.coord_var)),
        "cos_anchor_cpc1": float(g.cos_with_centered_pcs(1)[0]),
        "max_cos_anchor_cpc10": float(g.cos_with_centered_pcs(10).max()),
        "anchor_in_cpc1": g.anchor_in_centered_span(1),
        "anchor_in_cpc10": g.anchor_in_centered_span(10),
        "anchor_in_cpc16": g.anchor_in_centered_span(16),
        "anchor_in_cpc64": g.anchor_in_centered_span(64),
        "cpc1_variance_fraction": float(g.centered_eigvals[0] / g.total_centered_variance),
        "participation_ratio": g.participation_ratio,
        "tangent_participation_ratio": float(t_total**2 / (t_eig**2).sum()),
        "tangent_var_in_16": float(t_eig[:16].sum() / t_total),
        "tangent_var_in_64": float(t_eig[:64].sum() / t_total),
    }, g


def bootstrap_ci(rows: torch.Tensor, subjects: np.ndarray, source: int, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed + source)
    uniq = np.unique(subjects)
    blocks = {s: np.flatnonzero(subjects == s) for s in uniq}
    stats = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([blocks[s] for s in draw])
        block = rows[torch.from_numpy(idx), source]
        stats.append(summary(block.reshape(-1, block.shape[-1]).to(torch.float64))[0])
    return {k: [float(np.quantile([s[k] for s in stats], 0.025)), float(np.quantile([s[k] for s in stats], 0.975))]
            for k in BOOT_KEYS}


def unit_mean_direction(rows: torch.Tensor) -> torch.Tensor:
    m = torch.nn.functional.normalize(rows, dim=-1).mean(0)
    return m / m.norm()


def analyse(run: str, n_boot: int, n_split: int, seed: int) -> dict:
    export = torch.load(GEOM_ROOT / run / "export.pt", weights_only=False)
    rows, row_subjects = geometry_rows(export)
    uniq = np.unique(row_subjects)
    n_sources = rows.shape[1]
    per_source, geoms = [], []
    for s in range(n_sources):
        values, g = summary(source_rows(rows, s))
        values["source"] = s
        values["ci"] = bootstrap_ci(rows, row_subjects, s, n_boot, seed) if n_boot else {}
        per_source.append(values)
        geoms.append(g)
    anchors = torch.stack([g.anchor for g in geoms])
    cos = (anchors @ anchors.T).numpy()
    off = cos[~np.eye(n_sources, dtype=bool)]

    rng = np.random.default_rng(seed)
    split_cos = np.zeros((n_split, n_sources))
    for k in range(n_split):
        half = rng.permutation(uniq)[: len(uniq) // 2]
        mask = np.isin(row_subjects, half)
        for s in range(n_sources):
            split_cos[k, s] = float(unit_mean_direction(source_rows(rows, s, mask)) @ unit_mean_direction(source_rows(rows, s, ~mask)))
    loso = np.zeros((len(uniq), n_sources))
    for i, subj in enumerate(uniq):
        mask = row_subjects == subj
        for s in range(n_sources):
            loso[i, s] = float(unit_mean_direction(source_rows(rows, s, mask)) @ unit_mean_direction(source_rows(rows, s, ~mask)))

    out_dir = GEOM_ROOT / run
    torch.save({
        "anchor": anchors,
        "mean": torch.stack([g.mean for g in geoms]),
        "coord_mean": torch.tensor([g.coord_mean for g in geoms]),
        "centered_pcs": torch.stack([g.centered_pcs for g in geoms]),
        "centered_eigvals": torch.stack([g.centered_eigvals for g in geoms]),
        "tangent_pcs": torch.stack([g.tangent_pcs for g in geoms]),
        "tangent_eigvals": torch.stack([g.tangent_eigvals for g in geoms]),
        "tangent_mean": torch.stack([g.tangent_mean for g in geoms]),
        "uncentered_top": torch.stack([g.uncentered_top for g in geoms]),
    }, out_dir / "anchors.pt")
    table = pd.DataFrame([{k: v for k, v in row.items() if k != "ci"} for row in per_source])
    table["split_half_cos_min"] = split_cos.min(0)
    table["loso_cos_median"] = np.median(loso, 0)
    table["loso_cos_p05"] = np.quantile(loso, 0.05, axis=0)
    table["loso_cos_min"] = loso.min(0)
    table.to_csv(out_dir / "geometry_sources.csv", index=False)
    keys = [c for c in table.columns if c != "source"]
    result = {
        "run": run, "subjects": int(len(uniq)), "row_segments": int(rows.shape[0]),
        "rows_per_source": int(rows.shape[0] * rows.shape[2]),
        "per_source": per_source,
        "anchor_cosine_matrix": cos.tolist(),
        "anchor_cosine_offdiag": {"median": float(np.median(off)), "median_abs": float(np.median(np.abs(off))),
                                  "min": float(off.min()), "max": float(off.max()),
                                  "random_unit_sd": float(1.0 / np.sqrt(anchors.shape[1]))},
        "split_half_cos": {"mean": float(split_cos.mean()), "min": float(split_cos.min()), "n_splits": n_split},
        "loso_cos": {"median": float(np.median(loso)), "p05": float(np.quantile(loso, 0.05)), "min": float(loso.min())},
        "ranges": {k: [float(table[k].min()), float(table[k].max())] for k in keys},
        "medians": {k: float(table[k].median()) for k in keys},
        "n_boot": n_boot,
    }
    (out_dir / "geometry.json").write_text(json.dumps(result, indent=2))
    return result


def cross(runs: list[str]) -> dict:
    anchors = {r: torch.load(GEOM_ROOT / r / "anchors.pt", weights_only=False)["anchor"] for r in runs}
    out = {}
    for i, a in enumerate(runs):
        for b in runs[i + 1:]:
            cos = (anchors[a] * anchors[b]).sum(-1).numpy()
            out[f"{a}__{b}"] = {"per_source": cos.tolist(), "min": float(cos.min()), "median": float(np.median(cos))}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--n-split", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reuse", action="store_true", help="skip runs whose geometry.json and anchors.pt exist")
    args = parser.parse_args()
    for run in args.runs:
        if args.reuse and (GEOM_ROOT / run / "geometry.json").exists() and (GEOM_ROOT / run / "anchors.pt").exists():
            print(f"[geometry] {run}: reusing existing results")
            continue
        result = analyse(run, args.n_boot, args.n_split, args.seed)
        print(json.dumps({k: result[k] for k in ("run", "subjects", "anchor_cosine_offdiag", "split_half_cos", "loso_cos")}), flush=True)
    if len(args.runs) > 1:
        result = cross(args.runs)
        name = "cross_cohort.json" if all("__" not in r for r in args.runs) else "cross_" + "_".join(args.runs) + ".json"
        (GEOM_ROOT / name).write_text(json.dumps(result, indent=2))
        print(json.dumps({k: (round(v["min"], 4), round(v["median"], 4)) for k, v in result.items()}))


if __name__ == "__main__":
    main()
