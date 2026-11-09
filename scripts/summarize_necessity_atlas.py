"""Summarise necessity-atlas runs (scripts/run_necessity_atlas.py) with subject-level statistics.

Reads `results/necessity_atlas/<run>/atlas.csv` for every run given and writes per run
`summary.json` and `source_summary.csv`, plus `results/necessity_atlas/summary.json` with all runs.

Subjects are the unit of inference: changes in reconstruction error are averaged over segments
(and over random draws) within subject x source x edit, and every interval is a percentile interval
of a subject-cluster bootstrap that resamples the same subjects for all edits, so that differences
and ratios between edits are paired. Per source and edit: the mean over subjects of dMSE, dMAE and
d(1 - r), the relative change dMSE / baseline MSE, and bootstrap intervals. Across sources: median,
range, and the number of sources whose interval lies above (below) zero.

Derived quantities (per source, then median and range over sources, with bootstrap intervals for
the across-source median):
  * location: mean_anchor, resample_anchor and half_anchor as fractions of zero_anchor; the colatitude
    tuning curve (colat_m20 ... colat_p20);
  * axis specificity: zero_anchor minus the tangent rotations with the same geodesic displacement
    (rot_matched_on, rot_matched_off);
  * content: tpc_resample_k and rand_support_k as fractions of tangent_resample and the paired
    difference tpc_resample_k - rand_support_k; tangent_resample as a fraction of source_resample;
  * identity and routing: swap damage against the cosine between the two anchors; the shares of
    zero_anchor and swap reproduced when the edit reaches only the decoder's keys or only its values;
  * attention: the mean attention weight of the edited source before and after zero_anchor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from neurofm_mechinterp.paths import RESULTS_ROOT

ATLAS_ROOT = RESULTS_ROOT / "necessity_atlas"
METRICS = ["d_mse", "d_mae", "d_omr", "rel_mse"]
COLATITUDE = [("colat_m20", -20), ("colat_m10", -10), (None, 0), ("colat_p10", 10), ("colat_p20", 20)]
RATIOS = {
    "mean_over_zero": ("mean_anchor", "zero_anchor"),
    "resample_over_zero": ("resample_anchor", "zero_anchor"),
    "half_over_zero": ("half_anchor", "zero_anchor"),
    "keys_over_zero": ("zero_anchor_keys", "zero_anchor"),
    "values_over_zero": ("zero_anchor_values", "zero_anchor"),
    "keys_over_swap": ("swap_keys", "swap_p0"),
    "values_over_swap": ("swap_values", "swap_p0"),
    "keys_over_tangent": ("tangent_resample_keys", "tangent_resample"),
    "values_over_tangent": ("tangent_resample_values", "tangent_resample"),
    "keys_over_tpc4": ("tpc_resample_4_keys", "tpc_resample_4"),
    "values_over_tpc4": ("tpc_resample_4_values", "tpc_resample_4"),
    "tangent_over_source": ("tangent_resample", "source_resample"),
    "zero_over_source": ("zero_anchor", "source_resample"),
    "swap_over_source": ("swap", "source_resample"),
    "pole_over_source": ("to_pole", "source_resample"),
    "zero_over_tangent": ("zero_anchor", "tangent_resample"),
    "zero_over_rot_on": ("zero_anchor", "rot_matched_on"),
    "zero_over_rot_off": ("zero_anchor", "rot_matched_off"),
    "colat_p20_over_zero": ("colat_p20", "zero_anchor"),
    **{f"tpc{k}_over_tangent": (f"tpc_resample_{k}", "tangent_resample") for k in (1, 4, 16, 64)},
    **{f"rand{k}_over_tangent": (f"rand_support_{k}", "tangent_resample") for k in (1, 4, 16)},
    "randoff16_over_tangent": ("rand_off_16", "tangent_resample"),
    "tpcmean1_over_tangent": ("tpc_mean_1", "tangent_resample"),
    "tpcmean4_over_tangent": ("tpc_mean_4", "tangent_resample"),
}
DIFFERENCES = {
    "zero_minus_rot_on": ("zero_anchor", "rot_matched_on"),
    "zero_minus_rot_off": ("zero_anchor", "rot_matched_off"),
    "rot_on_minus_rot_off": ("rot_matched_on", "rot_matched_off"),
    "colat_p20_minus_m20": ("colat_p20", "colat_m20"),
    "colat_p10_minus_m10": ("colat_p10", "colat_m10"),
    **{f"tpc{k}_minus_rand{k}": (f"tpc_resample_{k}", f"rand_support_{k}") for k in (1, 4, 16)},
    "rand16_minus_randoff16": ("rand_support_16", "rand_off_16"),
    "swap_minus_zero": ("swap", "zero_anchor"),
    "keys_minus_values_zero": ("zero_anchor_keys", "zero_anchor_values"),
    "keys_minus_values_swap": ("swap_keys", "swap_values"),
    "keys_minus_values_tangent": ("tangent_resample_keys", "tangent_resample_values"),
    "keys_minus_values_tpc4": ("tpc_resample_4_keys", "tpc_resample_4_values"),
}


def interval(samples: np.ndarray) -> list[float]:
    return [float(np.nanquantile(samples, 0.025)), float(np.nanquantile(samples, 0.975))]


def subject_arrays(df: pd.DataFrame, metric: str) -> tuple[dict[str, np.ndarray], list[str], list[int]]:
    """{edit: subjects x sources array of subject means}."""
    sm = df.groupby(["subject", "source", "edit"], as_index=False)[metric].mean()
    subjects = sorted(sm["subject"].unique())
    sources = sorted(sm["source"].unique())
    out = {}
    for edit, block in sm.groupby("edit"):
        wide = block.pivot(index="subject", columns="source", values=metric).reindex(index=subjects, columns=sources)
        out[edit] = wide.to_numpy(dtype=float)
    return out, subjects, sources


def per_source_stats(arrays: dict[str, np.ndarray], boot_idx: np.ndarray) -> dict:
    stats = {}
    for edit, values in arrays.items():
        point = np.nanmean(values, axis=0)
        boots = np.stack([np.nanmean(values[idx], axis=0) for idx in boot_idx])          # n_boot x sources
        lo, hi = np.nanquantile(boots, 0.025, axis=0), np.nanquantile(boots, 0.975, axis=0)
        med_boot = np.nanmedian(boots, axis=1)
        stats[edit] = {
            "per_source": point.tolist(), "per_source_ci": np.stack([lo, hi], 1).tolist(),
            "median": float(np.nanmedian(point)), "median_ci": interval(med_boot),
            "mean": float(np.nanmean(point)), "mean_ci": interval(np.nanmean(boots, axis=1)),
            "min": float(np.nanmin(point)), "max": float(np.nanmax(point)),
            "n_above_zero": int((lo > 0).sum()), "n_below_zero": int((hi < 0).sum()),
            "n_sources": int(point.size),
        }
    return stats


def derived(arrays: dict[str, np.ndarray], boot_idx: np.ndarray, pairs: dict, kind: str) -> dict:
    out = {}
    for name, (a, b) in pairs.items():
        if a not in arrays or b not in arrays:
            continue
        x, y = arrays[a], arrays[b]

        def value(rows: np.ndarray | slice) -> np.ndarray:
            mx, my = np.nanmean(x[rows], axis=0), np.nanmean(y[rows], axis=0)
            return mx / my if kind == "ratio" else mx - my

        point = value(slice(None))
        boots = np.stack([value(idx) for idx in boot_idx])
        lo, hi = np.nanquantile(boots, 0.025, axis=0), np.nanquantile(boots, 0.975, axis=0)
        out[name] = {
            "per_source": point.tolist(), "per_source_ci": np.stack([lo, hi], 1).tolist(),
            "median": float(np.nanmedian(point)), "median_ci": interval(np.nanmedian(boots, axis=1)),
            "min": float(np.nanmin(point)), "max": float(np.nanmax(point)),
            "n_above_zero": int((lo > 0).sum()), "n_below_zero": int((hi < 0).sum()),
            "n_above_one": int((lo > 1).sum()), "n_below_one": int((hi < 1).sum()),
        }
    return out


def swap_alignment(df: pd.DataFrame) -> dict:
    swaps = df[(df["edit"] == "swap") & df["partner"].notna()]
    if swaps.empty:
        return {}
    pairs = swaps.groupby(["source", "partner"], as_index=False).agg(d_mse=("d_mse", "mean"), cos=("partner_cos", "mean"))
    rho, pval = spearmanr(pairs["cos"], pairs["d_mse"])
    within = []
    for _, block in pairs.groupby("source"):
        if len(block) >= 3 and block["cos"].nunique() > 1:
            within.append(spearmanr(block["cos"], block["d_mse"])[0])
    return {"n_pairs": int(len(pairs)), "spearman_rho": float(rho), "spearman_p": float(pval),
            "within_source_rho_median": float(np.nanmedian(within)) if within else None,
            "partner_cos_range": [float(pairs["cos"].min()), float(pairs["cos"].max())]}


def summarise_run(run: str, n_boot: int, seed: int, root=ATLAS_ROOT) -> dict:
    df = pd.read_csv(root / run / "atlas.csv")
    df["rel_mse"] = df["d_mse"] / df["base_mse"]
    # the keys-only and values-only swaps use the first partner only; "swap_p0" is the full swap to that partner
    first = df[(df["edit"] == "swap") & (df["rep"] == 0)].assign(edit="swap_p0")
    df = pd.concat([df, first], ignore_index=True)
    baseline = pd.read_csv(root / run / "baseline.csv")
    rng = np.random.default_rng(seed)
    result = {"run": run, "n_subjects": int(df["subject"].nunique()), "n_segments": int(df["path"].nunique()),
              "baseline_mse_median": float(baseline.groupby("subject")["mse"].mean().median()),
              "edits": {}, "ratios": {}, "differences": {}}
    first = True
    for metric in METRICS:
        arrays, subjects, sources = subject_arrays(df, metric)
        if first:
            boot_idx = rng.integers(0, len(subjects), size=(n_boot, len(subjects)))
            first = False
        result["edits"][metric] = per_source_stats(arrays, boot_idx)
        result["differences"][metric] = derived(arrays, boot_idx, DIFFERENCES, "difference")
        if metric == "d_mse":
            result["ratios"] = derived(arrays, boot_idx, RATIOS, "ratio")
            stats = result["edits"]["d_mse"]
            if any(e and e not in stats for e, _ in COLATITUDE):
                continue
            result["colatitude_curve"] = {
                "shift_deg": [d for _, d in COLATITUDE],
                "median": [stats[e]["median"] if e else 0.0 for e, _ in COLATITUDE],
                "per_source": [stats[e]["per_source"] if e else [0.0] * len(sources) for e, _ in COLATITUDE],
            }
    displacement = df.groupby("edit").agg(displacement_deg=("displacement_deg", "median"), dtheta_deg=("dtheta_deg", "median"),
                                          dtangent_deg=("dtangent_deg", "median"))
    result["displacement"] = displacement.to_dict(orient="index")
    result["swap_alignment"] = swap_alignment(df)
    zero = df[(df["edit"] == "zero_anchor") & df["att_base"].notna()]
    if not zero.empty:
        att = zero.groupby(["subject", "source"], as_index=False)[["att_base", "att_edit"]].mean()
        base = att.pivot(index="subject", columns="source", values="att_base").to_numpy()
        edit = att.pivot(index="subject", columns="source", values="att_edit").to_numpy()
        point = edit.mean(0) / base.mean(0) - 1.0
        boots = np.stack([edit[idx].mean(0) / base[idx].mean(0) - 1.0 for idx in boot_idx])
        hi = np.quantile(boots, 0.975, axis=0)
        result["attention_zero_anchor"] = {
            "base_median": float(np.median(base.mean(0))), "edit_median": float(np.median(edit.mean(0))),
            "relative_change_median": float(np.median(point)), "relative_change_range": [float(point.min()), float(point.max())],
            "edit_over_uniform_range": [float(edit.mean(0).min() * base.shape[1]), float(edit.mean(0).max() * base.shape[1])],
            "n_sources_decreased": int((hi < 0).sum()), "uniform": 1.0 / base.shape[1],
        }
    rows = []
    for edit, st in result["edits"]["d_mse"].items():
        for s, (v, ci) in enumerate(zip(st["per_source"], st["per_source_ci"])):
            rows.append({"edit": edit, "source": sources[s], "d_mse": v, "ci_low": ci[0], "ci_high": ci[1],
                         "rel_mse": result["edits"]["rel_mse"][edit]["per_source"][s]})
    pd.DataFrame(rows).to_csv(root / run / "source_summary.csv", index=False)
    (root / run / "summary.json").write_text(json.dumps(result, indent=1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", default=None, help="default: every run with an atlas.csv")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", type=Path, default=ATLAS_ROOT)
    args = parser.parse_args()
    runs = args.runs or sorted(p.parent.name for p in args.root.glob("*/atlas.csv"))
    combined_path = args.root / "summary.json"
    combined = json.loads(combined_path.read_text()) if combined_path.exists() else {}
    for run in runs:
        result = summarise_run(run, args.n_boot, args.seed, args.root)
        combined[run] = result
        e = result["edits"]["d_mse"]
        print(f"{run}: {result['n_subjects']} subjects, zero_anchor median {e['zero_anchor']['median']:.4f} "
              f"({e['zero_anchor']['n_above_zero']}/{e['zero_anchor']['n_sources']} > 0)", flush=True)
    combined_path.write_text(json.dumps(combined, indent=1))


if __name__ == "__main__":
    main()
