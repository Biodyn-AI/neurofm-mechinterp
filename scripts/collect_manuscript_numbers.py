"""Collect every quantitative value used in the manuscript into LaTeX macros and tables.

Reads the outputs under results/ and writes
  paper/numbers.tex     (\\newcommand macros)
  paper/tables/*.tex    (data-driven tables)
  results/manuscript_numbers.json                         (the same values, machine-readable)
Missing inputs produce a visible placeholder and are listed at the end; the manuscript is final
only when that list is empty.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neurofm_mechinterp.anchors import anchor_arrangement, off_sphere_coordinate
from neurofm_mechinterp.cohorts import get_cohort
from neurofm_mechinterp.paths import REPO_ROOT, RESULTS_ROOT
from neurofm_mechinterp.readout import bh_adjust, holm_adjust

OUT_DIR = REPO_ROOT / "paper"
TAB_DIR = OUT_DIR / "tables"
GEOM = RESULTS_ROOT / "source_geometry"
ATLAS = RESULTS_ROOT / "necessity_atlas"
TOKENS = RESULTS_ROOT / "token_structure"
READOUT = RESULTS_ROOT / "diagnostic_readout"
REPLICAS = RESULTS_ROOT / "tokenizer_replicas"
COHORTS = [("pd_ucsd", "PD-UCSD"), ("pd_iowa", "PD-Iowa"), ("dementia", "Dementia"), ("mdd_meg", "MDD-MEG")]
EEG = ["pd_ucsd", "pd_iowa", "dementia"]
CONTRASTS = [("pd_ucsd", "pd_vs_hc", "PD-UCSD, PD vs HC", "UCSDPD"), ("pd_iowa", "pd_vs_hc", "PD-Iowa, PD vs HC", "IowaPD"),
             ("dementia", "ad_vs_cn", "Dementia, AD vs CN", "AD"), ("dementia", "ftd_vs_cn", "Dementia, FTD vs CN", "FTD"),
             ("mdd_meg", "mdd_vs_hv", "MDD-MEG, MDD vs HV", "MDD")]
REPLICA_TAGS = [("seed1", "replica 1"), ("seed2", "replica 2"), ("seed1_srcemb", "variant 1"), ("seed2_srcemb", "variant 2")]
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
MISSING: list[str] = []
MACROS: dict[str, str] = {}


def placeholder(name: str) -> str:
    MISSING.append(name)
    return r"\textbf{[??]}"


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def f(x: float, nd: int = 2) -> str:
    if x is None or not math.isfinite(x):
        return r"\textbf{[??]}"
    text = f"{x:.{nd}f}"
    return text[1:] if text.startswith("-") and float(text) == 0.0 else text  # no "-0.0"


def pct(x: float, nd: int = 0) -> str:
    return f(100 * x if x is not None else None, nd)


def tnum(text: str) -> str:
    """Typeset a formatted number for a table (true minus sign)."""
    return "$-$" + text[1:] if text.startswith("-") else text


def floor_to(x: float, nd: int) -> float:
    return math.floor(x * 10**nd) / 10**nd if math.isfinite(x) else x


def ceil_to(x: float, nd: int) -> float:
    return math.ceil(x * 10**nd) / 10**nd if math.isfinite(x) else x


def span(lo: str, hi: str) -> str:
    return lo if lo == hi else f"{lo}--{hi}"


def ci(lo: float, hi: float, nd: int = 2) -> str:
    """'[lo, hi]' with nd decimals; a bound that would round to zero keeps one more digit and its sign."""
    def one(x: float) -> str:
        return f(x, nd + 1) if x != 0 and abs(x) < 0.5 * 10 ** -nd else f(x, nd)
    return f"[{one(lo)}, {one(hi)}]"


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def pval(p: float) -> str:
    return f"$p={p:.3f}$" if p >= 0.001 else r"$p<0.001$"


def setm(name: str, value: str) -> None:
    MACROS[name] = value


def cohort_runs(subdir: Path, filename: str, suffix: str = "") -> dict[str, dict]:
    out = {}
    for key, _ in COHORTS:
        data = load_json(subdir / f"{key}{suffix}" / filename)
        if data is None:
            placeholder(f"{subdir.name}:{key}{suffix}")
        else:
            out[key] = data
    return out


def arrangement(geometry: dict):
    """Arrangement of the 16 anchors of one run (effective dimension, leading shares, smallest angle)."""
    return anchor_arrangement(torch.tensor(geometry["anchor_cosine_matrix"]))


# ----------------------------------------------------------------------------- static and cohorts
def static_numbers() -> None:
    state = torch.load(REPO_ROOT / "external" / "checkpoints" / "BrainOmni" / "braintokenizer" / "BrainTokenizer.pt",
                       map_location="cpu", weights_only=True)
    n_params = sum(v.numel() for k, v in state.items() if "quantizer" not in k)
    setm("NumParams", f"{n_params / 1e6:.2f}~million")


GROUP_NAMES = {("pd_ucsd", "hc"): "HC", ("pd_ucsd", "pd"): "PD", ("pd_iowa", "Control"): "HC", ("pd_iowa", "PD"): "PD",
               ("dementia", "C"): "CN", ("dementia", "A"): "AD", ("dementia", "F"): "FTD",
               ("mdd_meg", "HV"): "HV", ("mdd_meg", "MDD"): "MDD"}
GROUP_ORDER = {("pd_ucsd", "hc"): 0, ("pd_ucsd", "pd"): 1, ("pd_iowa", "Control"): 0, ("pd_iowa", "PD"): 1,
               ("dementia", "C"): 0, ("dementia", "A"): 1, ("dementia", "F"): 2, ("mdd_meg", "HV"): 0, ("mdd_meg", "MDD"): 1}
RECORDING = {"pd_ucsd": "EEG", "pd_iowa": "EEG", "dementia": "EEG", "mdd_meg": "MEG"}


def cohort_numbers() -> None:
    rows, total_subj, total_seg = [], 0, 0
    for key, label in COHORTS:
        summary = load_json(GEOM / key / "export_summary.json")
        cohort = get_cohort(key)
        if summary is None:
            rows.append((label, cohort, None))
            placeholder(f"cohort:{key}")
            continue
        total_subj += summary["subjects"]
        total_seg += summary["segments"]
        rows.append((label, cohort, summary))
    complete = all(r[2] for r in rows)
    setm("NsubjTotal", str(total_subj) if complete else placeholder("NsubjTotal"))
    setm("NsegTotal", f"{total_seg:,}".replace(",", "{,}") if complete else placeholder("NsegTotal"))
    lines = [r"\begin{table}[tbp]", r"\renewcommand{\baselinestretch}{1}",
             r"\caption{Cohorts. All recordings are resting state (PD-Iowa eyes open, Dementia eyes closed); segments are "
             r"non-overlapping 10-s sections of a recording (five 2-s tokenizer windows each); group sizes count the subjects "
             r"analyzed. HC: healthy controls; PD: Parkinson's disease; CN: cognitively normal; AD: Alzheimer's disease; FTD: "
             r"frontotemporal dementia; HV: healthy volunteers; MDD: major depressive disorder.}",
             r"\label{tab:cohorts}", r"\centering\scriptsize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}lllrrr@{\hspace{6pt}}l@{}}", r"\toprule",
             r"\textbf{cohort} & \textbf{accession} & \textbf{modality} & \textbf{sensors} & \textbf{subjects} & \textbf{segments} & \textbf{groups} \\",
             r"\midrule"]
    for label, cohort, summary in rows:
        if summary is None:
            lines.append(f"{label} & {cohort.accession} & -- & -- & -- & -- & -- \\\\")
            continue
        # group of every analysed subject, from lists/ (written by scripts/export_subject_lists.py)
        listed = pd.read_csv(REPO_ROOT / "lists" / f"{cohort.name}_subjects.csv", dtype=str, keep_default_na=False)
        counts = {g: int(n) for g, n in listed["group"].replace("", "?").value_counts().items()}
        group_text = ", ".join(f"{GROUP_NAMES.get((cohort.name, g), g)} {n}" for g, n in
                               sorted(counts.items(), key=lambda kv: GROUP_ORDER.get((cohort.name, kv[0]), 9)))
        lines.append(f"{label} & {cohort.accession} & {RECORDING[cohort.name]} & {summary['channels']} & {summary['subjects']} & "
                     + f"{summary['segments']:,}".replace(",", "{,}") + f" & {group_text} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB_DIR / "cohorts.tex").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- geometry
def tangent_pc_alignment(run: str, k: int = 16) -> np.ndarray:
    """Per source, the mean squared cosine of the principal angles between the spans of the k leading centered PCs
    and the k leading tangent axes."""
    stored = torch.load(GEOM / run / "anchors.pt", weights_only=False)
    out = []
    for c, t in zip(stored["centered_pcs"][:, :k].double(), stored["tangent_pcs"][:, :k].double()):
        qc, _ = torch.linalg.qr(c.T)
        qt, _ = torch.linalg.qr(t.T)
        out.append(float((torch.linalg.svdvals(qc.T @ qt) ** 2).mean()))
    return np.array(out)


def geometry_numbers() -> None:
    geo = {k: g for k, g in cohort_runs(GEOM, "geometry.json").items() if "loso_cos" in g}
    if not geo:
        return

    def ext(key: str, which: str, cohorts=None) -> float:
        runs = [g for c, g in geo.items() if cohorts is None or c in cohorts]
        if not runs or any(key not in g["ranges"] for g in runs):
            placeholder(f"geometry:{key}")
            return float("nan")
        vals = [g["ranges"][key][0 if which == "min" else 1] for g in runs]
        return min(vals) if which == "min" else max(vals)

    setm("RbarMin", f(ext("R", "min")))
    setm("RbarMax", f(ext("R", "max")))
    setm("RbarMinEEG", f(ext("R", "min", EEG)))
    setm("RbarMaxEEG", f(ext("R", "max", EEG)))
    setm("RbarMinMEG", f(ext("R", "min", ["mdd_meg"])))
    setm("RbarMaxMEG", f(ext("R", "max", ["mdd_meg"])))
    setm("AngleMin", f(ext("mean_angle_deg", "min"), 0))
    setm("AngleMax", f(ext("mean_angle_deg", "max"), 0))
    setm("CosUncMin", f(floor_to(ext("cos_anchor_uncentered_top", "min"), 3), 3))
    setm("SinUncMax", f(ceil_to(ext("sin_anchor_uncentered_top", "max"), 3), 3))
    setm("ResidualBoundMax", f(ceil_to(ext("residual_sin_bound", "max"), 3), 3))
    setm("WeylBoundMax", f(ceil_to(ext("weyl_sin_bound", "max"), 3), 3))
    setm("DKBoundMin", f(ext("davis_kahan_sin_bound", "min"), 2))
    setm("DKBoundMax", f(ext("davis_kahan_sin_bound", "max"), 2))
    tight = [row["sin_anchor_uncentered_top"] / row["residual_sin_bound"] for g in geo.values() for row in g["per_source"]
             if row.get("residual_sin_bound")]
    setm("BoundTightnessMin", f(min(tight), 2))
    setm("BoundTightnessMedian", f(float(np.median(tight)), 2))
    setm("CosCPConeMax", f(ceil_to(ext("cos_anchor_cpc1", "max"), 2), 2))
    setm("CosCPCtenMax", f(ceil_to(ext("max_cos_anchor_cpc10", "max"), 2), 2))
    setm("SpanTenMin", f(ext("anchor_in_cpc10", "min"), 2))
    setm("SpanTenMax", f(ext("anchor_in_cpc10", "max"), 2))
    setm("SpanSixteenMin", f(ext("anchor_in_cpc16", "min"), 2))
    setm("SpanSixteenMax", f(ext("anchor_in_cpc16", "max"), 2))
    setm("SpanSixtyFourMin", f(floor_to(ext("anchor_in_cpc64", "min"), 3), 3))
    setm("AnchorVarFracMin", pct(ext("anchor_variance_fraction", "min"), 1))
    setm("AnchorVarFracMax", pct(ext("anchor_variance_fraction", "max"), 1))
    setm("AnchorRankMin", ordinal(int(ext("anchor_variance_rank", "min"))))
    setm("AnchorRankMax", ordinal(int(ext("anchor_variance_rank", "max"))))
    setm("CPConeVarFracMin", pct(ext("cpc1_variance_fraction", "min"), 0))
    setm("CPConeVarFracMax", pct(ext("cpc1_variance_fraction", "max"), 0))
    setm("PRMin", f(ext("participation_ratio", "min"), 0))
    setm("PRMax", f(ext("participation_ratio", "max"), 0))
    setm("TangentSixtyFourMin", pct(floor_to(ext("tangent_var_in_64", "min"), 3), 1))
    align = [tangent_pc_alignment(key) for key in geo if (GEOM / key / "anchors.pt").exists()]
    setm("TangentPCAlignMin", f(floor_to(min(a.min() for a in align), 2), 2) if align else placeholder("TangentPCAlignMin"))
    setm("TangentSixteenMin", pct(ext("tangent_var_in_16", "min"), 0))
    setm("TangentSixteenMax", pct(ext("tangent_var_in_16", "max"), 0))
    setm("SplitHalfMin", f(floor_to(min(g["split_half_cos"]["min"] for g in geo.values()), 3), 3))
    setm("LosoMedianMin", f(min(g["loso_cos"]["median"] for g in geo.values()), 3))
    setm("LosoMedianMax", f(max(g["loso_cos"]["median"] for g in geo.values()), 3))
    setm("LosoPfiveMin", f(floor_to(min(g["loso_cos"]["p05"] for g in geo.values()), 2), 2))
    setm("LosoPfiveMax", f(max(g["loso_cos"]["p05"] for g in geo.values()), 2))
    setm("CrossCovMin", f(ext("cross_covariance_norm", "min"), 3))
    setm("CrossCovMax", f(ext("cross_covariance_norm", "max"), 3))
    setm("GapMin", f(ext("eigengap", "min"), 2))
    setm("GapMax", f(ext("eigengap", "max"), 2))
    setm("CoordSDMin", f(ext("coord_sd", "min"), 2))
    setm("CoordSDMax", f(ext("coord_sd", "max"), 2))
    setm("ConcentrationBoundMax", f(ceil_to(ext("concentration_sin_bound", "max"), 2), 2))
    rows = [row for g in geo.values() for row in g["per_source"]]
    over_residual = [row["davis_kahan_sin_bound"] / row["residual_sin_bound"] for row in rows]
    over_concentration = [row["davis_kahan_sin_bound"] / row["concentration_sin_bound"] for row in rows]
    setm("DKOverResidualMedian", f(float(np.median(over_residual)), 0))
    setm("DKOverConcentrationMin", f(min(over_concentration), 1))
    setm("DKOverConcentrationMax", f(max(over_concentration), 1))
    setm("ConcentrationCosMin", f(floor_to(math.sqrt(max(0.0, 1 - ceil_to(ext("concentration_sin_bound", "max"), 3) ** 2)), 3), 3))
    setm("OffdiagMin", f(min(g["anchor_cosine_offdiag"]["min"] for g in geo.values()), 2))
    setm("OffdiagMax", f(max(g["anchor_cosine_offdiag"]["max"] for g in geo.values()), 2))
    setm("OffdiagAbsMedian", f(float(np.median([g["anchor_cosine_offdiag"]["median_abs"] for g in geo.values()])), 2))
    window_macros("", list(geo))
    best = [best_centered_pc(key) for key in geo]
    best_cos = np.concatenate([b[0] for b in best])
    best_rank = np.concatenate([b[1] for b in best])
    setm("BestCPCMin", f(best_cos.min(), 2))
    setm("BestCPCMax", f(best_cos.max(), 2))
    setm("BestCPCMedian", f(float(np.median(best_cos)), 2))
    setm("BestCPCRankMin", ordinal(int(best_rank.min())))
    setm("BestCPCRankMax", ordinal(int(best_rank.max())))
    arrangements = [arrangement(g) for g in geo.values()]
    setm("AnchorDimMin", f(min(a.participation_ratio for a in arrangements), 1))
    setm("AnchorDimMax", f(max(a.participation_ratio for a in arrangements), 1))
    setm("AnchorFourShareMin", pct(floor_to(min(a.leading_shares[3] for a in arrangements), 3), 1))
    setm("AnchorMinAngleMin", f(min(a.min_angle_deg for a in arrangements), 0))
    setm("AnchorMinAngleMax", f(max(a.min_angle_deg for a in arrangements), 0))

    lines = [r"\begin{table}[tbp]", r"\renewcommand{\baselinestretch}{1}",
             r"\caption{Geometry of the per-source unit vectors (released tokenizer); ranges over the 16 sources. "
             r"$\Rbar$: mean resultant length; colat.: mean colatitude; $\cos(\anchor,\bm{v}_1)$: anchor versus leading "
             r"uncentered direction; bound: residual bound~\eqref{eq:residual-bound} on $\sin\angle(\anchor,\bm{v}_1)$; "
             r"$|\cos(\anchor,\mathrm{PC}_k)|$: largest absolute cosine with the ten leading centered PCs; "
             r"$\|P_{16}\anchor\|^2$: squared norm of the anchor in the span of the 16 leading centered PCs; var.: share of "
             r"the centered variance along the anchor (rank among the centered eigenvalues); PR: participation ratio of the "
             r"centered covariance, $(\operatorname{tr}\Sigma_s)^2/\operatorname{tr}(\Sigma_s^2)$. "
             r"Upper limits ($\leq$) are rounded up and lower limits ($\geq$) down. Per-source values, with a bootstrap interval for "
             r"$\Rbar$, are given in \cref{app:tables}.}",
             r"\label{tab:geometry}", r"\centering\scriptsize", r"\setlength{\tabcolsep}{2.6pt}",
             r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
             r"\textbf{cohort} & $\Rbar$ & \textbf{colat.}\,($^\circ$) & $\cos(\anchor,\bm{v}_1)$ & \textbf{bound} & "
             r"$|\cos(\anchor,\mathrm{PC}_k)|$ & $\|P_{16}\anchor\|^2$ & \textbf{var.} (\textbf{rank}) & \textbf{PR} \\",
             r"\midrule"]
    for key, label in COHORTS:
        g = geo.get(key)
        if g is None:
            lines.append(f"{label} & \\multicolumn{{8}}{{c}}{{--}} \\\\")
            continue
        r = g["ranges"]
        lines.append(
            f"{label} & {f(r['R'][0])}--{f(r['R'][1])} & {f(r['mean_angle_deg'][0], 0)}--{f(r['mean_angle_deg'][1], 0)} & "
            f"$\\geq${f(floor_to(r['cos_anchor_uncentered_top'][0], 4), 4)} & $\\leq${f(ceil_to(r['residual_sin_bound'][1], 3), 3)} & "
            f"$\\leq${f(ceil_to(r['max_cos_anchor_cpc10'][1], 2), 2)} & {f(r['anchor_in_cpc16'][0], 2)}--{f(r['anchor_in_cpc16'][1], 2)} & "
            f"{pct(r['anchor_variance_fraction'][0], 1)}--{pct(r['anchor_variance_fraction'][1], 1)}\\% "
            f"({f(r['anchor_variance_rank'][0], 0)}--{f(r['anchor_variance_rank'][1], 0)}) & "
            f"{f(r['participation_ratio'][0], 0)}--{f(r['participation_ratio'][1], 0)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB_DIR / "geometry.tex").write_text("\n".join(lines) + "\n")

    # edits applied off the sphere: what the quantizer's normalization leaves of them
    effects = [off_sphere_effects(key) for key in geo]
    achieved = np.concatenate([e[0] for e in effects])
    left = np.concatenate([e[1] for e in effects])
    setm("OffSphereHalfMin", pct(float(achieved.min())))
    setm("OffSphereHalfMax", pct(float(achieved.max())))
    setm("OffSphereMeanLeftMin", pct(float(left.min())))
    setm("OffSphereMeanLeftMax", pct(float(left.max())))
    # vectors within 30 degrees of their anchor (no matched rotation exists)
    within = np.concatenate([within_degrees(key, 30.0) for key in geo])
    setm("WithinThirtyMedianPct", pct(float(np.median(within)), 1))
    setm("WithinThirtyMaxPct", pct(float(within.max()), 0))

    cross = load_json(GEOM / "cross_cohort.json")
    if cross is None or len(cross) < 6:
        placeholder("cross_cohort")
        return
    eeg_pairs = [v["min"] for k, v in cross.items() if all(c in EEG for c in k.split("__"))]
    meg_pairs = [v["min"] for k, v in cross.items() if "mdd_meg" in k.split("__")]
    setm("CrossCohortEEGMin", f(floor_to(min(eeg_pairs), 3), 3))
    setm("CrossModalMin", f(floor_to(min(meg_pairs), 2), 2))
    setm("CrossCohortEEGMaxAngle", f(math.ceil(math.degrees(math.acos(min(eeg_pairs)))), 0))
    setm("CrossModalMaxAngle", f(math.ceil(math.degrees(math.acos(min(meg_pairs)))), 0))
    setm("CrossCohortCosMedian", f(float(np.median([v["median"] for v in cross.values()])), 3))


def control_numbers() -> None:
    """Untrained tokenizers and the tokenizer embedded in the foundation-model checkpoints."""
    for run, prefix in (("pd_ucsd__random_init0", "Random"), ("pd_ucsd__seed1_init", "Init")):
        g = load_json(GEOM / run / "geometry.json")
        if g is None:
            for name in ("Rbar", "Angle", "OffdiagMin"):
                setm(prefix + name, placeholder(prefix + name))
            continue
        setm(prefix + "Rbar", span(f(g["ranges"]["R"][0], 2), f(g["ranges"]["R"][1], 2)))
        setm(prefix + "Angle", f(g["ranges"]["mean_angle_deg"][1], 0))
        setm(prefix + "OffdiagMin", f(floor_to(g["anchor_cosine_offdiag"]["min"], 2), 2))
    cos = []
    for key in ("pd_ucsd", "pd_iowa"):
        a = GEOM / key / "anchors.pt"
        b = GEOM / f"{key}__brainomni_tiny" / "anchors.pt"
        if a.exists() and b.exists():
            pa = torch.load(a, weights_only=False)["anchor"]
            pb = torch.load(b, weights_only=False)["anchor"]
            cos.append(float((pa * pb).sum(-1).min()))
    setm("EmbeddedAnchorCosMin", f(floor_to(min(cos), 4), 4) if cos else placeholder("EmbeddedAnchorCosMin"))
    toks = [load_json(TOKENS / f"{key}__brainomni_tiny" / "tokens.json") for key in ("pd_ucsd", "pd_iowa")]
    toks = [t for t in toks if t and "mi_bits" in t["usage"][0]]
    if toks:
        setm("EmbMILevelOneMin", f(min(t["usage"][0]["mi_bits"] for t in toks), 2))
        setm("EmbMILevelOneMax", f(max(t["usage"][0]["mi_bits"] for t in toks), 2))
        setm("EmbFlipZeroLevelOne", pct(float(np.median([t["code_change"]["zero_anchor"]["level1_median"] for t in toks]))))
    else:
        for name in ("EmbMILevelOneMin", "EmbMILevelOneMax", "EmbFlipZeroLevelOne"):
            setm(name, placeholder(name))
    atl = load_json(ATLAS / "pd_ucsd__brainomni_tiny" / "summary.json")
    if atl:
        setm("EmbZeroAnchorPositiveCount", str(atl["edits"]["d_mse"]["zero_anchor"]["n_above_zero"]))
        setm("EmbZeroAnchorRel", pct(atl["edits"]["rel_mse"]["zero_anchor"]["median"], 1))
        setm("EmbKeysZeroMedianPct", pct(atl["ratios"]["keys_over_zero"]["median"]))
    else:
        for name in ("EmbZeroAnchorPositiveCount", "EmbZeroAnchorRel", "EmbKeysZeroMedianPct"):
            setm(name, placeholder(name))


# ----------------------------------------------------------------------------- tokens
TOKEN_EDITS = [("zero_anchor", "zero-anchor"), ("colat_p20", r"colatitude $+20^\circ$"), ("colat_m20", r"colatitude $-20^\circ$"),
               ("mean_anchor", "mean-anchor"), ("tangent_resample", "tangent resample"), ("tpc_resample_16", "tangent-axis resample (16)"),
               ("rand_support_16", "random-support resample (16)"), ("rot_matched_on", "matched rotation (support)")]


def token_numbers() -> None:
    have = {k: t for k, t in cohort_runs(TOKENS, "tokens.json").items() if "mi_bits" in t["usage"][0]}
    if not have:
        return
    l1 = [t["usage"][0] for t in have.values()]
    deep = [u for t in have.values() for u in t["usage"][1:]]
    setm("MILevelOneMin", f(min(u["mi_bits"] for u in l1), 2))
    setm("MILevelOneMax", f(max(u["mi_bits"] for u in l1), 2))
    setm("MIShareLevelOneMin", pct(min(u["mi_share_of_source_entropy"] for u in l1)))
    setm("MIShareLevelOneMax", pct(max(u["mi_share_of_source_entropy"] for u in l1)))
    setm("SourceAccLevelOneMin", pct(min(u["source_accuracy"] for u in l1)))
    setm("SourceAccLevelOneMax", pct(max(u["source_accuracy"] for u in l1)))
    setm("MIDeepMax", f(ceil_to(max(u["mi_bits"] for u in deep), 2), 2))
    setm("CodeEntropyLevelOneMin", f(min(u["code_entropy_bits"] for u in l1), 1))
    setm("CodeEntropyLevelOneMax", f(max(u["code_entropy_bits"] for u in l1), 1))
    setm("MIShareCodeLevelOneMin", pct(min(u["mi_share_of_code_entropy"] for u in l1)))
    setm("MIShareCodeLevelOneMax", pct(max(u["mi_share_of_code_entropy"] for u in l1)))
    setm("PerplexLevelOneMin", f(min(u["per_source_perplexity_median"] for u in l1), 0))
    setm("PerplexLevelOneMax", f(max(u["per_source_perplexity_median"] for u in l1), 0))
    setm("PerplexDeepMin", f(min(u["per_source_perplexity_median"] for u in deep), 0))
    setm("PerplexDeepMax", f(max(u["per_source_perplexity_median"] for u in deep), 0))
    al = [t["level1_alignment_summary"] for t in have.values()]
    setm("CodewordCosMin", f(min(a["own_anchor_cos_range"][0] for a in al), 2))
    setm("CodewordCosMedianMin", f(min(a["own_anchor_cos_median"] for a in al), 2))
    setm("CodewordCosMedianMax", f(max(a["own_anchor_cos_median"] for a in al), 2))
    setm("CodewordCosMax", f(max(a["own_anchor_cos_range"][1] for a in al), 2))

    def change(edit: str, field: str, idx: int | None = None) -> float:
        vals = [t["code_change"][edit][field] if idx is None else t["code_change"][edit][field][idx] for t in have.values()]
        return float(np.median(vals))

    for edit, name in (("zero_anchor", "Zero"), ("colat_p20", "ColatPTwenty"), ("colat_m20", "ColatMTwenty"), ("mean_anchor", "Mean"),
                       ("tangent_resample", "Tangent"), ("tpc_resample_16", "TPCsixteen"), ("rand_support_16", "RandSixteen"),
                       ("rot_matched_on", "RotOn")):
        setm(f"Flip{name}LevelOne", pct(change(edit, "level1_median")))
        setm(f"Foreign{name}", pct(change(edit, "foreign_share_median")))
    setm("DeeperTangentLevelTwo", pct(change("tangent_resample", "deeper_given_same_level1_median", 0)))
    setm("ForeignColatMTwentyMax", pct(max(t["code_change"]["colat_m20"]["foreign_share_median"] for t in have.values())))
    setm("DeeperRandSixteenLevelTwo", pct(change("rand_support_16", "deeper_given_same_level1_median", 0)))

    labels = [lab for k, lab in COHORTS if k in have]
    lines = [r"\begin{table}[tbp]", r"\renewcommand{\baselinestretch}{1}",
             r"\caption{Token structure (released tokenizer). $I(S;C_\ell)$: mutual information between the source index and "
             r"the level-$\ell$ code (bits; $H(S)=4$~bits); accuracy: identification of the source from the level-1 code; "
             r"perplexity: per-source perplexity of code usage (median over sources; 512 codewords); codeword--anchor cosine: "
             r"usage-weighted cosine between a source's level-1 codewords and its anchor (median over sources). Lower part: "
             r"fraction of vectors whose level-1 code changes under an edit and, in parentheses, the fraction of these changes "
             r"that go to a code used most often by another source (medians over sources).}",
             r"\label{tab:tokens}", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{l" + "c" * len(labels) + "}", r"\toprule",
             r"& " + " & ".join(f"\\textbf{{{lab}}}" for lab in labels) + r" \\", r"\midrule"]
    rows = [(r"$I(S;C_1)$ (bits)", lambda t: f(t["usage"][0]["mi_bits"], 2)),
            (r"$I(S;C_\ell)$, $\ell=2,3,4$ (bits)", lambda t: f(floor_to(min(u["mi_bits"] for u in t["usage"][1:]), 2), 2) + "--" + f(ceil_to(max(u["mi_bits"] for u in t["usage"][1:]), 2), 2)),
            (r"source accuracy from $C_1$", lambda t: pct(t["usage"][0]["source_accuracy"]) + r"\%"),
            ("perplexity, level 1", lambda t: f(t["usage"][0]["per_source_perplexity_median"], 0)),
            ("perplexity, levels 2--4", lambda t: "--".join(f(x, 0) for x in (min(u["per_source_perplexity_median"] for u in t["usage"][1:]), max(u["per_source_perplexity_median"] for u in t["usage"][1:])))),
            ("codeword--anchor cosine", lambda t: f(t["level1_alignment_summary"]["own_anchor_cos_median"], 2)),
            (r"\midrule level-1 codes changed by", None)]
    for label, fn in rows:
        if fn is None:
            lines.append(label + " & " * len(labels) + r" \\")
            continue
        lines.append(label + " & " + " & ".join(fn(have[k]) for k, _ in COHORTS if k in have) + r" \\")
    for edit, label in TOKEN_EDITS:
        cells = [f"{pct(have[k]['code_change'][edit]['level1_median'])}\\% ({pct(have[k]['code_change'][edit]['foreign_share_median'])}\\%)"
                 for k, _ in COHORTS if k in have]
        lines.append(r"\quad " + label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB_DIR / "tokens.tex").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- interventions
NEC_EDITS = [("zero_anchor", "zero-anchor"), ("half_anchor", "half-anchor"), ("mean_anchor", "mean-anchor"),
             ("resample_anchor", "resample-anchor"), ("colat_m20", r"colatitude $-20^\circ$"), ("colat_m10", r"colatitude $-10^\circ$"),
             ("colat_p10", r"colatitude $+10^\circ$"), ("colat_p20", r"colatitude $+20^\circ$"), ("to_pole", "collapse"),
             ("rot_matched_on", "matched rotation (support)"), ("rot_matched_off", r"matched rotation ($\anchor^{\perp}$)"),
             ("tpc_resample_1", "tangent-axis resample (1)"), ("tpc_resample_4", "tangent-axis resample (4)"),
             ("tpc_resample_16", "tangent-axis resample (16)"), ("tpc_resample_64", "tangent-axis resample (64)"),
             ("rand_support_16", "random-support resample (16)"), ("rand_off_16", "random resample (16)"),
             ("tangent_resample", "tangent resample"), ("swap", "anchor swap (three partners)"),
             ("zero_anchor_keys", "zero-anchor, keys only"), ("zero_anchor_values", "zero-anchor, values only"),
             ("swap_p0", "anchor swap (first partner)"), ("swap_keys", "\\quad keys only"), ("swap_values", "\\quad values only"),
             ("source_resample", "source resample")]


def atlas_numbers(runs: dict[str, dict], prefix: str = "") -> None:
    """Macros pooled over the cohorts in ``runs`` (released tokenizer when prefix is empty)."""
    keys = list(runs)
    n_pairs = 16 * len(keys)
    setm(prefix + "SourceCohortPairs", str(n_pairs))

    def count(edit: str, field: str = "n_above_zero", metric: str = "d_mse") -> int:
        return sum(runs[k]["edits"][metric][edit][field] for k in keys if edit in runs[k]["edits"][metric])

    def dcount(name: str, field: str = "n_above_zero", metric: str = "d_mse") -> int:
        return sum(runs[k]["differences"][metric][name][field] for k in keys if name in runs[k]["differences"][metric])

    def pooled(edit: str, metric: str = "rel_mse") -> np.ndarray:
        return np.concatenate([runs[k]["edits"][metric][edit]["per_source"] for k in keys if edit in runs[k]["edits"][metric]])

    def ratio(name: str) -> np.ndarray:
        return np.concatenate([runs[k]["ratios"][name]["per_source"] for k in keys if name in runs[k]["ratios"]])

    def cohort_medians(edit: str, metric: str = "rel_mse") -> list[float]:
        return [runs[k]["edits"][metric][edit]["median"] for k in keys if edit in runs[k]["edits"][metric]]

    setm(prefix + "ZeroAnchorPositiveCount", str(count("zero_anchor")))
    for metric, name in (("d_mae", "MAE"), ("d_omr", "OMR")):
        setm(prefix + f"ZeroAnchorPositiveCount{name}", str(count("zero_anchor", metric=metric)))
    zr = cohort_medians("zero_anchor")
    setm(prefix + "ZeroAnchorRelMin", pct(min(zr), 1))
    setm(prefix + "ZeroAnchorRelMax", pct(max(zr), 1))
    setm(prefix + "ZeroAnchorRelPooledMin", pct(pooled("zero_anchor").min(), 1))
    setm(prefix + "ZeroAnchorRelPooledMax", pct(pooled("zero_anchor").max(), 1))
    setm(prefix + "HalfOverZeroMedianPct", pct(float(np.median(ratio("half_over_zero")))))
    setm(prefix + "MeanOverZeroMedianPct", pct(float(np.median(ratio("mean_over_zero")))))
    setm(prefix + "MeanOverZeroMaxPct", pct(float(np.max(ratio("mean_over_zero")))))
    setm(prefix + "ResampleOverZeroMedianPct", pct(float(np.median(ratio("resample_over_zero")))))
    setm(prefix + "MeanAnchorPositiveCount", str(count("mean_anchor")))
    setm(prefix + "MeanAnchorNegativeCount", str(count("mean_anchor", "n_below_zero")))
    setm(prefix + "ResampleAnchorPositiveCount", str(count("resample_anchor")))
    setm(prefix + "ColatPTwentyPositiveCount", str(count("colat_p20")))
    setm(prefix + "ColatMTwentyNegativeCount", str(count("colat_m20", "n_below_zero")))
    setm(prefix + "ColatMTwentyPositiveCount", str(count("colat_m20")))
    setm(prefix + "ColatMTenPositiveCount", str(count("colat_m10")))
    setm(prefix + "ColatMTenNegativeCount", str(count("colat_m10", "n_below_zero")))
    setm(prefix + "ColatAsymCount", str(dcount("colat_p20_minus_m20")))
    setm(prefix + "ColatPTwentyRelMedian", pct(float(np.median(pooled("colat_p20"))), 1))
    setm(prefix + "ColatMTwentyRelMedian", pct(float(np.median(pooled("colat_m20"))), 1))
    setm(prefix + "ColatPTwentyOverZeroPct", pct(float(np.median(ratio("colat_p20_over_zero")))))
    setm(prefix + "LatitudeSpecificCount", str(dcount("zero_minus_rot_on")))
    setm(prefix + "LatitudeSpecificOffCount", str(dcount("zero_minus_rot_off")))
    zo = [runs[k]["ratios"]["zero_over_rot_on"]["median"] for k in keys]
    setm(prefix + "LatitudeRatioMin", f(min(zo), 1))
    setm(prefix + "LatitudeRatioMax", f(max(zo), 1))
    disp = [runs[k]["displacement"]["zero_anchor"]["displacement_deg"] for k in keys]
    setm(prefix + "MatchedDisp", span(f(min(disp), 0), f(max(disp), 0)))
    for edit, name in (("tpc_resample_16", "TPCsixteen"), ("tpc_resample_4", "TPCfour"), ("rand_support_16", "Randsixteen"),
                       ("swap", "Swap")):
        d = [runs[k]["displacement"][edit]["displacement_deg"] for k in keys if edit in runs[k]["displacement"]]
        if d:
            setm(prefix + f"Disp{name}", span(f(min(d), 0), f(max(d), 0)))
    setm(prefix + "PoleOverSourcePct", pct(float(np.median(ratio("pole_over_source")))))
    # routing
    small = 1 if prefix == "Var" else 0          # the variants' shares through the keys are fractions of a percent
    setm(prefix + "KeysZeroMedianPct", pct(float(np.median(ratio("keys_over_zero"))), small))
    setm(prefix + "ValuesZeroMedianPct", pct(float(np.median(ratio("values_over_zero")))))
    kz = [runs[k]["ratios"]["keys_over_zero"]["median"] for k in keys]
    setm(prefix + "KeysZeroCohortMin", pct(min(kz)))
    setm(prefix + "KeysZeroCohortMax", pct(max(kz)))
    setm(prefix + "KeysOverValuesZeroCount", str(dcount("keys_minus_values_zero")))
    setm(prefix + "KeysSwapMedianPct", pct(float(np.median(ratio("keys_over_swap")))))
    setm(prefix + "ValuesSwapMedianPct", pct(float(np.median(ratio("values_over_swap")))))
    setm(prefix + "KeysOverValuesSwapCount", str(dcount("keys_minus_values_swap")))
    setm(prefix + "SwapOverSourceMedianPct", pct(float(np.median(ratio("swap_over_source"))), small))
    setm(prefix + "SwapPositiveCount", str(count("swap")))
    setm(prefix + "SwapMinusZeroCount", str(dcount("swap_minus_zero")))
    rhos = [runs[k]["swap_alignment"]["spearman_rho"] for k in keys if runs[k].get("swap_alignment")]
    ps = [runs[k]["swap_alignment"]["spearman_p"] for k in keys if runs[k].get("swap_alignment")]
    setm(prefix + "SwapRhoMin", f(min(rhos), 2))
    setm(prefix + "SwapRhoMax", f(max(rhos), 2))
    setm(prefix + "SwapRhoPMax", pval(max(ps)))
    att = [runs[k]["attention_zero_anchor"] for k in keys if runs[k].get("attention_zero_anchor")]
    setm(prefix + "AttnDropMedianPct", pct(-float(np.median([a["relative_change_median"] for a in att]))))
    setm(prefix + "AttnDropMin", pct(-max(a["relative_change_range"][1] for a in att)))
    setm(prefix + "AttnDropMax", pct(-min(a["relative_change_range"][0] for a in att)))
    setm(prefix + "AttnUniformMin", pct(min(a["edit_over_uniform_range"][0] for a in att)))
    setm(prefix + "AttnUniformMax", pct(max(a["edit_over_uniform_range"][1] for a in att)))
    setm(prefix + "AttnDecreasedCount", str(sum(a["n_sources_decreased"] for a in att)))
    # content
    setm(prefix + "TangentPositiveCount", str(count("tangent_resample")))
    setm(prefix + "TangentOverSourceMedianPct", pct(float(np.median(ratio("tangent_over_source")))))
    setm(prefix + "ZeroOverTangentMedianPct", pct(float(np.median(ratio("zero_over_tangent")))))
    setm(prefix + "ZeroOverSourceMedianPct", pct(float(np.median(ratio("zero_over_source")))))
    for k_dim, word in ((1, "one"), (4, "four"), (16, "sixteen"), (64, "sixtyfour")):
        setm(prefix + f"ShareTPC{word}MedianPct", pct(float(np.median(ratio(f"tpc{k_dim}_over_tangent")))))
    for k_dim, word in ((1, "one"), (4, "four"), (16, "sixteen")):
        setm(prefix + f"ShareRand{word}MedianPct", pct(float(np.median(ratio(f"rand{k_dim}_over_tangent"))), 1))
        setm(prefix + f"TPCvsRand{word}Count", str(dcount(f"tpc{k_dim}_minus_rand{k_dim}")))
    setm(prefix + "ShareRandOffMedianPct", pct(float(np.median(ratio("randoff16_over_tangent"))), 1))
    setm(prefix + "SupportVsOffCount", str(dcount("rand16_minus_randoff16")))
    setm(prefix + "ShareTPCMeanOneMedianPct", pct(float(np.median(ratio("tpcmean1_over_tangent")))))
    setm(prefix + "ShareTPCMeanFourMedianPct", pct(float(np.median(ratio("tpcmean4_over_tangent")))))
    if not prefix:
        # counts that the text states in words ("all"); a visible placeholder if the data disagree
        guards = {"GuardZeroAnchorAll": all(count("zero_anchor", metric=m) == n_pairs for m in ("d_mse", "d_mae", "d_omr")),
                  "GuardLatitudeAll": dcount("zero_minus_rot_on") == n_pairs and dcount("zero_minus_rot_off") == n_pairs,
                  "GuardTPCAll": all(dcount(f"tpc{k}_minus_rand{k}") == n_pairs for k in (1, 4, 16))}
        for name, ok in guards.items():
            setm(name, "" if ok else placeholder(name))


def routing_numbers() -> None:
    root = RESULTS_ROOT / "necessity_routing"
    runs = {k: load_json(root / k / "summary.json") for k, _ in COHORTS}
    runs = {k: v for k, v in runs.items() if v}
    for k, _ in COHORTS:
        if k not in runs:
            placeholder(f"necessity_routing:{k}")
    if not runs:
        return

    def share(name: str) -> np.ndarray:
        return np.concatenate([r["ratios"][name]["per_source"] for r in runs.values()])

    def dcount(name: str) -> int:
        return sum(r["differences"]["d_mse"][name]["n_above_zero"] for r in runs.values())

    for name, macro in (("keys_over_tangent", "KeysTangentMedianPct"), ("values_over_tangent", "ValuesTangentMedianPct"),
                        ("keys_over_tpc4", "KeysTPCfourMedianPct"), ("values_over_tpc4", "ValuesTPCfourMedianPct"),
                        ("keys_over_zero", "RoutKeysZeroMedianPct"), ("values_over_zero", "RoutValuesZeroMedianPct")):
        setm(macro, pct(float(np.median(share(name)))))
    setm("ValuesOverKeysTangentCount", str(sum(r["differences"]["d_mse"]["keys_minus_values_tangent"]["n_below_zero"] for r in runs.values())))
    setm("KeysOverValuesTangentCount", str(dcount("keys_minus_values_tangent")))
    setm("RoutingPairs", str(16 * len(runs)))
    # the same split for the tokenizers trained from scratch (replicas; source-embedding variants)
    for group, tags in (("Rep", [t for t, _ in REPLICA_TAGS if not t.endswith("srcemb")]), ("Var", [t for t, _ in REPLICA_TAGS if t.endswith("srcemb")])):
        g = [load_json(root / f"{k}__{t}" / "summary.json") for t in tags for k, _ in COHORTS]
        g = [x for x in g if x]
        if not g:
            continue
        def gshare(name: str) -> np.ndarray:
            return np.concatenate([r["ratios"][name]["per_source"] for r in g])
        for name, macro in (("keys_over_tangent", "KeysTangentMedianPct"), ("values_over_tangent", "ValuesTangentMedianPct"),
                            ("keys_over_tpc4", "KeysTPCfourMedianPct"), ("values_over_tpc4", "ValuesTPCfourMedianPct")):
            setm(group + macro, pct(float(np.median(gshare(name))), 1 if group == "Var" and macro.startswith("Keys") else 0))
        setm(group + "ValuesOverKeysTangentCount", str(sum(r["differences"]["d_mse"]["keys_minus_values_tangent"]["n_below_zero"] for r in g)))
        setm(group + "RoutingPairs", str(16 * len(g)))


def necessity_numbers() -> None:
    runs = cohort_runs(ATLAS, "summary.json")
    if not runs:
        return
    atlas_numbers(runs)
    if "mdd_meg" in runs:
        meg = runs["mdd_meg"]["edits"]["rel_mse"]
        setm("ColatMTwentyRelMEG", pct(meg["colat_m20"]["median"], 1))
        setm("ColatPTwentyRelMEG", pct(meg["colat_p20"]["median"], 1))
        eeg = {k: v for k, v in runs.items() if k != "mdd_meg"}
        setm("ColatMTwentyNegativeCountEEG", str(sum(v["edits"]["d_mse"]["colat_m20"]["n_below_zero"] for v in eeg.values())))
        n_pos = sum(v["edits"]["d_mse"]["colat_m20"]["n_above_zero"] for v in eeg.values())
        setm("ColatMTwentyPositiveCountEEG", str(n_pos))
        setm("GuardColatNone", "" if n_pos == 0 else placeholder("GuardColatNone"))   # the text says "none"
        setm("EEGSourceCohortPairs", str(16 * len(eeg)))
        setm("ColatMTwentyPositiveCountMEG", str(runs["mdd_meg"]["edits"]["d_mse"]["colat_m20"]["n_above_zero"]))
    base = [runs[k]["baseline_mse_median"] for k in runs]
    setm("BaselineMSEMin", f(min(base), 2))
    setm("BaselineMSEMax", f(max(base), 2))
    labels = [(k, lab) for k, lab in COHORTS if k in runs]
    lines = [r"\begin{sidewaystable}", r"\renewcommand{\baselinestretch}{1}",
             r"\caption{Change in reconstruction error under each edit, in percent of the unedited mean squared error (median "
             r"over the 16 sources of the subject-mean change, with a 95\% subject-cluster bootstrap interval), and numbers of "
             r"sources (out of 16) whose bootstrap interval of the subject-mean $\dmse$ lies above / below zero, released "
             r"tokenizer; cells in which any of the three values is below 0.1\% in magnitude are given with two decimals. Edits "
             r"are defined in \cref{tab:operators}; the median unedited errors of the edited segments (on the evaluated sensors) "
             r"were " + ", ".join(
                 f"{lab} {f(runs[k]['baseline_mse_median'], 2)}" for k, lab in labels) + r".}",
             r"\label{tab:necessity}", r"\centering\scriptsize", r"\setlength{\tabcolsep}{1.5pt}",
             r"\begin{tabular}{l" + "c" * len(labels) + "}", r"\toprule",
             r"\textbf{edit} & " + " & ".join(f"\\textbf{{{lab}}}" for _, lab in labels) + r" \\", r"\midrule"]
    for edit, label in NEC_EDITS:
        cells = []
        for k, _ in labels:
            e = runs[k]["edits"]["rel_mse"].get(edit)
            n = runs[k]["edits"]["d_mse"].get(edit)
            if e is None:
                cells.append("--")
                continue
            lo, hi = e["median_ci"]
            nd = 2 if min(abs(e["median"]), abs(lo), abs(hi)) < 1e-3 else 1
            cells.append(f"{tnum(pct(e['median'], nd))} [{tnum(pct(lo, nd))},\\,{tnum(pct(hi, nd))}] ({n['n_above_zero']}/{n['n_below_zero']})")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{sidewaystable}"]
    (TAB_DIR / "necessity.tex").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- diagnosis
READOUT_ROWS = [("anchor16", "anchor-16"), ("best_single", "best single anchor"), ("quant_anchor16", "quantized anchor-16"),
                ("rbar16", "concentration-16"), ("norm16", "norm-16"), ("pooled256", "pooled"), ("tangent4096", "tangent"),
                ("full4096", "full"), ("quant4096", "quantized"), ("bandpower", "band power"), ("anchor16_adj", "anchor-16, adjusted"),
                ("full4096_adj", "full, adjusted"), ("bandpower_adj", "band power, adjusted"), ("age_sex", "age and sex")]
PAIRED_NAMES = [("anchor16", "Anchor"), ("full4096", "Full"), ("tangent4096", "Tangent"), ("quant4096", "Quant"), ("pooled256", "Pooled")]


def readout_numbers() -> None:
    res = {}
    for cohort, contrast, label, tag in CONTRASTS:
        r = load_json(READOUT / f"{cohort}_{contrast}" / "readout.json")
        if r is None or "auc" not in r["results"][0]:
            placeholder(f"readout:{cohort}_{contrast}")
            continue
        res[tag] = r
    if not res:
        return
    tags = [c[3] for c in CONTRASTS if c[3] in res]
    holm = dict(zip(tags, holm_adjust(np.array([{e["feature"]: e for e in res[t]["results"]}["anchor16"]["perm_p"] for t in tags]))))
    mc_p = {}
    for tag in tags:
        r = res[tag]
        e = {x["feature"]: x for x in r["results"]}
        a = e["anchor16"]
        setm(f"Anchor{tag}AUC", f(a["auc"], 2))
        setm(f"Anchor{tag}CI", f"[{f(a['auc_ci'][0], 2)}, {f(a['auc_ci'][1], 2)}]")
        setm(f"Anchor{tag}P", pval(a["perm_p"]))
        setm(f"Anchor{tag}PHolm", pval(holm[tag]))
        setm(f"N{tag}", str(r["n_subjects"]))
        for feat, name in (("full4096", "Full"), ("tangent4096", "Tangent"), ("quant4096", "Quant"), ("bandpower", "Band"),
                           ("age_sex", "AgeSex"), ("anchor16_adj", "AnchorAdj"), ("best_single", "BestSingle"), ("norm16", "Norm"),
                           ("quant_anchor16", "QuantAnchor"), ("full4096_adj", "FullAdj"), ("bandpower_adj", "BandAdj"),
                           ("pooled256", "Pooled"), ("rbar16", "Rbar")):
            if feat in e:
                setm(f"{name}{tag}AUC", f(e[feat]["auc"], 2))
                setm(f"{name}{tag}CI", f"[{f(e[feat]['auc_ci'][0], 2)}, {f(e[feat]['auc_ci'][1], 2)}]")
                p = e[feat].get("perm_p_fdr", e[feat].get("perm_p"))
                if p is not None:
                    setm(f"{name}{tag}P", pval(p))
        singles = [x for x in r["results"] if x["feature"].startswith("single_")]
        best = min(singles, key=lambda x: (x["perm_p_maxstat"], -x["auc"]))
        setm(f"Single{tag}MaxAUC", f(max(x["auc"] for x in singles), 2))
        setm(f"Single{tag}MinP", pval(min(x["perm_p_maxstat"] for x in singles)))
        setm(f"Single{tag}BestSource", str(int(best["feature"].split("_")[1]) + 1))
        setm(f"Single{tag}BestAUC", f(best["auc"], 2))
        setm(f"Single{tag}SigCount", str(sum(x["perm_p_maxstat"] < 0.05 for x in singles)))
        null = r["random16_null"]
        setm(f"Rand{tag}Mean", f(null["mean"], 2))
        setm(f"Rand{tag}Range", f"{f(null['q025'], 2)}--{f(null['q975'], 2)}")
        setm(f"Rand{tag}GePct", pct(null["fraction_ge_anchor16"]))
        mc_p[tag] = (1 + round(null["fraction_ge_anchor16"] * null["n"])) / (1 + null["n"])
    if mc_p:
        setm("RandMCPMin", f"{min(mc_p.values()):.3f}")
    setm("NumContrasts", WORDS[len(tags)])
    if "UCSDPD" in res:
        e = {x["feature"]: x for x in res["UCSDPD"]["results"]}["age_sex"]
        setm("AgeSexUCSDPDNullMean", f(e["perm_null_mean"], 2))
    gaps = [abs(x["perm_statistic"] - x["auc"]) for t in tags for x in res[t]["results"]
            if "perm_statistic" in x and not x["feature"].startswith("single_")]
    setm("PermStatMaxDiff", f(max(gaps), 2))
    setm("AnchorSignificantCount", WORDS.get(sum(holm[t] < 0.05 for t in tags), "no"))
    # paired differences on the same subjects (repeat-averaged AUC, out-of-fold scores fixed)
    deviations = [p["reproduced_auc_max_abs_deviation"] for c, k, _, _ in CONTRASTS
                  if (p := load_json(READOUT / f"{c}_{k}" / "paired.json")) is not None]
    setm("PairedMaxDeviation", f(ceil_to(max(deviations), 3), 3) if deviations else placeholder("PairedMaxDeviation"))
    for cohort, contrast, _, tag in CONTRASTS:
        paired = load_json(READOUT / f"{cohort}_{contrast}" / "paired.json")
        if paired is None:
            placeholder(f"paired:{tag}")
            continue
        for feat, name in PAIRED_NAMES:
            d = paired["differences"][feat]["vs_bandpower"]
            setm(f"Paired{name}Band{tag}", f(d["difference"], 2))
            setm(f"Paired{name}Band{tag}CI", ci(*d["ci"]))
            setm(f"Paired{name}Band{tag}P", pval(d["p"]))
            if feat != "anchor16":
                d = paired["differences"][feat]["vs_anchor16"]
                setm(f"Paired{name}Anchor{tag}", f(d["difference"], 2))
                setm(f"Paired{name}Anchor{tag}CI", ci(*d["ci"]))
                setm(f"Paired{name}Anchor{tag}P", pval(d["p"]))
    lines = [r"\begin{sidewaystable}", r"\renewcommand{\baselinestretch}{1}",
             r"\caption{Subject-level diagnostic read-out (released tokenizer): AUC (mean over 20 repetitions of stratified five-fold "
             r"cross-validation) "
             r"with a 95\% stratified bootstrap interval over subjects in which the out-of-fold scores are held fixed (it does "
             r"not include the variability of model fitting; inference rests on the permutation tests). Markers: $^{*}$ permutation "
             r"$p<0.05$ (anchor-16: Holm-corrected across contrasts; secondary representations: Benjamini--Hochberg-corrected "
             r"within contrast; best single anchor, adjusted representations and age and sex: uncorrected). The permutation tests "
             r"use the pipeline with the inverse regularization strength fixed at 1 and two repetitions "
             r"(\cref{sec:methods-readout}), whose statistic differs from the AUC shown by at most \PermStatMaxDiff, "
             r"so a marked AUC can have an interval that includes 0.5. Best single anchor: "
             r"the anchor coordinate whose AUC on the training subjects is farthest from 0.5, selected inside every training fold "
             r"and evaluated out of fold (per-source results are given in the text); adjusted: age and sex regressed out within "
             r"training folds. Last row: AUC range (2.5--97.5\% quantiles) of 100 random 16-direction representations evaluated "
             r"like anchor-16.}",
             r"\label{tab:readout}", r"\centering\scriptsize", r"\setlength{\tabcolsep}{2pt}",
             r"\begin{tabular}{l" + "c" * len(tags) + "}", r"\toprule",
             r"& " + " & ".join(f"\\textbf{{{c[2].split(', ')[0]}}}" for c in CONTRASTS if c[3] in res) + r" \\",
             r"& " + " & ".join(f"\\textbf{{{c[2].split(', ')[1]}}} ($n={res[c[3]]['n_subjects']}$)" for c in CONTRASTS if c[3] in res) + r" \\",
             r"\midrule"]
    for feat, label in READOUT_ROWS:
        cells = []
        for tag in tags:
            e = {x["feature"]: x for x in res[tag]["results"]}.get(feat)
            if e is None:
                cells.append("--")
                continue
            p = holm[tag] if feat == "anchor16" else e.get("perm_p_fdr", e.get("perm_p", 1.0))
            star = "$^{*}$" if p < 0.05 else ""
            cells.append(f"{f(e['auc'], 2)}{star} [{f(e['auc_ci'][0], 2)}, {f(e['auc_ci'][1], 2)}]")
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule random 16 directions & " + " & ".join(
        f"{f(res[t]['random16_null']['q025'], 2)}--{f(res[t]['random16_null']['q975'], 2)}" for t in tags) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{sidewaystable}"]
    (TAB_DIR / "readout.tex").write_text("\n".join(lines) + "\n")


TRANSFER_FEATURES = (("anchor16", "Anchor"), ("bandpower_shared", "Band"), ("full4096", "Full"), ("tangent4096", "Tangent"),
                     ("quant4096", "Quant"), ("pooled256", "Pooled"), ("quant_anchor16", "QuantAnchor"), ("norm16", "Norm"),
                     ("rbar16", "Rbar"))


def transfer_numbers() -> None:
    """Transfer between the PD cohorts; p-values from test-label permutations of each fitted classifier, Holm-corrected
    across the two directions for anchor-16 and Benjamini-Hochberg-corrected within a direction for the others."""
    data = {}
    for train, test, tag in (("pd_iowa", "pd_ucsd", "IowaToUCSD"), ("pd_ucsd", "pd_iowa", "UCSDToIowa")):
        t = load_json(READOUT / f"transfer_{train}_to_{test}.json")
        if t is None or "p_test_perm" not in t["results"]["train_scaler"]["anchor16"]:
            placeholder(f"transfer:{tag}")
            continue
        data[tag] = t
        setm(f"Transfer{tag}NTest", str(t["n_test"]))
    if not data:
        return
    setm("TransferSharedChannels", str(len(next(iter(data.values()))["shared_channels"])))
    mc_all, q2_all = {}, {}
    for setting, sname in (("train_scaler", ""), ("per_cohort_standardisation", "Harm")):
        mc = {}
        for tag, t in data.items():
            null = t["random16_null"][setting]
            k = round(null["fraction_ge_anchor16"] * null["n"])
            mc[tag] = (1 + k) / (1 + null["n"])
        for tag, q in zip(mc, holm_adjust(np.array(list(mc.values())))):
            setm(f"Transfer{tag}{sname}RandP", pval(q))
            q2_all[tag + sname] = q
        mc_all.update({tag + sname: p for tag, p in mc.items()})
        holm = dict(zip(data, holm_adjust(np.array([data[g]["results"][setting]["anchor16"]["p_test_perm"] for g in data]))))
        for tag, t in data.items():
            r = t["results"][setting]
            secondary = [k for k in r if k != "anchor16"]
            bh = dict(zip(secondary, bh_adjust(np.array([r[k]["p_test_perm"] for k in secondary]))))
            for feat, name in TRANSFER_FEATURES:
                setm(f"Transfer{tag}{sname}{name}AUC", f(r[feat]["auc"], 2))
                setm(f"Transfer{tag}{sname}{name}CI", f"[{f(r[feat]['auc_ci'][0], 2)}, {f(r[feat]['auc_ci'][1], 2)}]")
                setm(f"Transfer{tag}{sname}{name}P", pval(holm[tag] if feat == "anchor16" else bh[feat]))
                setm(f"Transfer{tag}{sname}{name}RandGePct", pct(r[feat]["random16_fraction_ge"]))
                if feat != "anchor16":
                    d = r[feat]["vs_anchor16"]
                    setm(f"Transfer{tag}{sname}AnchorMinus{name}", f(d["difference"], 2))
                    setm(f"Transfer{tag}{sname}AnchorMinus{name}CI", ci(*d["ci"]))
                    setm(f"Transfer{tag}{sname}AnchorMinus{name}P", pval(d["p_holm"]))
            null = t["random16_null"][setting]
            setm(f"Transfer{tag}{sname}RandRange", f"{f(null['q025'], 2)}--{f(null['q975'], 2)}")
            setm(f"Transfer{tag}{sname}RandGePct", pct(null["fraction_ge_anchor16"], 1))
            setm(f"Transfer{tag}{sname}RandN", f"{null['n']:,}".replace(",", "{,}"))
            setm(f"Transfer{tag}{sname}RandGeCount", str(round(null["fraction_ge_anchor16"] * null["n"])))
            setm(f"Transfer{tag}{sname}MinDiffPHolm", pval(min(r[k]["vs_anchor16"]["p_holm"] for k in secondary)))
            setm(f"Transfer{tag}{sname}MaxSecondaryP", pval(max(bh.values())))
    if len(mc_all) == 4:
        # Holm across both scalings and both directions; the text states that the conclusions do not change
        q4 = dict(zip(mc_all, holm_adjust(np.array(list(mc_all.values())))))
        for key, q in q4.items():
            setm(f"Transfer{key}RandPHolmFour", pval(q))
        same = all((q4[k] < 0.05) == (q2_all[k] < 0.05) for k in q4)
        setm("GuardHolmFour", "" if same else placeholder("GuardHolmFour"))
    if len(data) == 2:
        transfer_table(data)


TRANSFER_ROWS = (("anchor16", "anchor-16"), ("quant_anchor16", "quantized anchor-16"), ("rbar16", "concentration-16"),
                 ("norm16", "norm-16"), ("pooled256", "pooled"), ("tangent4096", "tangent"), ("full4096", "full"),
                 ("quant4096", "quantized"), ("bandpower_shared", "band power"))


def transfer_table(data: dict) -> None:
    """Table of the transfer results: both directions (columns) and both feature scalings (row blocks)."""
    def pnum(q: float) -> str:
        return f"{q:.3f}" if q >= 0.001 else r"$<$0.001"

    tags = ("IowaToUCSD", "UCSDToIowa")
    n_rand = data["IowaToUCSD"]["random16_null"]["train_scaler"]["n"]
    lines = [r"\begin{table}[tbp]", r"\renewcommand{\baselinestretch}{1}",
             r"\caption{Transfer between the two Parkinson's cohorts (released tokenizer): a classifier trained on all subjects of "
             r"one cohort and applied to the other. AUC with 95\% bootstrap interval over the test subjects; $p$: permutation of the "
             r"test labels, Holm-corrected across the two directions for anchor-16 and Benjamini--Hochberg-corrected within a "
             r"direction for the other representations; rand.: number of the " + f"{n_rand:,}".replace(",", "{,}") +
             r" random 16-direction representations with at least the same AUC. Band power is restricted to the "
             + str(len(next(iter(data.values()))["shared_channels"])) + r" sensors shared by the two montages.}",
             r"\label{tab:transfer}", r"\centering\scriptsize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lcccccc}", r"\toprule",
             r"& \multicolumn{3}{c}{PD-Iowa $\to$ PD-UCSD} & \multicolumn{3}{c}{PD-UCSD $\to$ PD-Iowa} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"\textbf{representation} & AUC & $p$ & rand. & AUC & $p$ & rand. \\"]
    for setting, title in (("train_scaler", "features standardized with the training cohort's statistics"),
                           ("per_cohort_standardisation", "features standardized within each cohort")):
        holm = dict(zip(tags, holm_adjust(np.array([data[g]["results"][setting]["anchor16"]["p_test_perm"] for g in tags]))))
        bh = {}
        for tag in tags:
            r = data[tag]["results"][setting]
            secondary = [k for k in r if k != "anchor16"]
            bh[tag] = dict(zip(secondary, bh_adjust(np.array([r[k]["p_test_perm"] for k in secondary]))))
        lines += [r"\midrule", r"\multicolumn{7}{@{}l}{\emph{" + title + r"}} \\"]
        for feat, label in TRANSFER_ROWS:
            cells = []
            for tag in tags:
                r = data[tag]["results"][setting][feat]
                q = holm[tag] if feat == "anchor16" else bh[tag][feat]
                n = data[tag]["random16_null"][setting]["n"]
                cells += [f"{f(r['auc'], 2)} [{f(r['auc_ci'][0], 2)}, {f(r['auc_ci'][1], 2)}]", pnum(q),
                          str(round(r["random16_fraction_ge"] * n))]
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB_DIR / "transfer.tex").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- replication
def heldout_mse(tag: str) -> dict[str, tuple[float, float]]:
    """Median segment MSE on held-out subjects: replica vs released tokenizer, per cohort."""
    out = {}
    for key, _ in COHORTS:
        rep = GEOM / f"{key}__{tag}" / "export.pt"
        rel = GEOM / key / "export.pt"
        if not rep.exists() or not rel.exists():
            continue
        e_rep = torch.load(rep, weights_only=False)
        e_rel = torch.load(rel, weights_only=False)
        held = set(e_rep["subjects"])
        rel_mse = np.array([m for s, m in zip(e_rel["subjects"], e_rel["mse"].numpy()) if s in held])
        out[key] = (float(np.median(e_rep["mse"].numpy())), float(np.median(rel_mse)))
    return out


def centered_window(run: str) -> tuple[np.ndarray, np.ndarray]:
    """Per source: the number of centered eigenvalues within ||gamma|| of Var(a), and ||gamma|| / Var(a)."""
    stored = torch.load(GEOM / run / "anchors.pt", weights_only=False)
    rows = load_json(GEOM / run / "geometry.json")["per_source"]
    eig = stored["centered_eigvals"].double().numpy()
    counts = np.array([int(np.sum(np.abs(eig[i] - r["coord_var"]) <= r["cross_covariance_norm"])) for i, r in enumerate(rows)])
    ratio = np.array([r["cross_covariance_norm"] / r["coord_var"] for r in rows])
    return counts, ratio


def window_macros(prefix: str, runs: list[str]) -> None:
    per_run = [centered_window(r) for r in runs if (GEOM / r / "anchors.pt").exists()]
    if not per_run:
        return
    medians = [float(np.median(c)) for c, _ in per_run]
    setm(prefix + "WindowCountMin", f(min(medians), 0))
    setm(prefix + "WindowCountMax", f(max(medians), 0))
    setm(prefix + "GammaOverVarMedian", f(float(np.median(np.concatenate([q for _, q in per_run]))), 2))


def best_centered_pc(run: str) -> tuple[np.ndarray, np.ndarray]:
    """Per source, the largest |cos| between the anchor and one of the 64 leading centered PCs, and that PC's rank."""
    stored = torch.load(GEOM / run / "anchors.pt", weights_only=False)
    cos = torch.einsum("skd,sd->sk", stored["centered_pcs"].double(), stored["anchor"].double()).abs()
    best = cos.max(dim=1)
    return best.values.numpy(), best.indices.numpy() + 1


def anchor_spread(run: str) -> tuple[float, float]:
    """Largest angle (degrees) between an anchor and the normalized mean of the 16 anchors, and largest angle
    between two anchors."""
    anchors = torch.nn.functional.normalize(torch.load(GEOM / run / "anchors.pt", weights_only=False)["anchor"].double(), dim=-1)
    common = torch.nn.functional.normalize(anchors.mean(dim=0), dim=0)
    to_common = torch.rad2deg(torch.arccos((anchors @ common).clamp(-1, 1))).max()
    pairwise = torch.rad2deg(torch.arccos((anchors @ anchors.T).clamp(-1, 1))).max()
    return float(to_common), float(pairwise)


def off_sphere_effects(run: str) -> tuple[np.ndarray, np.ndarray]:
    """Per source, what the quantizer's normalization would leave of two edits applied off the sphere:
    the median share of the intended change achieved by halving the anchor component, and the share of the
    variance of the anchor coordinate left after setting the anchor component to its mean."""
    export = torch.load(GEOM / run / "export.pt", weights_only=False)
    anchors = torch.load(GEOM / run / "anchors.pt", weights_only=False)["anchor"].double()
    index = export["row_segment_index"].numpy()
    keep = torch.from_numpy(export["seg_in_subject"].numpy()[index] >= 1)
    rows = torch.nn.functional.normalize(export["rows"][keep].double(), dim=-1)
    a = torch.einsum("nsrd,sd->nsr", rows, anchors).permute(1, 0, 2).reshape(anchors.shape[0], -1)     # S x vectors
    halved = off_sphere_coordinate(a, 0.5 * a)
    achieved = torch.stack([((ai - hi) / (0.5 * ai))[ai > 0.05].median() for ai, hi in zip(a, halved)])
    to_mean = off_sphere_coordinate(a, a.mean(dim=1, keepdim=True))
    left = to_mean.var(dim=1) / a.var(dim=1)
    return achieved.numpy(), left.numpy()


def within_degrees(run: str, degrees: float) -> np.ndarray:
    """Per source, the share of the vectors (segments >= 1) whose colatitude is below ``degrees``."""
    export = torch.load(GEOM / run / "export.pt", weights_only=False)
    anchors = torch.load(GEOM / run / "anchors.pt", weights_only=False)["anchor"].float()
    index = export["row_segment_index"].numpy()
    keep = torch.from_numpy(export["seg_in_subject"].numpy()[index] >= 1)
    rows = torch.nn.functional.normalize(export["rows"][keep].float(), dim=-1)
    coord = torch.einsum("nsrd,sd->nsr", rows, anchors)
    return (coord > math.cos(math.radians(degrees))).float().mean(dim=(0, 2)).numpy()


def released_mse() -> dict[str, tuple[float, float]]:
    """Median segment MSE of the released tokenizer on all subjects, per cohort (no comparison value)."""
    out = {}
    for key, _ in COHORTS:
        path = GEOM / key / "export.pt"
        if path.exists():
            out[key] = (float(np.median(torch.load(path, weights_only=False)["mse"].numpy())), float("nan"))
    return out


def cell_range(values: list[float], nd: int) -> str:
    """'lo--hi' for a table cell, or a single value when both ends round to the same number."""
    lo, hi = f(min(values), nd), f(max(values), nd)
    return lo if lo == hi else f"{lo}--{hi}"


def replication_numbers() -> None:
    present = [(tag, lab) for tag, lab in REPLICA_TAGS if (GEOM / f"pd_ucsd__{tag}" / "geometry.json").exists()]
    n_seed = sum(1 for tag, _ in present if not tag.endswith("srcemb"))
    n_var = len(present) - n_seed
    setm("NumReplicas", WORDS.get(len(present), str(len(present))) if present else placeholder("NumReplicas"))
    setm("NumSeedReplicas", WORDS.get(n_seed, str(n_seed)).capitalize() if n_seed else placeholder("NumSeedReplicas"))
    setm("NumVariants", WORDS.get(n_var, str(n_var)) if n_var else placeholder("NumVariants"))
    setm("VariantCountCap", WORDS.get(n_var, str(n_var)).capitalize() if n_var else placeholder("VariantCountCap"))
    setm("NumIndependentTokenizers", WORDS.get(len(present) + 1, str(len(present) + 1)) if present else placeholder("NumIndependentTokenizers"))
    setm("NumTokenizers", WORDS.get(len(present) + 2, str(len(present) + 2)) if present else placeholder("NumTokenizers"))
    cfg = load_json(REPLICAS / "seed1" / "model_cfg.json")
    log = REPLICAS / "seed1" / "train_log.csv"
    setm("ReplicaSteps", f"{int(pd.read_csv(log)['step'].max()):,}".replace(",", "{,}") if log.exists() else placeholder("ReplicaSteps"))
    split = load_json(REPLICAS / "seed1" / "split.json")
    if split is not None:
        hours = 0.0
        for key, _ in COHORTS:
            n_train = len(split[key]["train"])
            summ = load_json(GEOM / key / "export_summary.json")
            if summ:
                hours += summ["segments"] * n_train / summ["subjects"] * 10 / 3600
        setm("ReplicaHours", f(hours, 0))
    else:
        setm("ReplicaHours", placeholder("ReplicaHours"))
    if cfg is None:
        placeholder("replica_cfg")
    rows = []
    for tag, label in [("", "released"), ("brainomni_tiny", "embedded")] + present:
        run = (lambda k: k) if not tag else (lambda k, t=tag: f"{k}__{t}")
        geo = {k: load_json(GEOM / run(k) / "geometry.json") for k, _ in COHORTS}
        atl = {k: load_json(ATLAS / run(k) / "summary.json") for k, _ in COHORTS}
        tok = {k: load_json(TOKENS / run(k) / "tokens.json") for k, _ in COHORTS}
        mse = released_mse() if not tag else heldout_mse(tag)
        rows.append((tag, label, geo, atl, tok, mse))
    emb_mse = next((m for t, _, _, _, _, m in rows if t == "brainomni_tiny"), {})
    emb_released = cell_range([v[1] for v in emb_mse.values()], 2) if emb_mse else placeholder("EmbReleasedMSE")
    lines = [r"\begin{table}[tbp]", r"\renewcommand{\baselinestretch}{1}",
             r"\caption{Tokenizers compared: released (all subjects of the four cohorts); embedded, the tokenizer stored in the "
             r"foundation-model checkpoints (PD-UCSD and PD-Iowa, all subjects; interventions on PD-UCSD only); replicas trained "
             r"from scratch and the "
             r"source-embedding variant" + ("s" if n_var != 1 else "") + r" (held-out subjects of the four cohorts). Ranges over cohorts. "
             r"MSE: median over all segments of the reconstruction error of a segment on all sensors (the released tokenizer "
             r"reached " + emb_released + r" on the subjects of the embedded tokenizer and \RepReleasedMSEMin--\RepReleasedMSEMax{} "
             r"on the held-out subjects of the tokenizers trained from scratch); "
             r"$\Rbar$: range over sources and cohorts; anchor dim.: effective number of directions spanned by the 16 anchors "
             r"(participation ratio of their cosine matrix); "
             r"zero-anchor: median $\dmse$ in percent of the unedited error and number of source--cohort pairs with an interval above zero, "
             r"out of all pairs evaluated; "
             r"keys, values: damage of zero-anchor applied to the decoder's keys alone or to its values alone, relative to the "
             r"damage of the full edit; "
             r"$I(S;C_1)$: bits.}",
             r"\label{tab:replication}", r"\centering\scriptsize", r"\setlength{\tabcolsep}{2pt}",
             r"\begin{tabular}{lccccccc}", r"\toprule",
             r"\textbf{tokenizer} & \textbf{MSE} & $\Rbar$ & \textbf{anchor dim.} & \textbf{zero-anchor} & \textbf{keys} & "
             r"\textbf{values} & $I(S;C_1)$ \\", r"\midrule"]
    for tag, label, geo, atl, tok, mse in rows:
        g = {k: v for k, v in geo.items() if v}
        a = {k: v for k, v in atl.items() if v}
        t = {k: v for k, v in tok.items() if v}
        if not mse:
            mse_cell = "--"
        else:
            mse_cell = cell_range([v[0] for v in mse.values()], 2)
        r_cell = "--" if not g else cell_range([x["ranges"]["R"][0] for x in g.values()] + [x["ranges"]["R"][1] for x in g.values()], 2)
        c_cell = "--" if not g else cell_range([arrangement(x).participation_ratio for x in g.values()], 1)
        if a:
            zs = [x["edits"]["rel_mse"]["zero_anchor"]["median"] for x in a.values()]
            npos = sum(x["edits"]["d_mse"]["zero_anchor"]["n_above_zero"] for x in a.values())
            z_cell = f"{cell_range([100 * z for z in zs], 1)}\\% ({npos}/{16 * len(a)})"
            ks = [x["ratios"]["keys_over_zero"]["median"] for x in a.values() if "keys_over_zero" in x["ratios"]]
            k_cell = cell_range(ks, 2) if ks else "--"
            vs = [x["ratios"]["values_over_zero"]["median"] for x in a.values() if "values_over_zero" in x["ratios"]]
            v_cell = cell_range(vs, 2) if vs else "--"
        else:
            z_cell = k_cell = v_cell = "--"
        i_cell = "--" if not t else cell_range([x["usage"][0]["mi_bits"] for x in t.values()], 2)
        lines.append(f"{label} & {mse_cell} & {r_cell} & {c_cell} & {z_cell} & {k_cell} & {v_cell} & {i_cell} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB_DIR / "replication.tex").write_text("\n".join(lines) + "\n")
    for key, name in (("pd_ucsd", "UCSD"), ("pd_iowa", "Iowa"), ("dementia", "Dementia"), ("mdd_meg", "MEG")):
        held = load_json(REPLICAS / f"heldout_{key}.json")
        setm(f"RepHeldOut{name}", str(len(held)) if held is not None else placeholder(f"RepHeldOut{name}"))
    # pilot with the upstream initialization (otherwise the replica recipe)
    pilot = load_json(GEOM / "pd_ucsd__seed1_upstream" / "geometry.json")
    pilot_log = REPLICAS / "seed1_upstream" / "train_log.csv"
    rep_log = REPLICAS / "seed1" / "train_log.csv"
    if pilot is not None and pilot_log.exists():
        pilot_pcc = float(pd.read_csv(pilot_log)["pcc"].tail(5).mean())
        rep_pcc = float(pd.read_csv(rep_log)["pcc"].tail(5).mean())
        setm("PilotRbar", span(f(pilot["ranges"]["R"][0], 2), f(pilot["ranges"]["R"][1], 2)))
        setm("PilotOffdiagMin", f(floor_to(pilot["anchor_cosine_offdiag"]["min"], 2), 2))
        setm("PilotPCC", f(pilot_pcc, 2))
        setm("RepPCC", f(rep_pcc, 2))
    else:
        for name in ("PilotRbar", "PilotOffdiagMin", "PilotPCC", "RepPCC"):
            setm(name, placeholder(name))
    # anchor arrangement of each replica separately (circle gaps; closest pair of anchors)
    for tag, _ in present:
        if tag.endswith("srcemb"):
            # per variant: largest angle between two anchors and the level-1 identity code
            name = "Variant" + WORDS.get(int(tag.replace("seed", "").replace("_srcemb", "")), tag).capitalize()
            runs = [f"{k}__{tag}" for k, _ in COHORTS if (GEOM / f"{k}__{tag}" / "anchors.pt").exists()]
            if runs:
                setm(name + "AnchorPairAngleMax", f(ceil_to(max(anchor_spread(r)[1] for r in runs), 0), 0))
            toks = [load_json(TOKENS / f"{k}__{tag}" / "tokens.json") for k, _ in COHORTS]
            toks = [x for x in toks if x]
            if toks:
                setm(name + "MILevelOneRange", cell_range([x["usage"][0]["mi_bits"] for x in toks], 2))
            continue
        name = "Replica" + WORDS.get(int(tag.replace("seed", "")), tag).capitalize()
        arr = [arrangement(g) for g in (load_json(GEOM / f"{k}__{tag}" / "geometry.json") for k, _ in COHORTS) if g]
        gaps = [x for a in arr if a.circle_gaps_deg for x in a.circle_gaps_deg]
        if gaps:
            setm(name + "GapMin", f(min(gaps), 0))
            setm(name + "GapMax", f(max(gaps), 0))
        setm(name + "MinAngleMin", f(min(a.min_angle_deg for a in arr), 0))
        setm(name + "MinAngleMax", f(max(a.min_angle_deg for a in arr), 0))
    # pooled macros for the replicas trained with the released architecture and for the variant
    for group, tags in (("Rep", [t for t, _ in present if not t.endswith("srcemb")]), ("Var", [t for t, _ in present if t.endswith("srcemb")])):
        runs = {f"{k}__{t}": load_json(ATLAS / f"{k}__{t}" / "summary.json") for t in tags for k, _ in COHORTS}
        runs = {k: v for k, v in runs.items() if v}
        if runs:
            atlas_numbers(runs, prefix=group)
        geos = [load_json(GEOM / f"{k}__{t}" / "geometry.json") for t in tags for k, _ in COHORTS]
        geos = [g for g in geos if g]
        if geos:
            setm(group + "RbarMin", f(min(g["ranges"]["R"][0] for g in geos), 2))
            setm(group + "RbarMax", f(max(g["ranges"]["R"][1] for g in geos), 2))
            setm(group + "CosUncMin", f(floor_to(min(g["ranges"]["cos_anchor_uncentered_top"][0] for g in geos), 3), 3))
            setm(group + "OffdiagAbsMedian", f(float(np.median([g["anchor_cosine_offdiag"]["median_abs"] for g in geos])), 2))
            arr = [arrangement(g) for g in geos]
            setm(group + "AnchorDimMin", f(min(a.participation_ratio for a in arr), 1))
            setm(group + "AnchorDimMax", f(max(a.participation_ratio for a in arr), 1))
            setm(group + "AnchorTwoShareMin", pct(floor_to(min(a.leading_shares[1] for a in arr), 3), 1))
            setm(group + "AnchorMinAngleMin", f(min(a.min_angle_deg for a in arr), 0))
            setm(group + "AnchorMinAngleMax", f(max(a.min_angle_deg for a in arr), 0))
            setm(group + "AngleMin", f(min(g["ranges"]["mean_angle_deg"][0] for g in geos), 0))
            setm(group + "AngleMax", f(max(g["ranges"]["mean_angle_deg"][1] for g in geos), 0))
            setm(group + "AnchorVarFracMin", pct(min(g["ranges"]["anchor_variance_fraction"][0] for g in geos), 1))
            setm(group + "AnchorVarFracMax", pct(max(g["ranges"]["anchor_variance_fraction"][1] for g in geos), 1))
            setm(group + "AnchorRankMin", ordinal(int(min(g["ranges"]["anchor_variance_rank"][0] for g in geos))))
            setm(group + "AnchorRankMax", ordinal(int(max(g["ranges"]["anchor_variance_rank"][1] for g in geos))))
            setm(group + "CosCPConeMax", f(ceil_to(max(g["ranges"]["cos_anchor_cpc1"][1] for g in geos), 2), 2))
            setm(group + "OffdiagMin", f(floor_to(min(g["anchor_cosine_offdiag"]["min"] for g in geos), 2), 2))
            setm(group + "OffdiagMax", f(max(g["anchor_cosine_offdiag"]["max"] for g in geos), 2))
            setm(group + "CrossCovMin", f(min(g["ranges"]["cross_covariance_norm"][0] for g in geos), 4))
            setm(group + "CrossCovMax", f(max(g["ranges"]["cross_covariance_norm"][1] for g in geos), 4))
            window_macros(group, [f"{k}__{t}" for t in tags for k, _ in COHORTS])
            spread = [anchor_spread(f"{k}__{t}") for t in tags for k, _ in COHORTS if (GEOM / f"{k}__{t}" / "anchors.pt").exists()]
            if spread:
                setm(group + "AnchorToCommonMax", f(ceil_to(max(x[0] for x in spread), 0), 0))
                setm(group + "AnchorPairAngleMax", f(ceil_to(max(x[1] for x in spread), 0), 0))
            best = [best_centered_pc(f"{k}__{t}") for t in tags for k, _ in COHORTS if (GEOM / f"{k}__{t}" / "anchors.pt").exists()]
            if best:
                best_cos = np.concatenate([b[0] for b in best])
                best_rank = np.concatenate([b[1] for b in best])
                setm(group + "BestCPCMin", f(best_cos.min(), 2))
                setm(group + "BestCPCMedian", f(float(np.median(best_cos)), 2))
                setm(group + "BestCPCRankMin", ordinal(int(best_rank.min())))
                setm(group + "BestCPCRankMax", ordinal(int(best_rank.max())))
            cross = [load_json(GEOM / ("cross_" + "_".join(f"{k}__{t}" for k, _ in COHORTS) + ".json")) for t in tags]
            cross = [v["min"] for c in cross if c for v in c.values()]
            setm(group + "CrossCohortMin", f(floor_to(min(cross), 3), 3) if cross else placeholder(group + "CrossCohortMin"))
            pcc = [float(pd.read_csv(REPLICAS / t / "train_log.csv")["pcc"].tail(5).mean()) for t in tags
                   if (REPLICAS / t / "train_log.csv").exists()]
            if pcc:
                setm(group + "PCCMin", f(min(pcc), 2))
                setm(group + "PCCMax", f(max(pcc), 2))
                setm(group + "PCCRange", cell_range(pcc, 2))
            mse = [v for t in tags for v in heldout_mse(t).values()]
            if mse:
                setm(group + "MSEMin", f(min(v[0] for v in mse), 2))
                setm(group + "MSEMax", f(max(v[0] for v in mse), 2))
                setm(group + "ReleasedMSEMin", f(min(v[1] for v in mse), 2))
                setm(group + "ReleasedMSEMax", f(max(v[1] for v in mse), 2))
            runs_here = [f"{k}__{t}" for t in tags for k, _ in COHORTS
                         if (GEOM / f"{k}__{t}" / "export.pt").exists() and (GEOM / f"{k}__{t}" / "anchors.pt").exists()]
            if runs_here:
                achieved = np.concatenate([off_sphere_effects(r)[0] for r in runs_here])
                setm(group + "OffSphereHalfMin", pct(float(achieved.min())))
                setm(group + "OffSphereHalfMax", pct(float(achieved.max())))
                within30 = np.concatenate([within_degrees(r, 30.0) for r in runs_here])
                setm(group + "WithinThirtyMedianPct", pct(float(np.median(within30))))
            near = [within_degrees(r, 20.0) for r in runs_here]
            near10 = [within_degrees(r, 10.0) for r in runs_here]
            if near10:
                setm(group + "WithinTenMedianPct", pct(float(np.median(np.concatenate(near10)))))
            if near:
                setm(group + "WithinTwentyMedianPct", pct(float(np.median(np.concatenate(near)))))
            on_circle = [a for a in arr if a.circle_gaps_deg]
            setm(group + "CircleRuns", f"{len(on_circle)} of {len(arr)}")
            if on_circle:
                gaps = [x for a in on_circle for x in a.circle_gaps_deg]
                setm(group + "CircleGapMin", f(min(gaps), 0))
                setm(group + "CircleGapMax", f(max(gaps), 0))
        toks = [load_json(TOKENS / f"{k}__{t}" / "tokens.json") for t in tags for k, _ in COHORTS]
        toks = [x for x in toks if x]
        if toks:
            setm(group + "MILevelOneMin", f(min(x["usage"][0]["mi_bits"] for x in toks), 2))
            setm(group + "MILevelOneMax", f(max(x["usage"][0]["mi_bits"] for x in toks), 2))
            setm(group + "MIShareCodeLevelOneMin", pct(min(x["usage"][0]["mi_share_of_code_entropy"] for x in toks)))
            setm(group + "MIShareCodeLevelOneMax", pct(max(x["usage"][0]["mi_share_of_code_entropy"] for x in toks)))
            setm(group + "FlipZeroLevelOne", pct(float(np.median([x["code_change"]["zero_anchor"]["level1_median"] for x in toks])), 1))
            setm(group + "ForeignZero", pct(float(np.median([x["code_change"]["zero_anchor"]["foreign_share_median"] for x in toks]))))


# ----------------------------------------------------------------------------- GPU versus CPU
def device_numbers() -> None:
    """Agreement of the GPU (Metal) computation with the CPU on the first segments of every export and atlas run."""
    checks = {}
    for path in GEOM.glob("*/export_summary.json"):
        summary = load_json(path) or {}
        if summary.get("device_check"):
            checks[path.parent.name] = summary["device_check"]
    for root in (ATLAS, RESULTS_ROOT / "necessity_routing"):
        for path in root.glob("*/atlas_meta.json"):
            meta = load_json(path) or {}
            if meta.get("device_check"):
                checks[f"{root.name}/{path.parent.name}"] = meta["device_check"]
    trained = {k: c for k, c in checks.items() if "random_init" not in k and not k.endswith("_init")}
    if not trained:
        placeholder("device_checks")
        return
    setm("DeviceUnitDiffMax", f(max(c["u_max_abs_diff"] for c in trained.values()) * 1e6, 1))
    released = [c["code_agreement"] for k, c in trained.items() if "__seed" not in k]
    replicas = [c["code_agreement"] for k, c in trained.items() if "__seed" in k]
    setm("DeviceCodeReleasedMinPct", pct(min(released), 2))
    setm("DeviceCodeReplicaMinPct", pct(floor_to(min(replicas), 4), 2) if replicas else placeholder("DeviceCodeReplicaMinPct"))


# ----------------------------------------------------------------------------- supplementary
def supplementary_tables() -> None:
    """One table per cohort with the geometry and intervention values of all 16 sources."""
    parts = []
    for key, label in COHORTS:
        geo = load_json(GEOM / key / "geometry.json")
        atl = load_json(ATLAS / key / "summary.json")
        if geo is None or atl is None:
            placeholder(f"supplementary:{key}")
            continue
        g_rows = {row["source"]: row for row in geo["per_source"]}
        rel = atl["edits"]["rel_mse"]
        ratios = atl["ratios"]
        lines = [r"\begin{table}[p]", r"\renewcommand{\baselinestretch}{1}",
                 f"\\caption{{Per-source values, {label} (released tokenizer). $\\Rbar$ with 95\\% subject-cluster bootstrap interval; "
                 r"colat.: mean colatitude; $\sin\angle(\anchor,\bm{v}_1)$ with the residual bound~\eqref{eq:residual-bound}; "
                 r"$\|P_{16}\anchor\|^2$ as in \cref{tab:geometry}; var.: share of the centered variance along the anchor; "
                 r"$\dmse$ in percent of the unedited error for zero-anchor (ZA), colatitude $+20^\circ$ (C+), mean-anchor (MA), "
                 r"matched rotation in the support (MR) and tangent resample (TR); K: share of the zero-anchor damage reproduced "
                 r"through the decoder's keys alone, relative to the damage of the full edit.}",
                 f"\\label{{tab:supp-{key}}}", r"\centering\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
                 r"\begin{tabular}{rcccccrrrrrrr}", r"\toprule",
                 r"\textbf{s} & $\Rbar$ & \textbf{colat.}\,($^\circ$) & $\sin\angle(\anchor,\bm{v}_1)$ & \textbf{bound} & $\|P_{16}\anchor\|^2$ & "
                 r"\textbf{var.} & \textbf{ZA} & \textbf{C+} & \textbf{MA} & \textbf{MR} & \textbf{TR} & \textbf{K} \\", r"\midrule"]
        for s in range(16):
            g = g_rows[s]
            ci_r = g["ci"].get("R", [float("nan")] * 2)
            lines.append(
                f"{s + 1} & {f(g['R'])} [{f(ci_r[0])}, {f(ci_r[1])}] & {f(g['mean_angle_deg'], 1)} & "
                f"{f(g['sin_anchor_uncentered_top'], 3)} & {f(g['residual_sin_bound'], 3)} & {f(g['anchor_in_cpc16'], 2)} & "
                f"{pct(g['anchor_variance_fraction'], 1)}\\% & {tnum(pct(rel['zero_anchor']['per_source'][s], 1))} & "
                f"{tnum(pct(rel['colat_p20']['per_source'][s], 1))} & {tnum(pct(rel['mean_anchor']['per_source'][s], 2))} & "
                f"{tnum(pct(rel['rot_matched_on']['per_source'][s], 1))} & {tnum(pct(rel['tangent_resample']['per_source'][s], 1))} & "
                f"{f(ratios['keys_over_zero']['per_source'][s], 2)} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        parts.append("\n".join(lines))
    (TAB_DIR / "supplementary.tex").write_text("\n\n".join(parts) + "\n")


def main() -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    static_numbers()
    cohort_numbers()
    geometry_numbers()
    control_numbers()
    token_numbers()
    necessity_numbers()
    routing_numbers()
    readout_numbers()
    transfer_numbers()
    replication_numbers()
    device_numbers()
    supplementary_tables()
    # analyses that have not finished yet: a visible placeholder for every capitalized macro the text uses that is
    # neither collected here nor defined in the manuscript (LaTeX's own capitalized commands are listed explicitly)
    tex = (OUT_DIR / "manuscript.tex").read_text()
    latex_commands = {"Cref", "Crefformat", "FloatBarrier", "Delta", "Sigma", "Theta", "Gamma", "Lambda", "Omega", "Phi", "Psi", "Pi"}
    defined = set(re.findall(r"\\(?:re)?newcommand\{\\([A-Za-z]+)\}", tex)) | latex_commands
    for name in sorted(set(re.findall(r"\\([A-Z][A-Za-z]+)", tex)) - set(MACROS) - defined):
        setm(name, placeholder(name))
    lines = ["% Generated by scripts/collect_manuscript_numbers.py -- do not edit by hand."]
    for name, value in list(MACROS.items()):            # true minus signs for negative numbers in text
        MACROS[name] = re.sub(r"(?<![\w$\-])-(?=\d)", "$-$", value)
    for name, value in sorted(MACROS.items()):
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    (OUT_DIR / "numbers.tex").write_text("\n".join(lines) + "\n")
    (RESULTS_ROOT / "manuscript_numbers.json").write_text(json.dumps(MACROS, indent=1))
    print(f"{len(MACROS)} macros; missing: {sorted(set(MISSING))}")


if __name__ == "__main__":
    main()
