"""Figures of the manuscript, generated from results/ (PDF, vector).

Every figure is drawn at the width at which it is printed (the 390-pt text width), so that no text is
smaller than 7 pt on paper.

fig_overview.pdf     (a) the tokenizer and the two pathways of the decoder's attention, (b) hyperspherical
                     coordinates of the unit vectors of one source
fig_geometry.pdf     (a) mean resultant length per source, (b) anchor vs uncentered and centered principal
                     directions, (c) per-source anchor agreement between cohorts
fig_interventions.pdf (a) damage vs geodesic displacement for families of edits, (b) colatitude tuning,
                     (c) share of the damage of location, identity and content edits reproduced through the
                     decoder's keys or values alone
fig_readout.pdf      subject-level AUC of each representation per contrast and for cross-site transfer, with the
                     random-direction null
fig_replication.pdf  key quantities for every tokenizer instance (geometry, anchor arrangement, interventions, tokens)
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from neurofm_mechinterp.anchors import anchor_arrangement  # noqa: E402
from neurofm_mechinterp.paths import REPO_ROOT, RESULTS_ROOT  # noqa: E402

OUT = REPO_ROOT / "paper" / "figures"
COHORTS = [("pd_ucsd", "PD-UCSD"), ("pd_iowa", "PD-Iowa"), ("dementia", "Dementia"), ("mdd_meg", "MDD-MEG")]
SHORT = {"pd_ucsd": "UCSD", "pd_iowa": "Iowa", "dementia": "Dem.", "mdd_meg": "MEG"}
# validated categorical palette (fixed order) + marker shapes as secondary encoding
COLORS = {"pd_ucsd": "#2a78d6", "pd_iowa": "#eb6834", "dementia": "#1baf7a", "mdd_meg": "#eda100"}
MARKERS = {"pd_ucsd": "o", "pd_iowa": "s", "dementia": "^", "mdd_meg": "D"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GEOM = RESULTS_ROOT / "source_geometry"
ATLAS = RESULTS_ROOT / "necessity_atlas"
READOUT = RESULTS_ROOT / "diagnostic_readout"

WIDTH = 5.4               # inches: the manuscript's text width (390 pt), at which every figure is printed
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def style(ax, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def panel(ax, letter: str, x: float = -0.16, y: float = 1.04) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def jitter(n: int, seed: int, width: float = 0.05) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-width, width, n)


def cohort_handles(keys, filled: bool = True) -> list:
    from matplotlib.lines import Line2D  # noqa: PLC0415
    return [Line2D([], [], linestyle="none", marker=MARKERS[key], markersize=4.5, label=dict(COHORTS)[key],
                   color=COLORS[key] if filled else "white", markeredgecolor="white" if filled else COLORS[key],
                   markeredgewidth=0.4 if filled else 0.9) for key in keys]


def cohort_legend(ax, keys, **kw) -> None:
    ax.legend(handles=cohort_handles(keys), **kw)


# ----------------------------------------------------------------------------- overview schematic
def _box(ax, x, y, w, h, text, face="#f4f3ef", edge=INK2, fontsize=7.5):
    from matplotlib.patches import FancyBboxPatch  # noqa: PLC0415
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02", facecolor=face,
                                edgecolor=edge, linewidth=0.7, clip_on=False))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=INK, linespacing=1.25)


def _arrow(ax, x0, y0, x1, y1, color=INK2, style="-|>", lw=0.8, ls="-"):
    from matplotlib.patches import FancyArrowPatch  # noqa: PLC0415
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=8, color=color, linewidth=lw,
                                 linestyle=ls, shrinkA=0, shrinkB=0, clip_on=False))


def fig_overview() -> None:
    """(a) the tokenizer and the two decoder pathways, (b) hyperspherical coordinates of one source."""
    fig = plt.figure(figsize=(WIDTH, 3.9))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], width_ratios=[1.0, 1.0], hspace=0.12, wspace=0.0,
                            left=0.03, right=0.97, top=0.95, bottom=0.02)
    ax = fig.add_subplot(grid[0, :])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # encoder and quantizer (left to right)
    top, height, width, step = 0.55, 0.28, 0.176, 0.206
    labels = ["EEG/MEG\nwindow\n($N_c$ sensors)", "SEANet\nencoder\nper sensor", "cross-\nattention,\n16 queries",
              "16 sources\n$\\boldsymbol{x}_s/\\|\\boldsymbol{x}_s\\|$", "residual VQ,\n4 levels\n(tokens)"]
    for i, text in enumerate(labels):
        _box(ax, i * step, top, width, height, text, face="#e8f0fb" if i >= 3 else "#f4f3ef")
        if i:
            _arrow(ax, (i - 1) * step + width, top + height / 2, i * step, top + height / 2)
    xc = 2 * step + width / 2                                                     # centre of the cross-attention box
    ax.text(xc, top + height + 0.11, "sensor embeddings enter\nthe attention keys", ha="center", va="bottom", fontsize=7,
            color=INK2, linespacing=1.2)
    _arrow(ax, xc, top + height + 0.10, xc, top + height + 0.012)
    # decoder: the quantized sources enter the decoder's attention as keys and values
    x_vq = 4 * step + width / 2
    _box(ax, 0.40, 0.02, 0.30, 0.38, "decoder attention\nquery: sensor embedding\nkeys $\\leftarrow$ $\\boldsymbol{q}_s$\n"
         "values $\\leftarrow$ $\\boldsymbol{q}_s$")
    _box(ax, 0.77, 0.07, 0.23, 0.28, "SEANet decoder\nper sensor\n$\\rightarrow$ reconstruction")
    _arrow(ax, x_vq, top, x_vq, 0.465, style="-")
    _arrow(ax, x_vq, 0.465, 0.55, 0.465, style="-")
    _arrow(ax, 0.55, 0.465, 0.55, 0.40)
    ax.text(x_vq - 0.02, 0.475, "quantized sources $\\boldsymbol{q}_s$ (no source index)", ha="right", va="bottom", fontsize=7,
            color=INK2)
    _arrow(ax, 0.70, 0.21, 0.77, 0.21)
    ax.text(0.175, 0.30, "keys: how much each sensor\nreads from a source\n(location, identity)", ha="center", va="center",
            fontsize=7, color="#b23a3a", linespacing=1.2)
    ax.text(0.175, 0.08, "values: what is read\n(content)", ha="center", va="center", fontsize=7, color="#4b3cb3", linespacing=1.2)
    _arrow(ax, 0.34, 0.30, 0.40, 0.30, color="#b23a3a", lw=0.8)
    _arrow(ax, 0.31, 0.09, 0.40, 0.12, color="#4b3cb3", lw=0.8)
    panel(ax, "a", x=-0.03, y=0.97)

    ax = fig.add_subplot(grid[1, 0])
    ax.set_aspect("equal")
    ax.axis("off")
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color=INK2, linewidth=0.8)                       # sphere outline
    ax.plot(np.cos(t), 0.28 * np.sin(t), color=INK2, linewidth=0.6, linestyle=(0, (3, 2)))   # equator
    cap = np.radians(50)
    ax.plot(np.sin(cap) * np.cos(t), np.cos(cap) + 0.15 * np.sin(cap) * np.sin(t), color="#2a78d6", linewidth=0.8)
    ax.text(-0.84, 0.80, "cap", fontsize=7.5, color="#2a78d6", ha="right")
    ax.annotate("", xy=(0, 1.0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0))
    ax.text(0.05, 1.03, "anchor $\\boldsymbol{p}_s$", fontsize=7.5, ha="left", va="bottom")
    theta = np.radians(38)
    u = (np.sin(theta), np.cos(theta))
    ax.annotate("", xy=u, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#2a78d6", lw=1.0))
    ax.text(u[0] + 0.05, u[1] + 0.02, "$\\boldsymbol{u}$", fontsize=8, color="#2a78d6")
    arc = np.linspace(0, theta, 30)
    ax.plot(0.32 * np.sin(arc), 0.32 * np.cos(arc), color=INK, linewidth=0.6)
    ax.text(0.10, 0.40, "$\\theta$", fontsize=7.5)
    ax.annotate("", xy=(0.62, 0.0), xytext=(0.0, 0.0), arrowprops=dict(arrowstyle="-|>", color="#6250d6", lw=0.9))
    ax.text(0.31, -0.08, "$\\boldsymbol{\\sigma}$", fontsize=8, color="#6250d6", ha="center", va="top")
    ax.text(1.04, 0.10, "equator", fontsize=7, color=INK2, ha="left")
    ax.set_xlim(-1.2, 1.55)
    ax.set_ylim(-1.08, 1.22)
    panel(ax, "b", x=-0.02, y=0.97)

    ax = fig.add_subplot(grid[1, 1])
    ax.axis("off")
    ax.text(0.0, 0.80, "$\\boldsymbol{u}=a\\,\\boldsymbol{p}_s+\\sqrt{1-a^{2}}\\;\\boldsymbol{\\sigma}$", fontsize=9, va="center")
    for y, line in ((0.58, "$\\boldsymbol{p}_s$: anchor (normalized mean of the vectors)"),
                    (0.44, "$\\theta$: colatitude (angle to the anchor)"),
                    (0.30, "$a=\\cos\\theta$: anchor coordinate"),
                    (0.16, "$\\boldsymbol{\\sigma}\\perp\\boldsymbol{p}_s$: tangent direction (unit vector)")):
        ax.text(0.0, y, line, fontsize=7.5, va="center")
    fig.savefig(OUT / "fig_overview.pdf", metadata={"CreationDate": None})  # no date: byte-reproducible
    plt.close(fig)


# ----------------------------------------------------------------------------- geometry
def fig_geometry() -> None:
    keys = [k for k, _ in COHORTS if (GEOM / k / "geometry_sources.csv").exists()]
    if not keys:
        return
    tables = {k: pd.read_csv(GEOM / k / "geometry_sources.csv") for k in keys}
    fig = plt.figure(figsize=(WIDTH, 4.0), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], width_ratios=[1.0, 1.0])
    ax = fig.add_subplot(grid[0, :])
    offsets = np.linspace(-0.27, 0.27, len(COHORTS))
    for i, (key, _) in enumerate(COHORTS):
        if key not in tables:
            continue
        t = tables[key]
        ax.plot(t["source"] + 1 + offsets[i], t["R"], linestyle="none", marker=MARKERS[key], markersize=4.2,
                color=COLORS[key], markeredgecolor="white", markeredgewidth=0.5)
    ax.set_xticks(range(1, 17))
    ax.set_xlim(0.4, 16.6)
    ax.set_xlabel("source")
    ax.set_ylabel(r"$\bar{R}_s$")
    ax.set_ylim(0, 1)
    style(ax)
    cohort_legend(ax, keys, loc="lower left", ncol=4, handletextpad=0.2, columnspacing=1.2)
    panel(ax, "a", x=-0.085, y=1.08)

    ax = fig.add_subplot(grid[1, 0])
    cats = [("cos_anchor_uncentered_top", "$\\boldsymbol{v}_1$\ncos"),
            ("cos_anchor_cpc1", "PC 1\n|cos|"),
            ("max_cos_anchor_cpc10", "PCs 1\u201310\nmax |cos|"),
            ("anchor_in_cpc16", "PCs 1\u201316\nsq. norm")]
    for j, (col, _) in enumerate(cats):
        for i, (key, _) in enumerate(COHORTS):
            if key not in tables or col not in tables[key]:
                continue
            t = tables[key]
            ax.plot(j + offsets[i] * 0.9 + jitter(len(t), i + 10 * j, 0.035), t[col], linestyle="none", marker=MARKERS[key],
                    markersize=3.0, color=COLORS[key], markeredgecolor="white", markeredgewidth=0.35)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([c[1] for c in cats])
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("anchor vs principal\ndirections")
    style(ax)
    panel(ax, "b", x=-0.27, y=1.02)

    ax = fig.add_subplot(grid[1, 1])
    cross = load(GEOM / "cross_cohort.json")
    if cross:
        order = [f"{a}__{b}" for i, (a, _) in enumerate(COHORTS) for (b, _) in COHORTS[i + 1:]]
        pairs = [p for p in order if p in cross]
        for j, pair in enumerate(pairs):
            vals = np.array(cross[pair]["per_source"])
            color = COLORS["mdd_meg"] if "mdd_meg" in pair else INK2
            ax.plot(np.full(len(vals), j) + jitter(len(vals), j, 0.12), vals, linestyle="none", marker="o", markersize=3.0,
                    color=color, markeredgecolor="white", markeredgewidth=0.35)
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels([f"{SHORT[p.split('__')[0]]}\n{SHORT[p.split('__')[1]]}" for p in pairs])
        lo = min(min(cross[p]["per_source"]) for p in pairs)
        ax.set_ylim(max(0.0, lo - 0.012), 1.001)
        ax.set_ylabel("anchor cosine\nbetween cohorts")
        from matplotlib.lines import Line2D  # noqa: PLC0415
        ax.legend(handles=[Line2D([], [], linestyle="none", marker="o", markersize=4, color=INK2, markeredgecolor="white",
                                  label="two EEG cohorts"),
                           Line2D([], [], linestyle="none", marker="o", markersize=4, color=COLORS["mdd_meg"],
                                  markeredgecolor="white", label="EEG cohort and MDD-MEG")],
                  loc="lower left", handletextpad=0.2, borderaxespad=0.2)
        style(ax)
    panel(ax, "c", x=-0.3, y=1.02)
    fig.savefig(OUT / "fig_geometry.pdf", metadata={"CreationDate": None})  # no date: byte-reproducible
    plt.close(fig)


# ----------------------------------------------------------------------------- interventions
def atlas_summary(run: str):
    return load(ATLAS / run / "summary.json")


def per_source(summary: dict, edit: str, metric: str = "rel_mse") -> np.ndarray | None:
    e = summary["edits"][metric].get(edit)
    return None if e is None else np.array(e["per_source"])


FAMILIES = [  # (label, edits, colour, line style, marker): edit families for damage vs displacement
    ("colatitude, away from anchor", ["colat_p10", "half_anchor", "colat_p20", "zero_anchor"], "#e34948", "-", "o"),
    ("leading tangent axes", ["tpc_resample_1", "tpc_resample_4", "tpc_resample_16", "tpc_resample_64"], "#6250d6", "-", "s"),
    ("random (support)", ["rand_support_1", "rand_support_4", "rand_support_16"], "#008300", (0, (4, 2)), "^"),
    ("rotation (support)", ["rot_on_30", "rot_matched_on", "rot_on_60", "rot_on_90"], "#8c564b", "-", "v"),
    ("random (whole tangent space)", ["rand_off_16", "rot_matched_off"], INK2, (0, (4, 2)), "D"),
]


def pooled_median(runs: list[dict], edit: str, metric: str = "rel_mse") -> float:
    vals = np.concatenate([r["edits"][metric][edit]["per_source"] for r in runs if edit in r["edits"][metric]])
    return float(np.median(vals))


def fig_interventions() -> None:
    keys = [k for k, _ in COHORTS if atlas_summary(k)]
    if not keys:
        return
    runs = [atlas_summary(k) for k in keys]
    fig = plt.figure(figsize=(WIDTH, 3.95), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.08], width_ratios=[0.95, 1.05])
    top_row = grid[0, :].subgridspec(1, 2, width_ratios=[1.6, 1.0])
    ax = fig.add_subplot(top_row[0])
    for label, edits, color, ls, marker in FAMILIES:
        fam = [e for e in edits if all(e in r["displacement"] for r in runs)]
        xs = np.array([np.median([r["displacement"][e]["displacement_deg"] for r in runs]) for e in fam])
        ys = np.array([100 * pooled_median(runs, e) for e in fam])
        order = np.argsort(xs)
        open_ = label == "random (whole tangent space)"          # open markers: it overlaps the random-support family
        mk = dict(marker=marker, markersize=4.0, markerfacecolor="none" if open_ else color,
                  markeredgecolor=color if open_ else "white", markeredgewidth=0.9 if open_ else 0.4)
        ax.plot(xs[order], ys[order], linestyle=ls, color=color, linewidth=1.2, label=label, **mk)
        ax.plot(xs[order], ys[order], linestyle="none", color=color, zorder=3, **mk)  # markers above every line
    for edit in ("colat_m10", "colat_m20"):
        x = np.median([r["displacement"][edit]["displacement_deg"] for r in runs])
        ax.plot(x, 100 * pooled_median(runs, edit), linestyle="none", marker="o", markersize=4.2, markerfacecolor="white",
                markeredgecolor="#e34948", markeredgewidth=0.9)
    ax.plot([], [], linestyle="none", marker="o", markerfacecolor="white", markeredgecolor="#e34948", markersize=4.2,
            label="colatitude, towards anchor")
    ax.axhline(0, color=INK2, linewidth=0.6)
    ax.set_xlabel("median geodesic displacement (degrees)")
    ax.set_ylabel("$\\Delta$MSE\n(% of unedited)")
    style(ax, "both")
    handles, labels = ax.get_legend_handles_labels()
    order = [0, len(labels) - 1] + list(range(1, len(labels) - 1))           # the two colatitude families first
    legend_ax = fig.add_subplot(top_row[1])
    legend_ax.axis("off")
    legend_ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="center left", handlelength=2.6,
                     labelspacing=0.55, borderaxespad=0.0)
    panel(ax, "a", x=-0.16, y=1.10)

    ax = fig.add_subplot(grid[1, 0])
    shifts = [-20, -10, 10, 20]
    for key, r in zip(keys, runs):
        med = [100 * r["edits"]["rel_mse"][e]["median"] for e in ("colat_m20", "colat_m10", "colat_p10", "colat_p20")]
        dx = {"pd_ucsd": -1.5, "pd_iowa": -0.5, "dementia": 0.5, "mdd_meg": 1.5}[key]  # horizontal offset for visibility
        xs = [x + dx for x in shifts]
        ax.plot(xs[:2] + [0] + xs[2:], med[:2] + [0.0] + med[2:], color=COLORS[key], linewidth=1.2)  # 0 deg: unedited
        ax.plot(xs, med, linestyle="none", color=COLORS[key], marker=MARKERS[key], markersize=4.2, markeredgecolor="white",
                markeredgewidth=0.4, zorder=3)
    ax.axhline(0, color=INK2, linewidth=0.6)
    ax.axvline(0, color=INK2, linewidth=0.6, linestyle=(0, (2, 2)))
    ax.set_xticks([-20, -10, 0, 10, 20])
    ax.set_xlabel("colatitude shift (degrees)")
    ax.set_ylabel(r"$\Delta$MSE (% of unedited)")
    style(ax, "both")
    cohort_legend(ax, keys, loc="upper left", handletextpad=0.2, borderaxespad=0.2)
    panel(ax, "b", x=-0.24, y=1.02)

    ax = fig.add_subplot(grid[1, 1])
    groups = [("zero", "zero-\nanchor"), ("swap", "anchor\nswap"), ("tangent", "tangent\nresample"), ("tpc4", "4 leading\naxes")]
    routing = {k: load(RESULTS_ROOT / "necessity_routing" / k / "summary.json") for k in keys}
    routing = {k: v for k, v in routing.items() if v}
    rkeys = [k for k in keys if k in routing]
    bar = 0.8 / max(1, len(rkeys))                      # keys block on [2j-0.85, 2j-0.05], values on [2j+0.05, 2j+0.85]
    for i, key in enumerate(rkeys):
        rr = routing[key]["ratios"]
        for j, (g, _) in enumerate(groups):
            for side, start, fill in (("keys", 2 * j - 0.85, COLORS[key]), ("values", 2 * j + 0.05, "white")):
                name = f"{side}_over_{g}"
                if name not in rr:
                    continue
                vals = np.array(rr[name]["per_source"])
                x = start + bar * (i + 0.5)
                med = float(np.median(vals))
                ax.bar(x, med, width=bar * 0.85, color=fill, edgecolor=COLORS[key], linewidth=0.8)
                ax.errorbar(x, med, yerr=[[med - np.quantile(vals, 0.25)], [np.quantile(vals, 0.75) - med]], fmt="none",
                            ecolor=INK2, elinewidth=0.5, capsize=0)
    ax.axhline(1, color=INK2, linewidth=0.6, linestyle=(0, (2, 2)))
    ax.axhline(0, color=INK2, linewidth=0.6)
    ax.set_xticks([2 * j for j in range(len(groups))])
    ax.set_xticklabels([g[1] for g in groups])
    ax.set_xlim(-1.0, 2 * len(groups) - 1.0)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    from matplotlib.patches import Patch  # noqa: PLC0415
    ax.legend(handles=[Patch(facecolor=INK2, edgecolor=INK2, label="keys only"),
                       Patch(facecolor="white", edgecolor=INK2, linewidth=0.8, label="values only")],
              loc="upper center", handlelength=1.2, ncol=2, borderaxespad=0.1, columnspacing=1.2)
    ax.set_ylabel("share of full-edit damage")
    style(ax)
    panel(ax, "c", x=-0.22, y=1.02)
    fig.savefig(OUT / "fig_interventions.pdf", metadata={"CreationDate": None})  # no date: byte-reproducible
    plt.close(fig)


# ----------------------------------------------------------------------------- read-out
READOUT_FEATURES = [("anchor16", "anchor-16"), ("best_single", "best single anchor"), ("quant_anchor16", "quantized anchor-16"),
                    ("rbar16", "concentration-16"), ("norm16", "norm-16"), ("pooled256", "pooled"),
                    ("tangent4096", "tangent"), ("full4096", "full"), ("quant4096", "quantized"),
                    ("bandpower", "band power"), ("age_sex", "age and sex")]
CONTRASTS = [("pd_ucsd_pd_vs_hc", "PD-UCSD\nPD vs HC", "pd_ucsd"), ("pd_iowa_pd_vs_hc", "PD-Iowa\nPD vs HC", "pd_iowa"),
             ("dementia_ad_vs_cn", "Dementia\nAD vs CN", "dementia"), ("dementia_ftd_vs_cn", "Dementia\nFTD vs CN", "dementia"),
             ("mdd_meg_mdd_vs_hv", "MDD-MEG\nMDD vs HV", "mdd_meg")]


TRANSFERS = [("transfer_pd_iowa_to_pd_ucsd.json", "transfer\nIowa \u2192 UCSD", "pd_ucsd"),
             ("transfer_pd_ucsd_to_pd_iowa.json", "transfer\nUCSD \u2192 Iowa", "pd_iowa")]
TRANSFER_NAMES = {"bandpower": "bandpower_shared"}


def _readout_panel(ax, title: str, results: dict, null: dict, key: str, transfer: bool) -> None:
    """AUCs with intervals of every representation (rows), the random-direction null band and chance."""
    ax.axvspan(null["q025"], null["q975"], color=GRID, linewidth=0)
    ax.axvline(0.5, color=INK2, linewidth=0.6, linestyle=(0, (2, 2)))
    for i, (feat, _) in enumerate(READOUT_FEATURES):
        e = results.get(TRANSFER_NAMES.get(feat, feat) if transfer else feat)
        if e is None:
            continue
        ax.plot([e["auc_ci"][0], e["auc_ci"][1]], [i, i], color=COLORS[key], linewidth=1.1, solid_capstyle="butt")
        ax.plot(e["auc"], i, marker=MARKERS[key], markersize=4.4, color="white" if transfer else COLORS[key],
                markeredgecolor=COLORS[key] if transfer else "white", markeredgewidth=0.9 if transfer else 0.4)
    ax.set_title(title, fontsize=7.5, linespacing=1.15)
    ax.set_xlim(0.15, 1.0)
    ax.set_xticks([0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("AUC")
    style(ax, "x")


def fig_readout() -> None:
    present = [c for c in CONTRASTS if (READOUT / c[0] / "readout.json").exists()
               and "auc" in load(READOUT / c[0] / "readout.json")["results"][0]]
    transfers = [t for t in TRANSFERS if (READOUT / t[0]).exists() and "random16_null" in load(READOUT / t[0])]
    if not present:
        return
    n_cols = 4
    n_rows = int(np.ceil((len(present) + len(transfers) + 1) / n_cols))               # + 1 cell for the legend
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(WIDTH, 2.25 * n_rows), sharey=True, layout="constrained")
    axes = axes.ravel()
    for ax, (run, title, key) in zip(axes, present):
        r = load(READOUT / run / "readout.json")
        _readout_panel(ax, title, {e["feature"]: e for e in r["results"]}, r["random16_null"], key, transfer=False)
    for ax, (fname, title, key) in zip(axes[len(present):], transfers):
        t = load(READOUT / fname)
        _readout_panel(ax, title, t["results"]["train_scaler"], t["random16_null"]["train_scaler"], key, transfer=True)
    used = len(present) + len(transfers)
    for ax in axes[used:]:
        ax.axis("off")
    axes[0].set_yticks(range(len(READOUT_FEATURES)))
    axes[0].set_yticklabels([lab for _, lab in READOUT_FEATURES])
    axes[0].set_ylim(len(READOUT_FEATURES) - 0.5, -0.5)
    for row in range(n_rows):                                    # row labels on the first panel of every row
        axes[row * n_cols].tick_params(labelleft=True)
    from matplotlib.lines import Line2D  # noqa: PLC0415
    from matplotlib.patches import Patch  # noqa: PLC0415
    keys = [k for k, _ in COHORTS if any(c[2] == k for c in present)]
    handles = cohort_handles(keys) + [
        Line2D([], [], linestyle="none", marker="o", markersize=4.4, color="white", markeredgecolor=INK2, markeredgewidth=0.9,
               label="transfer (test cohort)"),
        Patch(facecolor=GRID, edgecolor="none", label="random 16 directions\n(2.5\u201397.5%)"),
        Line2D([], [], color=INK2, linewidth=0.6, linestyle=(0, (2, 2)), label="chance")]
    axes[used].legend(handles=handles, loc="center left", borderaxespad=0.0, handletextpad=0.4, labelspacing=0.6)
    fig.savefig(OUT / "fig_readout.pdf", metadata={"CreationDate": None})  # no date: byte-reproducible
    plt.close(fig)


# ----------------------------------------------------------------------------- tokenizer instances
INSTANCES = [("", "released"), ("__brainomni_tiny", "embedded"), ("__seed1", "replica 1"), ("__seed2", "replica 2"),
             ("__seed1_srcemb", "variant 1"), ("__seed2_srcemb", "variant 2"), ("__seed1_init", "replica 1, init."),
             ("__random_init0", "random weights")]


def fig_replication() -> None:
    rows = []
    for suffix, label in INSTANCES:
        for key, _ in COHORTS:
            run = key + suffix
            g = load(GEOM / run / "geometry.json")
            if g is None or "medians" not in g:
                continue
            a = atlas_summary(run)
            tok = load(RESULTS_ROOT / "token_structure" / run / "tokens.json")
            rows.append({"instance": label, "cohort": key, "R": g["medians"]["R"],
                         "anchor_dim": anchor_arrangement(torch.tensor(g["anchor_cosine_matrix"])).participation_ratio,
                         "zero": None if a is None else 100 * a["edits"]["rel_mse"]["zero_anchor"]["median"],
                         "keys": None if a is None or "keys_over_zero" not in a["ratios"] else a["ratios"]["keys_over_zero"]["median"],
                         "mi": None if tok is None or "mi_bits" not in tok["usage"][0] or suffix.endswith("init0") or suffix.endswith("_init")
                         else tok["usage"][0]["mi_bits"]})
    if not rows:
        return
    t = pd.DataFrame(rows)
    labels = [lab for _, lab in INSTANCES if lab in set(t["instance"])]
    panels = [("R", "median $\\bar{R}_s$"), ("anchor_dim", "directions spanned\nby the 16 anchors"),
              ("zero", "zero-anchor $\\Delta$MSE\n(% of unedited)"), ("keys", "zero-anchor, keys only\n(relative to full edit)"),
              ("mi", "$I(S;C_1)$ (bits)")]
    n_cols = 3
    n_rows = int(np.ceil((len(panels) + 1) / n_cols))                                    # + 1 cell for the legend
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(WIDTH, 0.2 * len(labels) * n_rows + 0.75 * n_rows), sharey=True,
                             layout="constrained")
    axes = axes.ravel()
    offsets = {"pd_ucsd": -0.30, "pd_iowa": -0.10, "dementia": 0.10, "mdd_meg": 0.30}
    for ax, (col, title) in zip(axes, panels):
        for key, _ in COHORTS:
            sub = t[(t["cohort"] == key) & t[col].notna()]
            y = [labels.index(i) for i in sub["instance"]]
            ax.plot(sub[col], np.array(y) + offsets[key], linestyle="none", marker=MARKERS[key], color=COLORS[key], markersize=3.4,
                    markeredgecolor="white", markeredgewidth=0.4)
        ax.set_xlabel(title)
        style(ax, "x")
    for ax in axes[len(panels):]:
        ax.axis("off")
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels)
    axes[0].set_ylim(len(labels) - 0.5, -0.5)          # every row visible, first row on top
    for row in range(n_rows):
        axes[row * n_cols].tick_params(labelleft=True)
    axes[len(panels)].legend(handles=cohort_handles([k for k, _ in COHORTS if k in set(t["cohort"])]), loc="center left",
                             borderaxespad=0.0, labelspacing=0.6)
    fig.savefig(OUT / "fig_replication.pdf", metadata={"CreationDate": None})  # no date: byte-reproducible
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (fig_overview, fig_geometry, fig_interventions, fig_readout, fig_replication):
        fn()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
