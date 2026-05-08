#!/usr/bin/env python
"""Build publication-style figures from existing CSV/JSON results.

Outputs are saved as PNG and PDF (300 dpi) under `results/figures/`.

Each figure entrypoint is independent. Run all with `--figures all`, or one
at a time with e.g. `--figures source_dominance,prototype_damage`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neurofm_mechinterp.paths import RESULTS_ROOT, ensure_dir


FIGURES_ROOT = RESULTS_ROOT / "figures"

PD31_SEED_DIRS = {
    0: "pd31/encoder_pre_quantizer/linear_topk64_h1024_l1_0_seed0_pilot_topk64_ep10",
    1: "pd31/encoder_pre_quantizer/linear_topk64_h1024_l1_0_seed1_pilot_topk64_ep10",
    2: "pd31/encoder_pre_quantizer/linear_topk64_h1024_l1_0_seed2_pilot_topk64_ep10",
    3: "pd31/encoder_pre_quantizer/linear_topk64_h1024_l1_0_seed3_pilot_topk64_ep10",
}

MEG_SEED_DIRS = {
    0: "MEG-MMI/encoder_pre_quantizer/linear_topk64_h1024_l1_0_seed0_pilot_topk64",
    1: "MEG-MMI/encoder_pre_quantizer/linear_topk64_h1024_l1_0_seed1",
    2: "MEG-MMI/encoder_pre_quantizer/linear_topk64_h1024_l1_0_seed2",
    3: "MEG-MMI/encoder_pre_quantizer/linear_topk64_h1024_l1_0_seed3",
}

PD31_INTERVENTION_DIR = "source_prototype_intervention_source_3_rows_3_remove_projection_scale_1"
MEG_INTERVENTION_DIR = "source_prototype_intervention_source_2_rows_2_remove_projection_scale_1"


def save_fig(fig: plt.Figure, name: str) -> None:
    out_dir = ensure_dir(FIGURES_ROOT)
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_source_dominance() -> None:
    """F1: source dominance per dataset.

    Three panels:
      (a) PD31 EEG top-1 source count (76 segments, 4-subject sweep).
      (b) MEG-MMI top-1 source count from the original 2-subject 20-segment sweep.
      (c) MEG-MMI uncropped 6-subject source-ablation Δ MSE (per source) — the
          analysis basis for the paper's source-2 vs source-9 dissociation.
    """
    pd31 = pd.read_csv(RESULTS_ROOT / "source_stability" / "pd31" / "source_summary.csv")
    meg_2subj = pd.read_csv(RESULTS_ROOT / "source_stability" / "MEG-MMI" / "source_summary.csv")
    meg_6subj_path = RESULTS_ROOT / "real_data" / "MEG-MMI" / "tiny" / "source_ablation" / "source_ablation.csv"
    meg_6subj = pd.read_csv(meg_6subj_path) if meg_6subj_path.exists() else None

    n_panels = 3 if meg_6subj is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels, 3.4), constrained_layout=True)

    df = pd31.sort_values("times_ranked_first", ascending=False).reset_index(drop=True)
    axes[0].bar(
        [str(int(s)) for s in df["source_idx"]],
        df["times_ranked_first"],
        color=["#1f77b4" if x > 0 else "#aaaaaa" for x in df["times_ranked_first"]],
    )
    axes[0].set_title("(a) PD31 EEG — top-1 count\n(76 seg, 4 subj)", fontsize=10)
    axes[0].set_xlabel("source index")
    axes[0].set_ylabel("# segments where source ranks #1")
    axes[0].set_ylim(0, 76)

    df = meg_2subj.sort_values("times_ranked_first", ascending=False).reset_index(drop=True)
    axes[1].bar(
        [str(int(s)) for s in df["source_idx"]],
        df["times_ranked_first"],
        color=["#ff7f0e" if x > 0 else "#aaaaaa" for x in df["times_ranked_first"]],
    )
    axes[1].set_title("(b) MEG-MMI — top-1 count\n(20 seg, 2 subj, top-4 tested)", fontsize=10)
    axes[1].set_xlabel("source index")
    axes[1].set_ylabel("# segments where source ranks #1")
    axes[1].set_ylim(0, 20)

    if meg_6subj is not None:
        df = meg_6subj.sort_values("mse_delta", ascending=False).reset_index(drop=True)
        axes[2].bar(
            [str(int(s)) for s in df["source_idx"]],
            df["mse_delta"],
            color="#2ca02c",
        )
        axes[2].set_title("(c) MEG-MMI — source-ablation $\\Delta$MSE\n(uncropped, 6 subj)", fontsize=10)
        axes[2].set_xlabel("source index")
        axes[2].set_ylabel(r"$\Delta$ input MSE")

    fig.suptitle("Tokenizer source dominance across datasets and sample sizes", fontsize=12, fontweight="bold")
    save_fig(fig, "fig_source_dominance")


def _read_run_summary(model_dir: Path, intervention_dir: str) -> pd.DataFrame | None:
    csv_path = model_dir / intervention_dir / "run_summary.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def figure_prototype_damage() -> None:
    """F2: prototype removal mean delta MSE per seed, two datasets."""
    rows = []
    for seed, rel in PD31_SEED_DIRS.items():
        df = _read_run_summary(RESULTS_ROOT / "sae_models" / rel, PD31_INTERVENTION_DIR)
        if df is not None:
            rows.append({"dataset": "PD31 source 3", "seed": seed, "delta_mse": float(df.iloc[0]["mean_delta_input_mse"])})
    for seed, rel in MEG_SEED_DIRS.items():
        df = _read_run_summary(RESULTS_ROOT / "sae_models" / rel, MEG_INTERVENTION_DIR)
        if df is not None:
            rows.append({"dataset": "MEG source 2", "seed": seed, "delta_mse": float(df.iloc[0]["mean_delta_input_mse"])})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(5.6, 3.4), constrained_layout=True)
    width = 0.35
    seeds = sorted(df["seed"].unique())
    for i, dataset_name in enumerate(["PD31 source 3", "MEG source 2"]):
        sub = df[df["dataset"] == dataset_name].sort_values("seed")
        ax.bar(np.arange(len(sub)) + i * width, sub["delta_mse"], width=width, label=dataset_name)
    ax.set_xticks(np.arange(len(seeds)) + width / 2)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel(r"$\Delta$ input MSE under prototype removal")
    ax.set_title("Source-prototype removal damage is seed-stable across 4 seeds")
    ymax = float(df["delta_mse"].max()) * 1.32
    ax.set_ylim(0, ymax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    save_fig(fig, "fig_prototype_damage")


def figure_seed_stability() -> None:
    """F5: pairwise prototype-cosine matrix per dataset."""
    csv = RESULTS_ROOT / "multi_seed_stability" / "seed_pair_prototype_cosine.csv"
    if not csv.exists():
        print(f"missing: {csv}")
        return
    df = pd.read_csv(csv)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), constrained_layout=True)
    for ax, dataset_name in zip(axes, ["pd31", "MEG-MMI"]):
        sub = df[df["dataset"] == dataset_name]
        seeds = sorted(sub["left_seed"].unique())
        m = np.zeros((len(seeds), len(seeds)))
        for _, row in sub.iterrows():
            li = seeds.index(int(row["left_seed"]))
            ri = seeds.index(int(row["right_seed"]))
            m[li, ri] = row["prototype_cosine"]
        offdiag = m[~np.eye(len(seeds), dtype=bool)]
        vmin, vmax = float(offdiag.min()), 1.0
        im = ax.imshow(m, vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels([f"s{s}" for s in seeds])
        ax.set_yticks(range(len(seeds)))
        ax.set_yticklabels([f"s{s}" for s in seeds])
        ax.set_title(f"{dataset_name} (off-diag {offdiag.min():.4f}–{offdiag.max():.4f})", fontsize=10)
        threshold = vmin + 0.55 * (vmax - vmin)
        for i in range(len(seeds)):
            for j in range(len(seeds)):
                color = "black" if m[i, j] > threshold else "white"
                ax.text(j, i, f"{m[i, j]:.4f}", ha="center", va="center", color=color, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Source-prototype cosine across SAE seeds", fontsize=11, fontweight="bold")
    save_fig(fig, "fig_seed_stability")


def figure_patch_decomposition() -> None:
    """F3: projection vs residual cross-subject patching, expanded sets, three sources.

    Uses expanded10 PD31 source 3, expanded6 MEG source 2, and expanded6 MEG
    source 9 — matches the paper's §6 main table.
    """
    sae_root = RESULTS_ROOT / "sae_models"
    pd31_dir = sae_root / "pd31" / "encoder_pre_quantizer" / "linear_topk64_h1024_l1_0_seed0_expanded10_topk64_ep10"
    meg_dir = sae_root / "MEG-MMI" / "encoder_pre_quantizer" / "linear_topk64_h1024_l1_0.001_seed0_expanded6_topk64_ep10"

    pd31_proj = pd.read_csv(pd31_dir / "source_prototype_patch_source_3_rows_3_scale_1_expanded10_max40_projection" / "run_summary.csv")
    pd31_res = pd.read_csv(pd31_dir / "source_prototype_patch_source_3_rows_3_residual_scale_1_expanded10_max40_residual" / "run_summary.csv")
    meg2_proj = pd.read_csv(meg_dir / "source_prototype_patch_source_2_rows_2_scale_1_expanded6_max40_projection" / "run_summary.csv")
    meg2_res = pd.read_csv(meg_dir / "source_prototype_patch_source_2_rows_2_residual_scale_1_expanded6_max40_residual" / "run_summary.csv")
    meg9_proj = pd.read_csv(meg_dir / "source_prototype_patch_source_9_rows_9_scale_1_expanded6_src9_max40_projection" / "run_summary.csv")
    meg9_res = pd.read_csv(meg_dir / "source_prototype_patch_source_9_rows_9_residual_scale_1_expanded6_src9_max40_residual" / "run_summary.csv")

    def _val(df_):
        return float(df_.iloc[0]["mean_cosine_patch_delta_to_donor_delta"])

    rows = [
        ("PD31 src 3", "projection", _val(pd31_proj)),
        ("PD31 src 3", "residual", _val(pd31_res)),
        ("MEG src 2", "projection", _val(meg2_proj)),
        ("MEG src 2", "residual", _val(meg2_res)),
        ("MEG src 9", "projection", _val(meg9_proj)),
        ("MEG src 9", "residual", _val(meg9_res)),
    ]
    df = pd.DataFrame(rows, columns=["dataset", "patch_kind", "donor_alignment"])

    fig, ax = plt.subplots(figsize=(6.6, 3.6), constrained_layout=True)
    kinds = ["projection", "residual"]
    datasets = ["PD31 src 3", "MEG src 2", "MEG src 9"]
    width = 0.26
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, dataset_name in enumerate(datasets):
        sub = df[df["dataset"] == dataset_name]
        values = [float(sub[sub["patch_kind"] == k]["donor_alignment"].iloc[0]) for k in kinds]
        ax.bar(np.arange(len(kinds)) + (i - 1) * width, values, width=width, label=dataset_name, color=colors[i])
    ax.set_xticks(np.arange(len(kinds)))
    ax.set_xticklabels(kinds)
    ax.set_ylabel("donor-direction alignment (cosine)")
    ax.set_title("Donor-specific source variation lives in the\nprototype-orthogonal residual (PD31 src 3 / MEG src 2);\nMEG src 9 transports nothing.", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.5)
    y_min = min(0.0, min(r[2] for r in rows)) * 1.2
    y_max = max(r[2] for r in rows) * 1.18
    ax.set_ylim(y_min, y_max)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=False)
    save_fig(fig, "fig_patch_decomposition")


def figure_label_transport() -> None:
    """F4: similarity-matched same-vs-different label residual transport on PD31 source 3."""
    base_dir = RESULTS_ROOT / "sae_models" / "pd31/encoder_pre_quantizer"
    seed_to_dir = {
        0: base_dir / "linear_topk64_h1024_l1_0_seed0_pilot_topk64",
        1: base_dir / "linear_topk64_h1024_l1_0_seed1",
    }
    rows = []
    for seed, model_dir in seed_to_dir.items():
        same = pd.read_csv(model_dir / "source_prototype_patch_source_3_rows_3_residual_scale_1_seed{0}_same_label_matched10_tol0025".format(seed) / "run_summary.csv")
        diff = pd.read_csv(model_dir / "source_prototype_patch_source_3_rows_3_residual_scale_1_seed{0}_different_label_matched10_tol0025".format(seed) / "run_summary.csv")
        rows.append({"seed": seed, "label_kind": "same-label", "donor_alignment": float(same.iloc[0]["mean_cosine_patch_delta_to_donor_delta"]), "delta_mse": float(same.iloc[0]["mean_delta_input_mse"])})
        rows.append({"seed": seed, "label_kind": "diff-label", "donor_alignment": float(diff.iloc[0]["mean_cosine_patch_delta_to_donor_delta"]), "delta_mse": float(diff.iloc[0]["mean_delta_input_mse"])})
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), constrained_layout=True)
    width = 0.35
    seeds = sorted(df["seed"].unique())
    for i, kind in enumerate(["same-label", "diff-label"]):
        sub = df[df["label_kind"] == kind].sort_values("seed")
        axes[0].bar(np.arange(len(seeds)) + i * width, sub["donor_alignment"], width=width, label=kind)
        axes[1].bar(np.arange(len(seeds)) + i * width, sub["delta_mse"], width=width, label=kind)
    align_max = float(df["donor_alignment"].max())
    mse_max = float(df["delta_mse"].max())
    for ax, ylabel, title, ymax in [
        (axes[0], "donor-direction alignment", "Same-label residual transports more", align_max * 1.18),
        (axes[1], r"mean $\Delta$ input MSE", "Magnitude of intervention is matched", mse_max * 1.18),
    ]:
        ax.set_xticks(np.arange(len(seeds)) + width / 2)
        ax.set_xticklabels([f"seed {s}" for s in seeds])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, ymax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=2, frameon=False)
    fig.suptitle("PD31 source 3: similarity-matched residual transport, same-label vs different-label", fontsize=10, fontweight="bold")
    save_fig(fig, "fig_label_transport")


def figure_pca_ica_baseline() -> None:
    """F6: SAE prototype vs PCA/ICA baseline removal damage."""
    rows = []
    for dataset_label, run_dir, source_idx in [
        ("PD31 src 3", RESULTS_ROOT / "baseline_pca_ica" / "pd31_source3_seed0", 3),
        ("MEG src 2", RESULTS_ROOT / "baseline_pca_ica" / "MEG-MMI_source2_seed0", 2),
    ]:
        method_csv = run_dir / "method_summary.csv"
        if not method_csv.exists():
            continue
        method = pd.read_csv(method_csv)
        direction_csv = run_dir / "direction_summary.csv"
        direction = pd.read_csv(direction_csv) if direction_csv.exists() else pd.DataFrame()
        for _, row in method.iterrows():
            rows.append({"dataset": dataset_label, "method": row["method"], "delta_mse": float(row["mean_delta_input_mse"]), "direction_csv": str(direction_csv)})
    if not rows:
        print("PCA/ICA results not yet ready; skipping fig_pca_ica_baseline")
        return
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.0, 3.6), constrained_layout=True)
    methods = ["sae_prototype", "pca", "ica"]
    width = 0.32
    datasets = sorted(df["dataset"].unique())
    for i, dataset_name in enumerate(datasets):
        sub = df[df["dataset"] == dataset_name]
        values = []
        for m in methods:
            v = sub[sub["method"] == m]["delta_mse"]
            values.append(float(v.iloc[0]) if len(v) else 0.0)
        ax.bar(np.arange(len(methods)) + (i - (len(datasets) - 1) / 2) * width, values, width=width, label=dataset_name)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(["SAE prototype", "best PCA", "best ICA"])
    ax.set_ylabel(r"$\Delta$ input MSE under direction removal")
    ax.set_title("SAE prototype vs PCA/ICA baselines (best matched component)")
    ax.axhline(0, color="black", linewidth=0.5)
    y_min = min(df["delta_mse"].min(), 0.0) * 1.15
    y_max = df["delta_mse"].max() * 1.20
    ax.set_ylim(y_min, y_max)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    save_fig(fig, "fig_pca_ica_baseline")


def figure_downstream_probe() -> None:
    """F7: HC vs PD classifier — baseline / prototype-removed / prototype-only / random control."""
    summary_paths = [
        (
            "PD31 (10 subj)",
            RESULTS_ROOT
            / "sae_models"
            / "pd31"
            / "encoder_pre_quantizer"
            / "linear_topk64_h1024_l1_0_seed0_expanded10_topk64_ep10"
            / "downstream_expanded10_mean_full"
            / "summary.json",
        ),
        (
            "MEG-MMI (6 subj)",
            None,  # Resolved at call time from the expanded6 SAE if present.
        ),
    ]
    meg_summary = None
    for cand in (
        RESULTS_ROOT
        / "sae_models"
        / "MEG-MMI"
        / "encoder_pre_quantizer"
    ).glob("*expanded6*/downstream_expanded6_mean/summary.json"):
        meg_summary = cand
        break
    summary_paths[1] = ("MEG-MMI (6 subj)", meg_summary)

    rows = []
    for dataset_label, summary_path in summary_paths:
        if summary_path is None or not Path(summary_path).exists():
            continue
        data = json.loads(Path(summary_path).read_text())
        for cv_name, cv_block in data.get("results", {}).items():
            for setting in ("baseline", "prototype_removed", "prototype_only", "random_dir_removed"):
                if setting not in cv_block:
                    continue
                acc = cv_block[setting].get("best_accuracy")
                auc = cv_block[setting].get("best_auc")
                rows.append({
                    "dataset": dataset_label, "cv": cv_name, "setting": setting,
                    "accuracy": acc, "auc": auc,
                })
    if not rows:
        print("downstream-probe results not ready; skipping fig_downstream_probe")
        return
    df = pd.DataFrame(rows)

    settings_order = ["baseline", "prototype_removed", "prototype_only", "random_dir_removed"]
    setting_labels = ["Baseline", "Prototype\nremoved", "Prototype\nonly", "Random dir.\nremoved"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
    for ax, metric, ylabel in [(axes[0], "accuracy", "Stratified-5-fold accuracy"), (axes[1], "auc", "Stratified-5-fold AUC")]:
        sub = df[df["cv"] == "stratified5"]
        datasets = sorted(sub["dataset"].unique())
        width = 0.36 if len(datasets) == 2 else 0.6
        x = np.arange(len(settings_order))
        for i, dataset_name in enumerate(datasets):
            sub_d = sub[sub["dataset"] == dataset_name]
            values = []
            for setting in settings_order:
                v = sub_d[sub_d["setting"] == setting][metric]
                values.append(float(v.iloc[0]) if len(v) else 0.0)
            offset = (i - (len(datasets) - 1) / 2) * width
            ax.bar(x + offset, values, width=width, label=dataset_name)
        ax.set_xticks(x)
        ax.set_xticklabels(setting_labels)
        ax.set_ylabel(ylabel)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
        ax.set_ylim(0, 1.05)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)
    fig.suptitle("Downstream classifier under prototype-direction interventions")
    save_fig(fig, "fig_downstream_probe")


FIGURE_ENTRYPOINTS = {
    "source_dominance": figure_source_dominance,
    "prototype_damage": figure_prototype_damage,
    "seed_stability": figure_seed_stability,
    "patch_decomposition": figure_patch_decomposition,
    "label_transport": figure_label_transport,
    "pca_ica_baseline": figure_pca_ica_baseline,
    "downstream_probe": figure_downstream_probe,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures", default="all", help="Comma-separated figure names or 'all'.")
    args = parser.parse_args()
    selected = list(FIGURE_ENTRYPOINTS) if args.figures == "all" else [s.strip() for s in args.figures.split(",") if s.strip()]
    for name in selected:
        if name not in FIGURE_ENTRYPOINTS:
            print(f"unknown figure: {name}; skipping")
            continue
        try:
            FIGURE_ENTRYPOINTS[name]()
            print(f"wrote fig_{name}")
        except Exception as exc:
            print(f"failed: {name}: {exc}")


if __name__ == "__main__":
    main()
