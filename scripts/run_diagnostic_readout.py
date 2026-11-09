"""Subject-level diagnostic read-out from tokenizer source representations.

Reads `results/source_geometry/<cohort>/export.pt` and writes
`results/diagnostic_readout/<cohort>_<contrast>/readout.json` (+ `oof_scores.csv`).

Every feature is first averaged over a subject's segments (neurofm_mechinterp.subject_features);
classification is L2 logistic regression with C selected by inner CV, evaluated by stratified 5-fold
CV repeated ``--repeats`` times. The reported AUC is the mean over repeats of the out-of-fold AUC,
and its interval is a stratified subject bootstrap of that same statistic. Anchors are re-estimated
on the training subjects of every fold. Feature sets:

  primary
    anchor16        per-source anchor coordinates c_s = mean(u_s) . p_s                        (16)
  secondary (Benjamini-Hochberg within the contrast)
    full4096        concatenated per-source mean unit vectors                                  (4096)
    tangent4096     the same with every anchor coordinate removed                              (4096)
    pooled256       mean over sources of the per-source mean unit vectors                      (256)
    quant4096       per-source means of the quantized vectors (what the foundation model sees)  (4096)
    quant_anchor16  anchor coordinates of the quantized vectors                                (16)
    rbar16          per-source mean resultant length within segments                           (16)
    norm16          per-source mean norm before L2 normalisation (discarded by the quantizer)  (16)
    bandpower       log band power per channel and band                                        (5C)
  covariates
    age_sex         age and sex alone
    <set>_adj       anchor16, full4096 and bandpower with age and sex regressed out within each
                    training fold (missing values imputed with the training mean)
  single sources
    single_<s>      one anchor coordinate (C = 1); family-wise p-values by the maximum statistic
    best_single     the source with the highest training-fold AUC, selected inside every fold
  null
    rand16          16 random isotropic unit directions (one per source), --n-random draws, each
                    evaluated exactly like anchor16

Permutation tests permute labels across subjects and re-run the whole CV (with --perm-repeats
repeats and C fixed at 1, for observed and permuted labels alike; selection steps such as
best_single are repeated on the permuted labels).

Paired comparisons (`paired.json`): the difference between the AUCs of two representations on the same
subjects, with a paired stratified bootstrap of the repeat-averaged AUC (out-of-fold scores fixed), for
every representation against band power and against anchor-16. With --paired-only, only the CV point
estimates are recomputed (same folds and seeds; they must reproduce the AUCs in readout.json to within
0.001, and the largest deviation is recorded) and paired.json is written.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from neurofm_mechinterp.paths import RESULTS_ROOT
from neurofm_mechinterp.readout import (
    ReadoutConfig,
    auc_rows,
    bh_adjust,
    bootstrap_repeat_auc_ci,
    bootstrap_repeat_auc_difference_ci,
    max_statistic_pvalues,
    paired_auc_difference,
    permutation_pvalue,
    repeated_cv_readout,
)
from neurofm_mechinterp.subject_features import subject_table

SECONDARY = ["full4096", "tangent4096", "pooled256", "quant4096", "quant_anchor16", "rbar16", "norm16", "bandpower"]
PAIRED = ["anchor16"] + SECONDARY
ADJUSTED = ["anchor16_adj", "full4096_adj", "bandpower_adj"]


def fold_anchors(u: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    mean = u[train_idx].mean(axis=0)                                  # S x D
    return mean / np.linalg.norm(mean, axis=-1, keepdims=True)


def covariates(table: dict, train_idx: np.ndarray) -> np.ndarray:
    z = np.column_stack([table["age"], table["sex"]])
    fill = np.nanmean(z[train_idx], axis=0)
    fill = np.where(np.isnan(fill), 0.0, fill)
    return np.where(np.isnan(z), fill[None, :], z)


def residualise(x: np.ndarray, z: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(z)), z])
    beta, *_ = np.linalg.lstsq(design[train_idx], x[train_idx], rcond=None)
    return x - design @ beta


def base_features(table: dict, kind: str, train_idx: np.ndarray, directions: np.ndarray | None = None) -> np.ndarray:
    u = table["u"]
    if kind in ("anchor16", "tangent4096", "single"):
        p = fold_anchors(u, train_idx)
        coords = np.einsum("nsd,sd->ns", u, p)
        if kind == "tangent4096":
            return (u - coords[..., None] * p[None]).reshape(len(u), -1)
        return coords
    if kind == "quant_anchor16":
        return np.einsum("nsd,sd->ns", table["uq"], fold_anchors(table["uq"], train_idx))
    if kind == "rand16":
        return np.einsum("nsd,sd->ns", u, directions)
    fixed = {
        "full4096": lambda: u.reshape(len(u), -1),
        "pooled256": lambda: u.mean(axis=1),
        "quant4096": lambda: table["uq"].reshape(len(u), -1),
        "rbar16": lambda: table["rbar"],
        "norm16": lambda: table["norm"],
        "bandpower": lambda: table["band"].reshape(len(u), -1),
    }
    return fixed[kind]()


def make_transform(table: dict, kind: str, source: int | None = None, directions: np.ndarray | None = None):
    def transform(train_idx, test_idx):
        if kind == "age_sex":
            feats = covariates(table, train_idx)
        else:
            base = kind[:-4] if kind.endswith("_adj") else kind
            feats = base_features(table, base, train_idx, directions)
            if kind == "single":
                feats = feats[:, [source]]
            if kind.endswith("_adj"):
                feats = residualise(feats, covariates(table, train_idx), train_idx)
        return feats[train_idx], feats[test_idx]

    return transform


def best_single_factory(table: dict):
    def factory(labels: np.ndarray):
        def transform(train_idx, test_idx):
            coords = base_features(table, "anchor16", train_idx)
            inner = np.abs(auc_rows(labels[train_idx], coords[train_idx].T) - 0.5)
            s_best = int(np.argmax(inner))
            return coords[train_idx][:, [s_best]], coords[test_idx][:, [s_best]]
        return transform
    return factory


def paired_differences(labels: np.ndarray, scores: dict[str, np.ndarray], seed: int) -> dict:
    """Paired bootstrap differences of every representation to band power and to anchor-16."""
    out = {}
    for name in PAIRED:
        entry = {}
        if name != "bandpower":
            entry["vs_bandpower"] = paired_auc_difference(labels, scores[name], scores["bandpower"], n_boot=2000, seed=seed)
        if name != "anchor16":
            entry["vs_anchor16"] = paired_auc_difference(labels, scores[name], scores["anchor16"], n_boot=2000, seed=seed)
        out[name] = entry
    return out


def paired_only(table: dict, labels: np.ndarray, config: ReadoutConfig, out_dir, seed: int) -> None:
    stored = {e["feature"]: e["auc"] for e in json.loads((out_dir / "readout.json").read_text())["results"]}
    scores, deviation = {}, 0.0
    for name in PAIRED:
        result = repeated_cv_readout(None, labels, config, make_transform(table, name))
        deviation = max(deviation, abs(result.auc_mean - stored[name]))
        scores[name] = result.scores_per_repeat
        print(f"  {name:14s} AUC {result.auc_mean:.4f} (readout.json {stored[name]:.4f})", flush=True)
    if deviation > 1e-3:                         # solver tolerances: high-dimensional fits reproduce to ~1e-4
        raise RuntimeError(f"recomputed AUCs differ from readout.json by up to {deviation:.2e}")
    paired = {"reproduced_auc_max_abs_deviation": deviation, "n_boot": 2000, "differences": paired_differences(labels, scores, seed)}
    (out_dir / "paired.json").write_text(json.dumps(paired, indent=2))
    for name, entry in paired["differences"].items():
        d = entry.get("vs_bandpower")
        if d:
            print(f"  {name:14s} - bandpower {d['difference']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] p={d['p']:.3f}", flush=True)


def evaluate(name: str, family: str, labels: np.ndarray, config: ReadoutConfig, factory, n_perm: int, perm_repeats: int,
             seed: int, n_jobs: int) -> dict:
    result = repeated_cv_readout(None, labels, config, factory(labels))
    lo, hi = bootstrap_repeat_auc_ci(labels, result.scores_per_repeat, n_boot=2000, seed=seed)
    entry = {"feature": name, "family": family, "auc": result.auc_mean, "auc_ci": [lo, hi], "auc_sd_repeats": result.auc_sd,
             "balanced_accuracy": result.balanced_accuracy_mean, "_scores": result.scores_per_repeat}
    if n_perm:
        perm_cfg = ReadoutConfig(**{**config.__dict__, "n_repeats": perm_repeats, "nested": False, "C": 1.0})
        observed = repeated_cv_readout(None, labels, perm_cfg, factory(labels)).auc_mean
        p_value, null = permutation_pvalue(None, labels, observed, n_perm, perm_cfg, n_repeats=perm_repeats, seed=seed,
                                           fold_transform_factory=factory, n_jobs=n_jobs)
        entry.update(perm_p=p_value, perm_statistic=observed, perm_null_mean=float(null.mean()), n_perm=n_perm)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--contrast", required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--n-perm-secondary", type=int, default=500)
    parser.add_argument("--perm-repeats", type=int, default=2)
    parser.add_argument("--n-random", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--paired-only", action="store_true", help="only recompute the CV point estimates and write paired.json")
    args = parser.parse_args()

    table = subject_table(args.cohort, args.contrast)
    labels = table["labels"]
    n_sources = table["u"].shape[1]
    config = ReadoutConfig(n_repeats=args.repeats, seed=args.seed)
    single_cfg = ReadoutConfig(**{**config.__dict__, "nested": False})
    print(f"[readout] {args.cohort}/{args.contrast}: {len(labels)} subjects ({int(labels.sum())} positive)", flush=True)
    out_dir = RESULTS_ROOT / "diagnostic_readout" / f"{args.cohort}_{args.contrast}"
    if args.paired_only:
        paired_only(table, labels, config, out_dir, args.seed)
        return

    def fixed(kind, **kw):
        transform = make_transform(table, kind, **kw)
        return lambda _labels: transform

    common = dict(perm_repeats=args.perm_repeats, seed=args.seed, n_jobs=args.n_jobs)
    results = [evaluate("anchor16", "primary", labels, config, fixed("anchor16"), args.n_perm, **common)]
    print(f"  anchor16 AUC {results[0]['auc']:.3f} {results[0]['auc_ci']} p={results[0]['perm_p']:.4f}", flush=True)
    for name in SECONDARY + ADJUSTED + ["age_sex"]:
        family = "secondary" if name in SECONDARY else "covariates"
        results.append(evaluate(name, family, labels, config, fixed(name), args.n_perm_secondary, **common))
        e = results[-1]
        print(f"  {name:14s} AUC {e['auc']:.3f} [{e['auc_ci'][0]:.3f},{e['auc_ci'][1]:.3f}] p={e['perm_p']:.4f}", flush=True)
    secondary = [e for e in results if e["family"] == "secondary"]
    for e, q in zip(secondary, bh_adjust(np.array([e["perm_p"] for e in secondary]))):
        e["perm_p_fdr"] = float(q)

    # single sources: CV AUC of every anchor coordinate, family-wise p-values by the maximum statistic
    def single_aucs(y: np.ndarray, repeats: int, cfg: ReadoutConfig) -> np.ndarray:
        c = ReadoutConfig(**{**cfg.__dict__, "n_repeats": repeats})
        return np.array([repeated_cv_readout(None, y, c, make_transform(table, "single", source=s)).auc_mean for s in range(n_sources)])

    perm_single_cfg = ReadoutConfig(**{**single_cfg.__dict__, "C": 1.0})
    observed_perm, adjusted, null_max = max_statistic_pvalues(lambda y: single_aucs(y, args.perm_repeats, perm_single_cfg),
                                                              labels, args.n_perm, seed=args.seed, n_jobs=args.n_jobs)
    for s in range(n_sources):
        res = repeated_cv_readout(None, labels, single_cfg, make_transform(table, "single", source=s))
        lo, hi = bootstrap_repeat_auc_ci(labels, res.scores_per_repeat, n_boot=1000, seed=args.seed)
        results.append({"feature": f"single_{s}", "family": "single", "auc": res.auc_mean, "auc_ci": [lo, hi],
                        "auc_sd_repeats": res.auc_sd, "balanced_accuracy": res.balanced_accuracy_mean,
                        "perm_statistic": float(observed_perm[s]), "perm_p_maxstat": float(adjusted[s]), "n_perm": args.n_perm,
                        "_scores": res.scores_per_repeat})
    results.append(evaluate("best_single", "single", labels, single_cfg, best_single_factory(table), args.n_perm, **common))
    print(f"  singles: max AUC {max(e['auc'] for e in results if e['feature'].startswith('single_')):.3f}, "
          f"min max-stat p {adjusted.min():.4f}; best_single AUC {results[-1]['auc']:.3f} p={results[-1]['perm_p']:.4f}", flush=True)

    # random-direction null, evaluated exactly like anchor16
    rng = np.random.default_rng(args.seed + 1)
    draws = []
    for _ in range(args.n_random):
        d = rng.normal(size=table["u"].shape[1:])
        draws.append(d / np.linalg.norm(d, axis=-1, keepdims=True))
    rand = Parallel(n_jobs=args.n_jobs)(
        delayed(lambda d: repeated_cv_readout(None, labels, config, make_transform(table, "rand16", directions=d)).auc_mean)(d)
        for d in draws)
    rand = np.asarray(rand)
    anchor = results[0]

    reference = {e["feature"]: e["_scores"] for e in results}
    for e in results:
        if e["feature"] != "anchor16" and not e["feature"].startswith("single_"):
            e["delta_vs_anchor16"] = e["auc"] - anchor["auc"]
            e["delta_vs_anchor16_ci"] = list(bootstrap_repeat_auc_difference_ci(labels, reference[e["feature"]], reference["anchor16"],
                                                                                n_boot=2000, seed=args.seed))
    out_dir.mkdir(parents=True, exist_ok=True)
    paired = {"reproduced_auc_max_abs_deviation": 0.0, "n_boot": 2000,
              "differences": paired_differences(labels, {name: reference[name] for name in PAIRED}, args.seed)}
    (out_dir / "paired.json").write_text(json.dumps(paired, indent=2))
    pd.DataFrame({"subject": table["subjects"], "label": labels,
                  **{k: v.mean(axis=0) for k, v in reference.items()}}).to_csv(out_dir / "oof_scores.csv", index=False)
    age, sex = table["age"], table["sex"]
    summary = {
        "cohort": args.cohort, "contrast": args.contrast, "n_subjects": int(len(labels)), "n_positive": int(labels.sum()),
        "segments_per_subject_median": float(np.median(table["segments"])),
        "repeats": args.repeats, "n_perm": args.n_perm, "n_perm_secondary": args.n_perm_secondary, "perm_repeats": args.perm_repeats,
        "demographics": {
            "age_mean": {g: float(np.nanmean(age[labels == g])) for g in (0, 1)},
            "male_fraction": {g: float(np.nanmean(sex[labels == g])) for g in (0, 1)},
            "missing_age": int(np.isnan(age).sum()), "missing_sex": int(np.isnan(sex).sum()),
        },
        "results": [{k: v for k, v in e.items() if k != "_scores"} for e in results],
        "single_maxstat_null_quantiles": [float(q) for q in np.quantile(null_max, [0.5, 0.95])],
        "random16_null": {
            "n": int(len(rand)), "mean": float(rand.mean()), "q025": float(np.quantile(rand, 0.025)),
            "q975": float(np.quantile(rand, 0.975)), "fraction_ge_anchor16": float(np.mean(rand >= anchor["auc"])),
        },
    }
    (out_dir / "readout.json").write_text(json.dumps(summary, indent=2))
    print(f"  random16 null mean {rand.mean():.3f} [{summary['random16_null']['q025']:.3f}, {summary['random16_null']['q975']:.3f}]; "
          f"fraction >= anchor16 {summary['random16_null']['fraction_ge_anchor16']:.2f}", flush=True)


if __name__ == "__main__":
    main()
