"""Cross-cohort transfer of Parkinson's-disease read-outs (train on one site, test on the other).

Source representations have the same dimensionality for every montage, so a classifier fitted on
one cohort can be applied to another. Band power is compared on the EEG channels shared by both
montages. Two feature-scaling settings are reported: the training cohort's scaler (no adaptation)
and standardisation within each cohort (label-free site harmonisation).

The classifier is L2 logistic regression with C selected by 5-fold CV on the training cohort; anchors
are estimated on the training cohort. Each representation gives one fitted classifier, whose scores on
the test cohort are evaluated by
  * the AUC with a stratified bootstrap interval over test subjects (scores held fixed);
  * a test-label permutation p-value (exact under the null that the transferred scores are unrelated
    to the test labels);
  * the paired bootstrap difference to anchor-16 on the same test subjects (Holm over representations);
  * the random-direction null, which replaces the anchors by 16 random isotropic unit directions (the
    same directions in both cohorts) and is compared with every representation.
Writes `results/diagnostic_readout/transfer_<train>_to_<test>.json` and the test scores
(`transfer_<train>_to_<test>_scores.csv`).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from neurofm_mechinterp.cohorts import get_cohort
from neurofm_mechinterp.paths import RESULTS_ROOT
from neurofm_mechinterp.readout import bootstrap_auc_ci, holm_adjust, paired_auc_difference, test_label_permutation_pvalue
from neurofm_mechinterp.subject_features import subject_table


def channel_list(cohort_name: str) -> list[str]:
    """Channel names in processed order (written by the preprocessing scripts)."""
    meta = get_cohort(cohort_name).metadata_path.parent
    return json.loads((meta / "channels.json").read_text())


def anchor_coordinates(u_train: np.ndarray, u: np.ndarray) -> np.ndarray:
    mean = u_train.mean(0)
    p = mean / np.linalg.norm(mean, axis=-1, keepdims=True)
    return np.einsum("nsd,sd->ns", u, p)


def representations(train: dict, test: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    mean = train["u"].mean(0)
    p = mean / np.linalg.norm(mean, axis=-1, keepdims=True)
    coords_tr, coords_te = anchor_coordinates(train["u"], train["u"]), anchor_coordinates(train["u"], test["u"])
    flat = lambda x: x.reshape(len(x), -1)  # noqa: E731
    return {
        "anchor16": (coords_tr, coords_te),
        "full4096": (flat(train["u"]), flat(test["u"])),
        "tangent4096": (flat(train["u"] - coords_tr[..., None] * p), flat(test["u"] - coords_te[..., None] * p)),
        "pooled256": (train["u"].mean(1), test["u"].mean(1)),
        "quant4096": (flat(train["uq"]), flat(test["uq"])),
        "quant_anchor16": (anchor_coordinates(train["uq"], train["uq"]), anchor_coordinates(train["uq"], test["uq"])),
        "rbar16": (train["rbar"], test["rbar"]),
        "norm16": (train["norm"], test["norm"]),
        "bandpower_shared": (flat(train["band"]), flat(test["band"])),
    }


def fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, per_cohort: bool, seed: int) -> np.ndarray:
    mu, sd = x_train.mean(0), x_train.std(0) + 1e-12
    xt = (x_train - mu) / sd
    xs = (x_test - x_test.mean(0)) / (x_test.std(0) + 1e-12) if per_cohort else (x_test - mu) / sd
    if xt.shape[1] >= xt.shape[0]:
        _, _, vt = np.linalg.svd(xt - xt.mean(0), full_matrices=False)
        basis = vt[: xt.shape[0] - 1].T
        xt, xs = xt @ basis, xs @ basis
    model = LogisticRegressionCV(Cs=list(np.logspace(-4, 2, 13)), cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                                 scoring="roc_auc", max_iter=5000)
    model.fit(xt, y_train)
    return model.decision_function(xs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", default="pd_iowa")
    parser.add_argument("--test", default="pd_ucsd")
    parser.add_argument("--contrast", default="pd_vs_hc")
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--n-random", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()

    names_train, names_test = channel_list(args.train), channel_list(args.test)
    shared = [c for c in names_train if c in names_test]
    train = subject_table(args.train, args.contrast)
    test = subject_table(args.test, args.contrast)
    train["band"] = train["band"][:, [names_train.index(c) for c in shared]]
    test["band"] = test["band"][:, [names_test.index(c) for c in shared]]
    reps = representations(train, test)
    y_tr, y_te = train["labels"], test["labels"]
    rng = np.random.default_rng(args.seed)
    directions = []
    for _ in range(args.n_random):
        d = rng.normal(size=train["u"].shape[1:])
        directions.append(d / np.linalg.norm(d, axis=-1, keepdims=True))
    out = {"train": args.train, "test": args.test, "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
           "n_positive_train": int(y_tr.sum()), "n_positive_test": int(y_te.sum()), "shared_channels": shared,
           "n_perm": args.n_perm, "p_value": "test-label permutation of the fitted classifier's scores",
           "results": {}, "random16_null": {}}
    scores_table = {"subject": test["subjects"], "label": y_te}
    for setting, per_cohort in (("train_scaler", False), ("per_cohort_standardisation", True)):
        scores = {name: fit_predict(x_tr, y_tr, x_te, per_cohort, args.seed) for name, (x_tr, x_te) in reps.items()}
        rand = np.asarray(Parallel(n_jobs=args.n_jobs)(
            delayed(lambda d: roc_auc_score(y_te, fit_predict(np.einsum("nsd,sd->ns", train["u"], d), y_tr,
                                                              np.einsum("nsd,sd->ns", test["u"], d), per_cohort, args.seed)))(d)
            for d in directions))
        results = {}
        for name, sc in scores.items():
            auc = float(roc_auc_score(y_te, sc))
            lo, hi = bootstrap_auc_ci(y_te, sc, n_boot=2000, seed=args.seed)
            p_value, null = test_label_permutation_pvalue(y_te, sc, n_permutations=args.n_perm, seed=args.seed)
            results[name] = {"auc": auc, "auc_ci": [lo, hi], "p_test_perm": p_value,
                             "test_perm_null_q975": float(np.quantile(null, 0.975)),
                             "random16_fraction_ge": float(np.mean(rand >= auc))}
            if name != "anchor16":
                results[name]["vs_anchor16"] = paired_auc_difference(y_te, scores["anchor16"], sc, n_boot=2000, seed=args.seed)
            scores_table[f"{setting}:{name}"] = sc
        others = [name for name in scores if name != "anchor16"]
        for name, q in zip(others, holm_adjust(np.array([results[n]["vs_anchor16"]["p"] for n in others]))):
            results[name]["vs_anchor16"]["p_holm"] = float(q)
        out["results"][setting] = results
        out["random16_null"][setting] = {"n": int(len(rand)), "mean": float(rand.mean()), "q025": float(np.quantile(rand, 0.025)),
                                         "q975": float(np.quantile(rand, 0.975)),
                                         "fraction_ge_anchor16": results["anchor16"]["random16_fraction_ge"]}
        for name, r in results.items():
            d = r.get("vs_anchor16")
            extra = "" if d is None else f"  anchor16-{name} {d['difference']:+.3f} [{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}] p_holm={d['p_holm']:.3f}"
            print(f"[transfer] {setting:28s} {name:16s} AUC {r['auc']:.3f} [{r['auc_ci'][0]:.3f},{r['auc_ci'][1]:.3f}] "
                  f"p={r['p_test_perm']:.4f} random>= {r['random16_fraction_ge']:.2f}{extra}", flush=True)
        print(f"[transfer] {setting}: random16 null {out['random16_null'][setting]}", flush=True)
    path = RESULTS_ROOT / "diagnostic_readout" / f"transfer_{args.train}_to_{args.test}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    pd.DataFrame(scores_table).to_csv(path.with_name(path.stem + "_scores.csv"), index=False)


if __name__ == "__main__":
    main()
