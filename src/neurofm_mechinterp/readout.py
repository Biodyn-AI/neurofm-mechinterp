"""Subject-level diagnostic read-out: repeated stratified CV, permutation tests, bootstrap CIs.

The unit of analysis is the subject. Features are averaged over a subject's
segments before classification, so no subject ever contributes to both the
training and the test fold, and AUCs are computed over subjects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata


@dataclass
class ReadoutConfig:
    n_splits: int = 5
    n_repeats: int = 20
    nested: bool = True
    C: float = 1.0
    Cs: tuple[float, ...] = tuple(np.logspace(-4, 2, 13))
    inner_splits: int = 5
    seed: int = 0


@dataclass
class ReadoutResult:
    auc_per_repeat: np.ndarray
    balanced_accuracy_per_repeat: np.ndarray
    mean_scores: np.ndarray
    auc_mean: float
    auc_sd: float
    balanced_accuracy_mean: float
    scores_per_repeat: np.ndarray | None = None
    extra: dict = field(default_factory=dict)


def make_classifier(n_features: int, n_train: int, config: ReadoutConfig):
    """L2 logistic regression on standardised features, C chosen by inner CV.

    When there are more features than training subjects, the standardised data are
    first rotated onto their (training) principal axes keeping all ``n_train - 1``
    non-degenerate components. With an L2 penalty this rotation is lossless (the
    solution lies in the span of the training data) and only speeds up fitting.
    """
    steps = [StandardScaler()]
    if n_features >= n_train:
        steps.append(PCA(n_components=n_train - 1, svd_solver="full"))
    if config.nested:
        inner = StratifiedKFold(n_splits=config.inner_splits, shuffle=True, random_state=config.seed)
        steps.append(LogisticRegressionCV(Cs=list(config.Cs), cv=inner, scoring="roc_auc", max_iter=5000))
    else:
        steps.append(LogisticRegression(C=config.C, max_iter=5000))
    return make_pipeline(*steps)


def _decision_scores(model, x: np.ndarray) -> np.ndarray:
    return model.decision_function(x)


def cross_validated_scores(
    features: np.ndarray,
    labels: np.ndarray,
    config: ReadoutConfig,
    repeat: int,
    fold_transform=None,
) -> np.ndarray:
    """Out-of-fold decision scores for one repeat of stratified K-fold CV.

    ``fold_transform(train_idx, test_idx) -> (x_train, x_test)`` lets callers build
    features inside the fold (e.g. anchors estimated on training subjects only).
    """
    splitter = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed + repeat)
    scores = np.zeros(len(labels), dtype=float)
    for train_idx, test_idx in splitter.split(np.zeros(len(labels)), labels):
        if fold_transform is None:
            x_train, x_test = features[train_idx], features[test_idx]
        else:
            x_train, x_test = fold_transform(train_idx, test_idx)
        model = make_classifier(x_train.shape[1], len(train_idx), config)
        model.fit(x_train, labels[train_idx])
        scores[test_idx] = _decision_scores(model, x_test)
    return scores


def repeated_cv_readout(
    features: np.ndarray | None,
    labels: np.ndarray,
    config: ReadoutConfig | None = None,
    fold_transform=None,
) -> ReadoutResult:
    """Repeated stratified K-fold read-out with AUC and balanced accuracy per repeat."""
    config = config or ReadoutConfig()
    labels = np.asarray(labels).astype(int)
    aucs, baccs, all_scores = [], [], []
    for repeat in range(config.n_repeats):
        scores = cross_validated_scores(features, labels, config, repeat, fold_transform)
        aucs.append(roc_auc_score(labels, scores))
        baccs.append(balanced_accuracy_score(labels, (scores > 0).astype(int)))
        all_scores.append(scores)
    aucs = np.asarray(aucs)
    baccs = np.asarray(baccs)
    return ReadoutResult(
        auc_per_repeat=aucs,
        balanced_accuracy_per_repeat=baccs,
        mean_scores=np.mean(np.stack(all_scores), axis=0),
        auc_mean=float(aucs.mean()),
        auc_sd=float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
        balanced_accuracy_mean=float(baccs.mean()),
        scores_per_repeat=np.stack(all_scores),
    )


def auc_rows(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """AUC of every row of ``scores`` (repeats x subjects) via the Mann-Whitney rank sum (ties count 1/2)."""
    labels = np.asarray(labels).astype(int)
    scores = np.atleast_2d(scores)
    ranks = rankdata(scores, axis=1)
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    return (ranks[:, pos].sum(axis=1) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _stratified_resamples(labels: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    return np.stack([np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(n_boot)])


def bootstrap_repeat_auc_ci(
    labels: np.ndarray,
    scores_per_repeat: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI of the repeat-averaged AUC (the statistic reported as the point estimate).

    Subjects are resampled within class; on every resample the AUC of each CV repeat's
    out-of-fold scores is recomputed and averaged over repeats.
    """
    labels = np.asarray(labels).astype(int)
    stats = [auc_rows(labels[idx], scores_per_repeat[:, idx]).mean() for idx in _stratified_resamples(labels, n_boot, seed)]
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def bootstrap_repeat_auc_difference_ci(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Paired percentile CI of the difference between two repeat-averaged AUCs on the same subjects."""
    labels = np.asarray(labels).astype(int)
    stats = [auc_rows(labels[idx], scores_a[:, idx]).mean() - auc_rows(labels[idx], scores_b[:, idx]).mean()
             for idx in _stratified_resamples(labels, n_boot, seed)]
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_auc_difference(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Difference between two AUCs on the same subjects, with a paired stratified bootstrap.

    Scores are ``(n,)`` (one fitted classifier) or ``(repeats, n)`` (out-of-fold scores of repeated CV, in
    which case each AUC is the mean over repeats, as reported). Returns the difference AUC(a) - AUC(b),
    its percentile interval and a two-sided bootstrap p-value (twice the smaller tail of the bootstrap
    distribution at zero), with out-of-fold scores held fixed.
    """
    labels = np.asarray(labels).astype(int)
    a = np.atleast_2d(np.asarray(scores_a, dtype=float))
    b = np.atleast_2d(np.asarray(scores_b, dtype=float))
    observed = float(auc_rows(labels, a).mean() - auc_rows(labels, b).mean())
    stats = np.array([auc_rows(labels[idx], a[:, idx]).mean() - auc_rows(labels[idx], b[:, idx]).mean()
                      for idx in _stratified_resamples(labels, n_boot, seed)])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    p_value = min(1.0, 2.0 * min(np.mean(stats <= 0.0), np.mean(stats >= 0.0)))
    return {"difference": observed, "ci": [float(lo), float(hi)], "p": float(max(p_value, 1.0 / n_boot))}


def test_label_permutation_pvalue(
    labels: np.ndarray,
    scores: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 0,
) -> tuple[float, np.ndarray]:
    """One-sided p-value of the AUC of fixed scores against permutations of the labels they are tested on.

    Exact under the null hypothesis that the scores (e.g. of a classifier trained on another cohort) are
    unrelated to these labels; returns the p-value and the null distribution of the AUC.
    """
    labels = np.asarray(labels).astype(int)
    rng = np.random.default_rng(seed)
    ranks = rankdata(np.asarray(scores, dtype=float))
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    observed = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    permuted = np.stack([rng.permutation(labels) for _ in range(n_permutations)])
    null = ((permuted * ranks[None, :]).sum(axis=1) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float((1.0 + np.sum(null >= observed - 1e-12)) / (1.0 + n_permutations)), null


def bootstrap_auc_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the AUC, resampling subjects within each class."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).astype(int)
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        stats.append(roc_auc_score(labels[idx], scores[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def permutation_pvalue(
    features: np.ndarray | None,
    labels: np.ndarray,
    observed_auc: float,
    n_permutations: int,
    config: ReadoutConfig | None = None,
    n_repeats: int = 5,
    seed: int = 0,
    fold_transform_factory=None,
    n_jobs: int = 1,
) -> tuple[float, np.ndarray]:
    """Subject-level label-permutation p-value for the repeat-averaged CV AUC.

    The whole CV procedure is re-run on every permutation (with ``n_repeats``
    repeats), and the observed statistic must be computed with the same number of
    repeats for the test to be exact.
    """
    config = config or ReadoutConfig()
    perm_config = ReadoutConfig(**{**config.__dict__, "n_repeats": n_repeats})
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).astype(int)
    permutations = [rng.permutation(labels) for _ in range(n_permutations)]

    def one(permuted: np.ndarray) -> float:
        transform = fold_transform_factory(permuted) if fold_transform_factory is not None else None
        return repeated_cv_readout(features, permuted, perm_config, transform).auc_mean

    null = np.asarray(Parallel(n_jobs=n_jobs)(delayed(one)(perm) for perm in permutations))
    p_value = (1.0 + np.sum(null >= observed_auc)) / (1.0 + n_permutations)
    return float(p_value), null


def max_statistic_pvalues(
    statistic,
    labels: np.ndarray,
    n_permutations: int,
    seed: int = 0,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Family-wise permutation p-values from the maximum statistic (Westfall-Young single step).

    ``statistic(labels) -> array`` returns one statistic per hypothesis (e.g. one CV AUC per source).
    Returns (observed, adjusted p-values, null distribution of the maximum).
    """
    labels = np.asarray(labels).astype(int)
    observed = np.asarray(statistic(labels), dtype=float)
    rng = np.random.default_rng(seed)
    permutations = [rng.permutation(labels) for _ in range(n_permutations)]
    null_max = np.asarray(Parallel(n_jobs=n_jobs)(delayed(lambda y: float(np.max(statistic(y))))(y) for y in permutations))
    adjusted = (1.0 + (null_max[None, :] >= observed[:, None]).sum(axis=1)) / (1.0 + n_permutations)
    return observed, adjusted, null_max


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (false discovery rate), monotone and capped at 1."""
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def cluster_bootstrap_mean(
    values: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Mean of per-cluster means with a percentile CI from resampling clusters.

    Returns (estimate, low, high); the estimate weights every cluster (subject) equally.
    """
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    per_cluster = np.array([values[clusters == c].mean() for c in uniq])
    rng = np.random.default_rng(seed)
    boots = per_cluster[rng.integers(0, len(per_cluster), size=(n_boot, len(per_cluster)))].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(per_cluster.mean()), float(lo), float(hi)
