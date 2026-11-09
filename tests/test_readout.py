from __future__ import annotations

import unittest

import numpy as np

from sklearn.metrics import roc_auc_score

from neurofm_mechinterp.readout import (
    ReadoutConfig,
    auc_rows,
    bh_adjust,
    bootstrap_auc_ci,
    bootstrap_repeat_auc_ci,
    bootstrap_repeat_auc_difference_ci,
    cluster_bootstrap_mean,
    holm_adjust,
    max_statistic_pvalues,
    paired_auc_difference,
    permutation_pvalue,
    repeated_cv_readout,
    test_label_permutation_pvalue,
)


class ReadoutTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.labels = np.array([0] * 30 + [1] * 30)
        self.noise = rng.normal(size=(60, 5))
        self.signal = self.noise.copy()
        self.signal[:, 0] += 2.5 * self.labels

    def test_separable_features_score_high(self) -> None:
        result = repeated_cv_readout(self.signal, self.labels, ReadoutConfig(n_repeats=3))
        self.assertGreater(result.auc_mean, 0.85)
        self.assertEqual(result.auc_per_repeat.shape, (3,))

    def test_noise_features_score_near_chance(self) -> None:
        result = repeated_cv_readout(self.noise, self.labels, ReadoutConfig(n_repeats=5))
        self.assertLess(abs(result.auc_mean - 0.5), 0.2)

    def test_high_dimensional_path_runs(self) -> None:
        rng = np.random.default_rng(1)
        wide = rng.normal(size=(60, 400))
        wide[:, :3] += 2.0 * self.labels[:, None]
        result = repeated_cv_readout(wide, self.labels, ReadoutConfig(n_repeats=2))
        # A dense L2 read-out dilutes 3 informative dimensions among 397 noise ones;
        # the test only checks that the high-dimensional path runs and beats chance.
        self.assertGreater(result.auc_mean, 0.65)

    def test_permutation_pvalue_small_for_signal(self) -> None:
        config = ReadoutConfig(n_repeats=2)
        observed = repeated_cv_readout(self.signal, self.labels, config).auc_mean
        p_value, null = permutation_pvalue(self.signal, self.labels, observed, n_permutations=30, config=config, n_repeats=2)
        self.assertLess(p_value, 0.1)
        self.assertEqual(null.shape, (30,))

    def test_fold_transform_is_used(self) -> None:
        calls = []

        def transform(train_idx, test_idx):
            calls.append(len(train_idx))
            return self.signal[train_idx], self.signal[test_idx]

        repeated_cv_readout(None, self.labels, ReadoutConfig(n_repeats=1), fold_transform=transform)
        self.assertEqual(len(calls), 5)

    def test_bootstrap_ci_contains_point_estimate(self) -> None:
        scores = self.signal[:, 0]
        lo, hi = bootstrap_auc_ci(self.labels, scores, n_boot=200)
        self.assertLess(lo, hi)
        self.assertGreater(hi, 0.8)

    def test_auc_rows_matches_sklearn(self) -> None:
        rng = np.random.default_rng(3)
        scores = np.round(rng.normal(size=(4, 60)), 1)          # rounding creates ties
        expected = [roc_auc_score(self.labels, row) for row in scores]
        np.testing.assert_allclose(auc_rows(self.labels, scores), expected)

    def test_repeat_ci_brackets_reported_auc(self) -> None:
        result = repeated_cv_readout(self.signal, self.labels, ReadoutConfig(n_repeats=4, nested=False))
        lo, hi = bootstrap_repeat_auc_ci(self.labels, result.scores_per_repeat, n_boot=300)
        self.assertLessEqual(lo, result.auc_mean)
        self.assertGreaterEqual(hi, result.auc_mean)
        np.testing.assert_allclose(auc_rows(self.labels, result.scores_per_repeat), result.auc_per_repeat)
        d_lo, d_hi = bootstrap_repeat_auc_difference_ci(self.labels, result.scores_per_repeat, result.scores_per_repeat, n_boot=50)
        self.assertEqual((d_lo, d_hi), (0.0, 0.0))

    def test_max_statistic_controls_family(self) -> None:
        features = np.column_stack([self.signal[:, 0], self.noise[:, 1:]])

        def statistic(y):
            return np.array([roc_auc_score(y, features[:, j]) for j in range(features.shape[1])])

        observed, adjusted, null_max = max_statistic_pvalues(statistic, self.labels, 99, seed=1)
        self.assertEqual(null_max.shape, (99,))
        self.assertLess(adjusted[0], 0.05)
        self.assertTrue(np.all(adjusted[1:] > 0.05))
        self.assertTrue(np.all(adjusted >= 1.0 / 100))

    def test_bh_adjust(self) -> None:
        adjusted = bh_adjust(np.array([0.01, 0.04, 0.03, 0.5]))
        np.testing.assert_allclose(adjusted, [0.04, 0.0533333333, 0.0533333333, 0.5])

    def test_holm_adjust(self) -> None:
        adjusted = holm_adjust(np.array([0.01, 0.04, 0.03, 0.5]))
        np.testing.assert_allclose(adjusted, [0.04, 0.09, 0.09, 0.5])

    def test_cluster_bootstrap_weights_clusters_equally(self) -> None:
        values = np.array([1.0, 1.0, 1.0, 1.0, 3.0])
        clusters = np.array([0, 0, 0, 0, 1])
        estimate, lo, hi = cluster_bootstrap_mean(values, clusters, n_boot=200)
        self.assertAlmostEqual(estimate, 2.0)
        self.assertLessEqual(lo, estimate)
        self.assertGreaterEqual(hi, estimate)

    def test_paired_difference_of_identical_scores_is_zero(self) -> None:
        scores = self.signal[:, 0]
        result = paired_auc_difference(self.labels, scores, scores, n_boot=200, seed=0)
        self.assertEqual(result["difference"], 0.0)
        self.assertEqual(result["ci"], [0.0, 0.0])
        self.assertEqual(result["p"], 1.0)

    def test_paired_difference_detects_better_scores(self) -> None:
        result = paired_auc_difference(self.labels, self.signal[:, 0], self.noise[:, 1], n_boot=500, seed=0)
        self.assertAlmostEqual(result["difference"], roc_auc_score(self.labels, self.signal[:, 0]) - roc_auc_score(self.labels, self.noise[:, 1]))
        self.assertGreater(result["ci"][0], 0.0)
        self.assertLess(result["p"], 0.01)
        repeated = paired_auc_difference(self.labels, np.stack([self.signal[:, 0]] * 3), np.stack([self.noise[:, 1]] * 3), n_boot=200)
        self.assertAlmostEqual(repeated["difference"], result["difference"])

    def test_test_label_permutation(self) -> None:
        p_signal, null = test_label_permutation_pvalue(self.labels, self.signal[:, 0], n_permutations=2000, seed=0)
        self.assertLess(p_signal, 0.001)
        self.assertEqual(null.shape, (2000,))
        self.assertLess(abs(float(null.mean()) - 0.5), 0.02)
        p_noise, _ = test_label_permutation_pvalue(self.labels, self.noise[:, 1], n_permutations=2000, seed=0)
        self.assertGreater(p_noise, 0.05)


if __name__ == "__main__":
    unittest.main()
