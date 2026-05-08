from __future__ import annotations

import unittest

import pandas as pd

from scripts.run_sae_feature_patching import choose_patch_pairs


class RunSaeFeaturePatchingTests(unittest.TestCase):
    def test_choose_patch_pairs_matches_high_to_low_examples(self) -> None:
        score_table = pd.DataFrame(
            [
                {"feature_idx": 57, "path": "a", "conditioned_feature_mean": 3.0},
                {"feature_idx": 57, "path": "b", "conditioned_feature_mean": 2.0},
                {"feature_idx": 57, "path": "c", "conditioned_feature_mean": 0.2},
                {"feature_idx": 57, "path": "d", "conditioned_feature_mean": 0.1},
            ]
        )

        pairs = choose_patch_pairs(score_table, feature_idx=57, num_pairs=2)

        self.assertEqual(pairs, [("a", "d"), ("b", "c")])

    def test_choose_patch_pairs_skips_self_pairs(self) -> None:
        score_table = pd.DataFrame(
            [
                {"feature_idx": 57, "path": "a", "conditioned_feature_mean": 1.0},
                {"feature_idx": 57, "path": "b", "conditioned_feature_mean": 1.0},
            ]
        )

        pairs = choose_patch_pairs(score_table, feature_idx=57, num_pairs=2)

        self.assertEqual(pairs, [("a", "b"), ("b", "a")])


if __name__ == "__main__":
    unittest.main()
