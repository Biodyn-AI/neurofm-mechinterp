from __future__ import annotations

import unittest

import pandas as pd
import torch

from scripts.compare_sae_feature_reuse import build_match_table


class CompareSaeFeatureReuseTests(unittest.TestCase):
    def test_build_match_table_marks_mutual_best_pairs(self) -> None:
        left_directions = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        right_directions = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        left_features = pd.DataFrame(
            {
                "feature_idx": [0, 1],
                "mean_activation": [1.0, 1.0],
                "active_fraction": [0.5, 0.5],
                "top_source_idx": [3, 4],
                "source_entropy": [1.0, 1.0],
            }
        )
        right_features = pd.DataFrame(
            {
                "feature_idx": [0, 1],
                "mean_activation": [1.0, 1.0],
                "active_fraction": [0.5, 0.5],
                "top_source_idx": [3, 5],
                "source_entropy": [1.0, 1.0],
                "dead_feature": [False, False],
            }
        )
        left_features["dead_feature"] = [False, False]

        match_table = build_match_table(
            left_name="pd31",
            right_name="meg",
            left_directions=left_directions,
            right_directions=right_directions,
            left_features=left_features,
            right_features=right_features,
        )

        self.assertEqual(match_table["mutual_best"].tolist(), [True, True])
        self.assertEqual(match_table["same_top_source"].tolist(), [True, False])
        self.assertEqual(match_table["active_in_both"].tolist(), [True, True])


if __name__ == "__main__":
    unittest.main()
