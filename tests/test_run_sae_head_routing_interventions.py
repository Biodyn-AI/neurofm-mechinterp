from __future__ import annotations

import unittest

import pandas as pd

from scripts.run_sae_head_routing_interventions import build_feature_summary


class RunSaeHeadRoutingInterventionsTests(unittest.TestCase):
    def test_build_feature_summary_picks_head_with_largest_abs_delta(self) -> None:
        head_summary = pd.DataFrame(
            [
                {"feature_idx": 57, "head_idx": 0, "mean_abs_sensor_delta": 0.1, "mean_delta_mean_contribution": -0.01},
                {"feature_idx": 57, "head_idx": 2, "mean_abs_sensor_delta": 0.3, "mean_delta_mean_contribution": -0.02},
                {"feature_idx": 57, "head_idx": 1, "mean_abs_sensor_delta": 0.2, "mean_delta_mean_contribution": 0.03},
            ]
        )

        feature_summary = build_feature_summary(head_summary)

        self.assertEqual(feature_summary.iloc[0]["dominant_head_idx"], 2)


if __name__ == "__main__":
    unittest.main()
