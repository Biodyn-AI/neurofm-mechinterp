from __future__ import annotations

import unittest

import pandas as pd

from scripts.run_sae_source_prototype_head_routing import build_condition_summary


class RunSaeSourcePrototypeHeadRoutingTests(unittest.TestCase):
    def test_build_condition_summary_picks_head_with_largest_abs_delta(self) -> None:
        head_summary = pd.DataFrame(
            [
                {"head_idx": 0, "mean_abs_sensor_delta": 0.1, "mean_delta_mean_contribution": -0.01},
                {"head_idx": 3, "mean_abs_sensor_delta": 0.3, "mean_delta_mean_contribution": -0.02},
                {"head_idx": 1, "mean_abs_sensor_delta": 0.2, "mean_delta_mean_contribution": 0.03},
            ]
        )

        condition_summary = build_condition_summary(head_summary)

        self.assertEqual(condition_summary.iloc[0]["dominant_head_idx"], 3)


if __name__ == "__main__":
    unittest.main()
