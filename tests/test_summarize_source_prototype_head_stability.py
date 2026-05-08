from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.summarize_source_prototype_head_stability import (
    bootstrap_dominant_head_frequencies,
    compute_head_profile,
)


class SummarizeSourcePrototypeHeadStabilityTests(unittest.TestCase):
    def test_compute_head_profile_reports_dominant_head_and_margin(self) -> None:
        head_summary = pd.DataFrame(
            [
                {"head_idx": 0, "mean_abs_sensor_delta": 0.1},
                {"head_idx": 1, "mean_abs_sensor_delta": 0.4},
                {"head_idx": 2, "mean_abs_sensor_delta": 0.2},
            ]
        )

        profile = compute_head_profile(head_summary)

        self.assertEqual(profile["dominant_head_idx"], 1)
        self.assertEqual(profile["second_head_idx"], 2)
        self.assertAlmostEqual(profile["dominant_head_margin_abs"], 0.2)

    def test_bootstrap_dominant_head_is_deterministic_when_one_head_always_largest(self) -> None:
        head_matrix = np.array(
            [
                [0.1, 0.3, 0.2],
                [0.1, 0.4, 0.2],
                [0.1, 0.5, 0.2],
            ]
        )
        bootstrap = bootstrap_dominant_head_frequencies(
            head_matrix=head_matrix,
            head_idx=np.array([0, 1, 2]),
            num_bootstrap=200,
            seed=0,
        )

        winning_fraction = bootstrap.set_index("head_idx").loc[1, "bootstrap_win_fraction"]
        self.assertAlmostEqual(winning_fraction, 1.0)


if __name__ == "__main__":
    unittest.main()
