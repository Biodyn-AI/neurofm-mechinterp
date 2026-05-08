from __future__ import annotations

import unittest

import pandas as pd

from scripts.compare_sae_feature_routing import choose_source_idx, jaccard, overlap_count, parse_sensor_set


class CompareSaeFeatureRoutingTests(unittest.TestCase):
    def test_parse_sensor_set_handles_empty_values(self) -> None:
        self.assertEqual(parse_sensor_set(""), ())

    def test_overlap_and_jaccard_use_set_semantics(self) -> None:
        left = (1, 2, 3, 3)
        right = (2, 3, 4)
        self.assertEqual(overlap_count(left, right), 2)
        self.assertAlmostEqual(jaccard(left, right), 0.5)

    def test_choose_source_idx_reads_unique_row_source_idx(self) -> None:
        feature_summary = pd.DataFrame({"row_source_idx": [3, 3, 3]})
        self.assertEqual(choose_source_idx(feature_summary, explicit_source_idx=None), 3)


if __name__ == "__main__":
    unittest.main()
