from __future__ import annotations

import unittest

import pandas as pd

from scripts.compare_sae_source_prototype_routing import choose_source_idx, prototype_top_sensor_rows


class CompareSaeSourcePrototypeRoutingTests(unittest.TestCase):
    def test_choose_source_idx_reads_unique_source(self) -> None:
        sensor_summary = pd.DataFrame({"source_idx": [2, 2, 2]})
        self.assertEqual(choose_source_idx(sensor_summary, explicit_source_idx=None), 2)

    def test_prototype_top_sensor_rows_groups_by_mode_and_scale(self) -> None:
        sensor_summary = pd.DataFrame(
            [
                {"source_idx": 3, "mode": "remove_projection", "scale": 1.0, "sensor_idx": 29, "count": 7},
                {"source_idx": 3, "mode": "remove_projection", "scale": 1.0, "sensor_idx": 0, "count": 8},
                {"source_idx": 3, "mode": "remove_projection", "scale": 1.0, "sensor_idx": 1, "count": 3},
                {"source_idx": 3, "mode": "set_mean_projection", "scale": 1.0, "sensor_idx": 4, "count": 4},
                {"source_idx": 3, "mode": "set_mean_projection", "scale": 1.0, "sensor_idx": 2, "count": 5},
            ]
        )

        rows = prototype_top_sensor_rows(sensor_summary, top_k=2)

        self.assertEqual(rows.iloc[0]["prototype_top_sensors"], "0,29")
        self.assertEqual(rows.iloc[1]["prototype_top_sensors"], "2,4")


if __name__ == "__main__":
    unittest.main()
