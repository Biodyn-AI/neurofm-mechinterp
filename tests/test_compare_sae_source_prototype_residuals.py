from __future__ import annotations

import unittest

import pandas as pd
import torch

from scripts.compare_sae_source_prototype_residuals import (
    cross_run_nearest_neighbor,
    cross_run_same_path_summary,
    pairwise_segment_similarity,
    top_sensor_tuple,
    within_run_nearest_neighbor,
)


class CompareSaeSourcePrototypeResidualsTests(unittest.TestCase):
    def test_top_sensor_tuple_uses_absolute_magnitude(self) -> None:
        vector = torch.tensor([0.2, -3.0, 1.5, -2.5])

        result = top_sensor_tuple(vector, top_k=3)

        self.assertEqual(result, (1, 3, 2))

    def test_pairwise_segment_similarity_marks_same_subject(self) -> None:
        table = pd.DataFrame(
            [
                {"path": "a", "subject": "s1", "label": 0},
                {"path": "b", "subject": "s1", "label": 1},
                {"path": "c", "subject": "s2", "label": 0},
            ]
        )
        vectors = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])

        result = pairwise_segment_similarity("left", table, vectors)

        self.assertEqual(len(result), 3)
        self.assertEqual(result.iloc[0]["same_subject"], True)
        self.assertEqual(result.iloc[0]["same_label"], False)

    def test_within_run_nearest_neighbor_finds_same_subject(self) -> None:
        table = pd.DataFrame(
            [
                {"path": "a", "subject": "s1", "label": 0},
                {"path": "b", "subject": "s1", "label": 1},
                {"path": "c", "subject": "s2", "label": 0},
            ]
        )
        vectors = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])

        result = within_run_nearest_neighbor("left", table, vectors)

        self.assertEqual(result.iloc[0]["left_nearest_path"], "b")
        self.assertEqual(result.iloc[0]["same_subject"], True)

    def test_cross_run_same_path_summary_compares_matching_paths(self) -> None:
        left_table = pd.DataFrame(
            [
                {"path": "a", "subject": "s1", "label": 0, "top_residual_sensors": "0,1,2"},
                {"path": "b", "subject": "s2", "label": 1, "top_residual_sensors": "3,4,5"},
            ]
        )
        right_table = pd.DataFrame(
            [
                {"path": "b", "subject": "s2", "label": 1, "top_residual_sensors": "3,5,6"},
                {"path": "a", "subject": "s1", "label": 0, "top_residual_sensors": "0,2,7"},
            ]
        )
        left_vectors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        right_vectors = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

        result = cross_run_same_path_summary("left", "right", left_table, left_vectors, right_table, right_vectors)

        self.assertEqual(result.iloc[0]["left_path"], "a")
        self.assertEqual(result.iloc[0]["top_sensor_overlap_count"], 2)
        self.assertAlmostEqual(result.iloc[0]["top_sensor_jaccard"], 0.5)

    def test_cross_run_nearest_neighbor_marks_same_path(self) -> None:
        left_table = pd.DataFrame(
            [
                {"path": "a", "subject": "s1", "label": 0},
                {"path": "b", "subject": "s2", "label": 1},
            ]
        )
        right_table = pd.DataFrame(
            [
                {"path": "b", "subject": "s2", "label": 1},
                {"path": "a", "subject": "s1", "label": 0},
            ]
        )
        left_vectors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        right_vectors = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

        result = cross_run_nearest_neighbor("left", "right", left_table, left_vectors, right_table, right_vectors)

        self.assertEqual(result.iloc[0]["right_path"], "a")
        self.assertEqual(result.iloc[0]["same_path"], True)


if __name__ == "__main__":
    unittest.main()
