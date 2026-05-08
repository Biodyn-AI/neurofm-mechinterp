"""Tests for processed segment selection helpers."""

from __future__ import annotations

import unittest

from neurofm_mechinterp.processed_data import choose_examples


class ChooseExamplesTests(unittest.TestCase):
    def test_round_robins_across_subjects(self) -> None:
        metadata = [
            {"subject": "a", "channels": 32, "path": "a0"},
            {"subject": "a", "channels": 32, "path": "a1"},
            {"subject": "a", "channels": 32, "path": "a2"},
            {"subject": "b", "channels": 32, "path": "b0"},
            {"subject": "b", "channels": 32, "path": "b1"},
            {"subject": "c", "channels": 32, "path": "c0"},
        ]

        selected = choose_examples(metadata, max_segments=5)

        self.assertEqual([row["path"] for row in selected], ["a0", "b0", "c0", "a1", "b1"])

    def test_filters_by_channel_count_before_round_robin(self) -> None:
        metadata = [
            {"subject": "a", "channels": 32, "path": "a0"},
            {"subject": "b", "channels": 16, "path": "b0"},
            {"subject": "c", "channels": 32, "path": "c0"},
        ]

        selected = choose_examples(metadata, max_segments=3)

        self.assertEqual([row["path"] for row in selected], ["a0", "c0"])

    def test_falls_back_to_prefix_when_subject_missing(self) -> None:
        metadata = [
            {"channels": 32, "path": "x0"},
            {"channels": 32, "path": "x1"},
            {"channels": 32, "path": "x2"},
        ]

        selected = choose_examples(metadata, max_segments=2)

        self.assertEqual([row["path"] for row in selected], ["x0", "x1"])


if __name__ == "__main__":
    unittest.main()
