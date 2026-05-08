from __future__ import annotations

import unittest

from scripts.summarize_sae_interventions import relation_label


class SummarizeSaeInterventionsTests(unittest.TestCase):
    def test_relation_label_marks_matching_values_as_within(self) -> None:
        self.assertEqual(relation_label("sub-1", "sub-1", "within_subject", "cross_subject"), "within_subject")

    def test_relation_label_marks_different_values_as_cross(self) -> None:
        self.assertEqual(relation_label("sub-1", "sub-2", "within_subject", "cross_subject"), "cross_subject")


if __name__ == "__main__":
    unittest.main()
