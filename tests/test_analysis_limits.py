"""Unit tests for analysis-time source and block selection."""

from __future__ import annotations

import unittest

import torch

from neurofm_mechinterp.analysis.source_ablation import _select_source_indices
from neurofm_mechinterp.analysis.spatial_temporal import _select_block_indices


class SourceSelectionTests(unittest.TestCase):
    def test_source_limit_uses_top_energy_sources(self) -> None:
        feature = torch.zeros(1, 4, 1, 1, 1)
        feature[:, 0] = 1.0
        feature[:, 1] = 3.0
        feature[:, 2] = 2.0
        feature[:, 3] = 4.0

        selected = _select_source_indices(feature, max_sources=2)

        self.assertEqual(selected, [3, 1])

    def test_explicit_source_indices_are_deduplicated_and_preserved(self) -> None:
        feature = torch.zeros(1, 4, 1, 1, 1)

        selected = _select_source_indices(feature, source_indices=[2, 1, 2])

        self.assertEqual(selected, [2, 1])

    def test_invalid_source_index_raises(self) -> None:
        feature = torch.zeros(1, 4, 1, 1, 1)

        with self.assertRaisesRegex(ValueError, "out of range"):
            _select_source_indices(feature, source_indices=[4])


class BlockSelectionTests(unittest.TestCase):
    def test_block_limit_returns_first_blocks(self) -> None:
        selected = _select_block_indices(num_blocks=6, max_blocks=3)

        self.assertEqual(selected, [0, 1, 2])

    def test_explicit_block_indices_are_deduplicated_and_preserved(self) -> None:
        selected = _select_block_indices(num_blocks=6, block_indices=[4, 1, 4])

        self.assertEqual(selected, [4, 1])

    def test_invalid_block_index_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "out of range"):
            _select_block_indices(num_blocks=6, block_indices=[6])


if __name__ == "__main__":
    unittest.main()
