from __future__ import annotations

import unittest

import torch

from scripts.compare_sae_source_structure import mean_canonical_correlation, resolve_activation_path, source_prototypes


class CompareSaeSourceStructureTests(unittest.TestCase):
    def test_source_prototypes_aggregate_decoder_directions(self) -> None:
        source_mean = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        decoder = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        prototypes = source_prototypes(source_mean, decoder)

        self.assertTrue(torch.allclose(prototypes, torch.tensor([[1.0, 0.0], [0.0, 1.0]])))

    def test_mean_canonical_correlation_is_one_for_same_basis(self) -> None:
        basis = torch.eye(2)
        mean_cc, max_cc = mean_canonical_correlation(basis, basis)

        self.assertAlmostEqual(mean_cc, 1.0)
        self.assertAlmostEqual(max_cc, 1.0)

    def test_resolve_activation_path_falls_back_for_older_runs(self) -> None:
        checkpoint = {"config": {"activation_path": None, "dataset": "pd31", "hook": "encoder_pre_quantizer"}}
        path = resolve_activation_path(checkpoint)

        self.assertTrue(str(path).endswith("results/sae_exports/pd31/encoder_pre_quantizer/activations.pt"))


if __name__ == "__main__":
    unittest.main()
