from __future__ import annotations

import unittest

import torch

from scripts.run_sae_source_prototype_interventions import apply_source_prototype_intervention


class RunSaeSourcePrototypeInterventionsTests(unittest.TestCase):
    def test_remove_projection_zeroes_selected_component(self) -> None:
        rows = torch.tensor([[3.0, 4.0], [1.5, -2.0]])
        prototype = torch.tensor([1.0, 0.0])

        intervened = apply_source_prototype_intervention(
            standardized_rows=rows,
            prototype=prototype,
            mode="remove_projection",
            scale=1.0,
            target_projection=0.0,
        )

        self.assertTrue(torch.allclose(intervened, torch.tensor([[0.0, 4.0], [0.0, -2.0]])))

    def test_set_mean_projection_moves_rows_to_target_projection(self) -> None:
        rows = torch.tensor([[3.0, 4.0], [1.5, -2.0]])
        prototype = torch.tensor([1.0, 0.0])

        intervened = apply_source_prototype_intervention(
            standardized_rows=rows,
            prototype=prototype,
            mode="set_mean_projection",
            scale=1.0,
            target_projection=1.0,
        )

        self.assertTrue(torch.allclose(intervened, torch.tensor([[1.0, 4.0], [1.0, -2.0]])))


if __name__ == "__main__":
    unittest.main()
