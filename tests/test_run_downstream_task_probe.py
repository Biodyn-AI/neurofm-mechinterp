from __future__ import annotations

import unittest

import torch

from scripts.run_downstream_task_probe import (
    keep_only_projection,
    random_unit_direction,
    remove_projection,
)


class RunDownstreamTaskProbeTests(unittest.TestCase):
    def test_remove_projection_drops_only_the_prototype_component(self) -> None:
        rows = torch.tensor([[3.0, 4.0], [1.5, -2.0]])
        prototype = torch.tensor([1.0, 0.0])
        residual = remove_projection(rows, prototype)
        self.assertTrue(torch.allclose(residual, torch.tensor([[0.0, 4.0], [0.0, -2.0]])))

    def test_keep_only_projection_keeps_just_the_prototype_component(self) -> None:
        rows = torch.tensor([[3.0, 4.0], [1.5, -2.0]])
        prototype = torch.tensor([1.0, 0.0])
        keep = keep_only_projection(rows, prototype)
        self.assertTrue(torch.allclose(keep, torch.tensor([[3.0, 0.0], [1.5, 0.0]])))

    def test_remove_and_keep_only_partition_the_input(self) -> None:
        rows = torch.tensor([[2.0, -1.0, 3.0], [0.5, 1.5, -0.5]])
        prototype = torch.tensor([0.6, 0.8, 0.0])
        residual = remove_projection(rows, prototype)
        keep = keep_only_projection(rows, prototype)
        self.assertTrue(torch.allclose(residual + keep, rows, atol=1e-6))

    def test_random_unit_direction_is_unit_norm_and_seed_repeatable(self) -> None:
        v1 = random_unit_direction(dim=16, seed=7)
        v2 = random_unit_direction(dim=16, seed=7)
        v3 = random_unit_direction(dim=16, seed=8)
        self.assertAlmostEqual(float(v1.norm().item()), 1.0, places=5)
        self.assertTrue(torch.allclose(v1, v2))
        self.assertFalse(torch.allclose(v1, v3))


if __name__ == "__main__":
    unittest.main()
