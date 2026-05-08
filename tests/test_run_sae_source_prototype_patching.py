from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from scripts.run_sae_source_prototype_patching import (
    apply_source_component_patch,
    apply_source_prototype_patch,
    decompose_source_prototype_rows,
    load_explicit_pairs,
)


class RunSaeSourcePrototypePatchingTests(unittest.TestCase):
    def test_decompose_source_prototype_rows_splits_parallel_and_residual(self) -> None:
        rows = torch.tensor([[3.0, 4.0], [1.5, -2.0]])
        prototype = torch.tensor([1.0, 0.0])

        projection, parallel, residual = decompose_source_prototype_rows(rows, prototype)

        self.assertTrue(torch.allclose(projection, torch.tensor([3.0, 1.5])))
        self.assertTrue(torch.allclose(parallel, torch.tensor([[3.0, 0.0], [1.5, 0.0]])))
        self.assertTrue(torch.allclose(residual, torch.tensor([[0.0, 4.0], [0.0, -2.0]])))

    def test_full_patch_sets_projection_to_donor_values(self) -> None:
        rows = torch.tensor([[3.0, 4.0], [1.5, -2.0]])
        donor_projection = torch.tensor([1.0, -1.0])
        prototype = torch.tensor([1.0, 0.0])

        patched = apply_source_prototype_patch(
            standardized_rows=rows,
            donor_projection=donor_projection,
            prototype=prototype,
            scale=1.0,
        )

        self.assertTrue(torch.allclose(patched, torch.tensor([[1.0, 4.0], [-1.0, -2.0]])))

    def test_partial_patch_moves_projection_toward_donor(self) -> None:
        rows = torch.tensor([[3.0, 4.0], [1.5, -2.0]])
        donor_projection = torch.tensor([1.0, -1.0])
        prototype = torch.tensor([1.0, 0.0])

        patched = apply_source_prototype_patch(
            standardized_rows=rows,
            donor_projection=donor_projection,
            prototype=prototype,
            scale=0.5,
        )

        self.assertTrue(torch.allclose(patched, torch.tensor([[2.0, 4.0], [0.25, -2.0]])))

    def test_component_patch_can_replace_residual_only(self) -> None:
        recipient_rows = torch.tensor([[3.0, 4.0], [1.5, -2.0]])
        donor_rows = torch.tensor([[1.0, 8.0], [-1.0, 6.0]])
        prototype = torch.tensor([1.0, 0.0])

        patched = apply_source_component_patch(
            recipient_rows=recipient_rows,
            donor_rows=donor_rows,
            prototype=prototype,
            component="residual",
            scale=1.0,
        )

        self.assertTrue(torch.allclose(patched, torch.tensor([[3.0, 8.0], [1.5, 6.0]])))

    def test_component_patch_can_replace_full_rows(self) -> None:
        recipient_rows = torch.tensor([[3.0, 4.0], [1.5, -2.0]])
        donor_rows = torch.tensor([[1.0, 8.0], [-1.0, 6.0]])
        prototype = torch.tensor([1.0, 0.0])

        patched = apply_source_component_patch(
            recipient_rows=recipient_rows,
            donor_rows=donor_rows,
            prototype=prototype,
            component="full_rows",
            scale=1.0,
        )

        self.assertTrue(torch.allclose(patched, donor_rows))

    def test_load_explicit_pairs_reads_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pairs.csv"
            path.write_text("donor_path,recipient_path\nleft.pt,right.pt\nother.pt,target.pt\n")

            result = load_explicit_pairs(path)

        self.assertEqual(result, [("left.pt", "right.pt"), ("other.pt", "target.pt")])


if __name__ == "__main__":
    unittest.main()
