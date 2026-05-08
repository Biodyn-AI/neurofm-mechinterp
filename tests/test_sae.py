from __future__ import annotations

import unittest

import torch

from neurofm_mechinterp.sae import (
    SparseAutoencoder,
    choose_ablation_row_mask,
    flatten_source_activations,
    resolve_row_source_idx,
    restore_source_activations,
    select_row_indices,
    subset_activation_bundle,
    train_val_split_indices,
)


class SaeHelpersTests(unittest.TestCase):
    def test_flatten_source_activations_preserves_row_order(self) -> None:
        activations = torch.arange(1 * 2 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 2, 3, 4)
        rows = flatten_source_activations(activations)

        self.assertEqual(rows.activations.shape, (12, 4))
        self.assertTrue(torch.equal(rows.activations, activations.reshape(12, 4)))
        self.assertEqual(rows.source_idx.tolist(), [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        self.assertEqual(rows.patch_idx.tolist(), [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
        self.assertEqual(rows.frame_idx.tolist(), [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])

    def test_select_row_indices_is_deterministic_and_sorted(self) -> None:
        first = select_row_indices(num_rows=10, max_rows=4, seed=7)
        second = select_row_indices(num_rows=10, max_rows=4, seed=7)

        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.tolist(), sorted(first.tolist()))
        self.assertEqual(len(first.tolist()), 4)

    def test_restore_source_activations_round_trips(self) -> None:
        activations = torch.arange(1 * 2 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 2, 3, 4)
        rows = flatten_source_activations(activations)
        restored = restore_source_activations(rows.activations, tuple(activations.shape))

        self.assertTrue(torch.equal(restored, activations))

    def test_train_val_split_indices_are_disjoint(self) -> None:
        train_index, val_index = train_val_split_indices(num_rows=10, val_fraction=0.2, seed=3)

        self.assertEqual(len(train_index), 8)
        self.assertEqual(len(val_index), 2)
        self.assertEqual(sorted(set(train_index.tolist()) & set(val_index.tolist())), [])

    def test_sparse_autoencoder_forward_shapes(self) -> None:
        model = SparseAutoencoder(input_dim=6, hidden_dim=12)
        x = torch.randn(5, 6)
        reconstruction, features = model(x)

        self.assertEqual(reconstruction.shape, (5, 6))
        self.assertEqual(features.shape, (5, 12))
        self.assertTrue(torch.all(features >= 0))

    def test_sparse_autoencoder_topk_enforces_activation_cap(self) -> None:
        model = SparseAutoencoder(input_dim=6, hidden_dim=12, sparsifier="topk", topk=3)
        x = torch.randn(5, 6)
        _, features = model(x)

        self.assertTrue(torch.all((features > 0).sum(dim=-1) <= 3))

    def test_normalize_decoder_features_projects_columns_to_unit_norm(self) -> None:
        model = SparseAutoencoder(input_dim=3, hidden_dim=2)
        model.decoder.weight.data = torch.tensor(
            [
                [3.0, 0.0],
                [4.0, 0.0],
                [0.0, 5.0],
            ]
        )

        model.normalize_decoder_features()
        norms = model.decoder.weight.norm(dim=0)

        self.assertTrue(torch.allclose(norms, torch.ones_like(norms)))

    def test_resolve_row_source_idx_defaults_to_feature_source(self) -> None:
        self.assertEqual(resolve_row_source_idx(feature_source_idx=3, row_source_idx=None), 3)
        self.assertEqual(resolve_row_source_idx(feature_source_idx=3, row_source_idx=5), 5)
        self.assertIsNone(resolve_row_source_idx(feature_source_idx=None, row_source_idx=None))

    def test_choose_ablation_row_mask_can_condition_on_one_source(self) -> None:
        source_idx = torch.tensor([0, 1, 1, 2, 1, 3])
        mask = choose_ablation_row_mask(source_idx, row_source_idx=1)

        self.assertEqual(mask.tolist(), [False, True, True, False, True, False])

    def test_choose_ablation_row_mask_defaults_to_all_rows(self) -> None:
        source_idx = torch.tensor([0, 1, 1, 2])
        mask = choose_ablation_row_mask(source_idx, row_source_idx=None)

        self.assertEqual(mask.tolist(), [True, True, True, True])

    def test_subset_activation_bundle_preserves_tensor_and_list_alignment(self) -> None:
        bundle = {
            "activations": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
            "source_idx": torch.tensor([0, 1, 1, 2]),
            "segment_idx": torch.tensor([10, 11, 12, 13]),
            "subject": ["a", "b", "c", "d"],
            "path": ["p0", "p1", "p2", "p3"],
        }

        subset = subset_activation_bundle(bundle, torch.tensor([False, True, True, False]))

        self.assertTrue(torch.equal(subset["activations"], torch.tensor([[2.0], [3.0]])))
        self.assertEqual(subset["subject"], ["b", "c"])
        self.assertEqual(subset["path"], ["p1", "p2"])
        self.assertEqual(subset["segment_idx"].tolist(), [11, 12])

    def test_subset_activation_bundle_can_reindex_source_rows(self) -> None:
        bundle = {
            "activations": torch.tensor([[1.0], [2.0], [3.0]]),
            "source_idx": torch.tensor([3, 3, 3]),
            "subject": ["a", "b", "c"],
        }

        subset = subset_activation_bundle(bundle, torch.tensor([True, False, True]), reindex_source_idx=True)

        self.assertEqual(subset["source_idx"].tolist(), [0, 0])
        self.assertEqual(subset["original_source_idx"].tolist(), [3, 3])
