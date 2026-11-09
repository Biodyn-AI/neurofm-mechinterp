from __future__ import annotations

import math
import unittest

import torch

from neurofm_mechinterp.anchors import (
    anchor_arrangement,
    anchor_geometry,
    compose,
    decompose,
    geodesic_angle,
    matched_tangent_angle,
    mean_anchor,
    off_sphere_coordinate,
    random_orthonormal,
    resample_anchor,
    rotate_anchor_to,
    scale_anchor,
    shift_colatitude,
    tangent_mean_direction,
    tangent_rotation,
    tangent_subspace_mean,
    tangent_subspace_resample,
    unit_rows,
    zero_anchor,
)


def _concentrated_rows(n: int = 2000, dim: int = 16, offset: float = 2.0, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    direction = torch.zeros(dim, dtype=torch.float64)
    direction[0] = 1.0
    scales = torch.linspace(1.5, 0.2, dim, dtype=torch.float64)
    noise = torch.randn(n, dim, generator=gen, dtype=torch.float64) * scales
    return unit_rows(noise + offset * direction)


class GeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = _concentrated_rows()
        self.g = anchor_geometry(self.rows)

    def test_anchor_is_unit_mean_direction(self) -> None:
        self.assertAlmostEqual(float(self.g.anchor.norm()), 1.0, places=10)
        self.assertAlmostEqual(self.g.mean_resultant_length, float(self.rows.mean(0).norm()), places=10)

    def test_bounds_hold(self) -> None:
        cos = float(self.g.uncentered_top @ self.g.anchor)
        sin = math.sqrt(max(0.0, 1.0 - cos**2))
        self.assertLessEqual(sin, self.g.residual_sin_bound + 1e-9)
        self.assertLessEqual(sin, self.g.davis_kahan_sin_bound + 1e-9)
        self.assertLessEqual(self.g.residual_sin_bound, self.g.davis_kahan_sin_bound + 1e-9)
        self.assertLessEqual(self.g.residual_sin_bound, self.g.weyl_sin_bound + 1e-12)
        self.assertLessEqual(self.g.residual_sin_bound, self.g.concentration_sin_bound + 1e-12)

    def test_full_spectrum_summaries(self) -> None:
        g = anchor_geometry(self.rows, n_components=3)
        full = anchor_geometry(self.rows)
        self.assertAlmostEqual(g.participation_ratio, full.participation_ratio, places=10)
        self.assertEqual(g.anchor_variance_rank, full.anchor_variance_rank)

    def test_second_moment_identity(self) -> None:
        # M p = E[a^2] p + Cov(a, t)
        u = self.rows
        p = self.g.anchor
        m_p = (u.T @ u / u.shape[0]) @ p
        a = u @ p
        t = u - a.unsqueeze(-1) * p
        c = ((a - a.mean()).unsqueeze(-1) * t).mean(0)
        self.assertTrue(torch.allclose(m_p, (a**2).mean() * p + c, atol=1e-12))
        self.assertAlmostEqual(float(c.norm()), self.g.cross_covariance_norm, places=10)

    def test_trace_identity_and_tangent_pcs(self) -> None:
        self.assertAlmostEqual(self.g.total_centered_variance, 1.0 - self.g.mean_resultant_length**2, places=10)
        self.assertLess(float((self.g.tangent_pcs @ self.g.anchor).abs().max()), 1e-9)
        self.assertAlmostEqual(self.g.anchor_in_centered_span(self.rows.shape[1]), 1.0, places=8)


class OperatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = _concentrated_rows()
        self.g = anchor_geometry(self.rows)
        self.p = self.g.anchor
        self.a, self.s = decompose(self.rows, self.p)

    def _assert_unit(self, x: torch.Tensor) -> None:
        self.assertTrue(torch.allclose(x.norm(dim=-1), torch.ones(x.shape[0], dtype=x.dtype), atol=1e-10))

    def test_decompose_compose_round_trip(self) -> None:
        self.assertTrue(torch.allclose(compose(self.a, self.s, self.p), self.rows, atol=1e-10))
        self.assertLess(float((self.s @ self.p).abs().max()), 1e-10)

    def test_zero_and_scale_anchor(self) -> None:
        z = zero_anchor(self.rows, self.p)
        self._assert_unit(z)
        self.assertLess(float((z @ self.p).abs().max()), 1e-10)
        h = scale_anchor(self.rows, self.p, 0.5)
        self._assert_unit(h)
        self.assertTrue(torch.allclose(h @ self.p, 0.5 * self.a, atol=1e-10))
        _, s_h = decompose(h, self.p)
        self.assertTrue(torch.allclose(s_h, self.s, atol=1e-8))

    def test_mean_anchor_removes_all_colatitude_variation(self) -> None:
        m = mean_anchor(self.rows, self.p, self.g.coord_mean)
        self._assert_unit(m)
        self.assertLess(float((m @ self.p - self.g.coord_mean).abs().max()), 1e-10)
        _, s_m = decompose(m, self.p)
        self.assertTrue(torch.allclose(s_m, self.s, atol=1e-8))

    def test_resample_anchor_within_groups(self) -> None:
        groups = torch.arange(self.rows.shape[0]) // 100
        r = resample_anchor(self.rows, self.p, groups=groups, generator=torch.Generator().manual_seed(1))
        self._assert_unit(r)
        for g in range(int(groups.max()) + 1):
            idx = groups == g
            self.assertTrue(torch.allclose(torch.sort((r @ self.p)[idx]).values, torch.sort(self.a[idx]).values, atol=1e-10))

    def test_shift_colatitude(self) -> None:
        r = shift_colatitude(self.rows, self.p, math.radians(10))
        theta = torch.arccos(self.a)
        expected = (theta + math.radians(10)).clamp(0, math.pi)
        self.assertTrue(torch.allclose(torch.arccos((r @ self.p).clamp(-1, 1)), expected, atol=1e-6))

    def test_tangent_subspace_mean_keeps_colatitude(self) -> None:
        basis = self.g.tangent_pcs[:3]
        r = tangent_subspace_mean(self.rows, self.p, basis, self.g.tangent_mean)
        self._assert_unit(r)
        self.assertTrue(torch.allclose(r @ self.p, self.a, atol=1e-10))
        full = tangent_subspace_mean(self.rows, self.p, self.g.tangent_pcs, self.g.tangent_mean)
        target = tangent_mean_direction(self.rows, self.p, self.g.tangent_mean)
        self.assertTrue(torch.allclose(full, target, atol=1e-8))

    def test_tangent_subspace_resample(self) -> None:
        donor = self.s[torch.randperm(self.s.shape[0], generator=torch.Generator().manual_seed(4))]
        r = tangent_subspace_resample(self.rows, self.p, self.g.tangent_pcs[:3], donor)
        self._assert_unit(r)
        self.assertTrue(torch.allclose(r @ self.p, self.a, atol=1e-10))
        full = tangent_subspace_resample(self.rows, self.p, self.g.tangent_pcs, donor)
        _, s_full = decompose(full, self.p)
        self.assertTrue(torch.allclose(s_full, donor, atol=1e-8))

    def test_tangent_rotation_angle_and_support(self) -> None:
        gen = torch.Generator().manual_seed(5)
        r = tangent_rotation(self.rows, self.p, 0.7, generator=gen)
        self._assert_unit(r)
        self.assertTrue(torch.allclose(r @ self.p, self.a, atol=1e-10))
        _, s_r = decompose(r, self.p)
        self.assertTrue(torch.allclose(geodesic_angle(self.s, s_r), torch.full((self.rows.shape[0],), 0.7, dtype=torch.float64), atol=1e-7))
        support = self.g.tangent_pcs[:4]
        r2 = tangent_rotation(self.rows, self.p, 0.5, generator=gen, within=support)
        self.assertTrue(torch.allclose(r2 @ self.p, self.a, atol=1e-10))
        _, s2 = decompose(r2, self.p)
        self.assertTrue(torch.allclose(geodesic_angle(self.s, s2), torch.full((self.rows.shape[0],), 0.5, dtype=torch.float64), atol=1e-7))
        # the new direction lies in span(support, s): nothing outside that span is added
        for i in range(0, self.rows.shape[0], 397):
            span = torch.cat([support, self.s[i : i + 1]])
            q, _ = torch.linalg.qr(span.T)
            resid = s2[i] - q @ (q.T @ s2[i])
            self.assertLess(float(resid.norm()), 1e-8)

    def test_matched_angle_equalises_displacement(self) -> None:
        rows = _concentrated_rows(offset=0.8)
        g = anchor_geometry(rows)
        phi, attained = matched_tangent_angle(rows, g.anchor)
        self.assertGreater(int(attained.sum()), 0)
        rotated = tangent_rotation(rows, g.anchor, phi, generator=torch.Generator().manual_seed(2))
        zeroed = zero_anchor(rows, g.anchor)
        self.assertTrue(torch.allclose(geodesic_angle(rows, rotated)[attained], geodesic_angle(rows, zeroed)[attained], atol=1e-7))

    def test_off_sphere_coordinate_matches_renormalisation(self) -> None:
        a, s = decompose(self.rows, self.p)
        shifted = self.rows - 0.5 * a.unsqueeze(-1) * self.p          # halve the anchor component, off the sphere
        renormalised = unit_rows(shifted) @ self.p
        self.assertTrue(torch.allclose(renormalised, off_sphere_coordinate(a, 0.5 * a), atol=1e-10))
        self.assertAlmostEqual(float(off_sphere_coordinate(torch.tensor(0.6, dtype=torch.float64), torch.tensor(0.3, dtype=torch.float64))),
                               0.3 / (1 - 0.36 + 0.09) ** 0.5, places=12)

    def test_rotate_anchor_to(self) -> None:
        q = random_orthonormal(1, self.rows.shape[1], generator=torch.Generator().manual_seed(3))[0]
        r = rotate_anchor_to(self.rows, self.p, q)
        self._assert_unit(r)
        self.assertTrue(torch.allclose(rotate_anchor_to(self.p.unsqueeze(0), self.p, q)[0], q, atol=1e-10))
        self.assertTrue(torch.allclose(r @ q, self.a, atol=1e-10))

    def test_random_orthonormal_within_support(self) -> None:
        support = self.g.tangent_pcs[:6]
        basis = random_orthonormal(3, self.rows.shape[1], generator=torch.Generator().manual_seed(9), within=support, orthogonal_to=self.p)
        self.assertTrue(torch.allclose(basis @ basis.T, torch.eye(3, dtype=torch.float64), atol=1e-10))
        inside = (basis @ support.T) @ support
        self.assertTrue(torch.allclose(inside, basis, atol=1e-10))


class ArrangementTests(unittest.TestCase):
    def test_anchors_on_a_circle(self) -> None:
        angles = torch.arange(16, dtype=torch.float64) * (2 * math.pi / 16) + 0.3
        basis = random_orthonormal(2, 12, generator=torch.Generator().manual_seed(4))
        anchors = torch.cos(angles).unsqueeze(-1) * basis[0] + torch.sin(angles).unsqueeze(-1) * basis[1]
        arrangement = anchor_arrangement(anchors @ anchors.T)
        self.assertAlmostEqual(arrangement.participation_ratio, 2.0, places=8)
        self.assertAlmostEqual(arrangement.leading_shares[1], 1.0, places=8)
        self.assertAlmostEqual(arrangement.min_angle_deg, 22.5, places=5)
        self.assertEqual(len(arrangement.circle_gaps_deg), 16)
        for gap in arrangement.circle_gaps_deg:
            self.assertAlmostEqual(gap, 22.5, places=5)

    def test_orthonormal_anchors(self) -> None:
        anchors = random_orthonormal(4, 12, generator=torch.Generator().manual_seed(5))
        arrangement = anchor_arrangement(anchors @ anchors.T)
        self.assertAlmostEqual(arrangement.participation_ratio, 4.0, places=8)
        self.assertAlmostEqual(arrangement.leading_shares[1], 0.5, places=8)
        self.assertAlmostEqual(arrangement.min_angle_deg, 90.0, places=5)
        self.assertIsNone(arrangement.circle_gaps_deg)


if __name__ == "__main__":
    unittest.main()
