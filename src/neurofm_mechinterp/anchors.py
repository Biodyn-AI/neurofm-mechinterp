"""Hyperspherical anchor geometry and sphere-preserving edits of tokenizer source vectors.

BrainOmni's quantizer L2-normalises every 256-D source vector before residual vector
quantization (`BrainQuantizer.forward`), so the tokenizer only passes on the direction
``u = x / ||x||``. For one source with unit rows ``u``:

* the *anchor* ``p = E[u] / ||E[u]||`` is the directional mean and ``R = ||E[u]||`` its mean
  resultant length;
* every row decomposes as ``u = a p + sqrt(1 - a^2) s`` with *anchor coordinate*
  ``a = <u, p> = cos(theta)`` (``theta`` is the *colatitude*) and unit *tangent direction*
  ``s = t / ||t||``, where ``t = u - a p`` is the *tangent part* (``s`` is orthogonal to ``p``).

All edits below map the unit sphere onto itself: *location* edits change ``a`` and keep ``s``;
*content* edits change ``s`` and keep ``a``. Because their outputs are unit vectors, the
quantizer's re-normalisation leaves them unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

EPS = 1e-12


def unit_rows(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """L2-normalise the last dimension, matching the quantizer's own normalisation."""
    return F.normalize(x, p=2.0, dim=-1, eps=eps)


def anchor_coordinates(rows: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """Return ``a = u . p`` for every row."""
    return rows @ anchor.to(rows)


def tangent_component(rows: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """Return the tangent part ``t = u - (u . p) p`` of every row."""
    p = anchor.to(rows)
    return rows - (rows @ p).unsqueeze(-1) * p


def decompose(rows: torch.Tensor, anchor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split unit rows into anchor coordinates ``a`` and unit tangent directions ``s``."""
    u = unit_rows(rows)
    p = anchor.to(u)
    a = (u @ p).clamp(-1.0, 1.0)
    s = unit_rows(u - a.unsqueeze(-1) * p)
    return a, s


def compose(a: torch.Tensor, s: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`decompose`: ``u = a p + sqrt(1 - a^2) s`` (``s`` must be orthogonal to ``p``)."""
    a = a.clamp(-1.0, 1.0)
    return a.unsqueeze(-1) * anchor.to(s) + (1.0 - a**2).clamp_min(0.0).sqrt().unsqueeze(-1) * s


def off_sphere_coordinate(a: torch.Tensor, new_a: torch.Tensor) -> torch.Tensor:
    """Anchor coordinate after an off-sphere edit ``a -> new_a`` that keeps the tangent part, once the
    quantizer has re-normalised the vector: ``new_a / sqrt(1 - a^2 + new_a^2)``.

    Used to quantify how much of an edit applied off the sphere (e.g. scaling the anchor component)
    the normalisation undoes; the edits in this module avoid this by construction.
    """
    return new_a / torch.sqrt(1.0 - a**2 + new_a**2)


def geodesic_angle(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Row-wise angle (radians) between two sets of vectors after normalisation."""
    cos = (unit_rows(first) * unit_rows(second)).sum(dim=-1)
    return torch.arccos(cos.clamp(-1.0, 1.0))


# --------------------------------------------------------------------------------- geometry
@dataclass
class AnchorGeometry:
    """Summary of one source's directional distribution on the unit sphere."""

    anchor: torch.Tensor
    mean: torch.Tensor
    mean_resultant_length: float
    coord_mean: float
    coord_var: float
    coord_second_moment: float
    total_centered_variance: float
    centered_eigvals: torch.Tensor
    centered_pcs: torch.Tensor
    uncentered_top: torch.Tensor
    uncentered_top_eigval: float
    uncentered_second_eigval: float
    cross_covariance_norm: float
    tangent_mean: torch.Tensor
    tangent_eigvals: torch.Tensor
    tangent_pcs: torch.Tensor
    mean_angle_deg: float
    num_rows: int
    centered_eigvals_all: torch.Tensor | None = None

    @property
    def eigengap(self) -> float:
        """E[a^2] - lambda_2(M): the gap in the residual bound."""
        return self.coord_second_moment - self.uncentered_second_eigval

    @property
    def concentration_sin_bound(self) -> float:
        """sqrt(Var(a) ||Sigma||_op) / (E[a^2] - lambda_2(M)): the residual bound with ||Cov(a, t)|| replaced by its
        Cauchy-Schwarz bound sd(a) * lambda_max(Cov t)^(1/2) <= sd(a) * ||Sigma||_op^(1/2)."""
        gap = self.eigengap
        spectrum = self.centered_eigvals_all if self.centered_eigvals_all is not None else self.centered_eigvals
        return float("inf") if gap <= 0 else min(1.0, (self.coord_var * float(spectrum[0])) ** 0.5 / gap)

    @property
    def participation_ratio(self) -> float:
        """(tr Sigma)^2 / tr(Sigma^2) over the full spectrum."""
        spectrum = self.centered_eigvals_all if self.centered_eigvals_all is not None else self.centered_eigvals
        return float(self.total_centered_variance**2 / float((spectrum**2).sum()))

    @property
    def davis_kahan_sin_bound(self) -> float:
        """A-priori bound ``2 ||Sigma||_op / R^2`` (Yu-Wang-Samworth form of Davis-Kahan)."""
        return min(1.0, 2.0 * float(self.centered_eigvals[0]) / self.mean_resultant_length**2)

    @property
    def residual_sin_bound(self) -> float:
        """Residual bound ``||Cov(a, t)|| / (E[a^2] - lambda_2(M))`` on sin(angle(anchor, v1))."""
        gap = self.coord_second_moment - self.uncentered_second_eigval
        return float("inf") if gap <= 0 else min(1.0, self.cross_covariance_norm / gap)

    @property
    def weyl_sin_bound(self) -> float:
        """The residual bound with lambda_2(M) replaced by its Weyl upper bound ||Sigma||_op."""
        gap = self.coord_second_moment - float(self.centered_eigvals[0])
        return float("inf") if gap <= 0 else min(1.0, self.cross_covariance_norm / gap)

    @property
    def anchor_variance_fraction(self) -> float:
        """Share of the centered variance that lies along the anchor."""
        return self.coord_var / self.total_centered_variance

    @property
    def anchor_variance_rank(self) -> int:
        """Rank of Var(a) among the centered eigenvalues (1 = largest), over the full spectrum."""
        spectrum = self.centered_eigvals_all if self.centered_eigvals_all is not None else self.centered_eigvals
        return int((spectrum > self.coord_var).sum()) + 1

    def cos_with_centered_pcs(self, k: int | None = None) -> torch.Tensor:
        """Absolute cosine between the anchor and the leading centered principal axes."""
        pcs = self.centered_pcs if k is None else self.centered_pcs[:k]
        return (pcs @ self.anchor).abs()

    def anchor_in_centered_span(self, k: int) -> float:
        """Squared norm of the anchor's projection onto the span of the k leading centered PCs."""
        return float(((self.centered_pcs[:k] @ self.anchor) ** 2).sum())


def anchor_geometry(rows: torch.Tensor, n_components: int | None = None) -> AnchorGeometry:
    """Estimate the anchor geometry of one source from its rows ``(n, D)`` (float64)."""
    if rows.ndim != 2:
        raise ValueError("Expected rows shaped (n, D).")
    u = unit_rows(rows.detach().to(torch.float64).cpu())
    n, dim = u.shape
    mean = u.mean(dim=0)
    radius = float(mean.norm())
    if radius <= 0.0:
        raise ValueError("Mean resultant length is zero; the anchor is undefined.")
    anchor = mean / radius
    coords = u @ anchor
    tangent = u - coords.unsqueeze(-1) * anchor

    second_moment = u.T @ u / n
    eigvals_m, eigvecs_m = torch.linalg.eigh(second_moment)
    top = eigvecs_m[:, -1]
    if float(top @ anchor) < 0.0:
        top = -top

    centered = u - mean
    covariance = centered.T @ centered / n
    eigvals_c, eigvecs_c = torch.linalg.eigh(covariance)
    order = torch.argsort(eigvals_c, descending=True)
    eigvals_c = eigvals_c[order].clamp_min(0.0)
    pcs = eigvecs_c[:, order].T

    # tangent PCA: principal axes of the unit tangent directions, computed in an orthonormal basis
    # of the anchor's orthogonal complement (D - 1 axes, exactly orthogonal to p)
    s = unit_rows(tangent)
    complement = orthogonal_complement(anchor)                      # (D-1) x D
    z = s @ complement.T
    z_mean = z.mean(dim=0)
    z_cov = (z - z_mean).T @ (z - z_mean) / n
    eigvals_t, eigvecs_t = torch.linalg.eigh(z_cov)
    order_t = torch.argsort(eigvals_t, descending=True)
    eigvals_t = eigvals_t[order_t].clamp_min(0.0)
    tpcs = (complement.T @ eigvecs_t[:, order_t]).T
    s_mean = z_mean @ complement

    cross = (coords - coords.mean()).unsqueeze(-1) * tangent       # Cov(a, t); E[t] = 0
    eigvals_all = eigvals_c.clone()
    if n_components is not None:
        eigvals_c, pcs = eigvals_c[:n_components], pcs[:n_components]
        eigvals_t, tpcs = eigvals_t[:n_components], tpcs[:n_components]
    angles = torch.rad2deg(torch.arccos(coords.clamp(-1.0, 1.0)))
    return AnchorGeometry(
        anchor=anchor,
        mean=mean,
        mean_resultant_length=radius,
        coord_mean=float(coords.mean()),
        coord_var=float(coords.var(unbiased=False)),
        coord_second_moment=float((coords**2).mean()),
        total_centered_variance=float(torch.trace(covariance)),
        centered_eigvals=eigvals_c,
        centered_pcs=pcs,
        uncentered_top=top,
        uncentered_top_eigval=float(eigvals_m[-1]),
        uncentered_second_eigval=float(eigvals_m[-2]) if dim > 1 else 0.0,
        cross_covariance_norm=float(cross.mean(dim=0).norm()),
        tangent_mean=s_mean,
        tangent_eigvals=eigvals_t,
        tangent_pcs=tpcs,
        mean_angle_deg=float(angles.mean()),
        num_rows=n,
        centered_eigvals_all=eigvals_all,
    )


@dataclass
class AnchorArrangement:
    """How the unit anchors of all sources are arranged relative to one another."""

    participation_ratio: float                 # effective number of directions spanned by the anchors
    leading_shares: list[float]                # share of their squared norm in the k leading directions, k = 1, 2, ...
    min_angle_deg: float                       # smallest angle between the anchors of two sources
    circle_gaps_deg: list[float] | None        # gaps between consecutive anchors when two directions hold them


def anchor_arrangement(cosines: torch.Tensor, circle_share: float = 0.99) -> AnchorArrangement:
    """Describe the arrangement of unit anchors from their cosine (Gram) matrix ``(S, S)``.

    The eigenvalues of the Gram matrix are the squared singular values of the matrix of anchors, so
    their cumulative shares are the fractions of the anchors' squared norm in the leading k
    (uncentered) principal directions of the anchors. When two directions hold at least
    ``circle_share`` of it, the anchors lie close to a circle; the angular gaps between consecutive
    anchors along that circle are then returned as well.
    """
    gram = cosines.detach().to(torch.float64).cpu()
    gram = (gram + gram.T) / 2.0
    eigvals, eigvecs = torch.linalg.eigh(gram)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order].clamp_min(0.0)
    eigvecs = eigvecs[:, order]
    shares = (eigvals.cumsum(0) / eigvals.sum()).tolist()
    count = gram.shape[0]
    off_diagonal = gram[~torch.eye(count, dtype=torch.bool)]
    min_angle = float(torch.rad2deg(torch.arccos(off_diagonal.max().clamp(-1.0, 1.0))))
    gaps = None
    if count > 2 and shares[1] >= circle_share:
        plane = eigvecs[:, :2] * eigvals[:2].sqrt()
        angles = torch.sort(torch.rad2deg(torch.atan2(plane[:, 1], plane[:, 0])) % 360.0).values
        gaps = torch.diff(torch.cat([angles, angles[:1] + 360.0])).tolist()
    return AnchorArrangement(
        participation_ratio=float(eigvals.sum() ** 2 / (eigvals**2).sum()),
        leading_shares=shares,
        min_angle_deg=min_angle,
        circle_gaps_deg=gaps,
    )


def orthogonal_complement(vector: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis ``(D-1, D)`` of the subspace orthogonal to ``vector``."""
    v = vector / vector.norm()
    dim = v.shape[0]
    frame = torch.cat([v.unsqueeze(1), torch.eye(dim, dtype=v.dtype)], dim=1)
    q, _ = torch.linalg.qr(frame)
    basis = q[:, 1:dim].T
    return basis - (basis @ v).unsqueeze(-1) * v


def random_orthonormal(
    k: int,
    dim: int,
    generator: torch.Generator | None = None,
    orthogonal_to: torch.Tensor | None = None,
    within: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Draw a uniformly random orthonormal ``(k, dim)`` basis.

    ``within`` (an orthonormal ``(m, dim)`` basis) restricts the draw to its span;
    ``orthogonal_to`` (vector or ``(m, dim)``) removes those directions.
    """
    if within is not None:
        w = within.to(dtype)
        gaussian = w.T @ torch.randn(w.shape[0], k, generator=generator, dtype=dtype)
    else:
        gaussian = torch.randn(dim, k, generator=generator, dtype=dtype)
    if orthogonal_to is not None:
        constraint = orthogonal_to.reshape(-1, dim).to(dtype)
        q_c, _ = torch.linalg.qr(constraint.T)
        gaussian = gaussian - q_c @ (q_c.T @ gaussian)
    basis, _ = torch.linalg.qr(gaussian)
    return basis[:, :k].T.contiguous()


# --------------------------------------------------------------------------------- location edits
def set_anchor_coordinate(rows: torch.Tensor, anchor: torch.Tensor, new_a: torch.Tensor) -> torch.Tensor:
    """Give every row the anchor coordinate ``new_a`` and keep its tangent direction."""
    _, s = decompose(rows, anchor)
    return compose(new_a.to(s), s, anchor)


def zero_anchor(rows: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """Move every row to the anchor's equator (colatitude 90 degrees)."""
    return set_anchor_coordinate(rows, anchor, rows.new_zeros(rows.shape[0]))


def scale_anchor(rows: torch.Tensor, anchor: torch.Tensor, factor: float) -> torch.Tensor:
    """Multiply every anchor coordinate by ``factor``."""
    a, _ = decompose(rows, anchor)
    return set_anchor_coordinate(rows, anchor, factor * a)


def mean_anchor(rows: torch.Tensor, anchor: torch.Tensor, coord_mean: float) -> torch.Tensor:
    """Give every row the same anchor coordinate (no variation of the colatitude remains)."""
    return set_anchor_coordinate(rows, anchor, rows.new_full((rows.shape[0],), float(coord_mean)))


def resample_anchor(rows: torch.Tensor, anchor: torch.Tensor, groups: torch.Tensor | None = None,
                    generator: torch.Generator | None = None) -> torch.Tensor:
    """Permute anchor coordinates across rows (within each group, e.g. segment) and keep tangent directions."""
    a, s = decompose(rows, anchor)
    if groups is None:
        groups = torch.zeros(rows.shape[0], dtype=torch.long)
    groups = groups.cpu()
    new_a = a.clone()
    for g in torch.unique(groups):
        idx = torch.nonzero(groups == g).flatten()
        perm = idx[torch.randperm(len(idx), generator=generator)]
        new_a[idx.to(a.device)] = a[perm.to(a.device)]
    return compose(new_a, s, anchor)


def shift_colatitude(rows: torch.Tensor, anchor: torch.Tensor, delta: float) -> torch.Tensor:
    """Add ``delta`` radians to every colatitude (positive: towards the equator), clipped to [0, pi]."""
    a, s = decompose(rows, anchor)
    theta = torch.arccos(a).add(delta).clamp(0.0, torch.pi)
    return compose(torch.cos(theta), s, anchor)


# --------------------------------------------------------------------------------- content edits
def set_tangent_direction(rows: torch.Tensor, anchor: torch.Tensor, new_s: torch.Tensor) -> torch.Tensor:
    """Give every row the tangent direction ``new_s`` (unit, orthogonal to p) and keep its colatitude."""
    a, _ = decompose(rows, anchor)
    return compose(a, unit_rows(new_s.to(rows)), anchor)


def tangent_subspace_mean(rows: torch.Tensor, anchor: torch.Tensor, basis: torch.Tensor,
                          tangent_mean: torch.Tensor) -> torch.Tensor:
    """Mean-ablate the tangent direction inside ``span(basis)`` (basis orthonormal, orthogonal to p).

    Coordinates of ``s - E[s]`` on the basis are removed; the result is re-normalised within the
    tangent space and recombined with the original colatitude.
    """
    a, s = decompose(rows, anchor)
    b = basis.to(s)
    m = tangent_mean.to(s)
    s_new = s - ((s - m) @ b.T) @ b
    norms = s_new.norm(dim=-1, keepdim=True)
    fallback = unit_rows(m).expand_as(s_new)
    s_new = torch.where(norms > 1e-6, s_new / norms.clamp_min(EPS), fallback)
    return compose(a, s_new, anchor)


def tangent_subspace_resample(rows: torch.Tensor, anchor: torch.Tensor, basis: torch.Tensor,
                              donor_directions: torch.Tensor) -> torch.Tensor:
    """Resample ablation of the tangent direction inside ``span(basis)``.

    The coordinates of each row's tangent direction on the basis are replaced by those of a donor
    tangent direction (e.g. the corresponding row of another subject's segment); the result is
    re-normalised within the tangent space and recombined with the row's own colatitude. With a
    basis spanning the whole tangent space this replaces the tangent direction by the donor's.
    """
    a, s = decompose(rows, anchor)
    b = basis.to(s)
    donor = donor_directions.to(s)
    s_new = unit_rows(s - (s @ b.T) @ b + (donor @ b.T) @ b)
    return compose(a, s_new, anchor)


def tangent_mean_direction(rows: torch.Tensor, anchor: torch.Tensor, tangent_mean: torch.Tensor) -> torch.Tensor:
    """Remove all tangent variation: every row points along the mean tangent direction at its own colatitude."""
    a, s = decompose(rows, anchor)
    return compose(a, unit_rows(tangent_mean.to(s)).expand_as(s), anchor)


def tangent_rotation(rows: torch.Tensor, anchor: torch.Tensor, angle: float | torch.Tensor,
                     generator: torch.Generator | None = None, within: torch.Tensor | None = None) -> torch.Tensor:
    """Rotate each row's tangent direction by ``angle`` towards a random direction and keep its colatitude.

    The random direction is orthogonal to p and to the row's tangent direction; with ``within``
    (orthonormal basis orthogonal to p) it is drawn inside that subspace (e.g. the support of the data).
    """
    a, s = decompose(rows, anchor)
    if within is not None:
        w_basis = within.to(s)
        w = torch.randn(rows.shape[0], w_basis.shape[0], generator=generator, dtype=s.dtype).to(s.device) @ w_basis
    else:
        w = torch.randn(rows.shape, generator=generator, dtype=s.dtype).to(s.device)
        p = anchor.to(s)
        w = w - (w @ p).unsqueeze(-1) * p
    w = w - (w * s).sum(dim=-1, keepdim=True) * s
    w = unit_rows(w)
    angle = torch.as_tensor(angle, dtype=s.dtype, device=s.device)
    if angle.ndim == 1:
        angle = angle.unsqueeze(-1)
    return compose(a, torch.cos(angle) * s + torch.sin(angle) * w, anchor)


def matched_tangent_angle(rows: torch.Tensor, anchor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row tangent-rotation angle whose geodesic displacement equals that of ``zero_anchor``.

    Zero-anchor moves a unit row with anchor coordinate a by ``arccos(sqrt(1 - a^2))``; rotating its
    tangent direction by phi moves it by ``arccos(a^2 + (1 - a^2) cos(phi))``. Matching requires
    ``cos(phi) = (sqrt(1 - a^2) - a^2) / (1 - a^2)``, attainable when the row is at least 30 degrees from
    the anchor; otherwise phi = pi (the largest attainable displacement). Returns ``(phi, attained)``.
    """
    a, _ = decompose(rows, anchor)
    t_norm = (1.0 - a**2).clamp_min(EPS).sqrt()
    cos_phi = (t_norm - a**2) / t_norm**2
    attained = cos_phi >= -1.0
    return torch.arccos(cos_phi.clamp(-1.0, 1.0)), attained


# --------------------------------------------------------------------------------- identity edits
def rotate_anchor_to(rows: torch.Tensor, anchor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Apply the rotation in span{p, q} that maps the anchor p onto the target anchor q.

    Rows keep their colatitude, now measured from q, and their tangent geometry is carried along.
    """
    p = anchor.to(rows)
    q = target.to(rows)
    cos_alpha = float((p @ q).clamp(-1.0, 1.0))
    e2 = q - cos_alpha * p
    if float(e2.norm()) < 1e-9:
        return rows.clone()
    e2 = e2 / e2.norm()
    sin_alpha = (1.0 - cos_alpha**2) ** 0.5
    x1 = rows @ p
    x2 = rows @ e2
    return (rows + (cos_alpha - 1.0) * (x1.unsqueeze(-1) * p + x2.unsqueeze(-1) * e2)
            + sin_alpha * (x1.unsqueeze(-1) * e2 - x2.unsqueeze(-1) * p))
