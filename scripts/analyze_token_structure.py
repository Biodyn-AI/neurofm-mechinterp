"""Token-level structure: what the RVQ codes of each source encode.

Reads `results/source_geometry/<run>/{export.pt,anchors.pt}` and writes
`results/token_structure/<run>/tokens.json`:

* code usage per RVQ level: codes used, perplexity overall and per source, the mutual information
  I(S; C_l) between source identity and the level-l code (bits, and as a share of H(S) = 4 bits),
  the accuracy of identifying the source from the level-1 code alone, and the share of the level-1
  code entropy explained by source identity;
* alignment of the level-1 codewords a source selects with its anchor;
* residual norm after each RVQ level;
* code changes when one source's unit vectors are edited with sphere-preserving edits
  (neurofm_mechinterp.anchors): fraction of rows whose level-1 code changes, fraction of those whose
  new level-1 code is characteristic of another source (its most frequent user is another source),
  and change rates at levels 2-4 among rows whose level-1 code is unchanged.
Code usage statistics use the codes of all segments of the run; the edits use the rows of the second to
fourth segment of every recording, with statistics estimated on those rows and donor tangent directions
taken from a random segment of another subject.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch

from neurofm_mechinterp.anchors import (
    anchor_geometry,
    decompose,
    matched_tangent_angle,
    mean_anchor,
    random_orthonormal,
    resample_anchor,
    set_tangent_direction,
    shift_colatitude,
    tangent_rotation,
    tangent_subspace_resample,
    unit_rows,
    zero_anchor,
)
from neurofm_mechinterp.loading import load_tokenizer
from neurofm_mechinterp.paths import RESULTS_ROOT

GEOM_ROOT = RESULTS_ROOT / "source_geometry"


def entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def mutual_information(sources: np.ndarray, codes: np.ndarray, n_sources: int, size: int) -> tuple[float, float, float]:
    """I(S;C) in bits, H(S) and the Bayes accuracy of predicting S from C."""
    joint = np.zeros((n_sources, size))
    np.add.at(joint, (sources, codes), 1.0)
    h_s = entropy(joint.sum(1))
    h_c = entropy(joint.sum(0))
    h_sc = entropy(joint.ravel())
    accuracy = float(joint.max(0).sum() / joint.sum())
    return h_s + h_c - h_sc, h_s, accuracy


def donor_segments(subjects: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """For every segment, the index of a random segment of another subject (same position in the list is never used)."""
    idx = np.arange(len(subjects))
    donors = np.empty(len(subjects), dtype=int)
    for i, subject in enumerate(subjects):
        donors[i] = rng.choice(idx[subjects != subject])
    return donors


@torch.no_grad()
def rvq_encode(tokenizer, rows: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    x = unit_rows(rows.float())
    residual = x
    codes, residual_norms = [], []
    for layer in tokenizer.quantizer.rvq.layers:
        quantized, idx, _ = layer(residual)
        residual = residual - quantized
        codes.append(idx)
        residual_norms.append(residual.norm(dim=-1))
    return torch.stack(codes, dim=-1), residual_norms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--tokenizer", default="released")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    export = torch.load(GEOM_ROOT / args.run / "export.pt", weights_only=False)
    tokenizer = load_tokenizer(args.tokenizer, device="cpu").eval()
    codebooks = [layer._codebook.embed.detach().double() for layer in tokenizer.quantizer.rvq.layers]
    size = codebooks[0].shape[0]
    codes_all = export["codes"].long().numpy()                        # n_seg x S x rows x Q
    n_seg, n_src, n_rows, n_levels = codes_all.shape
    source_ids = np.broadcast_to(np.arange(n_src)[None, :, None], (n_seg, n_src, n_rows)).reshape(-1)

    usage = []
    for level in range(n_levels):
        codes = codes_all[..., level].reshape(-1)
        per_source = [math.exp(entropy(np.bincount(codes_all[:, s, :, level].reshape(-1), minlength=size)) * math.log(2)) for s in range(n_src)]
        info, h_s, acc = mutual_information(source_ids, codes, n_src, size)
        h_c = entropy(np.bincount(codes, minlength=size))
        usage.append({
            "level": level + 1, "codes_used": int(len(np.unique(codes))),
            "perplexity": float(2 ** h_c),
            "per_source_perplexity_median": float(np.median(per_source)),
            "per_source_perplexity_range": [float(min(per_source)), float(max(per_source))],
            "per_source_codes_used_median": float(np.median([len(np.unique(codes_all[:, s, :, level])) for s in range(n_src)])),
            "mi_bits": info, "mi_share_of_source_entropy": info / h_s, "source_entropy_bits": h_s,
            "code_entropy_bits": h_c, "mi_share_of_code_entropy": info / h_c, "source_accuracy": acc,
        })

    # majority source of every level-1 code (its most frequent user)
    joint1 = np.zeros((n_src, size))
    np.add.at(joint1, (source_ids, codes_all[..., 0].reshape(-1)), 1.0)
    owner = joint1.argmax(0)

    rows_all = export["rows"].double()
    row_index = export["row_segment_index"].numpy()
    keep = export["seg_in_subject"].numpy()[row_index] >= 1
    if not keep.any():
        keep = np.ones_like(keep)
    rows_all = rows_all[torch.from_numpy(keep)]
    row_subjects = np.array(export["subjects"])[row_index][keep]
    n_rowseg = rows_all.shape[0]
    gen = torch.Generator().manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    donor_segment = donor_segments(row_subjects, rng)                  # a segment of another subject for every segment

    alignment, change, residuals = [], {}, np.zeros((n_src, n_levels))
    for s in range(n_src):
        unit = unit_rows(rows_all[:, s].reshape(-1, rows_all.shape[-1]))
        g = anchor_geometry(unit, n_components=64)
        p = g.anchor
        cb = codebooks[0]
        counts = np.bincount(codes_all[:, s, :, 0].reshape(-1), minlength=size).astype(float)
        weights = torch.from_numpy(counts / counts.sum())
        cos_own = float((weights * (cb @ p) / cb.norm(dim=-1)).sum())
        alignment.append({"source": s, "weighted_cos_codeword_own_anchor": cos_own,
                          "weighted_codeword_norm": float((weights * cb.norm(dim=-1)).sum())})
        base_codes, residual_norms = rvq_encode(tokenizer, unit)
        residuals[s] = [float(r.mean()) for r in residual_norms]
        per_seg = unit.shape[0] // n_rowseg
        groups = torch.arange(n_rowseg).repeat_interleave(per_seg)
        perm = torch.from_numpy(donor_segment).repeat_interleave(per_seg) * per_seg + torch.arange(per_seg).repeat(n_rowseg)
        _, s_dir = decompose(unit, p)
        phi, _ = matched_tangent_angle(unit, p)
        support = g.tangent_pcs[:64]
        edits = {
            "zero_anchor": zero_anchor(unit, p),
            "mean_anchor": mean_anchor(unit, p, g.coord_mean),
            "resample_anchor": resample_anchor(unit, p, groups=groups, generator=gen),
            "colat_p20": shift_colatitude(unit, p, math.radians(20)),
            "colat_m20": shift_colatitude(unit, p, math.radians(-20)),
            "tpc_resample_16": tangent_subspace_resample(unit, p, support[:16], s_dir[perm]),
            "rand_support_16": tangent_subspace_resample(unit, p, random_orthonormal(16, unit.shape[1], generator=gen, within=support, orthogonal_to=p), s_dir[perm]),
            "tangent_resample": set_tangent_direction(unit, p, s_dir[perm]),
            "rot_matched_on": tangent_rotation(unit, p, phi, generator=gen, within=support),
        }
        for name, edited in edits.items():
            new_codes, _ = rvq_encode(tokenizer, edited)
            changed1 = (new_codes[:, 0] != base_codes[:, 0])
            foreign = float((torch.from_numpy(owner)[new_codes[changed1, 0]] != s).float().mean()) if changed1.any() else 0.0
            stay = ~changed1
            deeper = [float((new_codes[stay, l] != base_codes[stay, l]).float().mean()) if stay.any() else float("nan") for l in range(1, n_levels)]
            change.setdefault(name, []).append({"level1": float(changed1.float().mean()), "foreign_share": foreign, "deeper_given_same_level1": deeper})

    change_summary = {
        name: {
            "level1_median": float(np.median([v["level1"] for v in values])),
            "level1_range": [float(min(v["level1"] for v in values)), float(max(v["level1"] for v in values))],
            "foreign_share_median": float(np.median([v["foreign_share"] for v in values])),
            "deeper_given_same_level1_median": [float(np.nanmedian([v["deeper_given_same_level1"][l] for v in values])) for l in range(n_levels - 1)],
        }
        for name, values in change.items()
    }
    out_dir = RESULTS_ROOT / "token_structure" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "run": args.run, "tokenizer": args.tokenizer, "usage": usage, "level1_alignment": alignment,
        "level1_alignment_summary": {
            "own_anchor_cos_median": float(np.median([a["weighted_cos_codeword_own_anchor"] for a in alignment])),
            "own_anchor_cos_range": [float(min(a["weighted_cos_codeword_own_anchor"] for a in alignment)),
                                     float(max(a["weighted_cos_codeword_own_anchor"] for a in alignment))],
        },
        "residual_norm_after_level_median": [float(np.median(residuals[:, l])) for l in range(n_levels)],
        "code_change": change_summary,
    }
    (out_dir / "tokens.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("usage", "level1_alignment_summary", "residual_norm_after_level_median", "code_change")}, indent=1))


if __name__ == "__main__":
    main()
