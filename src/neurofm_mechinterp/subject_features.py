"""Subject-level feature tables built from exported tokenizer source vectors.

Every per-segment quantity of `results/source_geometry/<run>/export.pt` (scripts/export_source_vectors.py)
is averaged over the segments of each subject of a diagnostic contrast:

  u       per-source mean unit vector of the segment                      (S x D)
  uq      the same for the quantized vectors (sum of the four RVQ codewords) (S x D)
  rbar    per-source mean resultant length within the segment             (S)
  norm    per-source mean norm before L2 normalisation                    (S)
  band    log band power per channel and band                             (C x 5)

together with age and sex from the cohort's participants.tsv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .cohorts import get_cohort
from .loading import load_tokenizer
from .paths import RESULTS_ROOT

GEOM_ROOT = RESULTS_ROOT / "source_geometry"
DEMOGRAPHIC_COLUMNS = {
    "pd_ucsd": ("age", "gender"),
    "pd_iowa": ("AGE", "GENDER"),
    "dementia": ("Age", "Gender"),
    "mdd_meg": ("participant_age", "sex"),
}


def rvq_codebooks(tokenizer_key: str = "released") -> list[torch.Tensor]:
    tokenizer = load_tokenizer(tokenizer_key, device="cpu")
    return [layer._codebook.embed.detach().double().reshape(-1, layer._codebook.embed.shape[-1])
            for layer in tokenizer.quantizer.rvq.layers]


def quantized_means(codes: torch.Tensor, codebooks: list[torch.Tensor], chunk: int = 32) -> np.ndarray:
    """Per segment and source, the mean over rows of the quantized vector sum_l C_l[code_l]."""
    n_seg, n_src, _, n_levels = codes.shape
    out = np.zeros((n_seg, n_src, codebooks[0].shape[1]))
    for start in range(0, n_seg, chunk):
        block = codes[start : start + chunk].long()
        q = sum(codebooks[level][block[..., level]] for level in range(n_levels))
        out[start : start + chunk] = q.mean(dim=2).numpy()
    return out


def demographics(cohort_name: str, subjects: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Age (years) and sex (1 male, 0 female); NaN where missing."""
    cohort = get_cohort(cohort_name)
    table = pd.read_csv(cohort.participants_tsv, sep="\t", dtype=str)
    table.columns = [c.strip() for c in table.columns]
    table["participant_id"] = table["participant_id"].str.strip()
    table = table.set_index("participant_id")
    age_col, sex_col = DEMOGRAPHIC_COLUMNS[cohort_name]
    age = pd.to_numeric(table.reindex(subjects)[age_col], errors="coerce").to_numpy(dtype=float)
    sex = (table.reindex(subjects)[sex_col].astype(str).str.strip().str.upper().str[:1]
           .map({"M": 1.0, "F": 0.0}).to_numpy(dtype=float))
    return age, sex


def subject_table(cohort_name: str, contrast: str, run: str | None = None, tokenizer_key: str = "released",
                  quantized: bool = True) -> dict:
    """Subject means of the per-segment quantities of one run for the subjects of a contrast."""
    cohort = get_cohort(cohort_name)
    export = torch.load(GEOM_ROOT / (run or cohort_name) / "export.pt", weights_only=False)
    seg_subjects = np.array(export["subjects"])
    subjects, labels = cohort.contrast_labels(contrast, sorted(set(seg_subjects)))
    u = export["u_mean"].double().numpy()
    per_segment = {
        "u": u,
        "rbar": np.linalg.norm(u, axis=-1),
        "norm": export["norm_mean"].double().numpy(),
        "band": export["bandpower"].double().numpy(),
    }
    if quantized:
        per_segment["uq"] = quantized_means(export["codes"], rvq_codebooks(tokenizer_key))
    table = {"subjects": subjects, "labels": np.array(labels, dtype=int),
             "segments": np.array([int(np.sum(seg_subjects == s)) for s in subjects])}
    for key, values in per_segment.items():
        table[key] = np.stack([values[seg_subjects == s].mean(axis=0) for s in subjects])
    table["age"], table["sex"] = demographics(cohort_name, subjects)
    return table
