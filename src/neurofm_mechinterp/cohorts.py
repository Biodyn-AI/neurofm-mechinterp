"""Registry of the analysed cohorts (processed segments + participant groups)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .paths import REPO_ROOT

RAW_ROOT = Path.home() / "neurofm-data" / "openneuro"


@dataclass(frozen=True)
class Cohort:
    """One processed cohort and how to read its participant groups."""

    name: str
    accession: str
    modality: str
    metadata_path: Path
    participants_tsv: Path
    group_column: str
    contrasts: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = field(default_factory=dict)
    description: str = ""

    def load_metadata(self) -> list[dict]:
        rows = json.loads(self.metadata_path.read_text())
        return [row for row in rows if row.get("channels")]

    def groups(self) -> dict[str, str]:
        """Map participant_id -> group string from participants.tsv.

        ds002778 has no group column; its group is encoded in the ID (sub-hcN / sub-pdN).
        """
        table = pd.read_csv(self.participants_tsv, sep="\t", dtype=str)
        table.columns = [c.strip() for c in table.columns]
        ids = [str(v).strip() for v in table["participant_id"]]
        if self.group_column == "participant_id":
            return {pid: ucsd_group(pid) for pid in ids}
        return {pid: str(g).strip() for pid, g in zip(ids, table[self.group_column])}

    def contrast_labels(self, contrast: str, subjects: list[str]) -> tuple[list[str], list[int]]:
        """Subjects belonging to either side of ``contrast`` and their 0/1 labels."""
        negative, positive = self.contrasts[contrast]
        groups = self.groups()
        kept, labels = [], []
        for subject in subjects:
            group = groups.get(subject)
            if group in positive:
                kept.append(subject)
                labels.append(1)
            elif group in negative:
                kept.append(subject)
                labels.append(0)
        return kept, labels


def _meta(dataset: str) -> Path:
    return REPO_ROOT / "share" / "metadata" / "evaluate" / dataset / "BrainOmni" / "info.json"


COHORTS: dict[str, Cohort] = {
    "pd_ucsd": Cohort(
        name="pd_ucsd",
        accession="ds002778",
        modality="EEG",
        metadata_path=_meta("pd31_full"),
        participants_tsv=RAW_ROOT / "ds002778" / "participants.tsv",
        group_column="participant_id",
        contrasts={"pd_vs_hc": (("hc",), ("pd",))},
        description="32-channel resting EEG; 16 controls, 15 Parkinson's patients off medication",
    ),
    "pd_iowa": Cohort(
        name="pd_iowa",
        accession="ds004584",
        modality="EEG",
        metadata_path=_meta("pd_rest63"),
        participants_tsv=RAW_ROOT / "ds004584" / "participants.tsv",
        group_column="GROUP",
        contrasts={"pd_vs_hc": (("Control",), ("PD",))},
        description="63-channel eyes-open resting EEG; 49 controls, 100 Parkinson's patients",
    ),
    "dementia": Cohort(
        name="dementia",
        accession="ds004504",
        modality="EEG",
        metadata_path=_meta("dementia19"),
        participants_tsv=RAW_ROOT / "ds004504" / "participants.tsv",
        group_column="Group",
        contrasts={"ad_vs_cn": (("C",), ("A",)), "ftd_vs_cn": (("C",), ("F",))},
        description="19-channel eyes-closed resting EEG; 29 controls, 36 Alzheimer's, 23 frontotemporal dementia",
    ),
    "mdd_meg": Cohort(
        name="mdd_meg",
        accession="ds003568",
        modality="MEG",
        metadata_path=_meta("meg_rest"),
        participants_tsv=RAW_ROOT / "ds003568" / "participants.tsv",
        group_column="group",
        contrasts={"mdd_vs_hv": (("HV",), ("MDD",))},
        description="269-gradiometer CTF resting MEG; adolescents with major depression and healthy volunteers",
    ),
}


def ucsd_group(participant_id: str) -> str:
    """ds002778 encodes the group in the participant ID (sub-hcN / sub-pdN)."""
    return "pd" if participant_id.startswith("sub-pd") else "hc"


def get_cohort(name: str) -> Cohort:
    if name not in COHORTS:
        raise KeyError(f"Unknown cohort {name!r}; known: {sorted(COHORTS)}")
    return COHORTS[name]
