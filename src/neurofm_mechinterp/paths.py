"""Central path definitions for the local workspace."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = REPO_ROOT / "upstream" / "BrainOmni"
CHECKPOINT_ROOT = REPO_ROOT / "external" / "checkpoints" / "BrainOmni"
RESULTS_ROOT = REPO_ROOT / "results"


def ensure_dir(path: Path) -> Path:
    """Create a directory tree when needed and return the same path."""
    path.mkdir(parents=True, exist_ok=True)
    return path

