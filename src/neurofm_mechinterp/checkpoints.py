"""Checkpoint download helpers for the published BrainOmni releases."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

from .paths import CHECKPOINT_ROOT, ensure_dir


HF_REPO_ID = "OpenTSLab/BrainOmni"
# revision of the Hugging Face repository whose checkpoints all analyses used
HF_REVISION = "9a4d3c70495370397ccfbfd6d2496f25647545a5"
VARIANT_FILES = {
    "tiny": ["model_cfg.json", "BrainOmni.pt"],
    "base": ["model_cfg.json", "BrainOmni.pt"],
    "braintokenizer": ["model_cfg.json", "BrainTokenizer.pt"],
}


def download_variant(variant: str, root: Path | None = None) -> Path:
    """Download one published checkpoint directory into the local cache."""
    if variant not in VARIANT_FILES:
        known = ", ".join(sorted(VARIANT_FILES))
        raise ValueError(f"Unknown checkpoint variant '{variant}'. Expected one of: {known}.")

    root = ensure_dir(root or CHECKPOINT_ROOT)
    for filename in VARIANT_FILES[variant]:
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f"{variant}/{filename}",
            revision=HF_REVISION,
            local_dir=root,
        )
    return root / variant


def ensure_checkpoint_bundle(model_variant: str) -> dict[str, Path]:
    """Make sure the tokenizer and requested model checkpoint are both present."""
    bundle = {
        "braintokenizer": download_variant("braintokenizer"),
        model_variant: download_variant(model_variant),
    }
    return bundle

