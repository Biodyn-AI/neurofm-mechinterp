"""Model loading wrappers around the upstream BrainOmni implementation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from .checkpoints import ensure_checkpoint_bundle
from .compat import install_deepspeed_comm_stub
from .paths import CHECKPOINT_ROOT, UPSTREAM_ROOT


def _configure_upstream_imports() -> None:
    """Make the upstream repo importable and patch its DeepSpeed dependency."""
    install_deepspeed_comm_stub()
    upstream_str = str(UPSTREAM_ROOT)
    if upstream_str not in sys.path:
        sys.path.insert(0, upstream_str)


def resolve_device(device: str | None = None) -> torch.device:
    """Resolve the execution device. CPU is the conservative default for reproducibility."""
    if device:
        return torch.device(device)
    return torch.device("cpu")


def _load_config(checkpoint_dir: Path) -> dict:
    with (checkpoint_dir / "model_cfg.json").open("r") as handle:
        return json.load(handle)


def load_brainomni(
    variant: str = "tiny",
    device: str | None = None,
    checkpoint_root: Path | None = None,
) -> torch.nn.Module:
    """Load a published BrainOmni checkpoint and move it to eval mode."""
    if variant not in {"tiny", "base"}:
        raise ValueError("BrainOmni variant must be 'tiny' or 'base'.")

    _configure_upstream_imports()
    ensure_checkpoint_bundle(variant)

    from brainomni.model import BrainOmni  # imported lazily after path setup

    checkpoint_root = checkpoint_root or CHECKPOINT_ROOT
    checkpoint_dir = checkpoint_root / variant
    model = BrainOmni(**_load_config(checkpoint_dir))
    state_dict = torch.load(checkpoint_dir / "BrainOmni.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    for parameter in model.tokenizer.parameters():
        parameter.requires_grad = False
    model.eval()
    return model.to(resolve_device(device))


def load_braintokenizer(
    device: str | None = None,
    checkpoint_root: Path | None = None,
) -> torch.nn.Module:
    """Load the published BrainTokenizer checkpoint and move it to eval mode."""
    _configure_upstream_imports()
    ensure_checkpoint_bundle("tiny")

    from braintokenizer.model import BrainTokenizer  # imported lazily after path setup

    checkpoint_root = checkpoint_root or CHECKPOINT_ROOT
    checkpoint_dir = checkpoint_root / "braintokenizer"
    model = BrainTokenizer(**_load_config(checkpoint_dir))
    state_dict = torch.load(
        checkpoint_dir / "BrainTokenizer.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model.to(resolve_device(device))

