"""Independently trained BrainTokenizer replicas (same architecture and objective, new seeds).

One architectural variant is supported: a learned per-source embedding added to the quantized
source vectors before the decoder computes its attention keys and values. The released decoder
receives no source index, so it can tell sources apart only by the content of their vectors;
the variant supplies the identity explicitly.
"""

from __future__ import annotations

import json
import math
import types
from pathlib import Path

import torch

from .loading import _configure_upstream_imports, resolve_device


def add_decoder_source_embedding(model: torch.nn.Module) -> None:
    """Add ``decoder.source_embedding`` (S x D) to the decoder input of every source."""
    n_sources, dim = model.encoder.neuros.shape
    embedding = torch.nn.Parameter(torch.empty(n_sources, dim))
    torch.nn.init.trunc_normal_(embedding, std=1.0 / math.sqrt(dim))
    model.decoder.source_embedding = embedding
    original = model.decoder.forward

    def forward(self, x, sensor_embedding=None):
        return original(x + self.source_embedding[None, :, None, None, :].to(x), sensor_embedding)

    model.decoder.forward = types.MethodType(forward, model.decoder)


def load_replica(directory: Path, device: str | None = None, tag: str = "final") -> torch.nn.Module:
    """Load a replica trained by scripts/train_tokenizer_replica.py."""
    _configure_upstream_imports()
    from braintokenizer.model import BrainTokenizer  # imported lazily after path setup

    directory = Path(directory)
    config = json.loads((directory / "model_cfg.json").read_text())
    model = BrainTokenizer(**config)
    if config.get("decoder_source_embedding"):
        add_decoder_source_embedding(model)
    state = torch.load(directory / f"{tag}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model.to(resolve_device(device))
