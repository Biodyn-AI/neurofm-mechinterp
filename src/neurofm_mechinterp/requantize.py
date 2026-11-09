"""Encode once, then re-quantize and decode edited source vectors with a frozen tokenizer.

The BrainTokenizer decoder reconstructs each sensor from its own sensor embedding
and the 16 quantized sources (`ForwardSolution` is a cross-attention whose
queries are sensor embeddings), so reconstructing a subset of sensors gives
exactly the same per-sensor output as a full pass. Patches are encoded
independently. Both facts let interventions run on sensor and patch subsets.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import BrainBatch


@dataclass
class EncodedBatch:
    """Frozen-tokenizer state for one batch of segments."""

    feature: torch.Tensor           # B x S x N x T x D pre-quantizer source vectors
    sensor_embedding: torch.Tensor  # B x C x D
    target: torch.Tensor            # B x C x N x L, the tokenizer's per-window z-scored target

    @property
    def num_sources(self) -> int:
        return int(self.feature.shape[1])


class TokenizerProbe:
    """Thin wrapper exposing the encoder, quantizer and decoder of a BrainTokenizer."""

    def __init__(self, tokenizer: torch.nn.Module) -> None:
        self.tokenizer = tokenizer.eval()

    @property
    def device(self) -> torch.device:
        return next(self.tokenizer.parameters()).device

    @torch.no_grad()
    def encode(self, batch: BrainBatch) -> EncodedBatch:
        batch = batch.to(self.device)
        unfolded = self.tokenizer.unfold(batch.x)
        sensor_embedding = self.tokenizer.sensor_embed(batch.pos, batch.sensor_type)
        feature = self.tokenizer.encoder(unfolded, sensor_embedding)
        target = self.tokenizer.norm_target(unfolded)
        return EncodedBatch(feature=feature, sensor_embedding=sensor_embedding, target=target)

    @torch.no_grad()
    def quantize(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (quantized vectors, RVQ indices B x S x N x T x Q)."""
        quantized, indices, _ = self.tokenizer.quantizer(feature)
        return quantized, indices

    @torch.no_grad()
    def reconstruct(
        self,
        feature: torch.Tensor,
        sensor_embedding: torch.Tensor,
        channels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        quantized, _ = self.quantize(feature)
        if channels is not None:
            sensor_embedding = sensor_embedding[:, channels]
        return self.tokenizer.decoder(quantized, sensor_embedding).float()

    @torch.no_grad()
    def segment_mse(
        self,
        feature: torch.Tensor,
        encoded: EncodedBatch,
        channels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-segment reconstruction MSE against the tokenizer's own target."""
        reconstruction = self.reconstruct(feature, encoded.sensor_embedding, channels)
        target = encoded.target if channels is None else encoded.target[:, channels]
        return ((reconstruction - target.float()) ** 2).mean(dim=(1, 2, 3))

    @torch.no_grad()
    def segment_mse_with_source(
        self,
        encoded: EncodedBatch,
        source: int,
        new_vectors: torch.Tensor,
        channels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Replace one source's vectors (B x N x T x D) and return per-segment MSE."""
        feature = encoded.feature.clone()
        feature[:, source] = new_vectors.to(feature)
        return self.segment_mse(feature, encoded, channels)


@torch.no_grad()
def decode_split(
    tokenizer: torch.nn.Module,
    q_key: torch.Tensor,
    q_value: torch.Tensor,
    sensor_embedding: torch.Tensor,
    channels: torch.Tensor | None = None,
    return_attention: bool = False,
):
    """Decoder pass with separate inputs for the attention keys and values.

    Re-implements `BrainTokenizerDecoder.forward` with explicit attention (softmax(QK^T/sqrt(d)))
    so that the keys can be computed from ``q_key`` and the values from ``q_value`` (both
    B x S x N x T x D quantized sources). With ``q_key is q_value`` it equals the standard decoder.
    Returns the reconstruction (B x C x N x L) and, optionally, the attention of every sensor over
    the sources averaged over heads ((B*N*T) x C x S).
    """
    from einops import rearrange  # noqa: PLC0415

    decoder = tokenizer.decoder
    if hasattr(decoder, "source_embedding"):
        embedding = decoder.source_embedding[None, :, None, None, :].to(q_key)
        q_key = q_key + embedding
        q_value = q_value + embedding
    b, s, n, t, d = q_key.shape
    if channels is not None:
        sensor_embedding = sensor_embedding[:, channels]
    c = sensor_embedding.shape[1]
    keys_in = rearrange(q_key, "B S N T D -> (B N T) S D")
    values_in = rearrange(q_value, "B S N T D -> (B N T) S D")
    queries = rearrange(sensor_embedding.view(b, c, 1, 1, d).repeat(1, 1, n, t, 1), "B C N T D -> (B N T) C D")
    solution = decoder.forwardsolution
    heads = solution.n_head
    keys = solution.kv(keys_in)[..., :d]
    values = solution.kv(values_in)[..., d:]
    q = rearrange(queries, "M C (H E) -> M H C E", H=heads)
    k = rearrange(keys, "M S (H E) -> M H S E", H=heads)
    v = rearrange(values, "M S (H E) -> M H S E", H=heads)
    attention = torch.softmax(q @ k.transpose(-1, -2) / (d // heads) ** 0.5, dim=-1)
    out = (attention @ v).transpose(1, 2).reshape(-1, c, d)
    out = solution.proj(out)
    x = rearrange(out, "(B N T) C D -> (B C N) D T", B=b, N=n, T=t)
    x = decoder.seanet_decoder(x)
    reconstruction = rearrange(x, "(B C N) 1 L -> B C N L", B=b, N=n).float()
    if return_attention:
        return reconstruction, attention.mean(dim=1)
    return reconstruction


def reconstruction_errors(reconstruction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-segment MSE, mean absolute error and 1 - mean Pearson r over sensor windows."""
    target = target.float()
    diff = reconstruction - target
    rc = reconstruction - reconstruction.mean(dim=-1, keepdim=True)
    tc = target - target.mean(dim=-1, keepdim=True)
    r = (rc * tc).sum(-1) / (rc.norm(dim=-1) * tc.norm(dim=-1)).clamp_min(1e-8)
    return {
        "mse": (diff**2).mean(dim=(1, 2, 3)),
        "mae": diff.abs().mean(dim=(1, 2, 3)),
        "one_minus_r": 1.0 - r.mean(dim=(1, 2)),
    }


def edit_source_rows(
    encoded: EncodedBatch,
    source: int,
    edit,
) -> torch.Tensor:
    """Apply a row-wise edit ``(n, D) -> (n, D)`` to one source's unit vectors.

    Returns the edited ``B x N x T x D`` block. The edit receives unit rows (the
    quantizer's view); its output may have any norm.
    """
    block = encoded.feature[:, source]
    shape = block.shape
    rows = torch.nn.functional.normalize(block.reshape(-1, shape[-1]), p=2.0, dim=-1)
    return edit(rows).reshape(shape)
