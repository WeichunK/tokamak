"""Core layers shared by the supported decoder-only architectures.

Numerical conventions deliberately mirror the Hugging Face reference implementations
(normalization statistics in float32, rotary tables computed in float32 and applied in
the activation dtype) so that logits can be compared against ``transformers`` at tight
tolerances in the parity tests.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (Zhang & Sennrich, 2019).

    The variance is computed in float32 regardless of the activation dtype, matching
    the reference implementations of the Llama/Qwen families.
    """

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize over the last dimension."""
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding (Su et al., 2021).

    Uses the half-rotation ("NeoX") layout of the Hugging Face Llama implementation:
    the head dimension is split into two halves rather than interleaved pairs.
    """

    inv_freq: torch.Tensor

    def __init__(self, head_dim: int, base: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` tables of shape ``positions.shape + (head_dim,)``.

        Positions may be ``[seq_len]`` (one sequence) or ``[batch, seq_len]``
        (each row at its own position, as in batched decode).
        """
        freqs = positions.float().unsqueeze(-1) * self.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate query/key tensors by their positions.

    Args:
        q: Query tensor of shape ``[batch, num_heads, seq_len, head_dim]``.
        k: Key tensor of shape ``[batch, num_kv_heads, seq_len, head_dim]``.
        cos: Cosine table, ``[seq_len, head_dim]`` shared across the batch or
            ``[batch, seq_len, head_dim]`` with per-row positions (float32).
        sin: Sine table, same shape as ``cos``.
    """
    if cos.dim() == 2:
        cos = cos.to(q.dtype)[None, None, :, :]
        sin = sin.to(q.dtype)[None, None, :, :]
    else:
        cos = cos.to(q.dtype).unsqueeze(1)
        sin = sin.to(q.dtype).unsqueeze(1)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


class GatedMLP(nn.Module):
    """SwiGLU feed-forward block (Shazeer, 2020): ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated feed-forward transformation."""
        out: torch.Tensor = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return out
