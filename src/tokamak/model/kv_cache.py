"""Contiguous per-sequence KV cache — the M1 baseline.

One preallocated ``[num_layers, batch, num_kv_heads, max_seq_len, head_dim]`` buffer
per K and V. Simple and fast for a single sequence, but every sequence reserves its
worst-case length up front; the paged cache in M2 exists to remove exactly this
restriction. Memory cost per token is::

    2 * num_layers * num_kv_heads * head_dim * dtype_bytes

which for Qwen3-0.6B in bf16 is 2 * 28 * 8 * 128 * 2 = 112 KiB per token.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tokamak.config import ModelConfig


class KVCache:
    """Preallocated contiguous KV cache for a static batch of equal-length sequences.

    The cache is indexed by absolute position: callers write new keys/values at
    ``[start_pos, start_pos + seq_len)`` and receive views over ``[0, start_pos +
    seq_len)`` for attention. The engine drives positions strictly left to right
    (prefill once, then one token per decode step).
    """

    def __init__(
        self,
        config: ModelConfig,
        max_seq_len: int,
        *,
        batch_size: int = 1,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        shape = (
            config.num_layers,
            batch_size,
            config.num_kv_heads,
            max_seq_len,
            config.head_dim,
        )
        self.max_seq_len = max_seq_len
        self.k_cache = torch.zeros(shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(shape, device=device, dtype=dtype)

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write new keys/values for one layer and return the filled prefix.

        Args:
            layer_idx: Decoder layer index.
            k: New keys of shape ``[batch, num_kv_heads, seq_len, head_dim]``.
            v: New values, same shape as ``k``.
            start_pos: Absolute position of the first new token.

        Returns:
            ``(k, v)`` views of shape ``[batch, num_kv_heads, start_pos + seq_len,
            head_dim]`` covering every cached position including the new ones.
        """
        seq_len = k.shape[2]
        end_pos = start_pos + seq_len
        if end_pos > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: writing positions [{start_pos}, {end_pos}) "
                f"into a cache of length {self.max_seq_len}"
            )
        self.k_cache[layer_idx, :, :, start_pos:end_pos] = k
        self.v_cache[layer_idx, :, :, start_pos:end_pos] = v
        return (
            self.k_cache[layer_idx, :, :, :end_pos],
            self.v_cache[layer_idx, :, :, :end_pos],
        )
