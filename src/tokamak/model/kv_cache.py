"""KV cache interface and the contiguous (M1 baseline) implementation.

Attention layers depend only on :class:`KVCacheProtocol`; the engine picks the
concrete backend. Two implementations exist:

- :class:`ContiguousKVCache` (here): one preallocated buffer per sequence, indexed
  by absolute position. Zero-copy reads, but every sequence reserves its worst-case
  length up front.
- ``PagedKVCache`` (:mod:`tokamak.memory`): fixed-size blocks allocated on demand
  from a shared pool, in the style of vLLM's PagedAttention.

Memory cost per cached token is identical for both::

    2 * num_layers * num_kv_heads * head_dim * dtype_bytes

which for Qwen3-0.6B in bf16 is 2 * 28 * 8 * 128 * 2 = 112 KiB per token. What
differs is how much of it sits reserved but unused.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from tokamak.config import ModelConfig


class KVCacheProtocol(Protocol):
    """What attention layers and the engine require from a KV cache backend."""

    def ensure_capacity(self, num_tokens: int) -> None:
        """Guarantee storage for ``num_tokens`` total tokens, or raise."""
        ...

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store new keys/values for one layer and return all cached positions.

        Args:
            layer_idx: Decoder layer index.
            k: New keys of shape ``[batch, num_kv_heads, seq_len, head_dim]``.
            v: New values, same shape as ``k``.
            start_pos: Absolute position of the first new token.

        Returns:
            ``(k, v)`` of shape ``[batch, num_kv_heads, start_pos + seq_len,
            head_dim]`` covering every cached position including the new ones.
        """
        ...

    def release(self) -> None:
        """Return any pooled resources; the cache must not be used afterwards."""
        ...


class ContiguousKVCache:
    """Preallocated contiguous KV cache for a static batch of equal-length sequences.

    The cache is indexed by absolute position: callers write new keys/values at
    ``[start_pos, start_pos + seq_len)`` and receive views over ``[0, start_pos +
    seq_len)`` for attention — reads are zero-copy. The engine drives positions
    strictly left to right (prefill once, then one token per decode step).
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

    def ensure_capacity(self, num_tokens: int) -> None:
        """Validate the request fits the preallocated buffer (no-op otherwise)."""
        if num_tokens > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: {num_tokens} tokens exceed the preallocated "
                f"capacity of {self.max_seq_len}"
            )

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write new keys/values for one layer and return the filled prefix."""
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

    def release(self) -> None:
        """Nothing pooled to return; the buffer dies with the object."""


# Transitional alias so consumers migrate one commit at a time; removed once the
# engine switches to the protocol.
KVCache = ContiguousKVCache
