"""Per-step attention contexts: how one forward pass sees KV storage.

The engine builds a context per step and the attention layers consume it, which is
what lets one model implementation serve both phases of continuous batching:

- :class:`PrefillContext` — one sequence writing positions ``[0, seq_len)``;
  square causal attention, exactly the M1 code path.
- :class:`BatchedDecodeContext` — B sequences, each contributing one token at its
  own position and attending over its own history. Histories have different
  lengths, so the reference implementation right-pads gathered K/V to the batch
  maximum and masks the padding. Rows are mathematically independent; padding
  only buys one SDPA call instead of B.

Both work over any per-sequence :class:`~tokamak.model.kv_cache.KVCacheProtocol`
implementation (contiguous or paged) — batching strategy and storage layout are
independent axes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence as AbcSequence

    from tokamak.model.kv_cache import KVCacheProtocol


class StepContextProtocol(Protocol):
    """What attention layers require from a step context."""

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Store this step's keys/values and return what attention should see.

        Args:
            layer_idx: Decoder layer index.
            k: New keys of shape ``[batch, num_kv_heads, seq_len, head_dim]``.
            v: New values, same shape as ``k``.

        Returns:
            ``(k_all, v_all, attn_mask)``. When ``attn_mask`` is ``None`` the
            attention matrix is square and causal masking applies; otherwise it
            is a boolean mask (``True`` = attend) broadcastable to
            ``[batch, heads, q_len, kv_len]``.
        """
        ...


class PrefillContext:
    """One sequence's prompt (or recomputation) starting at position 0."""

    def __init__(self, cache: KVCacheProtocol) -> None:
        self._cache = cache

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Write the full prompt; attention over it is square and causal."""
        k_all, v_all = self._cache.update(layer_idx, k, v, start_pos=0)
        return k_all, v_all, None


class BatchedDecodeContext:
    """One decode step for a batch of sequences at heterogeneous positions.

    Args:
        caches: One per-sequence cache per batch row, in row order.
        seq_lens: Per-row total length *including* the token being decoded;
            row ``i``'s new token is written at position ``seq_lens[i] - 1``.
        device: Device for the padding mask.
    """

    def __init__(
        self,
        caches: AbcSequence[KVCacheProtocol],
        seq_lens: AbcSequence[int],
        device: torch.device,
    ) -> None:
        if len(caches) != len(seq_lens):
            raise ValueError(f"{len(caches)} caches for {len(seq_lens)} seq_lens")
        self._caches = list(caches)
        self._seq_lens = list(seq_lens)
        self.max_len = max(self._seq_lens)
        lens = torch.tensor(self._seq_lens, device=device)
        cols = torch.arange(self.max_len, device=device)
        # [batch, 1, 1, max_len]: True where the column is a real position.
        self.attn_mask = (cols[None, :] < lens[:, None])[:, None, None, :]

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Write each row's token at its own position; gather padded histories."""
        batch, num_kv_heads, seq_len, head_dim = k.shape
        if seq_len != 1:
            raise ValueError(f"decode steps carry exactly 1 token per row, got {seq_len}")
        if batch != len(self._caches):
            raise ValueError(f"batch of {batch} rows for {len(self._caches)} caches")

        k_pad = k.new_zeros(batch, num_kv_heads, self.max_len, head_dim)
        v_pad = k.new_zeros(batch, num_kv_heads, self.max_len, head_dim)
        rows = zip(self._caches, self._seq_lens, strict=True)
        for i, (cache, length) in enumerate(rows):
            k_i, v_i = cache.update(layer_idx, k[i : i + 1], v[i : i + 1], start_pos=length - 1)
            k_pad[i, :, :length] = k_i[0]
            v_pad[i, :, :length] = v_i[0]
        return k_pad, v_pad, self.attn_mask
