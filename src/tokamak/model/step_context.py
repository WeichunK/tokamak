"""Per-step attention contexts: each step's storage layout and attention math.

The engine builds a context per step; attention layers hand it rotated Q/K/V and
receive attention outputs. Owning the whole (store → attend) pipeline is what
makes contexts the kernel seam: the reference contexts here express attention
through SDPA, and kernel-backed contexts (``tokamak.kernels``) replace both the
storage traffic and the attention math without the model noticing.

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
import torch.nn.functional as F  # noqa: N812

if TYPE_CHECKING:
    from collections.abc import Sequence as AbcSequence

    from tokamak.model.kv_cache import KVCacheProtocol


class StepContextProtocol(Protocol):
    """What attention layers require from a step context."""

    def attend(
        self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Store this step's K/V and attend over everything each row may see.

        Args:
            layer_idx: Decoder layer index.
            q: Rotated queries of shape ``[batch, num_q_heads, seq_len, head_dim]``.
            k: Rotated keys of shape ``[batch, num_kv_heads, seq_len, head_dim]``.
            v: Values, same shape as ``k``.

        Returns:
            Attention output of shape ``[batch, num_q_heads, seq_len, head_dim]``.
        """
        ...


class PrefillContext:
    """One sequence's prompt (or recomputation) starting at position 0."""

    def __init__(self, cache: KVCacheProtocol) -> None:
        self._cache = cache

    def attend(
        self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Write the full prompt, then run square causal attention over it.

        SDPA's ``is_causal`` aligns the mask to the top-left corner, which is
        correct exactly because prefill queries and keys have equal length.
        """
        k_all, v_all = self._cache.update(layer_idx, k, v, start_pos=0)
        return F.scaled_dot_product_attention(
            q, k_all, v_all, is_causal=q.shape[2] > 1, enable_gqa=True
        )


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

    def attend(
        self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Write each row's token, gather padded histories, mask, and attend.

        Causality is implied — only past positions exist in each row's cache —
        so the only mask needed is the padding-length mask.
        """
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
        return F.scaled_dot_product_attention(
            q, k_pad, v_pad, attn_mask=self.attn_mask, enable_gqa=True
        )
