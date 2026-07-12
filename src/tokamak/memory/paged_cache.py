"""Paged KV cache storage and the per-sequence view attention layers consume.

``PagedKVCache`` owns the physical block pool: one preallocated tensor per K and V,
shaped ``[num_layers, num_blocks, block_size, num_kv_heads, head_dim]``. Which
blocks belong to which sequence is entirely the
:class:`~tokamak.memory.block_manager.BlockManager`'s business — this module only
translates (block table, position) into physical slots.

Reads use a gather: the sequence's blocks are copied into a contiguous ``[1, heads,
seq_len, head_dim]`` tensor and handed to SDPA. That copy per layer per step is the
deliberate cost of the *reference* implementation — it makes correctness trivial to
audit and test. The M4 Triton kernel removes it by reading block tables inside the
attention kernel, which is exactly what vLLM's PagedAttention kernel does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tokamak.memory.block_manager import OutOfBlocksError

if TYPE_CHECKING:
    from tokamak.config import ModelConfig
    from tokamak.memory.block_manager import BlockManager


class PagedKVCache:
    """Physical block-pool storage for keys and values, shared by all sequences."""

    def __init__(
        self,
        config: ModelConfig,
        num_blocks: int,
        block_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = device
        shape = (
            config.num_layers,
            num_blocks,
            block_size,
            config.num_kv_heads,
            config.head_dim,
        )
        self.k_cache = torch.zeros(shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(shape, device=device, dtype=dtype)

    def write(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        block_table: torch.Tensor,
        start_pos: int,
    ) -> None:
        """Scatter new keys/values into the sequence's blocks.

        Args:
            layer_idx: Decoder layer index.
            k: New keys of shape ``[1, num_kv_heads, seq_len, head_dim]``.
            v: New values, same shape as ``k``.
            block_table: Long tensor of block ids; must already cover positions
                ``[start_pos, start_pos + seq_len)``.
            start_pos: Absolute position of the first new token.
        """
        seq_len = k.shape[2]
        positions = torch.arange(start_pos, start_pos + seq_len, device=self.device)
        slots = block_table[positions // self.block_size] * self.block_size + (
            positions % self.block_size
        )
        # [num_blocks, block_size, heads, dim] flattened to slot-indexed rows.
        k_flat = self.k_cache[layer_idx].flatten(0, 1)
        v_flat = self.v_cache[layer_idx].flatten(0, 1)
        k_flat[slots] = k[0].permute(1, 0, 2)  # [seq_len, heads, dim]
        v_flat[slots] = v[0].permute(1, 0, 2)

    def gather(
        self,
        layer_idx: int,
        block_table: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize the sequence's cached K/V as contiguous tensors.

        Returns:
            ``(k, v)`` of shape ``[1, num_kv_heads, seq_len, head_dim]``. These are
            copies — see the module docstring for why the reference implementation
            accepts this cost.
        """
        k = self.k_cache[layer_idx, block_table].flatten(0, 1)[:seq_len]
        v = self.v_cache[layer_idx, block_table].flatten(0, 1)[:seq_len]
        return (
            k.permute(1, 0, 2).unsqueeze(0),
            v.permute(1, 0, 2).unsqueeze(0),
        )


class PagedKVCacheView:
    """Per-sequence facade over the shared pool, satisfying ``KVCacheProtocol``.

    Attention layers call :meth:`update` exactly as they would on a contiguous
    cache; the view resolves the sequence's block table on every call, so tables
    grown between steps by :meth:`ensure_capacity` are picked up automatically.
    """

    def __init__(self, cache: PagedKVCache, manager: BlockManager, seq_id: int) -> None:
        self._cache = cache
        self._manager = manager
        self._seq_id = seq_id
        self._table_tensor = torch.empty(0, dtype=torch.long, device=cache.device)

    def ensure_capacity(self, num_tokens: int) -> None:
        """Allocate blocks from the pool so ``num_tokens`` tokens fit."""
        self._manager.ensure_capacity(self._seq_id, num_tokens)

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write new keys/values through the block table and gather all positions."""
        if k.shape[0] != 1:
            raise NotImplementedError(
                "the paged KV backend supports batch size 1 until continuous batching lands (M3)"
            )
        seq_len = start_pos + k.shape[2]
        table = self._manager.block_table(self._seq_id)
        if len(table) * self._cache.block_size < seq_len:
            raise OutOfBlocksError(
                f"sequence {self._seq_id}: block table covers "
                f"{len(table) * self._cache.block_size} tokens but {seq_len} are "
                f"required; call ensure_capacity() before the forward pass"
            )
        if len(table) != self._table_tensor.shape[0]:
            self._table_tensor = torch.tensor(table, dtype=torch.long, device=self._cache.device)
        self._cache.write(layer_idx, k, v, self._table_tensor, start_pos)
        return self._cache.gather(layer_idx, self._table_tensor, seq_len)

    def release(self) -> None:
        """Return this sequence's blocks to the pool."""
        self._manager.free(self._seq_id)
