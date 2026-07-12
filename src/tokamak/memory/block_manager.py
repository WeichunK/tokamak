"""Fixed-size KV block allocation, in the style of vLLM's PagedAttention.

The physical KV cache is carved into blocks of ``block_size`` tokens. Sequences
never own contiguous memory; they own a *block table* — an ordered list of block
ids — that maps logical token positions to physical blocks:

    logical position p  →  block_table[p // block_size], slot p % block_size

Blocks are allocated on demand as a sequence grows and returned to the pool when
it finishes, so the reserved-but-unused memory per sequence is bounded by one
partially filled block instead of ``max_new_tokens`` worth of headroom.

This manager is deliberately minimal for M2: no block sharing between sequences
(prefix caching) and no copy-on-write; both belong to later milestones and would
add reference counting here.
"""

from __future__ import annotations


class OutOfBlocksError(RuntimeError):
    """The block pool cannot satisfy an allocation.

    With a single running sequence (M2) this is a hard error; the M3 scheduler
    turns it into a preemption signal.
    """


class BlockManager:
    """Allocates fixed-size KV blocks from a bounded pool to sequences.

    Args:
        num_blocks: Total blocks in the pool. Physical storage of this size is
            preallocated by :class:`~tokamak.memory.paged_cache.PagedKVCache`.
        block_size: Tokens per block. Smaller blocks waste less memory per
            sequence (at most ``block_size - 1`` slack tokens) but mean longer
            block tables and more gather indirection.
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
        if block_size <= 0:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        self.num_blocks = num_blocks
        self.block_size = block_size
        # LIFO free list: recently freed (cache-warm) blocks are reused first.
        self._free_blocks = list(range(num_blocks - 1, -1, -1))
        self._block_tables: dict[int, list[int]] = {}

    @property
    def num_free_blocks(self) -> int:
        """Blocks currently available for allocation."""
        return len(self._free_blocks)

    def blocks_needed(self, num_tokens: int) -> int:
        """Number of blocks required to store ``num_tokens`` tokens."""
        if num_tokens < 0:
            raise ValueError(f"num_tokens must be >= 0, got {num_tokens}")
        return -(-num_tokens // self.block_size)

    def block_table(self, seq_id: int) -> list[int]:
        """The sequence's current block table (treat as read-only).

        Unknown sequences have an empty table; entries are appended by
        :meth:`ensure_capacity` and never reordered, so previously written
        positions stay valid as the table grows.
        """
        return self._block_tables.get(seq_id, [])

    def ensure_capacity(self, seq_id: int, num_tokens: int) -> None:
        """Grow the sequence's block table to cover ``num_tokens`` tokens.

        Idempotent: existing capacity is kept, only the shortfall is allocated.

        Raises:
            OutOfBlocksError: If the pool has fewer free blocks than needed.
        """
        table = self._block_tables.setdefault(seq_id, [])
        shortfall = self.blocks_needed(num_tokens) - len(table)
        if shortfall <= 0:
            return
        if shortfall > len(self._free_blocks):
            raise OutOfBlocksError(
                f"sequence {seq_id} needs {shortfall} more block(s) for "
                f"{num_tokens} tokens, but only {len(self._free_blocks)} of "
                f"{self.num_blocks} are free"
            )
        for _ in range(shortfall):
            table.append(self._free_blocks.pop())

    def free(self, seq_id: int) -> None:
        """Return the sequence's blocks to the pool (no-op for unknown ids)."""
        table = self._block_tables.pop(seq_id, None)
        if table:
            self._free_blocks.extend(table)

    def reserved_tokens(self, seq_id: int) -> int:
        """Token capacity currently reserved by the sequence's blocks."""
        return len(self.block_table(seq_id)) * self.block_size
