import pytest

from tokamak.memory import BlockManager, OutOfBlocksError


def test_blocks_needed_rounds_up() -> None:
    manager = BlockManager(num_blocks=8, block_size=4)
    assert manager.blocks_needed(0) == 0
    assert manager.blocks_needed(1) == 1
    assert manager.blocks_needed(4) == 1
    assert manager.blocks_needed(5) == 2


def test_invalid_construction() -> None:
    with pytest.raises(ValueError):
        BlockManager(num_blocks=0, block_size=4)
    with pytest.raises(ValueError):
        BlockManager(num_blocks=4, block_size=0)


def test_ensure_capacity_allocates_on_demand() -> None:
    manager = BlockManager(num_blocks=8, block_size=4)
    manager.ensure_capacity(seq_id=0, num_tokens=5)
    assert len(manager.block_table(0)) == 2
    assert manager.num_free_blocks == 6
    assert manager.reserved_tokens(0) == 8


def test_ensure_capacity_is_idempotent_and_grows_stably() -> None:
    manager = BlockManager(num_blocks=8, block_size=4)
    manager.ensure_capacity(0, 5)
    table_before = list(manager.block_table(0))

    manager.ensure_capacity(0, 5)  # same capacity: no change
    assert list(manager.block_table(0)) == table_before

    manager.ensure_capacity(0, 12)  # growth appends, never reorders
    table_after = manager.block_table(0)
    assert table_after[: len(table_before)] == table_before
    assert len(table_after) == 3


def test_sequences_get_disjoint_blocks() -> None:
    manager = BlockManager(num_blocks=8, block_size=4)
    manager.ensure_capacity(0, 8)
    manager.ensure_capacity(1, 8)
    assert not set(manager.block_table(0)) & set(manager.block_table(1))


def test_exhaustion_raises_and_leaves_state_intact() -> None:
    manager = BlockManager(num_blocks=4, block_size=4)
    manager.ensure_capacity(0, 12)  # 3 blocks
    with pytest.raises(OutOfBlocksError, match="1 of 4 are free"):
        manager.ensure_capacity(1, 8)  # needs 2, only 1 free
    assert manager.num_free_blocks == 1
    assert manager.block_table(1) == []


def test_free_returns_blocks_and_is_idempotent() -> None:
    manager = BlockManager(num_blocks=4, block_size=4)
    manager.ensure_capacity(0, 16)
    assert manager.num_free_blocks == 0

    manager.free(0)
    assert manager.num_free_blocks == 4
    assert manager.block_table(0) == []

    manager.free(0)  # unknown/already-freed: no-op
    manager.free(42)
    assert manager.num_free_blocks == 4


def test_freed_blocks_are_reusable() -> None:
    manager = BlockManager(num_blocks=2, block_size=4)
    manager.ensure_capacity(0, 8)
    manager.free(0)
    manager.ensure_capacity(1, 8)  # would raise if blocks leaked
    assert len(manager.block_table(1)) == 2


def test_waste_is_bounded_by_one_block() -> None:
    """The M2 exit criterion at the allocator level: slack < block_size."""
    manager = BlockManager(num_blocks=64, block_size=16)
    for seq_id, num_tokens in enumerate([1, 15, 16, 17, 100, 255]):
        manager.ensure_capacity(seq_id, num_tokens)
        slack = manager.reserved_tokens(seq_id) - num_tokens
        assert 0 <= slack < 16, f"{num_tokens} tokens wasted {slack}"
