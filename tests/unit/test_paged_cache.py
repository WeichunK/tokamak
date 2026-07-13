"""Paged cache correctness: storage roundtrips and equivalence with the contiguous
backend on tiny random-weight models, including physically scattered blocks."""

import pytest
import torch

from tokamak.config import ModelConfig
from tokamak.memory import BlockManager, OutOfBlocksError, PagedKVCache, PagedKVCacheView
from tokamak.model.kv_cache import ContiguousKVCache
from tokamak.model.step_context import BatchedDecodeContext, PrefillContext
from tokamak.model.transformer import TransformerForCausalLM

CPU = torch.device("cpu")


def tiny_config() -> ModelConfig:
    return ModelConfig(
        architecture="Qwen3ForCausalLM",
        vocab_size=128,
        hidden_size=32,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=16,
        intermediate_size=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        attention_bias=False,
        use_qk_norm=True,
        eos_token_ids=(0,),
    )


def make_paged(
    config: ModelConfig, num_blocks: int = 16, block_size: int = 4
) -> tuple[PagedKVCache, BlockManager]:
    cache = PagedKVCache(config, num_blocks, block_size, device=CPU, dtype=torch.float32)
    return cache, BlockManager(num_blocks, block_size)


def test_write_gather_roundtrip_with_scattered_table() -> None:
    """Slot addressing must be exact even when blocks are physically out of order."""
    config = tiny_config()
    cache, _ = make_paged(config, num_blocks=8, block_size=4)
    heads, dim = config.num_kv_heads, config.head_dim

    torch.manual_seed(0)
    k = torch.randn(1, heads, 10, dim)
    v = torch.randn(1, heads, 10, dim)
    table = torch.tensor([5, 2, 7])  # deliberately non-monotonic

    cache.write(0, k, v, table, start_pos=0)
    k_out, v_out = cache.gather(0, table, seq_len=10)

    torch.testing.assert_close(k_out, k)
    torch.testing.assert_close(v_out, v)


def test_incremental_writes_reach_correct_slots() -> None:
    config = tiny_config()
    cache, _ = make_paged(config, num_blocks=8, block_size=4)
    heads, dim = config.num_kv_heads, config.head_dim
    table = torch.tensor([3, 0])

    torch.manual_seed(1)
    k_all = torch.randn(1, heads, 6, dim)
    v_all = torch.randn(1, heads, 6, dim)
    # Prefill 5, then decode one token that crosses into block 0's second slot.
    cache.write(1, k_all[:, :, :5], v_all[:, :, :5], table, start_pos=0)
    cache.write(1, k_all[:, :, 5:], v_all[:, :, 5:], table, start_pos=5)

    k_out, v_out = cache.gather(1, table, seq_len=6)
    torch.testing.assert_close(k_out, k_all)
    torch.testing.assert_close(v_out, v_all)


def test_view_requires_capacity() -> None:
    config = tiny_config()
    cache, manager = make_paged(config, block_size=4)
    view = PagedKVCacheView(cache, manager, seq_id=0)
    k = torch.randn(1, config.num_kv_heads, 5, config.head_dim)

    with pytest.raises(OutOfBlocksError, match="ensure_capacity"):
        view.update(0, k, k, start_pos=0)

    view.ensure_capacity(5)
    view.update(0, k, k, start_pos=0)  # now fits


def test_view_rejects_batches() -> None:
    config = tiny_config()
    cache, manager = make_paged(config)
    view = PagedKVCacheView(cache, manager, seq_id=0)
    view.ensure_capacity(4)
    k = torch.randn(2, config.num_kv_heads, 2, config.head_dim)
    with pytest.raises(NotImplementedError, match="batch size 1"):
        view.update(0, k, k, start_pos=0)


def test_view_release_returns_blocks() -> None:
    config = tiny_config()
    cache, manager = make_paged(config, num_blocks=4, block_size=4)
    view = PagedKVCacheView(cache, manager, seq_id=0)
    view.ensure_capacity(16)
    assert manager.num_free_blocks == 0
    view.release()
    assert manager.num_free_blocks == 4


@torch.inference_mode()
def test_paged_matches_contiguous_on_tiny_model() -> None:
    """M2 exit criterion: identical logits from both backends, prefill and decode."""
    config = tiny_config()
    torch.manual_seed(0)
    model = TransformerForCausalLM(config).eval()
    generator = torch.Generator().manual_seed(2)
    token_ids = torch.randint(0, config.vocab_size, (1, 13), generator=generator)

    contiguous = ContiguousKVCache(config, max_seq_len=13, device=CPU, dtype=torch.float32)
    paged_cache, manager = make_paged(config, num_blocks=8, block_size=4)
    paged = PagedKVCacheView(paged_cache, manager, seq_id=0)

    logits: dict[str, list[torch.Tensor]] = {"contiguous": [], "paged": []}
    for name, cache in (("contiguous", contiguous), ("paged", paged)):
        cache.ensure_capacity(6)
        hidden = model(token_ids[:, :6], torch.arange(6)[None], PrefillContext(cache))
        logits[name].append(model.compute_logits(hidden))
        for pos in range(6, 13):  # decode crosses block boundaries at 8 and 12
            cache.ensure_capacity(pos + 1)
            ctx = BatchedDecodeContext([cache], [pos + 1], CPU)
            hidden = model(token_ids[:, pos : pos + 1], torch.tensor([[pos]]), ctx)
            logits[name].append(model.compute_logits(hidden))

    torch.testing.assert_close(
        torch.cat(logits["paged"], dim=1),
        torch.cat(logits["contiguous"], dim=1),
    )


@torch.inference_mode()
def test_paged_matches_contiguous_with_fragmented_pool() -> None:
    """Equivalence must hold when the sequence's blocks are physically scattered."""
    config = tiny_config()
    torch.manual_seed(0)
    model = TransformerForCausalLM(config).eval()
    generator = torch.Generator().manual_seed(3)
    token_ids = torch.randint(0, config.vocab_size, (1, 12), generator=generator)

    paged_cache, manager = make_paged(config, num_blocks=8, block_size=4)
    # Fragment the pool: an earlier sequence grabs blocks, interleaved with ours,
    # then frees them, so our table ends up physically non-contiguous.
    manager.ensure_capacity(99, 8)
    view = PagedKVCacheView(paged_cache, manager, seq_id=0)
    view.ensure_capacity(4)
    manager.free(99)
    view.ensure_capacity(12)
    assert manager.block_table(0) != sorted(manager.block_table(0))

    contiguous = ContiguousKVCache(config, max_seq_len=12, device=CPU, dtype=torch.float32)
    positions = torch.arange(12)[None]
    hidden_paged = model(token_ids, positions, PrefillContext(view))
    hidden_contig = model(token_ids, positions, PrefillContext(contiguous))

    torch.testing.assert_close(
        model.compute_logits(hidden_paged), model.compute_logits(hidden_contig)
    )
