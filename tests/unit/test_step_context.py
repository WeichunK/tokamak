"""BatchedDecodeContext: padding, masks, and per-row cache dispatch."""

import pytest
import torch

from tokamak.config import ModelConfig
from tokamak.model.kv_cache import ContiguousKVCache
from tokamak.model.step_context import BatchedDecodeContext, PrefillContext

CPU = torch.device("cpu")


def tiny_config() -> ModelConfig:
    return ModelConfig(
        architecture="LlamaForCausalLM",
        vocab_size=64,
        hidden_size=16,
        num_layers=1,
        num_attention_heads=2,
        num_kv_heads=2,
        head_dim=8,
        intermediate_size=32,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        attention_bias=False,
        use_qk_norm=False,
        eos_token_ids=(0,),
    )


def make_cache(config: ModelConfig) -> ContiguousKVCache:
    return ContiguousKVCache(config, max_seq_len=32, device=CPU, dtype=torch.float32)


def test_prefill_context_is_causal_passthrough() -> None:
    config = tiny_config()
    cache = make_cache(config)
    k = torch.randn(1, 2, 5, 8)
    v = torch.randn(1, 2, 5, 8)

    k_all, v_all, mask = PrefillContext(cache).update(0, k, v)

    assert mask is None  # square causal attention
    torch.testing.assert_close(k_all, k)
    torch.testing.assert_close(v_all, v)


def test_batched_decode_pads_and_masks_by_length() -> None:
    config = tiny_config()
    caches = [make_cache(config), make_cache(config)]
    # Histories of different lengths: 3 and 6 tokens (including the new one).
    for cache, prior in zip(caches, (2, 5), strict=True):
        k_hist = torch.randn(1, 2, prior, 8)
        cache.update(0, k_hist, k_hist, start_pos=0)

    ctx = BatchedDecodeContext(caches, seq_lens=[3, 6], device=CPU)
    k_new = torch.randn(2, 2, 1, 8)
    k_all, _v_all, mask = ctx.update(0, k_new, k_new)

    assert k_all.shape == (2, 2, 6, 8)  # padded to the batch max
    assert mask is not None
    assert mask.shape == (2, 1, 1, 6)
    assert mask[0, 0, 0].tolist() == [True, True, True, False, False, False]
    assert mask[1, 0, 0].tolist() == [True] * 6
    # Each row's new token landed at its own position.
    torch.testing.assert_close(k_all[0, :, 2:3], k_new[0])
    torch.testing.assert_close(k_all[1, :, 5:6], k_new[1])
    # Padding stays zero (masked out anyway).
    torch.testing.assert_close(k_all[0, :, 3:], torch.zeros(2, 3, 8))


def test_batched_decode_rejects_multi_token_rows() -> None:
    config = tiny_config()
    ctx = BatchedDecodeContext([make_cache(config)], [4], CPU)
    k = torch.randn(1, 2, 2, 8)  # two tokens in a decode step
    with pytest.raises(ValueError, match="exactly 1 token"):
        ctx.update(0, k, k)


def test_batched_decode_validates_row_count() -> None:
    config = tiny_config()
    ctx = BatchedDecodeContext([make_cache(config)], [4], CPU)
    k = torch.randn(2, 2, 1, 8)  # two rows, one cache
    with pytest.raises(ValueError, match="2 rows"):
        ctx.update(0, k, k)

    with pytest.raises(ValueError, match="caches for"):
        BatchedDecodeContext([make_cache(config)], [4, 5], CPU)
