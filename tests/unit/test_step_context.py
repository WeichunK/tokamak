"""Step contexts: storage dispatch plus attention math against manual references."""

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

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
        num_attention_heads=4,
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


def test_prefill_attend_matches_manual_causal_sdpa() -> None:
    config = tiny_config()
    cache = make_cache(config)
    torch.manual_seed(0)
    q = torch.randn(1, 4, 5, 8)
    k = torch.randn(1, 2, 5, 8)
    v = torch.randn(1, 2, 5, 8)

    out = PrefillContext(cache).attend(0, q, k, v)

    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    torch.testing.assert_close(out, expected)
    # And the tokens were actually cached.
    torch.testing.assert_close(cache.k_cache[0, :, :, :5], k)
    torch.testing.assert_close(cache.v_cache[0, :, :, :5], v)


def test_batched_decode_attend_matches_per_row_sdpa() -> None:
    """Padding and masking must not change any row's attention output."""
    config = tiny_config()
    torch.manual_seed(1)
    caches = [make_cache(config), make_cache(config)]
    histories = []
    # Histories of different lengths: 2 and 5 prior tokens.
    for cache, prior in zip(caches, (2, 5), strict=True):
        k_hist = torch.randn(1, 2, prior, 8)
        v_hist = torch.randn(1, 2, prior, 8)
        cache.update(0, k_hist, v_hist, start_pos=0)
        histories.append((k_hist, v_hist))

    seq_lens = [3, 6]
    ctx = BatchedDecodeContext(caches, seq_lens=seq_lens, device=CPU)
    q = torch.randn(2, 4, 1, 8)
    k_new = torch.randn(2, 2, 1, 8)
    v_new = torch.randn(2, 2, 1, 8)

    out = ctx.attend(0, q, k_new, v_new)

    assert out.shape == (2, 4, 1, 8)
    for i, (k_hist, v_hist) in enumerate(histories):
        k_full = torch.cat([k_hist, k_new[i : i + 1]], dim=2)
        v_full = torch.cat([v_hist, v_new[i : i + 1]], dim=2)
        expected = F.scaled_dot_product_attention(q[i : i + 1], k_full, v_full, enable_gqa=True)
        torch.testing.assert_close(out[i : i + 1], expected, rtol=1e-5, atol=1e-6)


def test_batched_decode_rejects_multi_token_rows() -> None:
    config = tiny_config()
    ctx = BatchedDecodeContext([make_cache(config)], [4], CPU)
    q = torch.randn(1, 4, 2, 8)
    k = torch.randn(1, 2, 2, 8)  # two tokens in a decode step
    with pytest.raises(ValueError, match="exactly 1 token"):
        ctx.attend(0, q, k, k)


def test_batched_decode_validates_row_count() -> None:
    config = tiny_config()
    ctx = BatchedDecodeContext([make_cache(config)], [4], CPU)
    q = torch.randn(2, 4, 1, 8)
    k = torch.randn(2, 2, 1, 8)  # two rows, one cache
    with pytest.raises(ValueError, match="2 rows"):
        ctx.attend(0, q, k, k)

    with pytest.raises(ValueError, match="caches for"):
        BatchedDecodeContext([make_cache(config)], [4, 5], CPU)
