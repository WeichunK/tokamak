"""Attention policies: parsing, index arithmetic, and masked-context math.

The math tests compare policy-masked contexts against a brute-force reference:
attention computed over the full history with an explicit visibility mask built
directly from the policy's definition (``[0, sinks) plus [band_start(pos), pos]``).
"""

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from tokamak.config import ModelConfig
from tokamak.model.attention_policy import FULL_ATTENTION, AttentionPolicy
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


def visibility_mask(policy: AttentionPolicy, positions: list[int], kv_len: int) -> torch.Tensor:
    """[len(positions), kv_len] bool mask straight from the policy definition."""
    rows = []
    for pos in positions:
        cols = torch.arange(kv_len)
        causal = cols <= pos
        if policy.is_full:
            rows.append(causal)
        else:
            band = cols >= policy.band_start(pos)
            sink = cols < policy.sinks
            rows.append(causal & (band | sink))
    return torch.stack(rows)


class TestParsing:
    def test_full(self) -> None:
        assert AttentionPolicy.parse("full") == AttentionPolicy()
        assert AttentionPolicy.parse("full").is_full

    def test_window(self) -> None:
        assert AttentionPolicy.parse("window:512") == AttentionPolicy(window=512)

    def test_streaming(self) -> None:
        assert AttentionPolicy.parse("streaming:512+4") == AttentionPolicy(window=512, sinks=4)

    def test_instance_passthrough(self) -> None:
        policy = AttentionPolicy(window=8, sinks=2)
        assert AttentionPolicy.parse(policy) is policy

    @pytest.mark.parametrize(
        "spec", ["", "window", "window:", "window:abc", "streaming:512", "streaming:a+b", "band:4"]
    )
    def test_rejects_malformed_specs(self, spec: str) -> None:
        with pytest.raises(ValueError, match="attention_policy"):
            AttentionPolicy.parse(spec)

    def test_rejects_invalid_values(self) -> None:
        with pytest.raises(ValueError, match="window"):
            AttentionPolicy(window=0)
        with pytest.raises(ValueError, match="sinks"):
            AttentionPolicy(window=4, sinks=-1)
        with pytest.raises(ValueError, match="sinks require a window"):
            AttentionPolicy(sinks=2)


class TestBandStart:
    def test_full_sees_everything(self) -> None:
        assert FULL_ATTENTION.band_start(100) == 0

    def test_window_slides(self) -> None:
        policy = AttentionPolicy(window=4)
        assert policy.band_start(2) == 0  # window not yet full
        assert policy.band_start(3) == 0
        assert policy.band_start(10) == 7  # sees {7, 8, 9, 10}

    def test_band_clipped_to_sinks(self) -> None:
        policy = AttentionPolicy(window=8, sinks=2)
        # pos - window + 1 < sinks: the band never dips into the sink prefix,
        # keeping the two pieces of the visible set disjoint.
        assert policy.band_start(4) == 2


class TestPrefillContext:
    def prefill_out(self, policy: AttentionPolicy, seq_len: int = 7) -> torch.Tensor:
        config = tiny_config()
        torch.manual_seed(0)
        q = torch.randn(1, 4, seq_len, 8)
        k = torch.randn(1, 2, seq_len, 8)
        v = torch.randn(1, 2, seq_len, 8)
        out = PrefillContext(make_cache(config), policy=policy).attend(0, q, k, v)
        expected_mask = visibility_mask(policy, list(range(seq_len)), seq_len)
        expected = F.scaled_dot_product_attention(
            q, k, v, attn_mask=expected_mask[None, None], enable_gqa=True
        )
        torch.testing.assert_close(out, expected)
        return out

    def test_windowed_matches_masked_reference(self) -> None:
        self.prefill_out(AttentionPolicy(window=3))

    def test_streaming_matches_masked_reference(self) -> None:
        self.prefill_out(AttentionPolicy(window=3, sinks=2))

    def test_wide_window_equals_full(self) -> None:
        full = self.prefill_out(FULL_ATTENTION)
        wide = self.prefill_out(AttentionPolicy(window=32, sinks=1))
        torch.testing.assert_close(full, wide)

    def test_windowed_chunked_prefill(self) -> None:
        """A mid-cache chunk masks against absolute positions, not chunk-local."""
        config = tiny_config()
        cache = make_cache(config)
        torch.manual_seed(1)
        k_hist = torch.randn(1, 2, 4, 8)
        v_hist = torch.randn(1, 2, 4, 8)
        cache.update(0, k_hist, v_hist, start_pos=0)

        policy = AttentionPolicy(window=3, sinks=1)
        q = torch.randn(1, 4, 2, 8)
        k_new = torch.randn(1, 2, 2, 8)
        v_new = torch.randn(1, 2, 2, 8)
        out = PrefillContext(cache, start_pos=4, policy=policy).attend(0, q, k_new, v_new)

        k_full = torch.cat([k_hist, k_new], dim=2)
        v_full = torch.cat([v_hist, v_new], dim=2)
        mask = visibility_mask(policy, [4, 5], 6)
        expected = F.scaled_dot_product_attention(
            q, k_full, v_full, attn_mask=mask[None, None], enable_gqa=True
        )
        torch.testing.assert_close(out, expected)


class TestBatchedDecodeContext:
    def test_policy_matches_per_row_masked_reference(self) -> None:
        config = tiny_config()
        policy = AttentionPolicy(window=3, sinks=1)
        torch.manual_seed(2)
        caches = [make_cache(config), make_cache(config)]
        histories = []
        for cache, prior in zip(caches, (2, 6), strict=True):
            k_hist = torch.randn(1, 2, prior, 8)
            v_hist = torch.randn(1, 2, prior, 8)
            cache.update(0, k_hist, v_hist, start_pos=0)
            histories.append((k_hist, v_hist))

        seq_lens = [3, 7]
        ctx = BatchedDecodeContext(caches, seq_lens=seq_lens, device=CPU, policy=policy)
        q = torch.randn(2, 4, 1, 8)
        k_new = torch.randn(2, 2, 1, 8)
        v_new = torch.randn(2, 2, 1, 8)
        out = ctx.attend(0, q, k_new, v_new)

        for i, (k_hist, v_hist) in enumerate(histories):
            k_full = torch.cat([k_hist, k_new[i : i + 1]], dim=2)
            v_full = torch.cat([v_hist, v_new[i : i + 1]], dim=2)
            mask = visibility_mask(policy, [seq_lens[i] - 1], seq_lens[i])
            expected = F.scaled_dot_product_attention(
                q[i : i + 1], k_full, v_full, attn_mask=mask[None, None], enable_gqa=True
            )
            torch.testing.assert_close(out[i : i + 1], expected, rtol=1e-5, atol=1e-6)

    def test_wide_window_equals_full(self) -> None:
        config = tiny_config()
        torch.manual_seed(3)
        outs = []
        for policy in (FULL_ATTENTION, AttentionPolicy(window=32, sinks=2)):
            torch.manual_seed(3)
            cache = make_cache(config)
            k_hist = torch.randn(1, 2, 5, 8)
            v_hist = torch.randn(1, 2, 5, 8)
            cache.update(0, k_hist, v_hist, start_pos=0)
            ctx = BatchedDecodeContext([cache], seq_lens=[6], device=CPU, policy=policy)
            q = torch.randn(1, 4, 1, 8)
            k_new = torch.randn(1, 2, 1, 8)
            v_new = torch.randn(1, 2, 1, 8)
            outs.append(ctx.attend(0, q, k_new, v_new))
        torch.testing.assert_close(outs[0], outs[1])
