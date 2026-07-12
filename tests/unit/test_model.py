"""Model-level correctness tests on tiny random-weight configurations.

The key invariant: decoding token-by-token through the KV cache must produce the
same logits as one full forward pass over the whole sequence. This exercises cache
indexing, causal masking, GQA, and RoPE position handling without any checkpoint.
"""

import pytest
import torch

from tokamak.config import ModelConfig
from tokamak.model.kv_cache import KVCache
from tokamak.model.transformer import TransformerForCausalLM


def tiny_config(**overrides: object) -> ModelConfig:
    defaults: dict = {
        "architecture": "Qwen3ForCausalLM",
        "vocab_size": 128,
        "hidden_size": 32,
        "num_layers": 2,
        "num_attention_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 16,
        "intermediate_size": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 64,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "use_qk_norm": True,
        "eos_token_ids": (0,),
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)


def make_model(config: ModelConfig, seed: int = 0) -> TransformerForCausalLM:
    torch.manual_seed(seed)
    return TransformerForCausalLM(config).eval()


def make_cache(config: ModelConfig, max_seq_len: int = 64) -> KVCache:
    return KVCache(
        config,
        max_seq_len=max_seq_len,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


@pytest.mark.parametrize(
    "config",
    [
        tiny_config(),  # Qwen3-style: QK-norm, head_dim != hidden/heads
        tiny_config(
            architecture="LlamaForCausalLM",
            use_qk_norm=False,
            head_dim=8,
        ),  # Llama-style
        tiny_config(
            architecture="Qwen2ForCausalLM",
            use_qk_norm=False,
            attention_bias=True,
        ),  # Qwen2-style: QKV bias
    ],
    ids=["qwen3", "llama", "qwen2"],
)
@torch.inference_mode()
def test_incremental_decode_matches_full_forward(config: ModelConfig) -> None:
    model = make_model(config)
    generator = torch.Generator().manual_seed(1)
    token_ids = torch.randint(0, config.vocab_size, (1, 12), generator=generator)

    # One pass over the full sequence.
    full_hidden = model(token_ids, make_cache(config), start_pos=0)
    full_logits = model.compute_logits(full_hidden)

    # Prefill the first 5 tokens, then decode the rest one token at a time.
    cache = make_cache(config)
    step_logits = []
    hidden = model(token_ids[:, :5], cache, start_pos=0)
    step_logits.append(model.compute_logits(hidden))
    for pos in range(5, 12):
        hidden = model(token_ids[:, pos : pos + 1], cache, start_pos=pos)
        step_logits.append(model.compute_logits(hidden))

    incremental_logits = torch.cat(step_logits, dim=1)
    torch.testing.assert_close(incremental_logits, full_logits, rtol=1e-4, atol=1e-4)


@torch.inference_mode()
def test_logits_causality() -> None:
    """Changing a future token must not change logits at earlier positions."""
    config = tiny_config()
    model = make_model(config)
    generator = torch.Generator().manual_seed(2)
    tokens_a = torch.randint(0, config.vocab_size, (1, 10), generator=generator)
    tokens_b = tokens_a.clone()
    tokens_b[0, -1] = (tokens_b[0, -1] + 1) % config.vocab_size

    logits_a = model.compute_logits(model(tokens_a, make_cache(config), start_pos=0))
    logits_b = model.compute_logits(model(tokens_b, make_cache(config), start_pos=0))

    torch.testing.assert_close(logits_a[:, :-1], logits_b[:, :-1])
    assert not torch.allclose(logits_a[:, -1], logits_b[:, -1])


def test_tied_embeddings_share_storage() -> None:
    config = tiny_config(tie_word_embeddings=True)
    model = make_model(config)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()


def test_kv_cache_overflow_raises() -> None:
    config = tiny_config()
    model = make_model(config)
    cache = make_cache(config, max_seq_len=4)
    tokens = torch.randint(0, config.vocab_size, (1, 5))
    with pytest.raises(ValueError, match="KV cache overflow"):
        model(tokens, cache, start_pos=0)


def test_chunked_prefill_rejected() -> None:
    config = tiny_config()
    model = make_model(config)
    tokens = torch.randint(0, config.vocab_size, (1, 4))
    with pytest.raises(NotImplementedError, match="chunked prefill"):
        model(tokens, make_cache(config), start_pos=2)
