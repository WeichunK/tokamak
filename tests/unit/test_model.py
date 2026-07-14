"""Model-level correctness tests on tiny random-weight configurations.

Two invariants anchor everything:

1. Decoding token-by-token through the KV cache must produce the same logits as
   one full forward pass (cache indexing, causal masking, GQA, RoPE positions).
2. Decoding sequences batched together must produce the same logits as decoding
   them one at a time (padding, masks, per-row positions) — the invariant that
   makes continuous batching safe.
"""

import pytest
import torch

from tokamak.config import ModelConfig
from tokamak.model.kv_cache import ContiguousKVCache
from tokamak.model.step_context import BatchedDecodeContext, PrefillContext
from tokamak.model.transformer import TransformerForCausalLM

CPU = torch.device("cpu")


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


def make_cache(config: ModelConfig, max_seq_len: int = 64) -> ContiguousKVCache:
    return ContiguousKVCache(config, max_seq_len=max_seq_len, device=CPU, dtype=torch.float32)


def full_forward_logits(
    model: TransformerForCausalLM, config: ModelConfig, token_ids: torch.Tensor
) -> torch.Tensor:
    positions = torch.arange(token_ids.shape[1])[None]
    hidden = model(token_ids, positions, PrefillContext(make_cache(config)))
    return model.compute_logits(hidden)


def decode_one(
    model: TransformerForCausalLM,
    cache: ContiguousKVCache,
    token: torch.Tensor,
    position: int,
) -> torch.Tensor:
    """One single-sequence decode step; returns [1, 1, vocab] logits."""
    ctx = BatchedDecodeContext([cache], [position + 1], CPU)
    hidden = model(token.view(1, 1), torch.tensor([[position]]), ctx)
    return model.compute_logits(hidden)


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

    full_logits = full_forward_logits(model, config, token_ids)

    # Prefill the first 5 tokens, then decode the rest one token at a time.
    cache = make_cache(config)
    positions = torch.arange(5)[None]
    hidden = model(token_ids[:, :5], positions, PrefillContext(cache))
    step_logits = [model.compute_logits(hidden)]
    for pos in range(5, 12):
        step_logits.append(decode_one(model, cache, token_ids[:, pos], pos))

    incremental = torch.cat(step_logits, dim=1)
    torch.testing.assert_close(incremental, full_logits, rtol=1e-4, atol=1e-4)


@torch.inference_mode()
def test_batched_decode_matches_sequential() -> None:
    """The continuous-batching invariant: batching must not change any row's logits."""
    config = tiny_config()
    model = make_model(config)
    generator = torch.Generator().manual_seed(4)
    prompts = [torch.randint(0, config.vocab_size, (1, n), generator=generator) for n in (5, 9)]
    steps = 4

    # Sequential reference: each sequence decodes alone.
    sequential_logits: list[list[torch.Tensor]] = []
    seq_caches = []
    for prompt in prompts:
        cache = make_cache(config)
        model(prompt, torch.arange(prompt.shape[1])[None], PrefillContext(cache))
        seq_caches.append(cache)
        per_seq = []
        for step in range(steps):
            pos = prompt.shape[1] + step
            token = torch.tensor([step + 1])
            per_seq.append(decode_one(model, cache, token, pos))
        sequential_logits.append(per_seq)

    # Batched: same prompts prefilled separately, then decoded together.
    batch_caches = []
    for prompt in prompts:
        cache = make_cache(config)
        model(prompt, torch.arange(prompt.shape[1])[None], PrefillContext(cache))
        batch_caches.append(cache)

    for step in range(steps):
        lens = [prompt.shape[1] + step + 1 for prompt in prompts]
        ctx = BatchedDecodeContext(batch_caches, lens, CPU)
        input_ids = torch.tensor([[step + 1], [step + 1]])
        positions = torch.tensor([[lens[0] - 1], [lens[1] - 1]])
        hidden = model(input_ids, positions, ctx)
        logits = model.compute_logits(hidden)
        for row, per_seq in enumerate(sequential_logits):
            torch.testing.assert_close(logits[row : row + 1], per_seq[step], rtol=1e-4, atol=1e-4)


@torch.inference_mode()
def test_chunked_forward_matches_full_forward() -> None:
    """Mid-cache multi-token chunks (speculative verification) must be exact."""
    config = tiny_config()
    model = make_model(config)
    generator = torch.Generator().manual_seed(3)
    token_ids = torch.randint(0, config.vocab_size, (1, 12), generator=generator)

    full_logits = full_forward_logits(model, config, token_ids)

    cache = make_cache(config)
    chunks = [(0, 5), (5, 9), (9, 12)]  # prefill, then two mid-cache chunks
    chunk_logits = []
    for start, end in chunks:
        ctx = PrefillContext(cache, start_pos=start)
        positions = torch.arange(start, end)[None]
        hidden = model(token_ids[:, start:end], positions, ctx)
        chunk_logits.append(model.compute_logits(hidden))

    torch.testing.assert_close(torch.cat(chunk_logits, dim=1), full_logits, rtol=1e-4, atol=1e-4)


@torch.inference_mode()
def test_logits_causality() -> None:
    """Changing a future token must not change logits at earlier positions."""
    config = tiny_config()
    model = make_model(config)
    generator = torch.Generator().manual_seed(2)
    tokens_a = torch.randint(0, config.vocab_size, (1, 10), generator=generator)
    tokens_b = tokens_a.clone()
    tokens_b[0, -1] = (tokens_b[0, -1] + 1) % config.vocab_size

    logits_a = full_forward_logits(model, config, tokens_a)
    logits_b = full_forward_logits(model, config, tokens_b)

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
        model(tokens, torch.arange(5)[None], PrefillContext(cache))
