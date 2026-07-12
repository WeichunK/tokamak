from types import SimpleNamespace

import pytest
from transformers import LlamaConfig, Qwen2Config, Qwen3Config

from tokamak.config import ModelConfig


def namespace_config(**overrides: object) -> SimpleNamespace:
    """A minimal duck-typed HF config, independent of the transformers version."""
    fields: dict = {
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": 1000,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 128,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 512,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "rope_theta": 10000.0,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_from_hf_qwen3() -> None:
    hf_config = Qwen3Config(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,  # decoupled from hidden_size / num_heads, as in real Qwen3
        intermediate_size=128,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        max_position_embeddings=512,
        tie_word_embeddings=True,
        architectures=["Qwen3ForCausalLM"],
    )
    config = ModelConfig.from_hf(hf_config, eos_token_ids=[7])

    assert config.architecture == "Qwen3ForCausalLM"
    assert config.head_dim == 32
    assert config.rope_theta == 1e6
    assert config.use_qk_norm  # Qwen3 uses QK-norm ...
    assert not config.attention_bias  # ... instead of QKV biases
    assert config.tie_word_embeddings
    assert config.num_kv_groups == 2
    assert config.eos_token_ids == (7,)


def test_from_hf_qwen2_has_attention_bias() -> None:
    hf_config = Qwen2Config(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        architectures=["Qwen2ForCausalLM"],
    )
    config = ModelConfig.from_hf(hf_config, eos_token_ids=[0])

    assert config.attention_bias  # hardcoded in HF's Qwen2 modeling code
    assert not config.use_qk_norm
    assert config.head_dim == 16  # falls back to hidden_size / num_heads


def test_from_hf_llama() -> None:
    hf_config = LlamaConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=128,
        architectures=["LlamaForCausalLM"],
    )
    config = ModelConfig.from_hf(hf_config, eos_token_ids=[2])

    assert not config.attention_bias
    assert not config.use_qk_norm
    assert config.num_kv_groups == 1


def test_unsupported_architecture_raises() -> None:
    hf_config = LlamaConfig(architectures=["GPT2LMHeadModel"])
    with pytest.raises(ValueError, match="Unsupported architecture"):
        ModelConfig.from_hf(hf_config, eos_token_ids=[0])


def test_rope_theta_from_flat_attribute() -> None:
    """transformers 4.x layout: flat rope_theta attribute."""
    config = ModelConfig.from_hf(namespace_config(rope_theta=500000.0), eos_token_ids=[0])
    assert config.rope_theta == 500000.0


def test_rope_theta_from_rope_parameters() -> None:
    """transformers 5.x layout: rope_parameters dict."""
    hf_config = namespace_config(rope_parameters={"rope_type": "default", "rope_theta": 1e6})
    del hf_config.rope_theta
    config = ModelConfig.from_hf(hf_config, eos_token_ids=[0])
    assert config.rope_theta == 1e6


def test_missing_rope_theta_raises() -> None:
    """A silently-defaulted RoPE base corrupts every position — must be an error."""
    hf_config = namespace_config()
    del hf_config.rope_theta
    with pytest.raises(ValueError, match="rope_theta"):
        ModelConfig.from_hf(hf_config, eos_token_ids=[0])


@pytest.mark.parametrize(
    "overrides",
    [
        {"rope_scaling": {"rope_type": "yarn", "factor": 4.0}},
        {"rope_parameters": {"rope_type": "yarn", "rope_theta": 1e6}},
    ],
    ids=["legacy-rope_scaling", "rope_parameters-nondefault"],
)
def test_rope_scaling_rejected(overrides: dict) -> None:
    with pytest.raises(ValueError, match="not supported"):
        ModelConfig.from_hf(namespace_config(**overrides), eos_token_ids=[0])


def test_indivisible_kv_heads_raise() -> None:
    hf_config = LlamaConfig(
        num_attention_heads=4,
        num_key_value_heads=3,
        architectures=["LlamaForCausalLM"],
    )
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig.from_hf(hf_config, eos_token_ids=[0])
