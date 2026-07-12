"""Model and engine configuration.

``ModelConfig`` is tokamak's own architecture description: a frozen, explicit set of
hyperparameters extracted from a Hugging Face ``config.json``. Keeping the engine's
view of the model separate from the Hugging Face config object makes every field the
model code depends on visible in one place, and makes unsupported architectures fail
loudly at load time instead of silently misbehaving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

SUPPORTED_ARCHITECTURES = frozenset(
    {
        "LlamaForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
    }
)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Architecture hyperparameters for a decoder-only transformer.

    Attributes:
        architecture: Hugging Face architecture name (e.g. ``Qwen3ForCausalLM``).
        vocab_size: Token vocabulary size.
        hidden_size: Residual stream width.
        num_layers: Number of decoder layers.
        num_attention_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (< ``num_attention_heads`` under GQA).
        head_dim: Per-head dimension. Not necessarily ``hidden_size / num_heads``
            (Qwen3 sets it independently).
        intermediate_size: Hidden width of the gated MLP.
        rms_norm_eps: Epsilon used by all RMSNorm layers.
        rope_theta: Rotary embedding base frequency.
        max_position_embeddings: Maximum context length the model was trained for.
        tie_word_embeddings: Whether the LM head shares weights with the embedding.
        attention_bias: Whether Q/K/V projections carry a bias term (Qwen2).
        use_qk_norm: Whether per-head RMSNorm is applied to Q/K before RoPE (Qwen3).
        eos_token_ids: Token ids that terminate generation.
    """

    architecture: str
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    attention_bias: bool
    use_qk_norm: bool
    eos_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible "
                f"by num_kv_heads ({self.num_kv_heads})"
            )

    @property
    def num_kv_groups(self) -> int:
        """Number of query heads sharing each KV head."""
        return self.num_attention_heads // self.num_kv_heads

    @classmethod
    def from_hf(cls, hf_config: Any, eos_token_ids: Sequence[int]) -> ModelConfig:
        """Build a ``ModelConfig`` from a Hugging Face ``PretrainedConfig``.

        Args:
            hf_config: The config object returned by ``AutoConfig.from_pretrained``.
            eos_token_ids: EOS ids resolved from the checkpoint's generation config
                (the model config alone is often missing or wrong about these).

        Raises:
            ValueError: If the checkpoint architecture is not supported.
        """
        architectures = getattr(hf_config, "architectures", None) or []
        architecture = architectures[0] if architectures else "<missing>"
        if architecture not in SUPPORTED_ARCHITECTURES:
            supported = ", ".join(sorted(SUPPORTED_ARCHITECTURES))
            raise ValueError(f"Unsupported architecture {architecture!r}; supported: {supported}")

        num_heads = int(hf_config.num_attention_heads)
        head_dim = getattr(hf_config, "head_dim", None)
        if head_dim is None:
            head_dim = hf_config.hidden_size // num_heads

        # Qwen2 hardcodes Q/K/V bias in its modeling code rather than exposing it in
        # the config; Llama and Qwen3 expose `attention_bias` (default False).
        if architecture == "Qwen2ForCausalLM":
            attention_bias = True
        else:
            attention_bias = bool(getattr(hf_config, "attention_bias", False))

        return cls(
            architecture=architecture,
            vocab_size=int(hf_config.vocab_size),
            hidden_size=int(hf_config.hidden_size),
            num_layers=int(hf_config.num_hidden_layers),
            num_attention_heads=num_heads,
            num_kv_heads=int(getattr(hf_config, "num_key_value_heads", num_heads)),
            head_dim=int(head_dim),
            intermediate_size=int(hf_config.intermediate_size),
            rms_norm_eps=float(hf_config.rms_norm_eps),
            rope_theta=_resolve_rope_theta(hf_config),
            max_position_embeddings=int(hf_config.max_position_embeddings),
            tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
            attention_bias=attention_bias,
            use_qk_norm=architecture == "Qwen3ForCausalLM",
            eos_token_ids=tuple(eos_token_ids),
        )


def _resolve_rope_theta(hf_config: Any) -> float:
    """Extract the RoPE base frequency across transformers config layouts.

    transformers 5.x moved RoPE settings into a ``rope_parameters`` dict;
    4.x exposes a flat ``rope_theta`` attribute plus an optional ``rope_scaling``
    dict. A wrong base silently corrupts every position, so an unrecognized layout
    is an error, not a default.

    Raises:
        ValueError: If the config requests RoPE scaling (not implemented) or the
            base frequency cannot be found.
    """
    rope_parameters = getattr(hf_config, "rope_parameters", None)
    if isinstance(rope_parameters, dict):
        rope_type = rope_parameters.get("rope_type", "default")
        if rope_type != "default":
            raise ValueError(f"RoPE scaling type {rope_type!r} is not supported")
        theta = rope_parameters.get("rope_theta")
        if theta is not None:
            return float(theta)

    theta = getattr(hf_config, "rope_theta", None)
    if theta is not None:
        rope_scaling = getattr(hf_config, "rope_scaling", None)
        if rope_scaling is not None:
            raise ValueError(f"RoPE scaling is not supported (got {rope_scaling!r})")
        return float(theta)

    raise ValueError("Could not determine rope_theta from the model config")


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve a device spec, defaulting to CUDA when available."""
    if device == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(device)


def resolve_dtype(dtype: str | torch.dtype, device: torch.device) -> torch.dtype:
    """Resolve a dtype spec.

    ``"auto"`` picks bfloat16 on CUDA (matching how modern checkpoints are stored)
    and float32 on CPU (bf16 matmuls are slow or unsupported on many CPUs).
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    resolved = getattr(torch, dtype, None)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"Unknown dtype spec: {dtype!r}")
    return resolved
