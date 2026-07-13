"""Decoder-only transformer covering the Llama / Qwen2 / Qwen3 families.

Module names deliberately mirror the Hugging Face checkpoint layout
(``model.layers.N.self_attn.q_proj`` etc.) so that safetensors weights map onto
``named_parameters()`` without a translation table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from tokamak.model.layers import GatedMLP, RMSNorm, RotaryEmbedding, apply_rotary_emb

if TYPE_CHECKING:
    from tokamak.config import ModelConfig
    from tokamak.model.step_context import StepContextProtocol


class Attention(nn.Module):
    """Multi-head attention with grouped-query KV heads and rotary embeddings.

    Optionally applies per-head RMSNorm to queries and keys before rotation
    (QK-norm, used by Qwen3 in place of the Q/K/V biases of Qwen2).
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        hidden = config.hidden_size
        bias = config.attention_bias

        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)

        self.q_norm: RMSNorm | None = None
        self.k_norm: RMSNorm | None = None
        if config.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        ctx: StepContextProtocol,
    ) -> torch.Tensor:
        """Attend over every cached position visible to each row.

        Args:
            x: Input activations of shape ``[batch, seq_len, hidden]``.
            cos: RoPE cosine table for the current positions.
            sin: RoPE sine table for the current positions.
            ctx: Step context that stores this step's K/V and returns what each
                row may attend over (plus a padding mask for batched decode).
        """
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # The context owns storage and attention math (SDPA reference or a
        # fused kernel); this layer only projects.
        out = ctx.attend(self.layer_idx, q, k, v)

        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        projected: torch.Tensor = self.o_proj(out)
        return projected


class DecoderLayer(nn.Module):
    """Pre-norm decoder block: RMSNorm → attention → RMSNorm → gated MLP."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = GatedMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        ctx: StepContextProtocol,
    ) -> torch.Tensor:
        """Apply one decoder block with residual connections."""
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, ctx)
        out: torch.Tensor = x + self.mlp(self.post_attention_layernorm(x))
        return out


class TransformerModel(nn.Module):
    """Embedding, decoder stack, and final norm (the ``model.*`` checkpoint subtree)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(config, layer_idx) for layer_idx in range(config.num_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, base=config.rope_theta)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: StepContextProtocol,
    ) -> torch.Tensor:
        """Run the decoder stack, returning normalized hidden states.

        Args:
            input_ids: Token ids of shape ``[batch, seq_len]``.
            positions: Absolute position of every input token, ``[batch, seq_len]``
                (rows may sit at different positions during batched decode).
            ctx: Step context shared by all layers.
        """
        cos, sin = self.rotary_emb(positions)

        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, cos, sin, ctx)
        normalized: torch.Tensor = self.norm(x)
        return normalized


class TransformerForCausalLM(nn.Module):
    """A decoder-only language model with an (optionally tied) LM head."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = TransformerModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: StepContextProtocol,
    ) -> torch.Tensor:
        """Return final hidden states of shape ``[batch, seq_len, hidden]``.

        Logits are computed separately via :meth:`compute_logits` so the engine can
        project only the last position during generation instead of the full
        ``[batch, seq_len, vocab]`` tensor.
        """
        hidden: torch.Tensor = self.model(input_ids, positions, ctx)
        return hidden

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden states to float32 vocabulary logits."""
        logits: torch.Tensor = self.lm_head(hidden)
        return logits.float()
