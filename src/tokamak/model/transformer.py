"""Decoder-only transformer covering the Llama / Qwen2 / Qwen3 families.

Module names deliberately mirror the Hugging Face checkpoint layout
(``model.layers.N.self_attn.q_proj`` etc.) so that safetensors weights map onto
``named_parameters()`` without a translation table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from tokamak.model.layers import GatedMLP, RMSNorm, RotaryEmbedding, apply_rotary_emb

if TYPE_CHECKING:
    from tokamak.config import ModelConfig
    from tokamak.model.kv_cache import KVCacheProtocol


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
        kv_cache: KVCacheProtocol,
        start_pos: int,
    ) -> torch.Tensor:
        """Attend over all cached positions up to and including the current tokens.

        Args:
            x: Input activations of shape ``[batch, seq_len, hidden]``.
            cos: RoPE cosine table for the current positions.
            sin: RoPE sine table for the current positions.
            kv_cache: Cache holding keys/values for positions ``[0, start_pos)``.
            start_pos: Absolute position of ``x[:, 0]``.
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

        k, v = kv_cache.update(self.layer_idx, k, v, start_pos)

        # Two cases cover the M1 engine: full prefill (causal mask over a square
        # attention matrix) and single-token decode (the new token may attend to
        # everything cached, so no mask is needed). SDPA's `is_causal` aligns the
        # mask to the top-left corner, which is only correct when q and k have equal
        # length — guard against silently reusing it for chunked inputs.
        if seq_len > 1 and start_pos != 0:
            raise NotImplementedError(
                "chunked prefill is not supported: multi-token forward passes must "
                "start at position 0"
            )
        out = F.scaled_dot_product_attention(q, k, v, is_causal=seq_len > 1, enable_gqa=True)

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
        kv_cache: KVCacheProtocol,
        start_pos: int,
    ) -> torch.Tensor:
        """Apply one decoder block with residual connections."""
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, kv_cache, start_pos)
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
        kv_cache: KVCacheProtocol,
        start_pos: int,
    ) -> torch.Tensor:
        """Run the decoder stack, returning normalized hidden states."""
        seq_len = input_ids.shape[1]
        positions = torch.arange(start_pos, start_pos + seq_len, device=input_ids.device)
        cos, sin = self.rotary_emb(positions)

        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, cos, sin, kv_cache, start_pos)
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
        kv_cache: KVCacheProtocol,
        start_pos: int,
    ) -> torch.Tensor:
        """Return final hidden states of shape ``[batch, seq_len, hidden]``.

        Logits are computed separately via :meth:`compute_logits` so the engine can
        project only the last position during generation instead of the full
        ``[batch, seq_len, vocab]`` tensor.
        """
        hidden: torch.Tensor = self.model(input_ids, kv_cache, start_pos)
        return hidden

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden states to float32 vocabulary logits."""
        logits: torch.Tensor = self.lm_head(hidden)
        return logits.float()
