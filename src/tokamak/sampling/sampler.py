"""Logit filtering and token sampling.

Filters are applied in the order temperature → top-k → top-p (matching vLLM), then a
token is drawn from the surviving distribution with ``torch.multinomial``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tokamak.sampling_params import SamplingParams


def sampling_probs(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    """The post-filter sampling distribution (softmax after temperature/top-k/top-p).

    Speculative verification needs the full distribution — not just a draw from
    it — for both the draft and the target model, so the filtering pipeline is
    exposed separately from :func:`sample`.

    Args:
        logits: Float logits of shape ``[batch, vocab]``.
        params: Sampling configuration; must not be greedy (a greedy "distribution"
            is a delta — callers handle argmax explicitly).

    Returns:
        Probabilities of shape ``[batch, vocab]`` summing to 1 per row.
    """
    if params.is_greedy:
        raise ValueError("greedy decoding has no sampling distribution; use argmax")
    logits = logits / params.temperature
    if 0 < params.top_k < logits.shape[-1]:
        logits = _filter_top_k(logits, params.top_k)
    if params.top_p < 1.0:
        logits = _filter_top_p(logits, params.top_p)
    return torch.softmax(logits, dim=-1)


def sample(
    logits: torch.Tensor,
    params: SamplingParams,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token id per batch row.

    Args:
        logits: Float logits of shape ``[batch, vocab]``.
        params: Sampling configuration.
        generator: Optional RNG for reproducible draws; must live on the same device
            as ``logits``.

    Returns:
        Long tensor of shape ``[batch]``.
    """
    if params.is_greedy:
        return logits.argmax(dim=-1)
    probs = sampling_probs(logits, params)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def _filter_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Mask every logit below the k-th largest to ``-inf``."""
    kth = logits.topk(top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth, float("-inf"))


def _filter_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keep the smallest set of tokens whose cumulative probability reaches ``top_p``.

    The highest-probability token always survives, even when its probability alone
    exceeds ``top_p``.
    """
    sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
    probs = sorted_logits.softmax(dim=-1)
    cumulative = probs.cumsum(dim=-1)
    # Drop a token iff the cumulative mass *before* it has already reached top_p.
    drop = cumulative - probs >= top_p
    sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(
        dim=-1, index=sorted_indices, src=sorted_logits
    )
