"""Verification of draft proposals against the target distribution.

The guarantee that makes speculative decoding trustworthy: accepted-or-corrected
tokens are distributed *exactly* as if sampled from the target model alone. For a
draft token ``d ~ q``:

- accept with probability ``min(1, p(d) / q(d))``;
- on rejection, sample the correction from the residual ``norm(max(0, p - q))``.

Summing the two cases gives back ``p`` identically (Leviathan et al., 2023,
Theorem 1) — no approximation, any draft model, any (shared) filtering. The unit
suite checks this property empirically against synthetic distributions.

Both ``p`` and ``q`` must be the *post-filter* distributions (after temperature /
top-k / top-p, via :func:`tokamak.sampling.sampling_probs`); the guarantee is then
"identical to target-only sampling with those same parameters". Greedy decoding is
the degenerate delta-distribution case and gets the explicit argmax form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence as AbcSequence


def verify_greedy(
    draft_tokens: AbcSequence[int],
    target_logits: torch.Tensor,
) -> tuple[int, int]:
    """Verify drafts under greedy decoding: accept while they match the argmax.

    Args:
        draft_tokens: The ``k`` proposed token ids.
        target_logits: Target logits of shape ``[k + 1, vocab]`` — row ``i`` is
            the target's prediction for the position draft ``i`` occupies, and
            the final row predicts the bonus token after a full acceptance.

    Returns:
        ``(num_accepted, next_token)`` where ``next_token`` is the correction at
        the first mismatch, or the bonus token when everything was accepted.
    """
    if target_logits.shape[0] != len(draft_tokens) + 1:
        raise ValueError(
            f"expected {len(draft_tokens) + 1} logit rows, got {target_logits.shape[0]}"
        )
    targets = target_logits.argmax(dim=-1).tolist()
    accepted = 0
    for draft, target in zip(draft_tokens, targets[:-1], strict=True):
        if draft != target:
            break
        accepted += 1
    return accepted, targets[accepted]


def verify_rejection(
    draft_tokens: AbcSequence[int],
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[int, int]:
    """Verify drafts by rejection sampling, preserving the target distribution.

    Args:
        draft_tokens: The ``k`` proposed token ids, sampled from ``draft_probs``.
        draft_probs: Post-filter draft distributions, ``[k, vocab]``.
        target_probs: Post-filter target distributions, ``[k + 1, vocab]`` (the
            final row is the bonus-token distribution).
        generator: Optional RNG on the same device, for reproducible runs.

    Returns:
        ``(num_accepted, next_token)`` — ``next_token`` comes from the residual
        distribution at the first rejection, or from the target's bonus-row
        distribution when every draft was accepted.
    """
    num_drafts = len(draft_tokens)
    if draft_probs.shape[0] != num_drafts or target_probs.shape[0] != num_drafts + 1:
        raise ValueError(
            f"shape mismatch: {num_drafts} drafts, draft_probs {tuple(draft_probs.shape)}, "
            f"target_probs {tuple(target_probs.shape)}"
        )

    for i, draft in enumerate(draft_tokens):
        # d was sampled from q, so q[d] > 0; the ratio may exceed 1 (auto-accept).
        ratio = target_probs[i, draft] / draft_probs[i, draft]
        r = torch.rand((), device=target_probs.device, generator=generator)
        if r < ratio:
            continue
        residual = (target_probs[i] - draft_probs[i]).clamp(min=0)
        total = residual.sum()
        if total <= 0:
            # p == q exactly: the residual is empty and any correction from p
            # is faithful. (Rejection here has probability 0 up to rounding.)
            residual, total = target_probs[i], target_probs[i].sum()
        correction = torch.multinomial(residual / total, 1, generator=generator)
        return i, int(correction.item())

    bonus = torch.multinomial(target_probs[num_drafts], 1, generator=generator)
    return num_drafts, int(bonus.item())
