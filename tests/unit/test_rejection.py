"""Rejection-sampling verification: the distribution-preservation guarantee.

The headline test draws tens of thousands of speculative rounds against fixed
synthetic distributions and checks that the emitted token's empirical
distribution matches the *target* distribution regardless of how bad the draft
is — the property that makes speculative decoding trustworthy at all.
"""

import pytest
import torch

from tokamak.speculative import verify_greedy, verify_rejection


def test_greedy_accepts_matching_prefix() -> None:
    # vocab 4; argmax rows: 2, 0, 3, then bonus row argmax 1
    logits = torch.tensor(
        [
            [0.0, 1.0, 9.0, 2.0],
            [9.0, 1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0, 9.0],
            [0.0, 9.0, 2.0, 1.0],
        ]
    )
    assert verify_greedy([2, 0, 3], logits) == (3, 1)  # all accepted + bonus
    assert verify_greedy([2, 1, 3], logits) == (1, 0)  # mismatch at i=1 -> correction
    assert verify_greedy([0, 0, 3], logits) == (0, 2)  # immediate mismatch


def test_greedy_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="logit rows"):
        verify_greedy([1, 2], torch.zeros(2, 4))


def test_rejection_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        verify_rejection([1], torch.rand(2, 4), torch.rand(2, 4))


def test_identical_distributions_always_accept() -> None:
    p = torch.tensor([0.5, 0.3, 0.2])
    probs = torch.stack([p, p, p, p])  # 3 drafts + bonus
    generator = torch.Generator().manual_seed(0)
    for trial in range(200):
        drafts = torch.multinomial(p, 3, replacement=True, generator=generator).tolist()
        accepted, _ = verify_rejection(drafts, probs[:3], probs, generator)
        assert accepted == 3, f"trial {trial}: rejection despite p == q"


@pytest.mark.parametrize(
    "draft_dist",
    [
        [0.55, 0.25, 0.15, 0.05],  # close to the target
        [0.05, 0.15, 0.30, 0.50],  # nearly reversed
        [0.97, 0.01, 0.01, 0.01],  # overconfident draft
    ],
    ids=["close", "reversed", "overconfident"],
)
def test_output_distribution_matches_target(draft_dist: list[float]) -> None:
    """Accepted-or-corrected tokens must be distributed exactly as the target.

    Single-draft rounds (k=1): the emitted token is either the accepted draft or
    the residual correction; by Leviathan et al. Theorem 1 its law is exactly
    ``p``, for any ``q``. Checked empirically via total variation distance.
    """
    p = torch.tensor([0.40, 0.30, 0.20, 0.10])
    q = torch.tensor(draft_dist)
    generator = torch.Generator().manual_seed(42)

    trials = 40_000
    drafts = torch.multinomial(q, trials, replacement=True, generator=generator)
    counts = torch.zeros(4)
    q_rows = q[None]
    p_rows = torch.stack([p, p])  # bonus row unused at k=1 rejection, needed on accept
    for i in range(trials):
        accepted, token = verify_rejection([int(drafts[i])], q_rows, p_rows, generator)
        emitted = int(drafts[i]) if accepted == 1 else token
        counts[emitted] += 1

    empirical = counts / trials
    total_variation = 0.5 * (empirical - p).abs().sum().item()
    assert total_variation < 0.01, f"TV(empirical, target) = {total_variation:.4f}"


def test_acceptance_rate_matches_theory() -> None:
    """E[accept] for one draft is sum_x min(p(x), q(x)) — check empirically."""
    p = torch.tensor([0.40, 0.30, 0.20, 0.10])
    q = torch.tensor([0.10, 0.20, 0.30, 0.40])
    expected = torch.minimum(p, q).sum().item()  # 0.6
    generator = torch.Generator().manual_seed(7)

    trials = 40_000
    drafts = torch.multinomial(q, trials, replacement=True, generator=generator)
    accepted_total = 0
    q_rows = q[None]
    p_rows = torch.stack([p, p])
    for i in range(trials):
        accepted, _ = verify_rejection([int(drafts[i])], q_rows, p_rows, generator)
        accepted_total += accepted

    rate = accepted_total / trials
    assert abs(rate - expected) < 0.01, f"acceptance {rate:.4f} vs theory {expected:.4f}"


def test_rejection_stops_at_first_failure() -> None:
    """Tokens after the first rejection must be discarded, never emitted."""
    p = torch.tensor([1.0, 0.0, 0.0, 0.0])
    q = torch.tensor([0.0, 1.0, 0.0, 0.0])
    generator = torch.Generator().manual_seed(0)
    # Draft proposes token 1 (certain under q, impossible under p) three times.
    accepted, token = verify_rejection(
        [1, 1, 1], torch.stack([q, q, q]), torch.stack([p, p, p, p]), generator
    )
    assert accepted == 0
    assert token == 0  # the residual is exactly p
