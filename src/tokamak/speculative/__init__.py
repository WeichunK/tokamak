"""Speculative decoding: verification that preserves the target distribution.

Implements the rejection-sampling scheme of Leviathan et al. (2023) and
Chen et al. (2023).
"""

from tokamak.speculative.rejection import verify_greedy, verify_rejection

__all__ = ["verify_greedy", "verify_rejection"]
