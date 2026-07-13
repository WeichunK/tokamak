"""Custom attention kernels (optional; requires the ``triton`` extra and CUDA).

Importing this package is always safe — triton itself is only imported by the
kernel modules, which callers load lazily after checking :func:`is_available`.
"""

from __future__ import annotations

import importlib.util

import torch


def is_available() -> bool:
    """Whether the Triton kernel path can run on this machine."""
    if not torch.cuda.is_available():
        return False
    return importlib.util.find_spec("triton") is not None


__all__ = ["is_available"]
