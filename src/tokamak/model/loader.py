"""Checkpoint resolution, model construction, and safetensors weight loading."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from tokamak.model.transformer import TransformerForCausalLM

if TYPE_CHECKING:
    from tokamak.config import ModelConfig

logger = logging.getLogger(__name__)

_SNAPSHOT_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.model"]


def resolve_model_path(model: str) -> Path:
    """Resolve a model spec to a local directory.

    Args:
        model: Either a local directory containing a Hugging Face-format checkpoint
            or a Hub repo id (e.g. ``Qwen/Qwen3-0.6B``), which is downloaded into the
            local Hub cache on first use.
    """
    path = Path(model)
    if path.is_dir():
        return path
    logger.info("Downloading %s from the Hugging Face Hub", model)
    return Path(snapshot_download(model, allow_patterns=_SNAPSHOT_PATTERNS))


def build_model(
    config: ModelConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> TransformerForCausalLM:
    """Construct an empty model directly on the target device and dtype."""
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.device(device):
            model = TransformerForCausalLM(config)
    finally:
        torch.set_default_dtype(default_dtype)
    return model.eval()


def load_weights(model: TransformerForCausalLM, path: Path) -> None:
    """Copy safetensors weights into ``model``, validating full coverage.

    Weight names in the checkpoint must match ``model.named_parameters()`` exactly —
    the module tree mirrors the Hugging Face layout, so no renaming is performed.
    A checkpoint-level ``lm_head.weight`` is skipped when embeddings are tied
    (``named_parameters`` deduplicates shared tensors, and the copy into
    ``model.embed_tokens.weight`` already covers it).

    Raises:
        FileNotFoundError: If ``path`` contains no ``*.safetensors`` files.
        ValueError: On unexpected, shape-mismatched, or missing weights.
    """
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No *.safetensors files found under {path}")

    params = dict(model.named_parameters())
    tied = model.config.tie_word_embeddings
    loaded: set[str] = set()

    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():  # noqa: SIM118 — safe_open handles are not dicts
                if name not in params:
                    if name == "lm_head.weight" and tied:
                        continue
                    raise ValueError(f"Unexpected weight {name!r} in {file.name}")
                tensor = f.get_tensor(name)
                if tensor.shape != params[name].shape:
                    raise ValueError(
                        f"Shape mismatch for {name!r}: checkpoint {tuple(tensor.shape)} "
                        f"vs model {tuple(params[name].shape)}"
                    )
                with torch.no_grad():
                    params[name].copy_(tensor)
                loaded.add(name)

    missing = params.keys() - loaded
    if missing:
        raise ValueError(f"Weights missing from checkpoint: {sorted(missing)}")
    logger.info("Loaded %d tensors from %d file(s)", len(loaded), len(files))
