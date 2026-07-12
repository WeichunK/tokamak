"""User-facing sampling configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Parameters controlling token sampling for a single request.

    Filters are applied in the order temperature → top-k → top-p, matching vLLM.

    Attributes:
        temperature: Softmax temperature. ``0.0`` selects greedy decoding.
        top_k: Sample only from the ``top_k`` highest-probability tokens.
            ``0`` disables the filter.
        top_p: Nucleus sampling (Holtzman et al., 2020): sample from the smallest
            token set whose cumulative probability reaches ``top_p``. ``1.0``
            disables the filter.
        max_new_tokens: Hard cap on the number of generated tokens.
        seed: Per-request RNG seed for reproducible sampling. ``None`` draws from
            the global generator.
        ignore_eos: Keep generating past EOS until another stop condition hits.
            Useful for benchmarking fixed-length decodes.
    """

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    max_new_tokens: int = 128
    seed: int | None = None
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")

    @property
    def is_greedy(self) -> bool:
        """Whether this configuration decodes greedily."""
        return self.temperature == 0.0
