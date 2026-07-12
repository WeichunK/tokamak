"""Per-request token state.

``Sequence`` is deliberately engine-agnostic: it tracks tokens and lifecycle status
but knows nothing about KV memory or scheduling. The M3 scheduler will manage
collections of these without changing this module.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokamak.sampling_params import SamplingParams


class SequenceStatus(enum.Enum):
    """Lifecycle of a request inside the engine."""

    WAITING = enum.auto()
    RUNNING = enum.auto()
    FINISHED = enum.auto()


class FinishReason(enum.Enum):
    """Why generation stopped."""

    STOP = "stop"  # an EOS token was generated
    LENGTH = "length"  # max_new_tokens or the context limit was reached


class Sequence:
    """Token-level state of one generation request.

    Attributes:
        seq_id: Engine-assigned request id.
        prompt_token_ids: Tokenized prompt (immutable after construction).
        output_token_ids: Tokens generated so far, in order.
        sampling_params: Sampling configuration for this request.
        status: Current lifecycle status.
        finish_reason: Set exactly once, when the sequence finishes.
    """

    def __init__(
        self,
        seq_id: int,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
    ) -> None:
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        self.seq_id = seq_id
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids: list[int] = []
        self.sampling_params = sampling_params
        self.status = SequenceStatus.WAITING
        self.finish_reason: FinishReason | None = None

    @property
    def num_prompt_tokens(self) -> int:
        """Number of prompt tokens."""
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        """Number of generated tokens."""
        return len(self.output_token_ids)

    @property
    def num_tokens(self) -> int:
        """Total tokens (prompt + generated)."""
        return self.num_prompt_tokens + self.num_output_tokens

    @property
    def is_finished(self) -> bool:
        """Whether the sequence has terminated."""
        return self.status is SequenceStatus.FINISHED

    def append_output_token(self, token_id: int) -> None:
        """Record one newly generated token."""
        if self.is_finished:
            raise RuntimeError(f"Sequence {self.seq_id} is already finished")
        self.output_token_ids.append(token_id)

    def finish(self, reason: FinishReason) -> None:
        """Mark the sequence as finished."""
        self.status = SequenceStatus.FINISHED
        self.finish_reason = reason

    def __repr__(self) -> str:
        return (
            f"Sequence(id={self.seq_id}, prompt={self.num_prompt_tokens} tok, "
            f"output={self.num_output_tokens} tok, status={self.status.name})"
        )
