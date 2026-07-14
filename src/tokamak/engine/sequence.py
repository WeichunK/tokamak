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

    A preempted sequence goes back to ``WAITING`` with its generated tokens
    intact; resuming recomputes the KV cache over ``all_token_ids`` and continues.

    Attributes:
        seq_id: Engine-assigned request id (also its FCFS arrival priority).
        prompt_token_ids: Tokenized prompt (immutable after construction).
        output_token_ids: Tokens generated so far, in order.
        sampling_params: Sampling configuration for this request.
        status: Current lifecycle status.
        finish_reason: Set exactly once, when the sequence finishes.
        arrival_time: ``perf_counter`` timestamp when the request was submitted.
        first_token_time: Timestamp of the first sampled token (TTFT numerator).
        finish_time: Timestamp when the sequence finished.
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
        self.arrival_time: float | None = None
        self.first_token_time: float | None = None
        self.finish_time: float | None = None
        # Speculative-decoding statistics (stay 0 on the plain path).
        self.spec_proposed = 0
        self.spec_accepted = 0

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
    def all_token_ids(self) -> list[int]:
        """Prompt plus generated tokens — what a (re)prefill computes over."""
        return self.prompt_token_ids + self.output_token_ids

    @property
    def last_token_id(self) -> int:
        """The most recent token (input to the next decode step)."""
        if self.output_token_ids:
            return self.output_token_ids[-1]
        return self.prompt_token_ids[-1]

    @property
    def is_finished(self) -> bool:
        """Whether the sequence has terminated."""
        return self.status is SequenceStatus.FINISHED

    def append_output_token(self, token_id: int) -> None:
        """Record one newly generated token."""
        if self.is_finished:
            raise RuntimeError(f"Sequence {self.seq_id} is already finished")
        self.output_token_ids.append(token_id)

    def append_checked(
        self,
        token_id: int,
        *,
        eos_token_ids: tuple[int, ...],
        max_total_tokens: int,
    ) -> bool:
        """Append one token and apply the stop conditions; True when finished.

        Stop conditions, in order: EOS (unless ``ignore_eos``), the request's
        ``max_new_tokens``, and the engine's context budget ``max_total_tokens``.
        """
        self.append_output_token(token_id)
        params = self.sampling_params
        if not params.ignore_eos and token_id in eos_token_ids:
            self.finish(FinishReason.STOP)
        elif self.num_output_tokens >= params.max_new_tokens or self.num_tokens >= max_total_tokens:
            self.finish(FinishReason.LENGTH)
        return self.is_finished

    def finish(self, reason: FinishReason) -> None:
        """Mark the sequence as finished."""
        self.status = SequenceStatus.FINISHED
        self.finish_reason = reason

    def __repr__(self) -> str:
        return (
            f"Sequence(id={self.seq_id}, prompt={self.num_prompt_tokens} tok, "
            f"output={self.num_output_tokens} tok, status={self.status.name})"
        )
