"""Public output types returned by the engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokamak.engine.sequence import FinishReason


@dataclass(frozen=True, slots=True)
class RequestOutput:
    """Completed generation for one prompt.

    Attributes:
        request_id: Engine-assigned id, ordered by submission.
        prompt: The input prompt text.
        prompt_token_ids: Tokenized prompt.
        output_text: Detokenized completion (special tokens stripped).
        output_token_ids: Generated token ids, including any final EOS.
        finish_reason: Why generation stopped.
    """

    request_id: int
    prompt: str
    prompt_token_ids: list[int]
    output_text: str
    output_token_ids: list[int]
    finish_reason: FinishReason
