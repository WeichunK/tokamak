"""The draft-and-verify generation loop (speculative decoding, single sequence).

One iteration: the draft model proposes ``k`` tokens autoregressively (cheap), the
target model scores all of them plus one bonus position in a *single* chunked
forward (its per-step cost barely depends on chunk length in the decode regime),
and rejection sampling accepts a prefix of the proposals while provably preserving
the target distribution (:mod:`tokamak.speculative.rejection`).

Cache bookkeeping is the subtle part. Each model tracks ``cached_len`` — how many
positions hold *valid* K/V. Verification writes K/V for rejected draft positions
too; rather than erasing them, the frontier rolls back to ``committed - 1`` and the
next chunked forward overwrites the stale positions before anything reads them
(position-addressed caches make this safe). One invariant keeps every forward a
single contiguous chunk: a model's next input is always
``all_token_ids[cached_len:]``.

Deliberate scope: speculative decoding runs sequences one at a time with per-request
contiguous caches. Composing it with continuous batching and the paged pool is a
substantial engineering project on its own (vLLM's took several releases to
stabilize) and is out of scope for this milestone.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from tokamak.model.kv_cache import ContiguousKVCache
from tokamak.model.step_context import PrefillContext
from tokamak.sampling.sampler import sample, sampling_probs
from tokamak.speculative.rejection import verify_greedy, verify_rejection

if TYPE_CHECKING:
    from tokamak.config import ModelConfig
    from tokamak.engine.sequence import Sequence
    from tokamak.model.transformer import TransformerForCausalLM


@dataclass
class _ModelState:
    """One model's cache and its valid-prefix frontier for the current request."""

    model: TransformerForCausalLM
    cache: ContiguousKVCache
    cached_len: int = 0


class SpeculativeRunner:
    """Runs one sequence to completion with draft-model speculation.

    Args:
        target: The model whose distribution the output must follow.
        target_config: Its architecture config (for cache shapes and EOS ids).
        draft: The cheap proposal model (must share the target's vocabulary).
        draft_config: Its architecture config.
        num_speculative_tokens: Proposals per iteration (``k``).
        max_seq_len: Engine-level context cap.
        device: Device both models live on.
        dtype: KV cache dtype.
    """

    def __init__(
        self,
        *,
        target: TransformerForCausalLM,
        target_config: ModelConfig,
        draft: TransformerForCausalLM,
        draft_config: ModelConfig,
        num_speculative_tokens: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if num_speculative_tokens < 1:
            raise ValueError(f"num_speculative_tokens must be >= 1, got {num_speculative_tokens}")
        self._target = target
        self._target_config = target_config
        self._draft = draft
        self._draft_config = draft_config
        self._k = num_speculative_tokens
        self._max_seq_len = max_seq_len
        self._device = device
        self._dtype = dtype

    @torch.inference_mode()
    def run(self, seq: Sequence) -> None:
        """Generate until a stop condition, preserving the target distribution."""
        from tokamak.engine.sequence import SequenceStatus

        params = seq.sampling_params
        max_total = min(self._max_seq_len, seq.num_prompt_tokens + params.max_new_tokens)
        # Verification may write up to k rejected positions past the budget.
        cache_len = max_total + self._k
        target = _ModelState(
            self._target,
            ContiguousKVCache(
                self._target_config, cache_len, device=self._device, dtype=self._dtype
            ),
        )
        draft = _ModelState(
            self._draft,
            ContiguousKVCache(
                self._draft_config, cache_len, device=self._device, dtype=self._dtype
            ),
        )
        generator: torch.Generator | None = None
        if params.seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(params.seed)

        seq.status = SequenceStatus.RUNNING
        # The first token always comes straight from the target.
        logits = self._advance(target, seq.all_token_ids)
        first = int(sample(logits[0, -1:], params, generator).item())
        if self._commit(seq, first, max_total):
            return

        while True:
            committed = seq.all_token_ids
            k = min(self._k, max_total - len(committed) - 1)

            if k < 1:
                # No room to speculate inside the budget: plain decode step.
                logits = self._advance(target, committed)
                token = int(sample(logits[0, -1:], params, generator).item())
                if self._commit(seq, token, max_total):
                    return
                continue

            # 1) Draft proposes k tokens autoregressively, keeping its
            #    post-filter distributions for the acceptance test.
            draft_tokens: list[int] = []
            draft_rows: list[torch.Tensor] = []
            for _ in range(k):
                logits = self._advance(draft, committed + draft_tokens)
                row = logits[0, -1:]
                if params.is_greedy:
                    token = int(row.argmax().item())
                else:
                    q = sampling_probs(row, params)[0]
                    draft_rows.append(q)
                    token = int(torch.multinomial(q, 1, generator=generator).item())
                draft_tokens.append(token)

            # 2) Target scores every proposal plus the bonus position in one
            #    chunked forward.
            logits = self._advance(target, committed + draft_tokens)
            verify_logits = logits[0, -(k + 1) :]

            # 3) Accept a prefix; correct or extend by one target-sampled token.
            if params.is_greedy:
                num_accepted, next_token = verify_greedy(draft_tokens, verify_logits)
            else:
                target_rows = sampling_probs(verify_logits, params)
                num_accepted, next_token = verify_rejection(
                    draft_tokens, torch.stack(draft_rows), target_rows, generator
                )
            seq.spec_proposed += k
            seq.spec_accepted += num_accepted

            # 4) Commit; stop conditions may fire on any accepted token.
            finished = False
            for token in draft_tokens[:num_accepted]:
                if self._commit(seq, token, max_total):
                    finished = True
                    break
            if not finished:
                finished = self._commit(seq, next_token, max_total)

            # 5) Roll the valid-prefix frontier back past any rejected positions;
            #    the next chunk overwrites them before they are ever read.
            frontier = seq.num_tokens - 1
            target.cached_len = min(target.cached_len, frontier)
            draft.cached_len = min(draft.cached_len, frontier)
            if finished:
                return

    def _advance(self, state: _ModelState, tokens: list[int]) -> torch.Tensor:
        """Feed everything beyond the model's frontier as one chunk.

        Returns float32 logits ``[1, chunk_len, vocab]`` for the fed positions.
        """
        pending = tokens[state.cached_len :]
        input_ids = torch.tensor([pending], dtype=torch.long, device=self._device)
        positions = torch.arange(state.cached_len, len(tokens), device=self._device)[None]
        ctx = PrefillContext(state.cache, start_pos=state.cached_len)
        hidden = state.model(input_ids, positions, ctx)
        state.cached_len = len(tokens)
        return state.model.compute_logits(hidden)

    def _commit(self, seq: Sequence, token_id: int, max_total: int) -> bool:
        """Append one verified token, stamping TTFT and finish time."""
        finished = seq.append_checked(
            token_id,
            eos_token_ids=self._target_config.eos_token_ids,
            max_total_tokens=max_total,
        )
        if seq.first_token_time is None:
            seq.first_token_time = time.perf_counter()
        if finished:
            seq.finish_time = time.perf_counter()
        return finished
