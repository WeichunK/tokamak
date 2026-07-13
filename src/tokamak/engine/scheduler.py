"""Iteration-level scheduler for continuous batching (Orca / vLLM-v0 style).

Every call to :meth:`Scheduler.schedule` plans exactly one engine step: either one
**prefill** (a waiting sequence is admitted and its whole prompt — or its
accumulated tokens, after preemption — is computed) or one **decode** step covering
every running sequence. Requests therefore join and leave the batch at token
granularity instead of waiting for a batch to drain.

Policy choices, all deliberate and documented in ``docs/design/003``:

- **FCFS with prefill priority.** New requests are admitted as soon as a batch
  slot and enough KV blocks for their prompt exist. This favours time-to-first-
  token at some cost to inter-token latency (the same trade vLLM v0 makes).
- **Preemption by recomputation.** When the block pool cannot cover the next
  decode token, the newest-arrived running sequence is evicted: its blocks are
  freed and it rejoins the *front* of the waiting queue (it is older than
  anything else waiting, so FCFS order is preserved). Its generated tokens are
  kept; resuming re-prefills over prompt + generated so far. Recompute-only —
  there is no swap-to-CPU tier.
- **Static mode** (`admission="static"`) exists as the comparison baseline: the
  batch is filled once and no admission happens until it fully drains. This is a
  charitable static-batching model (finished sequences stop consuming compute,
  unlike padded implementations).

The engine guarantees at submission that any single request's worst case fits the
pool alone, so preemption always makes progress: in the extreme, one sequence
runs solo to completion.
"""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from tokamak.engine.sequence import Sequence, SequenceStatus
from tokamak.memory.block_manager import OutOfBlocksError

if TYPE_CHECKING:
    from tokamak.memory.block_manager import BlockManager


class StepKind(enum.Enum):
    """What the engine executes this step."""

    PREFILL = enum.auto()
    DECODE = enum.auto()


@dataclass
class ScheduledBatch:
    """One engine step's worth of work.

    Attributes:
        kind: Prefill (``seqs`` holds one sequence) or decode (all running).
        seqs: Sequences to run this step, in stable arrival order.
        preempted: Sequences evicted while planning this step. The engine must
            drop their per-sequence caches; their blocks are already freed.
    """

    kind: StepKind
    seqs: list[Sequence]
    preempted: list[Sequence] = field(default_factory=list)


class Scheduler:
    """FCFS iteration-level scheduler with preemption by recomputation.

    Args:
        block_manager: The paged pool to allocate from, or ``None`` for the
            contiguous backend (admission is then bounded only by batch size,
            and preemption never triggers).
        max_batch_size: Upper bound on concurrently running sequences.
        admission: ``"continuous"`` admits whenever a slot is free;
            ``"static"`` fills the batch once and waits for it to drain.
    """

    def __init__(
        self,
        block_manager: BlockManager | None,
        max_batch_size: int,
        admission: Literal["continuous", "static"] = "continuous",
    ) -> None:
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
        if admission not in ("continuous", "static"):
            raise ValueError(f"Unknown admission policy: {admission!r}")
        self._manager = block_manager
        self._max_batch_size = max_batch_size
        self._admission = admission
        self._filling = True  # static mode: currently filling the batch
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []

    def add(self, seq: Sequence) -> None:
        """Enqueue a new request (FCFS position = submission order)."""
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        """Whether any request still needs engine steps."""
        return bool(self.waiting) or any(not s.is_finished for s in self.running)

    def schedule(self) -> ScheduledBatch | None:
        """Plan the next engine step; ``None`` when everything is finished."""
        self.running = [s for s in self.running if not s.is_finished]

        preempted: list[Sequence] = []
        while True:
            if not self.running and self._admission == "static":
                self._filling = True

            if self._may_admit():
                candidate = self.waiting[0]
                if self._prompt_fits(candidate):
                    return self._admit(preempted)
                if not self.running:
                    # Nothing running holds blocks, yet the prompt does not fit:
                    # impossible under the submission-time pool invariant.
                    raise OutOfBlocksError(
                        f"sequence {candidate.seq_id} needs more blocks than the "
                        f"pool can ever provide"
                    )
                # Blocks are busy; fall through and decode until some free up.

            if self.running:
                preempted += self._ensure_decode_capacity()
                if self.running:
                    return ScheduledBatch(StepKind.DECODE, list(self.running), preempted)
                continue  # everything was preempted; retry admission with a free pool

            if self.waiting:
                # Unreachable under the invariants above; fail loudly rather
                # than spin if a policy change ever breaks them.
                raise RuntimeError("scheduler stalled with waiting sequences")
            return None

    def _may_admit(self) -> bool:
        if not self.waiting or len(self.running) >= self._max_batch_size:
            return False
        return self._admission != "static" or self._filling

    def _prompt_fits(self, seq: Sequence) -> bool:
        if self._manager is None:
            return True
        needed = self._manager.blocks_needed(seq.num_tokens)
        return needed <= self._manager.num_free_blocks

    def _admit(self, preempted: list[Sequence]) -> ScheduledBatch:
        seq = self.waiting.popleft()
        if self._manager is not None:
            self._manager.ensure_capacity(seq.seq_id, seq.num_tokens)
        seq.status = SequenceStatus.RUNNING
        self.running.append(seq)
        if self._admission == "static" and (
            len(self.running) >= self._max_batch_size or not self.waiting
        ):
            self._filling = False
        return ScheduledBatch(StepKind.PREFILL, [seq], preempted)

    def _ensure_decode_capacity(self) -> list[Sequence]:
        """Reserve every running sequence's next-token block, preempting on demand."""
        if self._manager is None:
            return []
        preempted: list[Sequence] = []
        for seq in list(self.running):
            if seq not in self.running:
                continue  # already evicted as a victim
            while True:
                try:
                    self._manager.ensure_capacity(seq.seq_id, seq.num_tokens)
                    break
                except OutOfBlocksError:
                    victim = self.running[-1]  # newest arrival = lowest priority
                    self._preempt(victim)
                    preempted.append(victim)
                    if victim is seq:
                        break  # the starving sequence evicted itself; skip it
        return preempted

    def _preempt(self, victim: Sequence) -> None:
        """Evict by recomputation: free blocks, requeue at the front."""
        assert self._manager is not None
        self._manager.free(victim.seq_id)
        self.running.remove(victim)
        victim.status = SequenceStatus.WAITING
        self.waiting.appendleft(victim)
