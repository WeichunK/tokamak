"""Scheduler policy tests: FCFS admission, prefill priority, preemption, static mode.

Pure bookkeeping — no torch, no model. Sequences are driven by marking tokens
appended / finished by hand, exactly as the engine would.
"""

import pytest

from tokamak.engine.scheduler import Scheduler, StepKind
from tokamak.engine.sequence import FinishReason, Sequence, SequenceStatus
from tokamak.memory import BlockManager, OutOfBlocksError
from tokamak.sampling_params import SamplingParams


def make_seq(seq_id: int, prompt_len: int = 4, max_new: int = 8) -> Sequence:
    return Sequence(
        seq_id,
        prompt_token_ids=[1] * prompt_len,
        sampling_params=SamplingParams(max_new_tokens=max_new),
    )


def make_scheduler(
    num_blocks: int = 16,
    block_size: int = 4,
    max_batch_size: int = 4,
    admission: str = "continuous",
) -> tuple[Scheduler, BlockManager]:
    manager = BlockManager(num_blocks, block_size)
    return Scheduler(manager, max_batch_size, admission), manager  # type: ignore[arg-type]


def test_prefills_come_first_in_fcfs_order() -> None:
    scheduler, _ = make_scheduler()
    seqs = [make_seq(i) for i in range(3)]
    for seq in seqs:
        scheduler.add(seq)

    for i in range(3):
        batch = scheduler.schedule()
        assert batch is not None
        assert batch.kind is StepKind.PREFILL
        assert batch.seqs == [seqs[i]]
        assert seqs[i].status is SequenceStatus.RUNNING
        seqs[i].append_output_token(0)  # engine samples one token per prefill

    batch = scheduler.schedule()
    assert batch is not None
    assert batch.kind is StepKind.DECODE
    assert batch.seqs == seqs  # arrival order preserved


def test_admission_respects_max_batch_size() -> None:
    scheduler, _ = make_scheduler(max_batch_size=2)
    seqs = [make_seq(i) for i in range(3)]
    for seq in seqs:
        scheduler.add(seq)

    assert scheduler.schedule().kind is StepKind.PREFILL
    assert scheduler.schedule().kind is StepKind.PREFILL
    batch = scheduler.schedule()
    assert batch.kind is StepKind.DECODE  # batch full: third request waits
    assert len(batch.seqs) == 2

    seqs[0].finish(FinishReason.STOP)
    batch = scheduler.schedule()
    assert batch.kind is StepKind.PREFILL  # slot freed: admit the third
    assert batch.seqs == [seqs[2]]


def test_admission_waits_for_free_blocks() -> None:
    # Pool of 4 blocks; the first sequence's prompt takes 3, the second needs 2.
    scheduler, manager = make_scheduler(num_blocks=4, block_size=4)
    first = make_seq(0, prompt_len=12)
    second = make_seq(1, prompt_len=8)
    scheduler.add(first)
    scheduler.add(second)

    assert scheduler.schedule().kind is StepKind.PREFILL
    batch = scheduler.schedule()
    assert batch.kind is StepKind.DECODE  # second's prompt does not fit yet
    assert batch.seqs == [first]

    # The engine frees a finished sequence's blocks immediately (before the
    # next schedule() call) — emulate that ordering here.
    first.finish(FinishReason.STOP)
    manager.free(first.seq_id)
    batch = scheduler.schedule()
    assert batch.kind is StepKind.PREFILL
    assert batch.seqs == [second]


def test_decode_allocates_next_token_capacity() -> None:
    scheduler, manager = make_scheduler(num_blocks=8, block_size=4)
    seq = make_seq(0, prompt_len=4)
    scheduler.add(seq)
    scheduler.schedule()  # prefill: 1 block for 4 tokens
    assert manager.reserved_tokens(0) == 4

    seq.append_output_token(0)  # now 5 tokens
    scheduler.schedule()  # decode step must cover token 5
    assert manager.reserved_tokens(0) == 8


def test_preemption_evicts_newest_and_requeues_front() -> None:
    # 4 blocks of 4: two sequences of 8 tokens fill the pool exactly.
    scheduler, manager = make_scheduler(num_blocks=4, block_size=4, max_batch_size=4)
    old, new = make_seq(0, prompt_len=8, max_new=16), make_seq(1, prompt_len=8, max_new=16)
    scheduler.add(old)
    scheduler.add(new)
    scheduler.schedule()
    old.append_output_token(0)
    scheduler.schedule()
    new.append_output_token(0)

    # Both now need a 3rd block for token 9; only 0 are free -> evict `new`.
    batch = scheduler.schedule()
    assert batch.kind is StepKind.DECODE
    assert batch.seqs == [old]
    assert batch.preempted == [new]
    assert new.status is SequenceStatus.WAITING
    assert scheduler.waiting[0] is new  # front of the queue, not the back
    assert manager.block_table(new.seq_id) == []  # blocks returned

    # Victim resumes: once `old` finishes and frees, `new` is re-prefilled
    # over prompt + its generated token.
    old.finish(FinishReason.STOP)
    manager.free(old.seq_id)
    batch = scheduler.schedule()
    assert batch.kind is StepKind.PREFILL
    assert batch.seqs == [new]
    assert new.num_tokens == 9  # generated token survived preemption


def test_static_mode_drains_before_refilling() -> None:
    scheduler, manager = make_scheduler(max_batch_size=2, admission="static")
    seqs = [make_seq(i) for i in range(3)]
    for seq in seqs:
        scheduler.add(seq)

    scheduler.schedule()
    scheduler.schedule()  # batch of two admitted
    for seq in seqs[:2]:
        seq.append_output_token(0)
    assert scheduler.schedule().kind is StepKind.DECODE

    seqs[0].finish(FinishReason.STOP)
    manager.free(0)
    batch = scheduler.schedule()
    assert batch.kind is StepKind.DECODE  # slot free, but static mode won't admit
    assert batch.seqs == [seqs[1]]

    seqs[1].finish(FinishReason.STOP)
    manager.free(1)
    batch = scheduler.schedule()
    assert batch.kind is StepKind.PREFILL  # batch drained: refill begins
    assert batch.seqs == [seqs[2]]


def test_oversized_prompt_raises_when_pool_is_empty_and_free() -> None:
    scheduler, _ = make_scheduler(num_blocks=2, block_size=4)
    scheduler.add(make_seq(0, prompt_len=12))  # 3 blocks > 2-block pool
    with pytest.raises(OutOfBlocksError, match="pool can ever provide"):
        scheduler.schedule()


def test_all_finished_returns_none() -> None:
    scheduler, _ = make_scheduler()
    seq = make_seq(0)
    scheduler.add(seq)
    scheduler.schedule()
    seq.finish(FinishReason.STOP)
    assert not scheduler.has_unfinished()
    assert scheduler.schedule() is None
