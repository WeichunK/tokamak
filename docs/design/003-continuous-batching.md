# 003 — Continuous Batching (M3): iteration-level scheduling

**Status:** implemented
**Scope:** batch-aware model forward (step contexts, per-row positions), FCFS
scheduler with prefill priority and preemption by recomputation, static-batching
baseline mode, throughput benchmark with TTFT/latency percentiles.

## The problem M2 left behind

The paged pool reclaimed ~2× KV capacity, but the engine still ran one request at
a time — reclaimed memory with no tenant. Single-sequence decode on this hardware
is launch-latency-bound (~19 tok/s while the GPU idles); the linear layers that
dominate compute cost nearly the same whether they process 1 row or 16. Batching
is where paged memory turns into throughput.

The classic approach — static batching — pads B requests together and runs them
to completion. Two structural problems: a batch admits nobody until it *fully
drains* (head-of-line blocking for TTFT), and early finishers hold their slot
while the longest request runs alone at batch-of-one efficiency.

## The idea, borrowed from Orca

Iteration-level scheduling (Yu et al., OSDI 2022): make the scheduling decision
*every model step*, not every batch. A step is either

- **one prefill** — a waiting request computes all its accumulated tokens in one
  forward pass and joins the running set, or
- **one batched decode** — every running request advances one token in a single
  forward pass, each at its own position, each over its own KV history.

Requests join the moment a slot and blocks are free and leave the moment they
finish. The batch composition changes token by token.

## What had to change in the model

Batched decode breaks two M1 assumptions: all rows share one position, and the
attention matrix is either square-causal or single-row-full. Both fixes live at
the model boundary and leave the layer math untouched:

- **Per-row positions.** `forward(input_ids, positions, ctx)` takes a
  `[batch, seq_len]` position tensor; RoPE tables are computed per row.
- **Step contexts.** Attention consumes a `StepContextProtocol` instead of a raw
  cache. `PrefillContext` is the M1 path (square, causal, mask-free).
  `BatchedDecodeContext` writes each row's token at its own position through its
  own per-sequence cache, gathers the histories right-padded to the batch
  maximum, and hands SDPA a boolean length mask. Rows are mathematically
  independent — padding buys one SDPA call instead of B.

Batching strategy and storage layout stay orthogonal: contexts work over any
`KVCacheProtocol` (contiguous or paged), so the M2 equivalence guarantees carry
over unchanged.

## Scheduler policy (and why)

- **FCFS, prefill-prioritized.** A waiting request is admitted as soon as a batch
  slot and enough free blocks for its whole prompt exist — before the running
  batch decodes. This favors TTFT at some ITL cost; it is vLLM v0's trade, and
  the benchmark quantifies it.
- **Admission gate = blocks for the prompt, on hand now.** No lookahead
  reservation for future growth; growth is handled by demand allocation plus the
  safety valve below. (vLLM adds a watermark heuristic; deliberately omitted
  until the benchmark shows churn.)
- **Preemption by recomputation.** When the pool cannot cover someone's next
  token, the newest-arrived running sequence is evicted: blocks freed, generated
  tokens kept, requeued at the *front* of the waiting queue (it is older than
  everything else waiting, so FCFS order is preserved). Resuming re-prefills
  prompt + generated-so-far. There is no swap-to-CPU tier — recompute is the
  only eviction, matching vLLM's default.
- **Livelock safety.** Submission-time validation guarantees any single request's
  worst case fits the pool alone (`kv_pool_tokens >= max_seq_len` is enforced at
  construction). Preemption therefore always makes progress: in the extreme, one
  sequence runs solo to completion, then the queue drains FCFS.
- **Static mode as the baseline.** `scheduling="static"` fills the batch once and
  admits nothing until it drains — deliberately *charitable* to static batching
  (finished rows stop consuming compute, unlike a padded implementation), so the
  measured continuous-batching win is a lower bound.

## Ownership boundaries

| Component | Owns | Does not know about |
|---|---|---|
| `Scheduler` | queues, admission, preemption, block *accounting* | tensors, sampling, caches |
| `LLM` engine | cache/generator lifecycle, forwards, sampling, stop conditions | block arithmetic, queue order |
| Step contexts | (position → storage) dispatch, padding, masks | scheduling, lifecycle |

The one subtle handoff: on preemption the scheduler frees the victim's *blocks*,
but the engine owns the victim's cache view, so `ScheduledBatch.preempted` carries
evicted sequences and the engine drops their views before running the step. A
fresh view is built on every (re)prefill, so a stale block table can never be
reused — this is also why views are cheap by design.

## Correctness strategy

1. **Batched-vs-sequential equivalence** (tiny models, CPU): decoding two
   sequences of different lengths batched together must reproduce each one's
   solo logits — padding, masks, and per-row RoPE all covered.
2. **Scheduler policy tests** (no torch): FCFS order, prefill priority, batch-size
   and block-availability admission gates, preemption victim choice and
   front-of-queue requeue, static drain, livelock guards.
3. **Preemption invisibility** (real weights): a 3-block pool forces two growing
   sequences to evict each other repeatedly; greedy outputs must be identical to
   a roomy-pool run. Recomputation is only correct if the paged cache, block
   manager, scheduler, and engine lifecycle agree — this test crosses all of them.
4. **HF parity unchanged**: the parametrized parity suite now exercises the
   continuous-batching engine end to end on both KV backends.

## Known limitations

- Prefills run one sequence per step (no prefill batching or chunked prefill);
  long prompts stall decodes for everyone — visible as ITL spikes under load.
- The decode gather + pad copies every sequence's K/V each step; at large batch ×
  long context this dominates. Both are the M4 kernel's job.
- Per-row Python loops (cache updates, sampling) put a scheduler-overhead floor
  under each step; acceptable for a reference engine, measured in the benchmark.
- No prefix sharing between sequences (blocks are private), no multi-`generate`
  concurrency (one call at a time).

## References

- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative
  Models*, OSDI 2022.
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with
  PagedAttention*, SOSP 2023 — scheduler/preemption design.
