# Your GPU Serves One Request at a Time. It Shouldn't: Continuous Batching From Scratch

> **Status: draft.** Part 3 of a series building [tokamak](https://github.com/WeichunK/tokamak),
> a minimal LLM inference engine in PyTorch. Part 1 built a correct single-sequence
> engine; part 2 rebuilt vLLM's paged KV memory and reclaimed 2× cache capacity.

Part 2 ended with an awkward fact: I'd reclaimed half my KV memory and had nothing
to put in it. The engine still served one request at a time at 19 tokens/second
while a 16 GiB GPU idled. This post adds the piece that turns reclaimed memory
into throughput — an iteration-level scheduler in the style of Orca — and measures
what it buys over both sequential serving and classic static batching.

The headline: on the same 32-request workload and the same GPU, **4.2× the
throughput and 14× better mean time-to-first-token** than the sequential engine —
and it beats static batching at every batch size while being *nicer* to latency,
not worse.

## Why decode hates running alone

Generating one token means one full forward pass: for a 0.6B model, ~600M weights
stream from HBM to compute one row of activations. The arithmetic intensity is
absurdly low — decode is memory-bandwidth-bound (and on a laptop GPU in eager
PyTorch, also kernel-launch-bound). Feeding 16 rows through the same weights costs
nearly the same wall time as feeding 1. Single-sequence decode throws that factor
away.

The classic fix, static batching, pads B requests together and runs the batch to
completion. It works, but it has two structural problems:

1. **Nobody boards a moving bus.** New requests wait for the whole batch to drain
   — time-to-first-token inflates by whoever happens to be longest.
2. **Early finishers hold their seats.** A request that stops after 20 tokens
   occupies its slot while the 200-token straggler finishes the ride alone.

## Scheduling per iteration, not per batch

Orca's insight (OSDI 2022): the natural scheduling quantum is not the request,
it's the *model step*. Every step, the scheduler decides anew:

- **Prefill step**: one waiting request computes its whole prompt in one forward
  pass and joins the running set.
- **Decode step**: every running request advances one token in a single batched
  forward pass — each at its own position, over its own KV history.

Requests join the moment a slot and KV blocks are free, and leave the moment they
emit EOS. The batch reshapes itself token by token. That's the whole idea;
"continuous batching" is scheduling at token granularity.

## What the model has to learn

Batched decode breaks two assumptions my M1 forward baked in: that every row
shares one position, and that attention is either square-causal (prefill) or
single-row (decode). Both fixes live at the model boundary:

**Per-row positions.** `forward(input_ids, positions, ctx)` now takes a
`[batch, seq_len]` positions tensor; RoPE tables are computed per row. Row 0 can
be at position 731 while row 1 is at position 12.

**Step contexts.** Attention layers consume a small protocol instead of a raw
cache:

```python
class BatchedDecodeContext:
    def update(self, layer_idx, k, v):
        for i, (cache, length) in enumerate(zip(self._caches, self._seq_lens)):
            k_i, v_i = cache.update(layer_idx, k[i:i+1], v[i:i+1],
                                    start_pos=length - 1)
            k_pad[i, :, :length] = k_i[0]
            v_pad[i, :, :length] = v_i[0]
        return k_pad, v_pad, self.attn_mask   # True = may attend
```

Each row writes through its *own* per-sequence cache (contiguous or paged — the
context doesn't care), histories are gathered right-padded to the batch max, and
SDPA gets a boolean length mask. Rows stay mathematically independent; padding
just buys one SDPA call instead of B. The invariant that makes all of this safe
is tested directly: decoding two sequences batched together must reproduce each
one's solo logits.

## The scheduler is a queue, a list, and three rules

```
waiting: deque (FCFS)          running: list (arrival order)
```

1. **Admit first.** If a batch slot is free and the pool has blocks for the whole
   prompt, the oldest waiting request prefills *before* the batch decodes.
   Prefill priority favors TTFT; the benchmark shows the cost.
2. **Otherwise decode everyone.** One batched step; finished sequences leave and
   free their blocks immediately.
3. **When the pool runs dry, evict the newest.** Preemption by recomputation:
   the youngest running sequence's blocks are freed, its *generated tokens are
   kept*, and it rejoins the front of the waiting queue (it's older than
   anything there, so FCFS order survives). Resuming re-prefills prompt +
   everything it had generated.

Preemption sounds scary — you're deleting a sequence's working memory mid-thought.
The test that keeps it honest runs two sequences against a pool of *three* KV
blocks, forcing them to evict each other repeatedly, and asserts greedy output
identical to a run with a roomy pool. Recomputation is only invisible if the
paged cache, block manager, scheduler, and engine lifecycle all agree; this one
test crosses every seam.

One design detail worth stealing: a submission-time check guarantees any single
request's worst case fits the pool alone. That invariant is what makes preemption
livelock-free — in the extreme, one sequence runs solo to completion and the
queue drains FCFS.

## The numbers

Workload: 32 chat-like requests (log-normal prompts, exponential generation
lengths, ~2,900 output tokens total), Qwen3-0.6B in bf16 on an RTX 3080 Laptop,
greedy decoding, identical seeded workload for every row. "Static" here is
charitable — finished sequences stop consuming compute, which real padded
implementations don't get.

| Config | Wall (s) | Out tok/s | TTFT mean (s) | TTFT p95 (s) | Latency p95 (s) |
|---|---|---|---|---|---|
| sequential | 187.9 | 15.4 | 106.4 | 179.7 | 185.7 |
| static, batch 4 | 106.9 | 27.1 | 55.8 | 100.5 | 103.3 |
| continuous, batch 4 | 69.0 | 42.0 | 32.6 | 61.7 | 67.2 |
| static, batch 16 | 57.2 | 50.7 | 16.9 | 33.7 | 48.7 |
| continuous, batch 16 | 44.9 | 64.6 | 7.4 | 23.4 | 42.8 |

Three things worth noticing:

**Throughput scales the way the bandwidth argument predicts.** 15.4 → 42.0 →
64.6 tok/s for batch 1 → 4 → 16. Rows through the same weights are nearly free
until the GPU's actual compute limit shows up — a 4.2× win from scheduling alone,
zero new kernels.

**Continuous beats static at the same batch size, and the gap is pure seat
utilization.** 1.55× at batch 4, 1.27× at batch 16. Every one of those percentage
points comes from replacing finished requests at token granularity instead of
letting slots sit empty during the drain. Remember the static baseline here is
charitable; padded static batching would look worse.

**TTFT is the quiet headline.** Mean time-to-first-token at batch 16: 7.4 s
continuous vs 16.9 s static vs 106 s sequential. Prefill-priority admission
starts a new request the moment a slot and blocks free up, instead of making it
wait for the slowest passenger on the previous bus. If you're building anything
interactive, this row of the table is the one that matters.

One honest caveat: continuous batching does not make any *single* request faster
— sequential mode still matches the M2 single-sequence numbers, scheduler
overhead included. It makes a fleet of requests share hardware that one request
can't saturate. Different problem, different fix.

## What's still on the books

The reference implementation's honesty ledger, carried forward from part 2:
every decode step still *copies* each sequence's K/V out of the paged pool
(gather) and into a padded batch tensor. At batch 16 × growing contexts that copy
is the dominant non-GEMM cost, and prefills still run one at a time, stalling
decodes behind long prompts. Both belong to the next milestone: a Triton
paged-attention kernel that reads block tables in place — no gather, no padding.

Code, tests, and reproduction commands:
[github.com/WeichunK/tokamak](https://github.com/WeichunK/tokamak).

## References

- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, OSDI 2022.
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023.
