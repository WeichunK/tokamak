# Four Tokens That Buy Back a Perplexity Collapse

> **Status: draft.** Part 7 (final) of a series building [tokamak](https://github.com/WeichunK/tokamak),
> a minimal LLM inference engine in PyTorch. Parts 2–6 built paged KV memory,
> continuous batching, a Triton paged-attention kernel, speculative decoding,
> and an honest benchmark against vLLM.

Here is a number that should bother you more than it does: cut a language
model's attention down to the most recent 1,024 tokens — a quarter of a
4,096-token context, and *recent* tokens are obviously the important ones —
and perplexity doesn't drift. It detonates, from 26.7 to 218. Now pin just
four extra tokens, the *first four of the document*, tokens that are mostly a
title and some punctuation. Perplexity: 27.5.

A 8× collapse, repaired almost entirely by four tokens from the wrong end of
the context. This post builds the machinery to measure that (it's the
StreamingLLM effect, and it reproduces beautifully), and then follows the
consequence through the memory system, because bounding what attention can
*see* also bounds what the KV cache must *keep* — and this engine's paged
allocator can finally collect rent on that.

## Policies, not architectures

The roadmap called this milestone "experimental attention backends" and named
sliding windows, DeepSeek sparse attention, and Gated DeltaNet as candidates.
Two of those didn't survive contact with this project's one rule: prove it
correct, then measure what it buys.

DSA and Gated DeltaNet are *training-time* architectures. Bolt them onto a
model trained with dense softmax attention and you get an engine that runs
fast and emits garbage — a throughput table with a quality column you'd have
to leave blank or lie about. Sliding windows are different: they're
*inference-time* restrictions of the same computation the model was trained
on, which means a pretrained dense model can run them, and the damage they do
is honestly measurable against the exact baseline. That's the milestone:
policies over a fixed model, priced in perplexity.

A policy answers one question — which cached positions may a query at position
`p` see? Three answers: everything (`full`); the last `W` positions
(`window:W`); the last `W` plus the first `S`, always (`streaming:W+S`).

## Where a policy plugs into an engine

Since Part 4, every attention layer in tokamak talks to a per-step context
through one method: `attend(layer_idx, q, k, v)` — the context owns storage
and attention math. A policy is thirty lines of index arithmetic threaded into
the three implementations of that seam:

- the SDPA reference paths turn their causal masks into *banded* causal masks
  with sink columns re-enabled;
- the Triton decode kernel walks each sequence's block table in **two passes**
  — sink blocks first, then the recency band — and never loads a block between
  the two. Full attention is the degenerate case (empty first pass, band from
  zero), so one kernel serves everything.

That "never loads" is a stronger property than masking, and it's what makes
the next section possible. The test that pins it down is my favorite in the
repo: point the dead block-table entries at a NaN-poisoned block; any load
would poison the output through `0 × NaN`; the output comes back finite and
exactly right.

## The measurement: a collapse and a four-token repair

Teacher-forced perplexity, 16,384 tokens of *War and Peace*, each token seeing
exactly what the policy would have let a windowed decode see:

| Policy | KV budget | PPL | vs. full |
|---|---|---|---|
| full | 4,096 | 26.65 | — |
| window:1024 | 1,024 | 218.26 | +719% |
| window:256 | 256 | 726.78 | +2,627% |
| streaming:1024+4 | 1,028 | 27.47 | **+3.1%** |
| streaming:256+4 | 260 | 31.76 | **+19.2%** |

Why do four early tokens matter this much? Softmax attention must distribute
exactly 1.0 of probability mass every step, relevant or not. Trained models
learn to park the surplus on the earliest positions — attention *sinks*
(Xiao et al., 2023). A plain window evicts the parking lot; the mass lands on
whatever recent tokens are nearest, distorting every head, every layer, every
step. Four pinned tokens restore the parking lot for 0.1% of the context.

This table is also the sharpest correctness test the policy stack has. Masks,
kernels, and reclamation all conspire in these numbers; no plausible
implementation bug reproduces *this* precise pattern — collapse without sinks,
near-baseline with them, ordered by budget.

## Collecting the rent: eviction as a table edit

Bounded visibility means bounded *residency*: once the recency band passes a
block (and it isn't a sink block), no present or future query will ever read
it. In a contiguous cache that insight is worthless — the memory is one slab.
In a paged cache (Part 2), eviction is removing an entry from a list. The
block manager returns dead blocks to the pool mid-generation; monotonic,
idempotent, and double-free-safe against the sequence's final cleanup, with
the stale table entries left dangling — which is fine, because the kernel
structurally never dereferences them.

Measured, single sequence, 3,072 generated tokens: peak residency 3,088 →
544 tokens, **5.7× less**. And throughput: 22.3 → 22.6 tok/s. Nothing.

That flat line is the honest row in the table, and Part 1 predicted it: batch-1
decode on this stack sits on a ~45 ms/step launch-overhead floor, and reading
3k tokens of KV for a 0.6B model costs a rounding error against that floor.
Windowing buys memory, not time. Memory becomes time only when something
*contends* for it:

| 8 concurrent × 2,048 tokens, one 8,192-token pool | Wall (s) | Out tok/s | |
|---|---|---|---|
| full | 144.9 | 113.1 | ~4 sequences fit; the rest thrash through preemption |
| streaming:512+4 | 95.7 | 171.2 | all 8 run concurrently — **1.51×** |

Full attention's sequences each grow to ~2,100 tokens of residency; the pool
admits four, and the scheduler burns steps recomputing preempted prefixes.
The windowed pool never exceeds ~540 per sequence, admits all eight, and the
preemption machinery goes quiet. Price of the 1.51×: +9% perplexity, stated
next to it, as it should be.

## What the series adds up to

Seven milestones, one rule throughout. The engine went from 15.4 to 176.6
tok/s on its fixed workload (11.5×, each factor measured); learned where it
stands against vLLM (5.5×, decomposed one flag at a time); shipped one
algorithm that *loses* on this hardware (speculative decoding, 0.74×, with the
arithmetic that predicted it) and one approximation that costs +9% quality for
1.51× throughput — both reported with the price tag attached, because an
engineering claim without its cost isn't a measurement, it's marketing.

What I'd build next, in measured-impact order: a CUDA-graph-captured decode
step (the vLLM ablation bounds it at ≈2.3×), fused non-attention ops, and a
natively-trained hybrid-attention model family — the honest version of the
linear-attention backend this milestone declined to fake.
