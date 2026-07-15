# 007 — Experimental attention backends (M7): policies, not architectures

**Status:** in progress
**Scope:** inference-time attention *policies* — restrictions on which cached
positions a query may see — as pluggable backends behind the engine's existing
attention seam, with measured quality/memory/throughput trade-offs on a
pretrained dense-attention model (Qwen3-0.6B).

## The scoping decision

The roadmap names three candidate variants: sliding-window attention, DeepSeek
sparse attention, and Gated DeltaNet-style linear attention. This milestone
implements the first family — sliding window and its StreamingLLM refinement
(window + attention sinks) — and deliberately excludes the other two.

The reason is this project's own rule: prove it correct, then measure what it
buys. Sliding-window and sink policies are *inference-time approximations* of
dense attention: applied to a pretrained dense model they produce a measurable
quality curve (perplexity vs. budget) against an exact baseline, which is
precisely the trade-off measurement M7 exists to make. DSA and Gated DeltaNet
are *training-time architectures*: retrofitting them onto a model trained with
dense softmax attention yields meaningless quality numbers, so "measuring the
trade-off" would reduce to benchmarking throughput of an engine that emits
garbage. The honest version of that milestone is supporting a model *trained*
with those mechanisms (e.g. a natively hybrid GDN model family), which is a
model-loader milestone, not an attention-backend one — noted as a future
direction.

## Policies

A policy answers one question per query position `p`: which cached positions
are visible?

| Policy | Visible set | Cache growth |
|---|---|---|
| `full` (default, today's behavior) | `[0, p]` | unbounded |
| `window(W)` | `[max(0, p−W+1), p]` | bounded at `W` |
| `streaming(W, S)` | `[0, S) ∪ [max(S, p−W+1), p]` | bounded at `S + W` |

The StreamingLLM observation gives the family its measurable story: softmax
attention dumps surplus probability mass on the first few positions ("sinks"),
so a plain window that evicts them degrades badly once the context outgrows
the window, while keeping a handful of sink positions restores stability at
essentially no memory cost. The quality harness should reproduce exactly that
curve shape or the implementation is wrong.

Scope limit, documented: positions stay absolute (no StreamingLLM position
rolling), so evaluation stays within the model's trained context length. The
policies bound *cache*, not extrapolation.

## Where policies plug in

The engine's attention seam is `StepContextProtocol.attend(layer_idx, q, k, v)`
(design 004): contexts own storage and attention math. Policies slot in as a
value object (`AttentionPolicy`) resolved once at engine construction and
threaded to the three context implementations:

- **PrefillContext (SDPA reference):** the causal mask becomes a *banded*
  causal mask (plus always-visible sink columns). One mask expression covers
  full/window/streaming — `full` is `window(∞)` with zero sinks.
- **BatchedDecodeContext (SDPA reference):** the padded gather already builds a
  per-row column mask; the policy shrinks it (and the gather itself only copies
  the visible slice, so the reference path gets cheaper too).
- **TritonPagedDecodeContext (M4 kernel):** the kernel's block loop gains a
  per-sequence start block and a sink-block prefix pass; blocks wholly outside
  the visible set are never loaded. Masking within boundary blocks reuses the
  existing `token_idx < seq_len` pattern.

## The paged dividend

Bounded visibility means bounded *residency*: once every query that will ever
run can no longer see a block, the block manager can return it to the pool.
This is the paged-KV synergy (design 002) paying out a second time — the same
block table that made allocation elastic makes *eviction* a table edit, no
copies. Freed logical prefix blocks leave holes the kernel must never read;
the per-sequence window start is what makes the skip safe.

The measured claims to produce:

1. **Quality:** teacher-forced perplexity on long text vs. policy and budget
   (`full` baseline; `window(W)` for several `W`; `streaming(W, S)`); expect
   the StreamingLLM curve — window-only collapses past the budget, sinks
   restore it.
2. **Memory:** peak KV blocks held per sequence vs. policy (bounded vs.
   linear).
3. **Throughput:** long-context decode tok/s vs. context length — full
   attention's per-step cost grows with context, windowed stays flat.

## Results

_(pending implementation)_
