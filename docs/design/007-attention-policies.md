# 007 — Experimental attention backends (M7): policies, not architectures

**Status:** complete
**Headline:** on a pretrained dense model, a sliding window alone is
catastrophic (+719% perplexity at a 1024-token budget) while the same window
plus 4 sink tokens costs +3.1% — and bounded visibility converts, via paged
block reclamation, into a measured **1.51× throughput win** when concurrent
long generations outgrow a shared pool (5.7× lower per-sequence KV residency).
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

RTX 3080 Laptop (WSL2 numbers are not comparable — these ran on Windows),
Qwen3-0.6B bf16, Triton decode backend.

**Quality** (`benchmark_quality.py`): teacher-forced perplexity over 16,384
tokens of *War and Peace* in 4,096-token segments.

| Policy | KV budget | PPL | vs. full |
|---|---|---|---|
| full | 4,096 | 26.65 | — |
| window:1024 | 1,024 | 218.26 | +719% |
| window:512 | 512 | 397.89 | +1,393% |
| window:256 | 256 | 726.78 | +2,627% |
| streaming:1024+4 | 1,028 | 27.47 | **+3.1%** |
| streaming:512+4 | 516 | 29.05 | **+9.0%** |
| streaming:256+4 | 260 | 31.76 | **+19.2%** |

The StreamingLLM curve reproduces exactly: evicting the earliest positions is
what breaks a window (softmax has nowhere to park surplus attention mass), and
4 pinned tokens — 0.1% of the context — buy back almost the entire collapse.
This doubles as an end-to-end correctness check on every mask and kernel path:
no plausible masking bug produces *this* pattern.

**Single sequence, deep context** (`benchmark_streaming.py`): 3,072 greedy
tokens; "deep" isolates tokens 1,024–3,072.

| Policy | tok/s overall | tok/s deep | Peak KV tokens |
|---|---|---|---|
| full | 22.3 | 22.3 | 3,088 |
| window:512 | 21.9 | 21.4 | 528 |
| streaming:512+4 | 22.6 | 24.4 | **544** |

Residency drops 5.7×; throughput does not move. Honest and expected: at batch
1 this stack sits on the M1 launch-overhead floor (~45 ms/step on Windows), and
a 3k-context attention read on a 0.6B model costs a rounding error against it.
Windowing buys *memory*, and memory only becomes *time* when something
contends for it —

**Concurrent generations against a shared pool**: 8 requests × 2,048 new
tokens racing for an 8,192-token pool (each full-attention sequence peaks at
~2,100 tokens of residency, so at most ~4 fit; windowed residency is ~540).

| Policy | Wall (s) | Out tok/s | |
|---|---|---|---|
| full | 144.9 | 113.1 | pool thrash: admission stalls + preemption-by-recompute |
| window:512 | 97.5 | 168.1 | 1.49× |
| streaming:512+4 | 95.7 | 171.2 | **1.51×** |

The paged dividend, measured: reclamation keeps every sequence's footprint
near ``sinks + window``, the same pool admits all 8 at once, and the scheduler
stops burning steps on recompute. Quality cost of that 1.51×: +9% PPL.

## What M7 does not claim

- No position rolling: evaluation stays within the trained context (4,096 ≪
  Qwen3's 40k); these policies bound cache, they do not extend context.
- The single-sequence table does not show windowed attention beating full on
  *speed* — on this launch-bound stack at this model scale it cannot, and the
  numbers say so plainly.
- DSA / Gated DeltaNet remain out of scope for the reason in the scoping
  section: without natively-trained weights their quality column would be
  meaningless. Supporting a hybrid-architecture model family is the honest
  follow-up.
