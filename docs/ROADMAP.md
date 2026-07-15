# Roadmap

tokamak is built milestone by milestone, where each milestone is a working, tested
system that adds one inference-serving technique and measures what it buys. The goal
is not to compete with production engines but to understand them: every subsystem is
implemented from scratch in PyTorch, validated for correctness against a reference
implementation, and benchmarked before and after.

Legend: ✅ done · 🚧 in progress · ⬜ planned

## M0 — Project scaffolding ✅

Tooling and repository hygiene: `uv` for environments and locking, `ruff` for linting
and formatting, `mypy --strict` for type checking, `pytest` with `gpu`/`model` markers
so CI stays CPU-only, GitHub Actions CI, pre-commit hooks.

**Exit criteria:** CI is green on a clean checkout; `uv sync && uv run pytest` works
on Windows and Linux.

## M1 — Correct single-sequence engine ✅

A from-scratch decoder-only transformer (RMSNorm, RoPE, GQA, SwiGLU, optional QK-norm)
covering the Llama / Qwen2 / Qwen3 families, loading safetensors checkpoints directly
from the Hugging Face Hub. Contiguous per-sequence KV cache, greedy and
temperature/top-k/top-p sampling, and an offline `LLM.generate()` API. No batching —
this milestone is the correctness anchor and the performance baseline that every later
milestone is measured against.

**Exit criteria:** logits match the Hugging Face reference implementation (fp32) within
tolerance; greedy decoding is token-identical to `transformers.generate`; a latency
baseline (prefill ms, decode tok/s) is recorded.

## M2 — Paged KV cache ✅

Block-based KV cache management in the style of vLLM's PagedAttention: a block
manager that allocates fixed-size KV blocks from a global pool, per-sequence block
tables, and a reference paged-attention implementation in pure PyTorch. Removes the
"one contiguous buffer per sequence" restriction and the memory fragmentation that
comes with it.

**Exit criteria (met):** paged attention is numerically equivalent to the contiguous
implementation (bitwise on tiny models incl. fragmented pools; HF-parity on real
weights for both backends); measured reservation waste drops from 50.1% to 2.0%
(< 1 block/sequence). Known cost: reference gather decodes ~17% slower
single-sequence — the recorded motivation for M4. Design notes:
[design/002-paged-kv-cache.md](design/002-paged-kv-cache.md).

## M3 — Continuous batching ✅

An iteration-level scheduler (in the style of Orca): requests join and leave the
running batch at every engine step instead of waiting for the batch to drain.
Separate prefill and decode phases, FCFS admission with preemption when KV blocks
run out.

**Exit criteria (met):** throughput scales with concurrent requests (see
[benchmarks/README.md](../benchmarks/README.md) for the recorded curve vs.
sequential and static batching); preemption by recomputation is provably invisible
in greedy outputs (tested against a 3-block pool on real weights). Design notes:
[design/003-continuous-batching.md](design/003-continuous-batching.md).

## M4 — Custom attention kernels ✅

A Triton paged-attention decode kernel to replace the PyTorch reference
implementation, plus a benchmark harness comparing both against
`torch.nn.functional.scaled_dot_product_attention` on contiguous inputs.

**Exit criteria (met):** the kernel passes the reference equivalence suite
(fp64 explicit-math comparison over scattered tables, both dtypes, 4k-token
online-softmax stability; engine-level greedy token-identical to the SDPA path)
and microbenchmarks are recorded: 4.0–32.7× over the reference gather path,
faster than no-gather eager SDPA at every measured shape, single-sequence decode
21.0 tok/s vs the 15.8 reference / 19.0 contiguous baselines. Design notes:
[design/004-triton-paged-attention.md](design/004-triton-paged-attention.md).

## M5 — Speculative decoding ✅

Draft-model speculative decoding with rejection sampling (Leviathan et al., 2023;
Chen et al., 2023): a small draft model proposes k tokens, the target model verifies
them in a single forward pass, and rejection sampling preserves the target
distribution exactly.

**Exit criteria (met):** output distribution provably unchanged — greedy outputs
token-identical to plain decoding for self- and foreign drafts; 40k-round empirical
distribution tests within TV 0.01 of the target with acceptance matching the
analytic Σ min(p, q). Acceptance and speedup recorded — and honest: on this
launch-latency-bound stack the draft:target step-cost ratio is ~0.7 (vs. the ~0.2
the theory wants), so speculation measures 0.74× at k=2 with 57% acceptance; the
predicted and measured slowdowns agree, which is the point. Design notes:
[design/005-speculative-decoding.md](design/005-speculative-decoding.md).

## M6 — Benchmark against vLLM ✅

Throughput and latency comparison against vLLM on identical hardware and
byte-identical workloads (the seeded chat-like trace both engines import from
`benchmarks/workload.py`; raw token ids, so content and tokenization are
controlled), run under WSL2, with the gap decomposed by vLLM's own switches:
5.5× total = 2.12× kernels/engine (eager vs. eager, matched concurrency)
× 2.33 CUDA graphs × 1.11 admission headroom. Per-request TTFT/ITL is reported
tokamak-side only — vLLM 0.10's offline API (the newest this machine's driver
runs) does not expose it. Analysis and what closing each factor would take:
[design/006-vllm-gap-analysis.md](design/006-vllm-gap-analysis.md).

**Exit criteria:** reproducible benchmark scripts, published numbers, and a written
gap analysis.

## M7 — Experimental attention backends (stretch) ✅

Attention *policies* (sliding window, window + StreamingLLM sinks) behind the
engine's attention seam: banded masks in the SDPA contexts, a two-phase
block-table walk in the Triton kernel, and mid-flight reclamation of blocks
behind the window. Measured trade-offs: window-only collapses perplexity
(+719% at budget 1024) while +4 sinks holds it to +3.1%; residency drops 5.7×;
and when 8 concurrent long generations share a pool the reclaimed capacity is
worth 1.51× throughput at +9% PPL. DSA / Gated DeltaNet were deliberately
descoped — retrofitting training-time architectures onto a dense checkpoint
yields meaningless quality numbers; a natively-trained hybrid model family is
the honest follow-up. Design notes:
[design/007-attention-policies.md](design/007-attention-policies.md).
