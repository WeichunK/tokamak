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

## M5 — Speculative decoding ⬜

Draft-model speculative decoding with rejection sampling (Leviathan et al., 2023;
Chen et al., 2023): a small draft model proposes k tokens, the target model verifies
them in a single forward pass, and rejection sampling preserves the target
distribution exactly.

**Exit criteria:** output distribution is provably unchanged (greedy outputs
token-identical; sampled outputs pass a distribution test); acceptance-rate and
speedup measurements are recorded.

## M6 — Benchmark against vLLM ⬜

Throughput, latency (TTFT / ITL), and goodput comparison against vLLM on identical
hardware and workloads (ShareGPT-style traces), with an honest analysis of where the
gap comes from — kernels, scheduling, memory management — and what it would take to
close it.

**Exit criteria:** reproducible benchmark scripts, published numbers, and a written
gap analysis.

## M7 — Experimental attention backends (stretch) ⬜

The engine's attention abstraction reused for efficiency-oriented attention variants
(e.g. sliding-window attention, DeepSeek sparse attention, Gated DeltaNet-style linear
attention) as pluggable experimental backends, with quality/throughput trade-off
measurements.
