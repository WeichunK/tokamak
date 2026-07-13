# 004 — Triton Paged-Attention Kernel (M4): deleting the gather

**Status:** implemented
**Scope:** a Triton decode kernel that reads the paged pool in place through block
tables, a kernel-backed step context, an `attention_backend` engine switch with
auto-detection, and an attention microbenchmark.

## The problem M2/M3 left behind

The reference paged decode path pays for its auditability twice per step and per
layer: every sequence's K/V is **gathered** out of the block pool into a fresh
contiguous tensor, then **padded** to the batch's maximum length, with a Python
loop over rows doing per-sequence bookkeeping — roughly double the KV bytes moved,
plus allocator and launch overhead that grows with batch size. Microbenchmarked in
isolation (16 sequences at 512-token context, Qwen3-0.6B attention shape), the
reference costs **7.3 ms per layer per step**; 28 layers of that dwarfs the GEMMs.

## The idea, borrowed from vLLM's PagedAttention kernel

Move the block-table walk *inside* the attention kernel. If the kernel translates
logical positions to physical blocks itself, nothing needs to be contiguous,
nothing is copied, and padding disappears — attention reads exactly the bytes
that exist.

Kernel design (`tokamak/kernels/paged_attention.py`, ~80 lines of Triton):

- **Grid = (sequence, kv_head).** Each program computes one KV head's entire
  query group (GQA): all `num_q_heads / num_kv_heads` queries that share a KV
  head reuse the same K/V tile loads. Rows of the batch never interact — exactly
  the independence the equivalence tests established in M3.
- **Stream one block at a time.** The program loops over the sequence's block
  table, loading one `[block_size, head_dim]` K tile, scoring it against the
  query group, and folding it into a flash-attention-style **online softmax**
  (running max + normalizer), then the matching V tile into the accumulator.
  Memory high-water mark per program: one tile, regardless of context length.
- **fp32 math over bf16 storage.** Scores, softmax statistics, and the output
  accumulator live in fp32; only loads and the final store touch bf16.
- **Non-power-of-two query groups** are padded to the next power of two with
  masked loads/stores (`tl.arange` sizes must be powers of two), so shapes like
  7 query heads per KV head work — covered explicitly in the kernel tests.

Deliberate scope limits, stated where they bite:

- **Decode only.** Prefill stays on SDPA, which is already a fused flash kernel
  for the square causal case; rewriting it would be effort without a measured
  problem.
- **No split-K.** A very long sequence serializes inside one program instead of
  being split across SMs and reduced. Fine at this scale; flash-decoding-style
  splitting is the known next step if long-context decode ever dominates.
- **Element-wise scores instead of `tl.dot`.** Query groups (1–8) are far below
  tensor-core tile sizes; decode attention is bandwidth-bound, so tensor cores
  buy little here. vLLM pads query groups to 16 to use them; measured against
  the alternatives below, this simple kernel already wins by enough.

## The seam: contexts own attention

M3's step contexts returned (K, V, mask) for the model to run SDPA on. That shape
cannot express "the kernel computes attention itself", so the protocol collapsed
to one method:

```python
def attend(self, layer_idx, q, k, v) -> Tensor  # store new K/V, return attention
```

`Attention.forward` now projects, rotates, delegates, projects back — it cannot
tell SDPA from Triton. The kernel context also batches the *write* path: one
scatter per layer over precomputed slot indices, replacing the reference's
per-row Python loop.

Backend selection is an engine constructor argument:
`attention_backend="auto" | "sdpa" | "triton"`, where auto picks the kernel when
the paged backend, CUDA, and triton are all present, and explicit `"triton"`
fails loudly when they are not. Triton itself is an optional extra
(`tokamak-llm[triton]`) — on Windows via the community `triton-windows` port,
which passed every kernel test on the first run.

## Measured results

Microbenchmark (bf16, RTX 3080 Laptop, Qwen3-0.6B attention shape, scattered
block tables; µs per decode step per layer, median of 50):

| batch | context | reference (gather+SDPA) | kernel | contiguous SDPA (no gather) | speedup |
|---|---|---|---|---|---|
| 1 | 512 | 965 | 241 | 368 | 4.0× |
| 8 | 512 | 3,954 | 271 | 790 | 14.6× |
| 16 | 512 | 7,251 | 323 | 1,487 | 22.5× |
| 16 | 2,048 | 11,495 | 806 | 5,760 | 14.3× |
| 32 | 512 | 14,401 | 440 | 2,864 | 32.7× |
| 32 | 2,048 | 22,599 | 1,409 | 11,331 | 16.0× |

Two readings worth writing down:

1. The kernel beats not only the reference but the **no-gather contiguous SDPA
   "upper bound"** — eager-mode SDPA at decode shapes (query length 1) pays
   padded-tensor math and separate softmax passes that a fused single-pass
   kernel simply doesn't. The "upper bound" framing from M2 was too generous to
   SDPA.
2. Kernel cost grows sub-linearly with batch until bandwidth saturates
   (241 → 440 µs for 1 → 32 sequences at 512 context): launches amortize, and
   GQA tile reuse does its job.

Engine-level numbers (same workloads as M2/M3) are recorded in
`benchmarks/README.md` next to the reference rows they replace.

## Correctness strategy

1. **Kernel vs. explicit-math reference** (fp64 reference, scattered non-monotonic
   tables): three GQA shapes including a non-power-of-two group, partial blocks,
   exact block boundaries, single-token sequences, fp32 and bf16, plus a
   4,089-token sequence over 256 shuffled blocks for online-softmax stability.
2. **Context equivalence at engine level**: greedy decoding with
   `attention_backend="triton"` must be token-identical to `"sdpa"` on real
   Qwen3-0.6B weights.
3. **Everything from M2/M3 still holds**: the parity and preemption suites run
   on the reference path unchanged; the kernel path composes with the same block
   manager and scheduler state.

## Known limitations

- Decode-only; prefill and chunked prefill remain SDPA.
- No split-K: single-program-per-sequence serializes very long contexts.
- Kernel-side register pressure grows with `GROUP_PAD × head_dim`; fine for
  groups ≤ 8 at head_dim 128, revisit before supporting e.g. MQA with 32 heads.
- `head_dim` and `block_size` must be powers of two.

## References

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with
  PagedAttention*, SOSP 2023 — the in-kernel block-table walk.
- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with
  IO-Awareness*, NeurIPS 2022 — the online softmax.
- Dao, *FlashAttention-2*, 2023 — work partitioning background.
- Tillet et al., *Triton: an intermediate language and compiler for tiled neural
  network computations*, MAPL 2019.
