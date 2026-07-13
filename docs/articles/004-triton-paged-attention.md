# The Bill Comes Due: Writing a Paged-Attention Kernel in Triton

> **Status: draft.** Part 4 of a series building [tokamak](https://github.com/WeichunK/tokamak),
> a minimal LLM inference engine in PyTorch. Parts 2–3 rebuilt vLLM's paged KV
> memory and Orca's continuous batching; both carried a deliberate debt.

Two milestones ago I made a trade and wrote it down: the paged KV cache would
*copy* every sequence's cache out of the block pool on every decode step — a
"gather" — because a copy you can audit beats a kernel you can't debug. Part 3
stacked continuous batching on top and added a padding copy and a per-row Python
loop. The debt compounded: at batch 16, the reference decode attention costs
**7.3 milliseconds per layer per step** on my GPU. Times 28 layers, that's the
whole engine.

This post pays it off with ~80 lines of Triton: a paged-attention decode kernel
that walks block tables *inside* the kernel, reads the pool in place, and turns
out to be **4–33× faster than the reference — and, surprisingly, faster than the
"ideal" no-gather SDPA baseline I'd been treating as the ceiling.**

## Why the reference path is slow, precisely

Per decode step, per layer, the reference does:

1. **Gather**: for each sequence, index its blocks out of the pool into a fresh
   contiguous tensor (a copy of the entire cache-so-far — double the KV bytes).
2. **Pad**: right-pad all sequences to the batch maximum so one SDPA call can
   run (more copies, plus compute on padding that a mask then throws away).
3. **Loop**: steps 1–2 happen in a Python loop over rows, so kernel-launch
   overhead scales with batch size.

None of this is *wrong* — it passed every equivalence test and it's ten
auditable lines. It's just expensive in exactly the dimension serving cares
about: bytes moved per token.

## The kernel: block tables walk into an online softmax

The fix is the same one vLLM's PagedAttention kernel uses: don't make memory
contiguous for the kernel — teach the kernel the page table.

Each Triton program handles one (sequence, KV head) pair. GQA does the first
trick: all the query heads that share a KV head ride along in the same program
as a `[group, head_dim]` block, so every K/V tile loaded from HBM is reused
`group` times.

The main loop is the flash-attention online softmax, one physical block at a
time:

```python
for b in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
    block_id = tl.load(block_tables_ptr + seq * stride_bts + b * stride_btb)
    k_tile = tl.load(k_ptr + block_id * stride_kb + ...)      # [BLOCK_SIZE, D]
    scores = tl.sum(q[:, None, :] * k_tile[None, :, :], 2) * scale
    scores = tl.where(token_idx < seq_len, scores, float("-inf"))

    m_new = tl.maximum(m, tl.max(scores, 1))                  # running max
    rescale = tl.exp(m - m_new)                               # re-normalize past
    p = tl.exp(scores - m_new[:, None])
    acc_norm = acc_norm * rescale + tl.sum(p, 1)
    acc = acc * rescale[:, None] + tl.sum(p[:, :, None] * v_tile[None], 1)
    m = m_new
```

The `block_id` load is the whole idea — the "pointer chase" that makes paged
memory free to read. Everything else is textbook flash attention, in fp32 over
bf16 storage, with one tile of memory per program no matter how long the
context.

Details that earn their keep:

- **Non-power-of-two GQA groups.** `tl.arange` demands powers of two, and a
  model with 7 query heads per KV head exists sooner or later. Pad the group,
  mask loads and stores. There's a test for exactly this.
- **No `tl.dot`.** Tensor cores want 16×16 tiles; decode query groups are 1–8
  rows. vLLM pads to make tensor cores applicable; here, element-wise
  multiply-and-sum is simpler and the kernel is bandwidth-bound anyway.
- **Decode only.** Prefill stays on SDPA — for square causal attention, SDPA
  *is* already a fused flash kernel. Rewriting it is effort without a measured
  problem, which is this project's definition of waste.

## Where the kernel plugs in

M3 left one seam: attention layers consume a "step context" object. But the old
contract — context returns (K, V, mask), model runs SDPA — can't express "the
kernel does the attention". So the contract collapsed to a single method:

```python
class StepContextProtocol(Protocol):
    def attend(self, layer_idx, q, k, v) -> Tensor: ...
```

The model projects, rotates, delegates. Reference contexts implement `attend`
with SDPA; the Triton context implements it with one batched slot-scatter (the
write) and one kernel launch (the read). Swapping is a constructor argument:
`LLM(..., attention_backend="triton")`, with `"auto"` picking the kernel when
CUDA, the paged backend, and triton are all present.

A note for Windows users, since I am one: official Triton has no Windows
wheels, but the community `triton-windows` port compiled and passed every
kernel test on the first attempt. The alternative was moving development into
WSL2 — still reserved for the vLLM benchmark milestone, which needs Linux
anyway.

## Numbers

Microbenchmark, isolating one decode step's attention (write + attend) at
Qwen3-0.6B shapes, bf16, scattered block tables, RTX 3080 Laptop:

| batch | context | reference (µs) | kernel (µs) | no-gather SDPA (µs) | speedup |
|---|---|---|---|---|---|
| 1 | 512 | 965 | 241 | 368 | 4.0× |
| 8 | 512 | 3,954 | 271 | 790 | 14.6× |
| 16 | 512 | 7,251 | 323 | 1,487 | 22.5× |
| 16 | 2,048 | 11,495 | 806 | 5,760 | 14.3× |
| 32 | 512 | 14,401 | 440 | 2,864 | 32.7× |
| 32 | 2,048 | 22,599 | 1,409 | 11,331 | 16.0× |

The column I didn't expect: the kernel beats the *no-gather contiguous SDPA*
"upper bound" everywhere. Eager SDPA at decode shapes (query length 1) pays
padded-batch bookkeeping and separate softmax passes; a fused single-pass kernel
just… doesn't. The ceiling I'd been benchmarking against since M2 was actually a
floor with good PR. Lesson recorded: baselines are claims, and claims need
measuring.

Also worth seeing: kernel time grows sub-linearly with batch (241 → 440 µs for
1 → 32 sequences) — launch overhead amortizes and GQA reuse works, which is the
microscopic version of why continuous batching pays.

At the engine level, on the same 32-request workload as part 3 (out tok/s,
sdpa → triton):

| Config | Reference (SDPA) | Triton kernel | Kernel gain |
|---|---|---|---|
| sequential | 15.4 | 21.0 | 1.36× |
| continuous, batch 4 | 42.0 | 77.9 | 1.85× |
| continuous, batch 16 | 64.6 | **176.6** | **2.7×** |

The gain grows with batch size because the thing the kernel deletes — the
gather — scales with rows. Compounded across the series: the engine started at
15.4 tok/s and now serves the same workload at **176.6 tok/s, an 11.5×
improvement** on identical hardware, with mean time-to-first-token down from
106 seconds to 2.7. Every factor in that product is a separate, measured,
tested idea: paged memory made concurrency affordable, continuous batching
turned it into throughput, and the kernel removed the tax the first two paid
for correctness-first implementations.

## What correctness looks like for a kernel

The kernel faces the same three-layer gauntlet everything else in this repo
does, adapted to kernel failure modes:

1. **Against explicit math** (fp64 reference, no SDPA anywhere): scattered
   non-monotonic block tables, partial blocks, exact block boundaries,
   single-token sequences, both dtypes, and a 4,089-token context spread over
   256 shuffled blocks to stress the online softmax's numerical stability.
2. **Against the reference backend, end to end**: greedy generation with
   `attention_backend="triton"` must be token-identical to `"sdpa"` on real
   Qwen3-0.6B weights.
3. **Against the rest of the system**: the M2 fragmented-pool tests and the M3
   preemption-invisibility test keep running unchanged — the kernel composes
   with the same block manager and scheduler state, or those fail.

## What's next

The engine now has the full vLLM core trio — paged memory, continuous batching,
and a fused paged-attention kernel — each measured against the version before
it. Next milestone: speculative decoding, where a draft model proposes tokens
and the target model verifies them in one pass, provably changing nothing about
the output distribution. After that, the reckoning: benchmarking the whole
stack against vLLM itself and writing down honestly where the gap comes from.

Code, tests, and reproduction commands:
[github.com/WeichunK/tokamak](https://github.com/WeichunK/tokamak).

## References

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023.
- Dao et al., *FlashAttention*, NeurIPS 2022; Dao, *FlashAttention-2*, 2023.
- Tillet et al., *Triton: an intermediate language and compiler for tiled neural network computations*, MAPL 2019.
