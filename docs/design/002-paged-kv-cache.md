# 002 — Paged KV Cache (M2): block-based memory management

**Status:** implemented
**Scope:** block manager, paged physical KV storage, gather-based reference paged
attention, engine integration behind a `kv_backend` switch.

## The problem M1 left behind

The contiguous cache sizes every request's buffer at `prompt + max_new_tokens`
before generation starts, because a contiguous allocation cannot grow. Generation
length is unknowable up front, so the engine reserves for the worst case and most
of it goes unused: on a simulated chat-like workload (log-normal prompts, early
stopping, `max_new_tokens=512`), **50.1% of all reserved KV capacity is never
touched** (`benchmarks/benchmark_kv_memory.py`). vLLM's paper reports the same
figure at 60–80% for production serving systems that predate it.

That waste is invisible with one sequence on a 16 GiB card. It becomes the
binding constraint the moment sequences run concurrently (M3): reserved-but-unused
memory is exactly the memory that cannot admit another request.

## The idea, borrowed from virtual memory

PagedAttention (Kwon et al., SOSP 2023) applies the oldest trick in operating
systems: decouple logical addresses from physical placement.

- The physical KV pool is carved into fixed-size **blocks** of `block_size` tokens,
  preallocated once at engine startup.
- Each sequence owns a **block table** — an ordered list of block ids. Logical
  position `p` lives in `block_table[p // block_size]`, slot `p % block_size`.
- Blocks are allocated on demand as the sequence crosses block boundaries and
  returned to the pool the moment it finishes.

Per-sequence waste collapses from `max_new_tokens`-scale headroom to **less than
one block** (measured 2.0% on the same simulated workload, block_size=16), and the
same pool admits ~2× the concurrent sequences.

## Module responsibilities

| Module | Owns | Does not know about |
|---|---|---|
| `memory/block_manager.py` | which blocks belong to which sequence; free list | tensors, devices, attention |
| `memory/paged_cache.py` | physical storage; (table, position) → slot translation | scheduling, sequence lifecycle |
| `PagedKVCacheView` | one sequence's façade satisfying `KVCacheProtocol` | other sequences |
| `engine/llm.py` | when to grow (`ensure_capacity`) and free (`release`) | block arithmetic |

The `KVCacheProtocol` extracted in this milestone (`ensure_capacity` / `update` /
`release`) is what keeps `Attention.forward` byte-identical across backends — the
model never learns that paging exists.

## Key decisions

### Fixed-size blocks, LIFO free list, no sharing yet

`block_size=16` matches vLLM's default: small enough that slack stays negligible,
large enough that block tables stay short. The free list is LIFO so recently freed
(cache-warm) blocks are reused first. Reference counting for prefix sharing and
copy-on-write are deliberately absent — they belong with the scheduler work (M3+)
and would be untestable speculation today.

### Reads are a gather — on purpose

`PagedKVCache.gather` copies the sequence's blocks into a contiguous
`[1, heads, seq_len, head_dim]` tensor and hands it to SDPA. A copy per layer per
step is the *reference implementation* trade: correctness reduces to "does the
gather reconstruct the logical sequence," which is auditable in ten lines and
testable to bitwise equality. The M4 Triton kernel eliminates the copy by walking
block tables inside the attention kernel — which is precisely what vLLM's
PagedAttention kernel is.

The measured cost of the reference gather on the M1 latency workload is recorded
in `benchmarks/README.md`; it is the price paid until M4.

### The pool is preallocated at startup

`LLM(kv_backend="paged")` allocates `ceil(max_seq_len / block_size)` blocks once,
vLLM-style, rather than growing tensors during serving. Capacity planning becomes
explicit: `max_seq_len` bounds the pool, `OutOfBlocksError` is the signal that M3
turns into preemption instead of a crash.

### Slot addressing must survive scattered blocks

Nothing guarantees a sequence's blocks are physically adjacent or ordered — that
is the entire point. The equivalence tests therefore deliberately fragment the
pool (an interfering sequence allocates and frees interleaved blocks) and assert
logits identical to the contiguous backend with a block table like `[2, 1, 0]`.

## Correctness strategy

1. **Storage roundtrips** (no model): write through a non-monotonic block table,
   gather, compare bitwise — including incremental writes that straddle block
   boundaries.
2. **Backend equivalence on tiny models**: identical logits (prefill + 7 decode
   steps crossing two block boundaries) between paged and contiguous backends,
   with and without a fragmented pool.
3. **Real weights**: the Hugging Face parity suite (logits < 1e-3, greedy
   token-identical) now runs parametrized over both backends, so the paged
   read/write path is validated against the reference implementation end to end.
4. **Allocator invariants**: on-demand growth never reorders existing table
   entries; slack stays `< block_size`; exhaustion raises without corrupting
   state; freed blocks are reusable.

## Known limitations

- Batch size 1 per `update` call; batched paged attention arrives with the M3
  scheduler.
- The gather copies K/V every step (~2× KV bandwidth per token at long context)
  — accepted until the M4 kernel.
- No prefix sharing or copy-on-write; every sequence's blocks are private.
- Block tables live on CPU and are re-uploaded when they grow; fine at one
  sequence, revisited in M3.

## References

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with
  PagedAttention*, SOSP 2023. arXiv:2309.06180.
- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative
  Models*, OSDI 2022 — the scheduler this memory model exists to serve (M3).
