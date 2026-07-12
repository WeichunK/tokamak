# Half Your KV Cache Is Empty: Reimplementing vLLM's Paged Memory From Scratch

> **Status: draft.** Part 2 of a series building [tokamak](https://github.com/WeichunK/tokamak),
> a minimal LLM inference engine in PyTorch. Part 1 covered the single-sequence
> engine and its correctness anchor against Hugging Face.

I simulated 2,000 chat-style requests through the naive KV cache I built in the
last post. The result: **50.1% of every byte reserved for attention keys and values
was never touched.** Half the cache, allocated and idle.

This post walks through why that happens, and through reimplementing vLLM's answer
— PagedAttention's block-based memory management — in ~150 lines of readable
PyTorch, tested to bitwise equivalence against the naive implementation.

## Why the naive cache wastes half its memory

A decoder-only transformer generating token by token caches the key/value
projections of every previous position. For Qwen3-0.6B in bf16 that costs

```
2 · 28 layers · 8 kv-heads · 128 head-dim · 2 bytes = 112 KiB per token
```

The naive design gives each request one contiguous buffer. Contiguous buffers
can't grow, so you must size them *before* knowing how long the model will talk:
`prompt_len + max_new_tokens`, the worst case. But real generation stops early —
an EOS after 40 tokens against a 512-token budget strands 472 tokens × 112 KiB of
reserved memory for the request's entire lifetime.

With one sequence on a 16 GiB GPU, nobody notices. The moment you batch requests
(the next milestone), reserved-but-unused memory is precisely the memory that
can't admit another request. vLLM's paper measured 60–80% waste in pre-paging
serving systems; my simulation of a log-normal-prompt, early-stopping workload
lands at 50.1% — same disease, smaller patient.

## The fix is virtual memory, minus the hardware

The insight of PagedAttention (Kwon et al., SOSP 2023) is that operating systems
solved this exact problem fifty years ago: stop requiring logical adjacency to be
physical adjacency.

1. Carve the physical KV pool into fixed-size **blocks** (16 tokens each here,
   matching vLLM's default).
2. Give each sequence a **block table** — an ordered list of block ids. Token
   position `p` lives at `block_table[p // 16]`, slot `p % 16`.
3. Allocate a block only when a sequence actually crosses into it. Return all of
   them the moment the sequence finishes.

A sequence that stops at 40 tokens holds ⌈40/16⌉ = 3 blocks — 48 tokens of
capacity for 40 tokens of content. Waste is bounded by one partial block, ever.
On the same simulated workload the integrated waste drops from **50.1% to 2.0%**,
which means the same pool admits **~2× the concurrent sequences** before anything
else changes.

## The allocator is smaller than you think

The entire block manager is a free list and a dict of tables:

```python
def ensure_capacity(self, seq_id: int, num_tokens: int) -> None:
    table = self._block_tables.setdefault(seq_id, [])
    shortfall = self.blocks_needed(num_tokens) - len(table)
    if shortfall <= 0:
        return
    if shortfall > len(self._free_blocks):
        raise OutOfBlocksError(...)
    for _ in range(shortfall):
        table.append(self._free_blocks.pop())
```

Two properties matter more than the code: growth **appends and never reorders**
(already-written positions must stay addressable), and exhaustion **fails loudly
without corrupting state** (the scheduler milestone turns this exception into
preemption instead of a crash).

The physical side is one tensor per K and V, shaped
`[layers, num_blocks, block_size, kv_heads, head_dim]`, preallocated once at
startup. Writing new tokens is a scatter through the block table; reading is a
gather:

```python
def gather(self, layer_idx, block_table, seq_len):
    k = self.k_cache[layer_idx, block_table].flatten(0, 1)[:seq_len]
    v = self.v_cache[layer_idx, block_table].flatten(0, 1)[:seq_len]
    return k.permute(1, 0, 2).unsqueeze(0), v.permute(1, 0, 2).unsqueeze(0)
```

That gather *copies* the sequence's K/V every layer, every step. This is a
deliberate reference-implementation trade: correctness reduces to "does the gather
reconstruct the logical sequence," which is ten auditable lines. Production
engines don't copy — vLLM's PagedAttention CUDA kernel walks block tables *inside*
the attention kernel. That kernel is a later milestone; this post is about getting
the memory model provably right first.

One abstraction keeps the model code oblivious: attention layers depend on a
three-method protocol (`ensure_capacity` / `update` / `release`). The transformer
never learns that paging exists — swapping backends is a constructor argument.

## Proving equivalence, not eyeballing it

Paged memory has one failure mode that plausible-looking text will happily hide:
a wrong slot calculation that reads position 17 where position 12 should be. So
the tests are adversarial about physical layout:

- **Scattered tables.** Write through a deliberately non-monotonic block table
  like `[5, 2, 7]`, gather, compare bitwise.
- **Fragmented pools.** An interfering sequence allocates and frees interleaved
  blocks so the real sequence's table comes out as `[2, 1, 0]` — then logits must
  match the contiguous backend exactly, prefill and decode, across block
  boundaries.
- **Real weights.** The Hugging Face parity suite from part 1 (max logit diff
  < 1e-3, 32-token greedy generation token-identical) now runs parametrized over
  *both* backends. The paged path faces the same reference the naive path did.

The lesson from part 1 still applies — my RoPE base bug generated fluent text for
19 tokens before diverging. Memory bugs are sneakier than modeling bugs. Test the
allocator like it's trying to lie to you.

## What it costs today

Honest numbers on the same single-sequence workload (Qwen3-0.6B, bf16, RTX 3080
Laptop, 512 prompt + 128 greedy tokens):

| Backend | Decode tok/s | Inter-token |
|---|---|---|
| Contiguous (zero-copy views) | 19.0 | 52.6 ms |
| Paged, reference gather | 15.8 | 63.3 ms |

The gather costs ~17% of single-sequence decode throughput. I'm keeping that
regression on the books deliberately: it is the precise, measured motivation for
the custom attention kernel milestone, and it's honest about what a reference
implementation buys (auditable correctness) and what it doesn't (speed).

Memory, meanwhile, is transformed:

| Policy | Reserved-but-unused KV |
|---|---|
| Contiguous, worst-case sizing | 50.1% |
| Paged, block_size=16 | 2.0% |

## What's next

The paged pool is pointless with one tenant. The next milestone is continuous
batching — an iteration-level scheduler in the style of Orca that admits and
retires requests every engine step, which is where those reclaimed blocks turn
into actual throughput. Then a Triton kernel to delete the gather.

Code, tests, and reproduction commands: [github.com/WeichunK/tokamak](https://github.com/WeichunK/tokamak).

## References

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023.
- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, OSDI 2022.
