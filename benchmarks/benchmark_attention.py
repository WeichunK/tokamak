"""Decode-attention microbenchmark: reference gather+SDPA vs. the Triton kernel.

Measures one decode step's attention cost in isolation (write new K/V + attend),
across batch sizes and context lengths, on scattered block tables. A third
variant — SDPA over pre-materialized contiguous K/V, no gather, no write — is the
"what if memory were free" upper bound the kernel is chasing.

Requires CUDA + the triton extra.

Usage:
    uv run python benchmarks/benchmark_attention.py
    uv run python benchmarks/benchmark_attention.py --batches 1,16,64 --contexts 512,4096
"""

import argparse
import statistics
import time

import torch
import torch.nn.functional as F  # noqa: N812

from tokamak import kernels
from tokamak.config import ModelConfig
from tokamak.memory import BlockManager, PagedKVCache, PagedKVCacheView
from tokamak.model.step_context import BatchedDecodeContext

NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 16
CUDA = torch.device("cuda")


def bench_config(num_kv_heads: int, head_dim: int) -> ModelConfig:
    return ModelConfig(
        architecture="Qwen3ForCausalLM",
        vocab_size=1,
        hidden_size=1,
        num_layers=1,
        num_attention_heads=NUM_Q_HEADS,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_size=1,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        max_position_embeddings=1 << 20,
        tie_word_embeddings=False,
        attention_bias=False,
        use_qk_norm=True,
        eos_token_ids=(0,),
    )


def timeit(fn, warmup: int = 10, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1e6  # µs


@torch.inference_mode()
def run_case(batch: int, context: int, dtype: torch.dtype) -> dict[str, float]:
    config = bench_config(NUM_KV_HEADS, HEAD_DIM)
    blocks_per_seq = -(-context // BLOCK_SIZE)
    num_blocks = blocks_per_seq * batch + 8
    pool = PagedKVCache(config, num_blocks, BLOCK_SIZE, device=CUDA, dtype=dtype)
    torch.manual_seed(0)
    pool.k_cache.normal_()
    pool.v_cache.normal_()

    # Interleaved allocation scatters each sequence's blocks across the pool
    # (table of sequence i is [i, batch+i, 2*batch+i, ...]).
    manager = BlockManager(num_blocks, BLOCK_SIZE)
    for round_idx in range(blocks_per_seq):
        for i in range(batch):
            manager.ensure_capacity(i, (round_idx + 1) * BLOCK_SIZE)
    views = [PagedKVCacheView(pool, manager, i) for i in range(batch)]
    tables = [manager.block_table(i) for i in range(batch)]

    q = torch.randn(batch, NUM_Q_HEADS, 1, HEAD_DIM, device=CUDA, dtype=dtype)
    k_new = torch.randn(batch, NUM_KV_HEADS, 1, HEAD_DIM, device=CUDA, dtype=dtype)
    v_new = torch.randn(batch, NUM_KV_HEADS, 1, HEAD_DIM, device=CUDA, dtype=dtype)
    seq_lens = [context] * batch

    def reference() -> torch.Tensor:
        ctx = BatchedDecodeContext(views, seq_lens, CUDA)
        return ctx.attend(0, q, k_new, v_new)

    from tokamak.kernels.paged_attention import TritonPagedDecodeContext

    block_tables = torch.tensor(tables, dtype=torch.int32, device=CUDA)
    lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=CUDA)
    pos = context - 1
    slots = torch.tensor(
        [t[pos // BLOCK_SIZE] * BLOCK_SIZE + pos % BLOCK_SIZE for t in tables],
        dtype=torch.int64,
        device=CUDA,
    )

    def kernel() -> torch.Tensor:
        ctx = TritonPagedDecodeContext(pool, block_tables, lens_t, slots)
        return ctx.attend(0, q, k_new, v_new)

    # Upper bound: contiguous K/V already materialized, no writes.
    k_contig = torch.randn(batch, NUM_KV_HEADS, context, HEAD_DIM, device=CUDA, dtype=dtype)
    v_contig = torch.randn(batch, NUM_KV_HEADS, context, HEAD_DIM, device=CUDA, dtype=dtype)

    def contiguous_sdpa() -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k_contig, v_contig, enable_gqa=True)

    # Cross-check the two real paths before timing them.
    torch.testing.assert_close(kernel().float(), reference().float(), rtol=2e-2, atol=1e-2)

    return {
        "reference_us": timeit(reference),
        "kernel_us": timeit(kernel),
        "contiguous_us": timeit(contiguous_sdpa),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="1,8,16,32")
    parser.add_argument("--contexts", default="512,2048")
    args = parser.parse_args()

    if not kernels.is_available():
        raise SystemExit("CUDA + triton required")

    print(
        f"shapes: {NUM_Q_HEADS} q-heads / {NUM_KV_HEADS} kv-heads, head_dim {HEAD_DIM}, "
        f"block_size {BLOCK_SIZE}, bf16, {torch.cuda.get_device_name(0)}"
    )
    header = (
        f"{'batch':>5} {'context':>8} {'reference (us)':>15} {'kernel (us)':>12} "
        f"{'no-gather SDPA (us)':>20} {'speedup':>8}"
    )
    print("\n" + header)
    print("-" * len(header))
    for batch in (int(b) for b in args.batches.split(",")):
        for context in (int(c) for c in args.contexts.split(",")):
            r = run_case(batch, context, torch.bfloat16)
            print(
                f"{batch:>5} {context:>8} {r['reference_us']:>15.0f} {r['kernel_us']:>12.0f} "
                f"{r['contiguous_us']:>20.0f} {r['reference_us'] / r['kernel_us']:>7.1f}x"
            )


if __name__ == "__main__":
    main()
