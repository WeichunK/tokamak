"""Long-generation cost of attention policies: decode throughput + KV residency.

Full attention pays more per decode step as context grows (more KV to read)
and holds every block it has ever written. A windowed policy's per-step cost
and residency are bounded by ``sinks + window``, so its late-context decode
rate should stay flat where full attention's sags — and its peak KV footprint
should be a small constant instead of the whole sequence.

Late-phase rate is isolated with two runs per policy: a shallow run and a deep
run from the same prompt; ``(deep - shallow) tokens / (deep - shallow) wall``
prices exactly the tokens generated at depth.

Usage:
    python benchmarks/benchmark_streaming.py
    python benchmarks/benchmark_streaming.py --policies full,streaming:512+4
"""

import argparse
import time

import torch

from tokamak import LLM, SamplingParams

PROMPT = "Write a very long story about a lighthouse keeper.\n"


def timed_generate(llm: LLM, max_new_tokens: int) -> tuple[float, int]:
    """Greedy-generate; return (wall seconds, peak reserved KV tokens)."""
    assert llm.block_manager is not None
    manager = llm.block_manager
    peak = 0
    original = manager.ensure_capacity

    def spying_ensure(seq_id: int, num_tokens: int) -> None:
        nonlocal peak
        original(seq_id, num_tokens)
        peak = max(peak, manager.reserved_tokens(seq_id))

    manager.ensure_capacity = spying_ensure  # type: ignore[method-assign]
    try:
        params = SamplingParams(temperature=0.0, max_new_tokens=max_new_tokens, ignore_eos=True)
        if llm.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        llm.generate(PROMPT, params, use_tqdm=False)
        if llm.device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - start, peak
    finally:
        manager.ensure_capacity = original  # type: ignore[method-assign]


def concurrent_run(
    model: str, spec: str, num_requests: int, new_tokens: int, max_seq_len: int, pool: int
) -> tuple[float, int]:
    """N long generations racing for a pool that full attention cannot share.

    Returns (wall seconds, total output tokens). Under full attention each
    sequence's residency grows to prompt + new_tokens, the pool admits only a
    few at a time, and the scheduler preempts-by-recompute; under a windowed
    policy reclamation caps residency near sinks + window, so the same pool
    runs everything concurrently.
    """
    llm = LLM(
        model,
        max_seq_len=max_seq_len,
        kv_pool_tokens=pool,
        attention_policy=spec,
        max_batch_size=num_requests,
    )
    params = SamplingParams(temperature=0.0, max_new_tokens=new_tokens, ignore_eos=True)
    llm.generate(PROMPT, SamplingParams(temperature=0.0, max_new_tokens=8), use_tqdm=False)
    if llm.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = llm.generate([PROMPT] * num_requests, params, use_tqdm=False)
    if llm.device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - start
    total = sum(len(o.output_token_ids) for o in outputs)
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return wall, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--policies", default="full,window:512,streaming:512+4")
    parser.add_argument("--shallow", type=int, default=1024)
    parser.add_argument("--deep", type=int, default=3072)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--concurrent-requests", type=int, default=8)
    parser.add_argument("--concurrent-new", type=int, default=2048)
    parser.add_argument("--pool", type=int, default=8192)
    parser.add_argument("--skip-concurrent", action="store_true")
    args = parser.parse_args()
    policies = [spec.strip() for spec in args.policies.split(",")]

    print("single sequence, deep context:")
    print(f"{'policy':<20} {'tok/s overall':>13} {'tok/s deep':>11} {'peak KV tokens':>15}")
    print("-" * 62)
    for spec in policies:
        llm = LLM(args.model, max_seq_len=args.max_seq_len, attention_policy=spec)
        timed_generate(llm, 32)  # warmup: allocator + kernel compilation
        shallow_wall, _ = timed_generate(llm, args.shallow)
        deep_wall, peak = timed_generate(llm, args.deep)
        overall = args.deep / deep_wall
        late = (args.deep - args.shallow) / (deep_wall - shallow_wall)
        print(f"{spec:<20} {overall:>13.1f} {late:>11.1f} {peak:>15}")
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.skip_concurrent:
        return
    print(
        f"\n{args.concurrent_requests} concurrent x {args.concurrent_new} new tokens, "
        f"{args.pool}-token pool:"
    )
    print(f"{'policy':<20} {'wall (s)':>9} {'out tok/s':>10}")
    print("-" * 42)
    for spec in policies:
        wall, total = concurrent_run(
            args.model,
            spec,
            args.concurrent_requests,
            args.concurrent_new,
            args.max_seq_len,
            args.pool,
        )
        print(f"{spec:<20} {wall:>9.1f} {total / wall:>10.1f}")


if __name__ == "__main__":
    main()
