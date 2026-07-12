"""KV-memory reservation waste: contiguous worst-case sizing vs. paged blocks.

Pure allocator arithmetic — no GPU, no model. Simulates a stream of requests with
realistic length variance and measures, for each policy, how much KV capacity sat
reserved-but-unused over each request's lifetime:

- contiguous: a request reserves ``prompt + max_new_tokens`` for its entire life,
  because the buffer must be sized before generation length is known.
- paged: a request holds ``ceil(current_tokens / block_size)`` blocks at any moment,
  so waste is bounded by one partial block (plus scheduling headroom it never asks for).

Waste is integrated over time (token-steps), not just measured at completion:
early-stopping sequences hold their worst-case contiguous reservation for their
whole life, which is exactly what limits how many sequences fit concurrently (M3).

Usage:
    uv run python benchmarks/benchmark_kv_memory.py
    uv run python benchmarks/benchmark_kv_memory.py --requests 5000 --block-size 32
"""

import argparse
import random
import statistics

# Qwen3-0.6B in bf16: 2 * 28 layers * 8 kv-heads * 128 head-dim * 2 bytes.
KIB_PER_TOKEN = 112


def simulate_request(rng: random.Random, max_new_tokens: int) -> tuple[int, int]:
    """Sample (prompt_tokens, generated_tokens) for one request.

    Prompt lengths are log-normal (many short, few very long), clipped to
    [16, 1024]. Generation stops early with the geometric-ish behaviour of real
    chat traffic: most responses end well before the cap.
    """
    prompt = int(min(max(rng.lognormvariate(5.0, 0.8), 16), 1024))
    generated = min(int(rng.expovariate(1 / 180)) + 1, max_new_tokens)
    return prompt, generated


def waste_stats(
    requests: list[tuple[int, int]], max_new_tokens: int, block_size: int
) -> dict[str, float]:
    """Integrate reserved and used capacity over each request's decode steps."""
    contiguous_reserved = 0.0
    paged_reserved = 0.0
    used = 0.0
    for prompt, generated in requests:
        reservation = prompt + max_new_tokens
        for step in range(1, generated + 1):
            tokens_now = prompt + step
            contiguous_reserved += reservation
            paged_reserved += -(-tokens_now // block_size) * block_size
            used += tokens_now
    return {
        "contiguous_waste_pct": 100 * (contiguous_reserved - used) / contiguous_reserved,
        "paged_waste_pct": 100 * (paged_reserved - used) / paged_reserved,
        "paged_over_contiguous": paged_reserved / contiguous_reserved,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    requests = [simulate_request(rng, args.max_new_tokens) for _ in range(args.requests)]
    prompts = [p for p, _ in requests]
    gens = [g for _, g in requests]

    stats = waste_stats(requests, args.max_new_tokens, args.block_size)

    print(f"\nworkload: {args.requests} requests, max_new_tokens={args.max_new_tokens}")
    print(
        f"prompt tokens:    median {statistics.median(prompts):.0f}, "
        f"p95 {sorted(prompts)[int(0.95 * len(prompts))]}"
    )
    print(
        f"generated tokens: median {statistics.median(gens):.0f}, "
        f"p95 {sorted(gens)[int(0.95 * len(gens))]}"
    )
    print("\nKV capacity reserved but unused (integrated over request lifetimes):")
    print(f"  contiguous (prompt + max_new): {stats['contiguous_waste_pct']:.1f}%")
    print(f"  paged (block_size={args.block_size}):          {stats['paged_waste_pct']:.1f}%")
    ratio = stats["paged_over_contiguous"]
    print(
        f"\npaged holds {100 * ratio:.1f}% of the contiguous reservation, i.e. "
        f"{1 / ratio:.2f}x more sequences fit in the same KV pool"
    )
    mean_reservation = statistics.mean(prompts) + args.max_new_tokens
    saved_gib = (1 - ratio) * 128 * mean_reservation * KIB_PER_TOKEN / 2**20
    print(
        f"at {KIB_PER_TOKEN} KiB/token (Qwen3-0.6B bf16), 128 concurrent worst-case "
        f"reservations shrink by {saved_gib:.1f} GiB"
    )


if __name__ == "__main__":
    main()
