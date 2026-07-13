"""Throughput under concurrency: sequential vs. static vs. continuous batching.

The M3 exit-criteria benchmark. A fixed, seeded workload of chat-like requests
(log-normal prompt lengths, exponential-ish generation lengths, EOS ignored so the
workload shape is model-independent) is replayed under each scheduling
configuration, and the table reports throughput plus latency percentiles.

"sequential" is continuous batching with max_batch_size=1 (the M1/M2 behaviour);
"static" fills a batch and lets it drain before admitting more (the classic
baseline, charitably modelled — finished sequences stop consuming compute).

Usage:
    uv run python benchmarks/benchmark_throughput.py
    uv run python benchmarks/benchmark_throughput.py --num-requests 64 --batch-sizes 8,32
"""

import argparse
import random
import statistics
import time

import torch

from tokamak import LLM, SamplingParams


def build_workload(
    num_requests: int, max_new_cap: int, vocab_size: int, seed: int
) -> tuple[list[list[int]], list[SamplingParams]]:
    rng = random.Random(seed)
    prompts = []
    params = []
    for _ in range(num_requests):
        prompt_len = int(min(max(rng.lognormvariate(5.0, 0.7), 16), 512))
        new_tokens = min(int(rng.expovariate(1 / 120)) + 8, max_new_cap)
        prompts.append([rng.randrange(vocab_size) for _ in range(prompt_len)])
        params.append(SamplingParams(temperature=0.0, max_new_tokens=new_tokens, ignore_eos=True))
    return prompts, params


def percentile(values: list[float], p: float) -> float:
    return sorted(values)[min(int(p * len(values)), len(values) - 1)]


def run_config(
    model: str,
    scheduling: str,
    max_batch_size: int,
    kv_pool_tokens: int,
    prompts: list[list[int]],
    params: list[SamplingParams],
    attention_backend: str = "auto",
) -> dict[str, float]:
    llm = LLM(
        model,
        max_seq_len=1024,
        kv_backend="paged",
        kv_pool_tokens=kv_pool_tokens,
        max_batch_size=max_batch_size,
        scheduling=scheduling,  # type: ignore[arg-type]
        attention_backend=attention_backend,  # type: ignore[arg-type]
    )
    start = time.perf_counter()
    outputs = llm.generate(sampling_params=params, prompt_token_ids=prompts, use_tqdm=True)
    wall_s = time.perf_counter() - start

    out_tokens = sum(len(o.output_token_ids) for o in outputs)
    ttfts = [o.ttft_s for o in outputs if o.ttft_s is not None]
    latencies = [o.latency_s for o in outputs if o.latency_s is not None]
    result = {
        "wall_s": wall_s,
        "output_tok_per_s": out_tokens / wall_s,
        "requests_per_min": 60 * len(outputs) / wall_s,
        "ttft_mean_s": statistics.mean(ttfts),
        "ttft_p95_s": percentile(ttfts, 0.95),
        "latency_mean_s": statistics.mean(latencies),
        "latency_p95_s": percentile(latencies, 0.95),
    }
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--max-new-cap", type=int, default=256)
    parser.add_argument("--batch-sizes", default="4,16")
    parser.add_argument("--kv-pool-tokens", type=int, default=16384)
    parser.add_argument("--attention-backend", choices=["auto", "sdpa", "triton"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    configs = [("sequential", "continuous", 1)]
    for batch in batch_sizes:
        configs.append((f"static b={batch}", "static", batch))
        configs.append((f"continuous b={batch}", "continuous", batch))

    # Workload is built once and replayed identically for every config.
    probe = LLM(args.model, max_seq_len=1024, kv_pool_tokens=16384)
    vocab_size = probe.model_config.vocab_size
    del probe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    prompts, params = build_workload(args.num_requests, args.max_new_cap, vocab_size, args.seed)
    total_new = sum(p.max_new_tokens for p in params)
    print(
        f"workload: {args.num_requests} requests, {sum(map(len, prompts))} prompt tokens, "
        f"{total_new} output tokens (seed {args.seed})"
    )

    header = (
        f"{'config':<18} {'wall (s)':>9} {'out tok/s':>10} {'req/min':>8} "
        f"{'TTFT mean':>10} {'TTFT p95':>9} {'lat mean':>9} {'lat p95':>8}"
    )
    print("\n" + header)
    print("-" * len(header))
    for label, scheduling, batch in configs:
        r = run_config(
            args.model,
            scheduling,
            batch,
            args.kv_pool_tokens,
            prompts,
            params,
            args.attention_backend,
        )
        print(
            f"{label:<18} {r['wall_s']:>9.1f} {r['output_tok_per_s']:>10.1f} "
            f"{r['requests_per_min']:>8.1f} {r['ttft_mean_s']:>10.2f} "
            f"{r['ttft_p95_s']:>9.2f} {r['latency_mean_s']:>9.1f} {r['latency_p95_s']:>8.1f}"
        )


if __name__ == "__main__":
    main()
