"""vLLM side of the M6 comparison (run inside a vLLM environment on Linux/WSL2).

Serves the byte-identical seeded workload of ``benchmark_throughput.py`` (via the
shared ``workload`` module) and reports the same headline metrics. Two switches
decompose the gap against tokamak:

- ``--enforce-eager`` disables CUDA graphs, matching tokamak's eager-PyTorch
  execution model;
- ``--max-num-seqs`` caps concurrency to match tokamak's ``max_batch_size``.

Usage (inside a venv with vllm, from the repo root):
    python benchmarks/benchmark_vllm.py --enforce-eager --max-num-seqs 16
    python benchmarks/benchmark_vllm.py --max-num-seqs 16
    python benchmarks/benchmark_vllm.py
"""

import argparse
import statistics
import time

from vllm import LLM, SamplingParams
from workload import build_workload

try:  # location varies across vLLM versions
    from vllm import TokensPrompt
except ImportError:  # pragma: no cover
    from vllm.inputs import TokensPrompt


def percentile(values: list[float], p: float) -> float:
    return sorted(values)[min(int(p * len(values)), len(values) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--max-new-cap", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    prompts, new_tokens = build_workload(args.num_requests, args.max_new_cap, args.seed)
    inputs = [TokensPrompt(prompt_token_ids=p) for p in prompts]
    sampling = [SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True) for n in new_tokens]

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Warmup outside the timed region (graph capture / allocator settling).
    llm.generate(
        [TokensPrompt(prompt_token_ids=prompts[0][:16])],
        SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True),
        use_tqdm=False,
    )

    start = time.perf_counter()
    outputs = llm.generate(inputs, sampling, use_tqdm=True)
    wall = time.perf_counter() - start

    out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    ttfts = []
    for o in outputs:
        metrics = getattr(o, "metrics", None)
        if metrics and getattr(metrics, "first_token_time", None) and metrics.arrival_time:
            ttfts.append(metrics.first_token_time - metrics.arrival_time)

    mode = "eager" if args.enforce_eager else "cudagraphs"
    print(
        f"\nvllm ({mode}, max_num_seqs={args.max_num_seqs}): "
        f"{args.num_requests} requests, {out_tokens} output tokens"
    )
    print(f"wall:        {wall:.1f} s")
    print(f"out tok/s:   {out_tokens / wall:.1f}")
    print(f"req/min:     {60 * len(outputs) / wall:.1f}")
    if ttfts:
        print(f"TTFT mean:   {statistics.mean(ttfts):.2f} s")
        print(f"TTFT p95:    {percentile(ttfts, 0.95):.2f} s")
    else:
        print("TTFT:        (per-request metrics not exposed by this vLLM version)")


if __name__ == "__main__":
    main()
