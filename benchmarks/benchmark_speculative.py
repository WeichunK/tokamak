"""Speculative decoding: acceptance rate and end-to-end speedup vs. plain decoding.

Single-sequence generation (the regime speculative decoding targets), greedy, on
natural-language continuation prompts. The baseline is the same engine without a
draft model, best available attention backend. Speculation runs on per-request
contiguous caches with SDPA (see docs/design/005), so the comparison includes
that integration gap — reported numbers are honest end-to-end, not kernel-boosted.

Usage:
    uv run python benchmarks/benchmark_speculative.py
    uv run python benchmarks/benchmark_speculative.py --target Qwen/Qwen3-4B --gammas 2,4,6
"""

import argparse
import time

import torch

from tokamak import LLM, SamplingParams

PROMPTS = [
    "The history of the Roman Empire is a story of expansion and collapse. It begins",
    "To train a neural network from scratch, the first thing you need is",
    "The Pacific Ocean covers more of the Earth's surface than all land combined, and",
    "In the theory of computation, a Turing machine consists of",
]


def run(llm: LLM, max_new: int) -> tuple[float, int, float | None]:
    params = SamplingParams(temperature=0.0, max_new_tokens=max_new, ignore_eos=True)
    start = time.perf_counter()
    outputs = llm.generate(PROMPTS, params, use_tqdm=True)
    wall = time.perf_counter() - start
    out_tokens = sum(len(o.output_token_ids) for o in outputs)
    proposed = sum(o.spec_proposed or 0 for o in outputs)
    accepted = sum(o.spec_accepted or 0 for o in outputs)
    acceptance = accepted / proposed if proposed else None
    return wall, out_tokens, acceptance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--gammas", default="2,4,6")
    args = parser.parse_args()

    common = {"max_seq_len": 1024, "max_batch_size": 1}

    baseline = LLM(args.target, **common)  # type: ignore[arg-type]
    wall, tokens, _ = run(baseline, args.max_new)
    base_tps = tokens / wall
    print(
        f"\ntarget {args.target}, draft {args.draft}, greedy, {args.max_new} new tokens x 4 prompts"
    )
    header = f"{'config':<16} {'wall (s)':>9} {'tok/s':>7} {'speedup':>8} {'acceptance':>11}"
    print("\n" + header)
    print("-" * len(header))
    print(f"{'baseline':<16} {wall:>9.1f} {base_tps:>7.1f} {'1.00x':>8} {'-':>11}")
    del baseline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for gamma in (int(g) for g in args.gammas.split(",")):
        llm = LLM(
            args.target,
            draft_model=args.draft,
            num_speculative_tokens=gamma,
            **common,  # type: ignore[arg-type]
        )
        wall, tokens, acceptance = run(llm, args.max_new)
        tps = tokens / wall
        print(
            f"{f'spec k={gamma}':<16} {wall:>9.1f} {tps:>7.1f} "
            f"{f'{tps / base_tps:.2f}x':>8} {f'{100 * (acceptance or 0):.1f}%':>11}"
        )
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
