"""Single-sequence latency benchmark: prefill time and decode throughput.

Measures each milestone against the M1 baseline. Prompts are synthetic random token
ids — content does not affect compute — and decoding is greedy so sampling cost stays
out of the measurement. `--kv-backend` selects the cache implementation; the paged
backend's capacity is granted upfront so the timed region measures steady-state
write/gather cost rather than allocator calls.

Usage:
    uv run python benchmarks/benchmark_latency.py
    uv run python benchmarks/benchmark_latency.py --kv-backend contiguous
    uv run python benchmarks/benchmark_latency.py --prompt-tokens 1024 --new-tokens 256
"""

import argparse
import json
import statistics
import time

import torch

from tokamak import LLM
from tokamak.memory import PagedKVCacheView
from tokamak.model.kv_cache import ContiguousKVCache, KVCacheProtocol
from tokamak.model.step_context import BatchedDecodeContext, PrefillContext


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_cache(llm: LLM, total_tokens: int) -> KVCacheProtocol:
    if llm.kv_backend == "paged":
        assert llm.paged_cache is not None and llm.block_manager is not None
        view = PagedKVCacheView(llm.paged_cache, llm.block_manager, seq_id=0)
        view.ensure_capacity(total_tokens)
        return view
    return ContiguousKVCache(
        llm.model_config, max_seq_len=total_tokens, device=llm.device, dtype=llm.dtype
    )


@torch.inference_mode()
def run_once(llm: LLM, prompt_ids: list[int], new_tokens: int) -> dict[str, float]:
    device = llm.device
    cache = make_cache(llm, len(prompt_ids) + new_tokens)
    try:
        # Prefill.
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        positions = torch.arange(len(prompt_ids), device=device)[None]
        synchronize(device)
        start = time.perf_counter()
        hidden = llm.model(input_ids, positions, PrefillContext(cache))
        token = int(llm.model.compute_logits(hidden[:, -1]).argmax().item())
        synchronize(device)
        prefill_s = time.perf_counter() - start

        # Decode.
        start = time.perf_counter()
        pos = len(prompt_ids)
        for _ in range(new_tokens):
            step_ids = torch.tensor([[token]], dtype=torch.long, device=device)
            step_pos = torch.tensor([[pos]], dtype=torch.long, device=device)
            ctx = BatchedDecodeContext([cache], [pos + 1], device)
            hidden = llm.model(step_ids, step_pos, ctx)
            token = int(llm.model.compute_logits(hidden[:, -1]).argmax().item())
            pos += 1
        synchronize(device)
        decode_s = time.perf_counter() - start
    finally:
        cache.release()

    return {
        "prefill_ms": prefill_s * 1000,
        "decode_tok_per_s": new_tokens / decode_s,
        "inter_token_ms": decode_s * 1000 / new_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--kv-backend", choices=["contiguous", "paged"], default="paged")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args()

    total_tokens = args.prompt_tokens + args.new_tokens
    llm = LLM(
        args.model,
        max_seq_len=total_tokens,
        kv_backend=args.kv_backend,
        block_size=args.block_size,
    )
    generator = torch.Generator().manual_seed(0)
    prompt_ids = torch.randint(
        0, llm.model_config.vocab_size, (args.prompt_tokens,), generator=generator
    ).tolist()

    for _ in range(args.warmup):
        run_once(llm, prompt_ids, args.new_tokens)

    runs = [run_once(llm, prompt_ids, args.new_tokens) for _ in range(args.iters)]
    result = {
        "model": args.model,
        "device": str(llm.device),
        "dtype": str(llm.dtype),
        "kv_backend": args.kv_backend,
        "block_size": args.block_size if args.kv_backend == "paged" else None,
        "prompt_tokens": args.prompt_tokens,
        "new_tokens": args.new_tokens,
        "iters": args.iters,
        "prefill_ms": statistics.median(r["prefill_ms"] for r in runs),
        "decode_tok_per_s": statistics.median(r["decode_tok_per_s"] for r in runs),
        "inter_token_ms": statistics.median(r["inter_token_ms"] for r in runs),
    }
    if llm.device.type == "cuda":
        result["gpu"] = torch.cuda.get_device_name(llm.device)
        result["peak_mem_gib"] = torch.cuda.max_memory_allocated(llm.device) / 2**30

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        hardware = result.get("gpu", "cpu")
        print(f"\nmodel:             {result['model']}")
        print(f"device / dtype:    {result['device']} ({hardware}), {result['dtype']}")
        print(f"kv backend:        {args.kv_backend}")
        print(f"workload:          {args.prompt_tokens} prompt + {args.new_tokens} new tokens")
        print(f"prefill (median):  {result['prefill_ms']:.1f} ms")
        print(
            f"decode (median):   {result['decode_tok_per_s']:.1f} tok/s "
            f"({result['inter_token_ms']:.2f} ms/token)"
        )
        if "peak_mem_gib" in result:
            print(f"peak GPU memory:   {result['peak_mem_gib']:.2f} GiB")


if __name__ == "__main__":
    main()
