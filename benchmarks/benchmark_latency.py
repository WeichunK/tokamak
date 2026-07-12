"""Single-sequence latency benchmark: prefill time and decode throughput.

Measures the M1 baseline that later milestones (paged KV cache, continuous batching,
custom kernels, speculative decoding) are compared against. Prompts are synthetic
random token ids — content does not affect compute — and EOS is ignored so every run
decodes exactly the requested number of tokens.

Usage:
    uv run python benchmarks/benchmark_latency.py
    uv run python benchmarks/benchmark_latency.py --prompt-tokens 1024 --new-tokens 256
"""

import argparse
import json
import statistics
import time

import torch

from tokamak import LLM
from tokamak.model.kv_cache import KVCache


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def run_once(llm: LLM, prompt_ids: list[int], new_tokens: int) -> dict[str, float]:
    device = llm.device
    cache = KVCache(
        llm.model_config,
        max_seq_len=len(prompt_ids) + new_tokens,
        device=device,
        dtype=llm.dtype,
    )

    # Prefill (greedy decoding throughout: sampling cost is not what we measure).
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    synchronize(device)
    start = time.perf_counter()
    hidden = llm.model(input_ids, cache, start_pos=0)
    token = int(llm.model.compute_logits(hidden[:, -1]).argmax().item())
    synchronize(device)
    prefill_s = time.perf_counter() - start

    # Decode.
    start = time.perf_counter()
    pos = len(prompt_ids)
    for _ in range(new_tokens):
        step_ids = torch.tensor([[token]], dtype=torch.long, device=device)
        hidden = llm.model(step_ids, cache, start_pos=pos)
        token = int(llm.model.compute_logits(hidden[:, -1]).argmax().item())
        pos += 1
    synchronize(device)
    decode_s = time.perf_counter() - start

    return {
        "prefill_ms": prefill_s * 1000,
        "decode_tok_per_s": new_tokens / decode_s,
        "inter_token_ms": decode_s * 1000 / new_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args()

    llm = LLM(args.model)
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
