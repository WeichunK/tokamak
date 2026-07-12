# Benchmarks

Every milestone is measured against the previous one on the same hardware and
workload, so the value of each serving technique is quantified rather than assumed.
Numbers below are medians over 3 iterations after 1 warmup run.

## Reference hardware

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 Laptop (16 GiB, Ampere) |
| CPU / OS | Windows 11, Python 3.13 |
| PyTorch | 2.13.0 + cu126 |
| Model | Qwen/Qwen3-0.6B, bfloat16 |

## Single-sequence latency (`benchmark_latency.py`)

Workload: 512 synthetic prompt tokens, 128 greedy decode steps, EOS ignored.

| Milestone / backend | Prefill (ms) | Decode (tok/s) | Inter-token (ms) | Peak mem (GiB) |
|---|---|---|---|---|
| M1 — contiguous baseline | 66.3 | 19.0 | 52.6 | 1.40 |
| M2 — paged, reference gather (block 16) | 73.5 | 15.8 | 63.3 | 1.47 |

The M1 decode number is the point: a 0.6B model on a 16 GiB GPU decoding at
19 tok/s means the GPU is idle most of every step — single-sequence, eager-mode
decoding is launch-latency-bound, not compute-bound. Quantifying how much of that
gap each technique closes (batching, kernels, speculative decoding) is what the
following milestones are for.

The M2 paged backend is ~17% *slower* per sequence, deliberately: the reference
implementation materializes K/V through a gather every layer and step so that
correctness stays auditable. That regression is the measured motivation for the
M4 attention kernel, which reads block tables in-kernel instead of copying.

## KV reservation waste (`benchmark_kv_memory.py`)

Allocator-policy simulation, no GPU: 2,000 requests, log-normal prompts
(median 150 tokens), early-stopping generation (median 128 of a 512-token budget),
waste integrated over each request's lifetime.

| Policy | Reserved-but-unused KV |
|---|---|
| Contiguous (`prompt + max_new_tokens` up front) | 50.1% |
| Paged (block_size = 16) | 2.0% |

Paged holds 50.9% of the contiguous reservation — the same pool fits ~1.96×
the concurrent sequences, which is the capacity that continuous batching (M3)
converts into throughput.

```bash
uv run python benchmarks/benchmark_latency.py --kv-backend paged
uv run python benchmarks/benchmark_latency.py --kv-backend contiguous
uv run python benchmarks/benchmark_kv_memory.py
```

Reproduce with:

```bash
uv run python benchmarks/benchmark_latency.py --prompt-tokens 512 --new-tokens 128
```

## Planned

- **Throughput under concurrency** (M3): requests/s and token throughput vs. number
  of concurrent requests, compared against static batching.
- **vLLM comparison** (M6): TTFT / ITL / throughput on ShareGPT-style traces, run on
  Linux (WSL2 or cloud) since vLLM does not support native Windows. tokamak and vLLM
  will run the same model, dtype, and trace; the gap analysis lives in
  `docs/design/` when it exists.

## Methodology notes

- Synthetic random-token prompts: compute cost does not depend on token content.
- Greedy decoding: sampling cost is excluded from the measurement on purpose.
- `torch.cuda.synchronize` brackets every timed region.
- Medians, not means: first-iteration jitter (allocator warmup, autotuning) is real
  but is not what these tables compare.
