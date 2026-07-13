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

## Throughput under concurrency (`benchmark_throughput.py`)

The M3 exit-criteria curve. Fixed seeded workload replayed identically for every
configuration: 32 chat-like requests (log-normal prompts, 4,915 prompt tokens
total; exponential generation lengths, 2,901 output tokens total), greedy
decoding, EOS ignored, paged KV backend with a 16,384-token pool.

"Sequential" is continuous batching at `max_batch_size=1` (the M1/M2 behaviour).
"Static" fills a batch and drains it before admitting more — modelled charitably
(finished sequences stop consuming compute, which padded implementations don't
get), so the continuous-batching win below is a lower bound.

| Config | Wall (s) | Out tok/s | Req/min | TTFT mean (s) | TTFT p95 (s) | Latency mean (s) | Latency p95 (s) |
|---|---|---|---|---|---|---|---|
| sequential | 187.9 | 15.4 | 10.2 | 106.4 | 179.7 | 112.2 | 185.7 |
| static b=4 | 106.9 | 27.1 | 18.0 | 55.8 | 100.5 | 62.8 | 103.3 |
| continuous b=4 | 69.0 | 42.0 | 27.8 | 32.6 | 61.7 | 40.9 | 67.2 |
| static b=16 | 57.2 | 50.7 | 33.6 | 16.9 | 33.7 | 30.5 | 48.7 |
| continuous b=16 | 44.9 | 64.6 | 42.7 | 7.4 | 23.4 | 24.4 | 42.8 |

Reading the curve:

- **Throughput scales with concurrency** (the M3 exit criterion): 15.4 → 42.0 →
  64.6 tok/s as the batch goes 1 → 4 → 16. Decode is bandwidth- and launch-bound,
  so extra rows through the same weights are nearly free until compute saturates.
- **Continuous beats static at every batch size** — 1.55× at b=4, 1.27× at b=16 —
  purely from keeping seats filled: finished requests are replaced at token
  granularity instead of waiting for the batch to drain. The static baseline here
  is charitable; against a real padded implementation the gap widens.
- **TTFT is where iteration-level admission dominates**: mean 7.4 s vs 16.9 s
  (static, b=16) vs 106.4 s (sequential). Prefill-priority admission starts new
  requests the moment blocks free up.
- Continuous batching does not make a *single* request faster — sequential mode
  matches the M2 single-sequence numbers. It makes the fleet share hardware that
  one request cannot saturate.

```bash
uv run python benchmarks/benchmark_throughput.py
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
