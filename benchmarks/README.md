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
| M4 — paged, Triton kernel (block 16) | 70.7 | 21.0 | 47.6 | 1.47 |

The M1 decode number is the point: a 0.6B model on a 16 GiB GPU decoding at
19 tok/s means the GPU is idle most of every step — single-sequence, eager-mode
decoding is launch-latency-bound, not compute-bound. Quantifying how much of that
gap each technique closes (batching, kernels, speculative decoding) is what the
following milestones are for.

The M2 paged backend is ~17% *slower* per sequence, deliberately: the reference
implementation materializes K/V through a gather every layer and step so that
correctness stays auditable. That regression was the measured motivation for the
M4 attention kernel, which reads block tables in-kernel instead of copying — and
repays the debt with interest: the kernel path beats even the contiguous
zero-copy baseline (21.0 vs 19.0 tok/s single-sequence).

## Decode attention in isolation (`benchmark_attention.py`)

One decode step's attention (write new K/V + attend), Qwen3-0.6B shapes
(16 q-heads / 8 kv-heads, head_dim 128, block 16), bf16, scattered block tables;
median µs per call. "No-gather SDPA" is eager SDPA over pre-materialized
contiguous K/V — the cost with memory movement taken off the books.

| Batch | Context | Reference (gather+SDPA) | Triton kernel | No-gather SDPA | Speedup |
|---|---|---|---|---|---|
| 1 | 512 | 965 | 241 | 368 | 4.0× |
| 8 | 512 | 3,954 | 271 | 790 | 14.6× |
| 16 | 512 | 7,251 | 323 | 1,487 | 22.5× |
| 16 | 2,048 | 11,495 | 806 | 5,760 | 14.3× |
| 32 | 512 | 14,401 | 440 | 2,864 | 32.7× |
| 32 | 2,048 | 22,599 | 1,409 | 11,331 | 16.0× |

Two notes: the kernel beats the "no-gather" SDPA column everywhere — eager SDPA
at decode shapes pays padded-batch math a fused single-pass kernel avoids — and
kernel cost grows sub-linearly with batch (241 → 440 µs, 1 → 32 sequences)
because launches amortize and GQA tile reuse works.

```bash
uv run python benchmarks/benchmark_attention.py
```

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

With the M4 Triton kernel (`--attention-backend triton`), same workload:

| Config | Wall (s) | Out tok/s | Req/min | TTFT mean (s) | TTFT p95 (s) | Latency mean (s) | Latency p95 (s) |
|---|---|---|---|---|---|---|---|
| sequential | 138.4 | 21.0 | 13.9 | 78.2 | 132.6 | 82.5 | 136.8 |
| static b=4 | 72.1 | 40.2 | 26.6 | 37.3 | 67.6 | 41.7 | 69.2 |
| continuous b=4 | 37.2 | 77.9 | 51.6 | 17.5 | 33.0 | 22.0 | 35.9 |
| static b=16 | 26.2 | 110.7 | 73.2 | 7.1 | 14.1 | 11.8 | 19.7 |
| continuous b=16 | 16.4 | 176.6 | 116.9 | 2.7 | 7.4 | 8.0 | 14.9 |

The kernel's win grows with batch size exactly as the microbenchmark predicts
(the gather it deletes scales with rows): 1.36× at batch 1, 1.85× at continuous
b=4, **2.7× at continuous b=16**. Compounded across milestones, the engine went
from 15.4 tok/s (sequential, reference attention) to **176.6 tok/s** — 11.5× —
on identical hardware and workload.

```bash
uv run python benchmarks/benchmark_throughput.py
uv run python benchmarks/benchmark_throughput.py --attention-backend triton
```

## Speculative decoding (`benchmark_speculative.py`)

Qwen3-4B target, Qwen3-0.6B draft, greedy, single-sequence, natural-language
continuation prompts (4 × 128 new tokens). The baseline is the same engine
without a draft (best backend, i.e. the M4 kernel).

| Config | Wall (s) | Tok/s | Speedup | Acceptance |
|---|---|---|---|---|
| baseline (target only) | 35.1 | 14.6 | 1.00× | — |
| speculative k=2 | 47.3 | 10.8 | 0.74× | 57.0% |
| speculative k=4 | 64.9 | 7.9 | 0.54× | 37.2% |
| speculative k=6 | 76.1 | 6.7 | 0.46× | 30.9% |

**Speculative decoding loses on this stack, and the arithmetic says it must.**
The economics of speculation depend on the draft:target *step-cost* ratio ``c``;
the theory pays off around ``c ≲ 0.2``. Here every decode step — 0.6B or 4B —
sits on the same ~50 ms Python/kernel-launch overhead floor (M1's finding), so
``c ≈ 0.7`` despite a 7× parameter gap. Sanity check at k=2: an iteration costs
about 2 × 48 ms (draft) + 68 ms (verify) ≈ 164 ms and yields ≈ 1.9 tokens at 57%
acceptance → 86 ms/token vs. the baseline's 68 ms — predicting 0.79×, measuring
0.74× (the residual is the mode's integration gap: contiguous caches + SDPA
verify, no M4 kernel).

The algorithm itself is verified exactly (distribution tests, token-identical
greedy); what fails is the cost model this hardware offers it. Production
engines lower the overhead floor (CUDA graphs, fused runners) *before*
speculation multiplies — a measured argument for why serving-stack overhead
work precedes algorithmic acceleration.

```bash
uv run python benchmarks/benchmark_speculative.py
```

Reproduce with:

```bash
uv run python benchmarks/benchmark_latency.py --prompt-tokens 512 --new-tokens 128
```

## vLLM comparison (`benchmark_vllm.py`, M6)

Same GPU, same model and dtype, byte-identical requests: both engines import
the seeded workload from `workload.py` and receive raw prompt token ids, so
tokenization is out of the picture. Run under WSL2 (vLLM does not support
native Windows), which changes tokamak's own numbers versus the Windows tables
above — the identical configuration decodes 34% faster under Linux (236.8 vs
176.6 tok/s) because WDDM's kernel-submission path is more expensive, a free
preview of the comparison's conclusion. vLLM is pinned to 0.10.0, the last
release whose torch build matches this machine's CUDA 12.7 driver cap. vLLM's
offline API does not expose per-request TTFT at this version, so the
cross-engine columns are wall clock and throughput only.

| Config | Wall (s) | Out tok/s | vs. tokamak best |
|---|---|---|---|
| tokamak continuous b=16 (Triton) | 12.3 | 236.8 | 1.00× |
| vLLM eager, max_num_seqs=16 | 5.8 | 502.1 | 2.12× |
| vLLM CUDA graphs, max_num_seqs=16 | 2.5 | 1172.0 | 4.95× |
| vLLM defaults (graphs, max_num_seqs=256) | 2.2 | 1303.2 | 5.50× |

The two vLLM switches decompose the 5.5×: **2.12×** with execution model and
concurrency matched (kernels + fused ops + engine loop), **×2.33** more from
CUDA graphs alone (the launch-overhead floor a 0.6B decode step lives on),
**×1.11** from admitting more than 16 sequences. The full decomposition and
what it would take to close each factor:
[docs/design/006-vllm-gap-analysis.md](../docs/design/006-vllm-gap-analysis.md).

```bash
# tokamak side (WSL2 venv with CUDA torch):
python benchmarks/benchmark_throughput.py --attention-backend triton --batch-sizes 16
# vLLM side (separate venv, vllm==0.10.0):
python benchmarks/benchmark_vllm.py --enforce-eager --max-num-seqs 16
python benchmarks/benchmark_vllm.py --max-num-seqs 16
python benchmarks/benchmark_vllm.py
```

## Attention policies (`benchmark_quality.py`, `benchmark_streaming.py`, M7)

Windowed attention policies (`window:W`, `streaming:W+S` with S always-visible
sink tokens) are inference-time approximations of a dense model, so the first
measurement is what they *cost*: teacher-forced perplexity over 16,384 tokens
of long text (4,096-token segments, banded-causal masks reproducing exactly
the visibility a windowed decode would have).

| Policy | KV budget | PPL | vs. full |
|---|---|---|---|
| full | 4,096 | 26.65 | — |
| window:1024 | 1,024 | 218.26 | +719% |
| window:512 | 512 | 397.89 | +1,393% |
| window:256 | 256 | 726.78 | +2,627% |
| streaming:1024+4 | 1,028 | 27.47 | **+3.1%** |
| streaming:512+4 | 516 | 29.05 | **+9.0%** |
| streaming:256+4 | 260 | 31.76 | **+19.2%** |

The StreamingLLM result reproduces exactly: a plain window collapses (softmax
attention needs the earliest positions as a sink for surplus probability
mass), and pinning 4 tokens buys nearly all of it back at 1/4 the KV.

What bounded visibility buys back (`benchmark_streaming.py`): per-sequence KV
residency is capped near `sinks + window` — dead blocks return to the pool
mid-flight — which is memory at batch 1 and *throughput* under contention:

| Scenario | full | streaming:512+4 | |
|---|---|---|---|
| 1 × 3,072 tokens: tok/s | 22.3 | 22.6 | flat — batch-1 decode sits on the launch-overhead floor |
| 1 × 3,072 tokens: peak KV | 3,088 | 544 | **5.7× less residency** |
| 8 × 2,048 tokens, 8,192-token pool: tok/s | 113.1 | 171.2 | **1.51×** — reclamation ends pool thrash |

Full attention fits ~4 of the 8 sequences and preempts-by-recompute; the
windowed pool runs all 8 concurrently. The 1.51× costs +9% PPL — a trade
stated, not hidden. Analysis: [docs/design/007-attention-policies.md](../docs/design/007-attention-policies.md).

```bash
uv run python benchmarks/benchmark_quality.py
uv run python benchmarks/benchmark_streaming.py
```

## Planned

Milestones M1–M7 are complete; further backends (top-k block sparsity,
natively-trained hybrid architectures) are future work beyond the roadmap.

## Methodology notes

- Synthetic random-token prompts: compute cost does not depend on token content.
- Greedy decoding: sampling cost is excluded from the measurement on purpose.
- `torch.cuda.synchronize` brackets every timed region.
- Medians, not means: first-iteration jitter (allocator warmup, autotuning) is real
  but is not what these tables compare.
