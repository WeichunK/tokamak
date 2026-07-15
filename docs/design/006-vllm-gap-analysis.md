# 006 — The vLLM Gap Analysis (M6): measuring the distance honestly

**Status:** complete
**Scope:** a reproducible cross-engine benchmark (shared workload module, vLLM
runner script, WSL2 environment notes) and a written decomposition of the
performance gap between tokamak and vLLM on identical hardware and workloads.

**Headline:** vLLM is **5.5× faster** than tokamak's best configuration on this
workload — and the decomposition says the single biggest factor is not kernels
or scheduling but **CUDA graphs** (2.3×), with kernel/engine quality second
(2.1×) and admission headroom a distant third (1.1×). At 0.6B scale on a laptop
GPU, the gap is overhead-shaped, not FLOPs-shaped.

## Method

Same GPU (RTX 3080 Laptop, 16 GiB, inside WSL2), same model (Qwen3-0.6B, bf16),
same requests: the seeded workload both engines import from
`benchmarks/workload.py` — 32 chat-like requests, 4,915 prompt / 2,901 output
tokens, greedy, EOS ignored. Prompts are raw token ids fed to both engines
(vLLM via `TokensPrompt`), so tokenization differences are out of the picture.

The comparison is decomposed with two vLLM switches:

| Row | What it isolates |
|---|---|
| tokamak (continuous b=16, Triton kernel) | our best configuration |
| vLLM `--enforce-eager --max-num-seqs 16` | same execution model (eager) and same concurrency cap: the *scheduler + kernels + engine overhead* gap |
| vLLM `--max-num-seqs 16` | adds CUDA graphs: the *launch-overhead floor* gap |
| vLLM (defaults) | adds admission headroom: what full tuning buys |

## Results

All rows: same GPU, same OS (WSL2), same 32 requests (4,915 prompt / 2,901
output tokens), greedy, EOS ignored. vLLM is 0.10.0 (driver-capped; see
environment notes). TTFT is reported for tokamak only — vLLM 0.10's offline
API does not expose per-request timing, so the cross-engine comparison sticks
to wall clock and throughput.

| Config | Wall (s) | Out tok/s | vs. tokamak best | step over previous row |
|---|---|---|---|---|
| tokamak sequential | 102.1 | 28.4 | 0.12× | — |
| tokamak static b=16 | 20.3 | 142.6 | 0.60× | 5.0× |
| tokamak continuous b=16 (Triton) | 12.3 | 236.8 | 1.00× | 1.66× |
| vLLM eager, max_num_seqs=16 | 5.8 | 502.1 | **2.12×** | 2.12× |
| vLLM CUDA graphs, max_num_seqs=16 | 2.5 | 1172.0 | **4.95×** | 2.33× |
| vLLM defaults (graphs, max_num_seqs=256) | 2.2 | 1303.2 | **5.50×** | 1.11× |

(tokamak continuous b=16: TTFT mean 2.08 s / p95 5.48 s, request latency mean
5.9 s / p95 11.1 s.)

```bash
# tokamak side (WSL2, CUDA torch + triton):
python benchmarks/benchmark_throughput.py --attention-backend triton --batch-sizes 16
# vLLM side (separate venv):
python benchmarks/benchmark_vllm.py --enforce-eager --max-num-seqs 16
python benchmarks/benchmark_vllm.py --max-num-seqs 16
python benchmarks/benchmark_vllm.py
```

## Where the gap comes from

**2.33× — CUDA graphs (the launch-overhead floor).** The largest single factor,
isolated by one flag on one engine: flipping `--enforce-eager` off takes vLLM
from 502 to 1172 tok/s with kernels, scheduler, and batch size held fixed. This
is the M1 finding closing the loop: a 0.6B decode step does so little compute
that eager-mode launch latency dominates the step, and capturing the step as a
CUDA graph deletes exactly that. The same effect shows up OS-side: tokamak's
identical configuration decodes 34% faster under WSL2 than under Windows
(236.8 vs 176.6 tok/s, sequential 35 vs 48 ms/step) purely because WDDM's
submission path is more expensive — overhead of this kind, not FLOPs, is the
binding constraint at this model scale.

**2.12× — kernels + engine overhead, eager vs. eager.** The fair fight:
`--enforce-eager --max-num-seqs 16` gives vLLM the same execution model and the
same concurrency cap as tokamak, and it still wins 502 vs 237 tok/s. This
bucket mixes (a) attention kernels — FlashAttention prefill and a fused paged
decode kernel vs. our single Triton decode kernel plus eager SDPA prefill;
(b) everything *between* kernels — vLLM fuses RMSNorm/RoPE/activation into
custom CUDA ops where tokamak chains eager PyTorch ops, each a separate launch;
and (c) the per-step Python engine loop — scheduling, block-table assembly,
sampler. The M4 microbenchmark shows our decode-attention kernel itself is
competitive at these shapes, so most of this bucket lives in (b) and (c) — the
long tail of ops that are individually trivial and collectively 2×.

**1.11× — admission headroom.** Lifting `max_num_seqs` 16 → 256 lets vLLM admit
the whole trace at once (its default 0.85 GPU-memory utilization sizes a KV pool
far larger than tokamak's 16,384-token pool). Worth 11% here because 32
requests only reach ~2× the concurrency cap; on a deeper queue this factor
grows. It is also the cheapest to copy — a config choice, not engineering.

## What it would take to close it

In measured-impact order:

1. **A captured decode step (≈2.3×, bounded by the graphs row).** Static
   buffers for the decode batch + `torch.cuda.CUDAGraph` capture (or
   `torch.compile(mode="reduce-overhead")`) around the model forward. Paged
   attention makes this tractable: block tables are tensor inputs, so the graph
   is shape-stable at a fixed batch size. The engine loop's dynamic parts
   (admission, block allocation) stay eager between replays. This is M7-adjacent
   work and the single highest-leverage change the numbers point at.
2. **Fused non-attention ops and a prefill kernel (part of the 2.1×).**
   RMSNorm+residual, RoPE, and SwiGLU fusions; FlashAttention-style prefill
   instead of per-sequence SDPA. Individually small, collectively the eager gap.
3. **Raise the KV pool / admission cap (1.1×, config).** Size the pool from free
   VRAM like vLLM's `gpu_memory_utilization` instead of a fixed token count.

What this milestone deliberately does *not* conclude: that the remaining gap is
"just engineering." The 2.1× eager-vs-eager bucket is years of accumulated
kernel and engine work; the point of the decomposition is that it is *ranked
second* behind a single, well-understood mechanism at this model scale.

## Environment notes (WSL2)

- Ubuntu-20.04 under WSL2 sees the GPU through the Windows driver
  (`/usr/lib/wsl/lib/nvidia-smi`); no Linux driver install needed.
- tokamak's `tool.uv.sources` maps Linux to CPU torch wheels (for CI); the WSL
  environment must override with the cu126 index. `uv pip` *does* honor a
  project's sources when installing it editable from its directory — install
  torch explicitly with `--no-config --index-url
  https://download.pytorch.org/whl/cu126`.
- The Windows Hugging Face cache is reused via
  `HF_HOME=/mnt/c/Users/<user>/.cache/huggingface` (Windows caches store full
  copies, not symlinks, so they read fine from Linux).
- The Windows NVIDIA driver caps the CUDA *driver* API at 12.7 on this machine,
  and WSL inherits it. Current vLLM wheels ship torch built for CUDA ≥ 12.8 and
  refuse to initialize ("NVIDIA driver ... too old (found version 12070)"), so
  the comparison pins `vllm==0.10.0`, the last release whose pinned torch
  (2.7.1) is a cu126 build. Updating the Windows driver lifts this cap.
