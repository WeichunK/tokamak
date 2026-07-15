# Losing to vLLM by 5.5×, and Knowing Exactly Why

> **Status: draft.** Part 6 of a series building [tokamak](https://github.com/WeichunK/tokamak),
> a minimal LLM inference engine in PyTorch. Parts 2–5 built paged KV memory,
> continuous batching, a Triton paged-attention kernel, and speculative decoding.

Every from-scratch inference engine eventually has to answer the question it
was built to ask: how far is this from the real thing? The lazy answer is a
single embarrassing ratio. The useful answer is a decomposition — *which*
techniques account for *how much* of the distance — because that turns an
embarrassment into a prioritized roadmap.

This post benchmarks tokamak against vLLM on the same GPU, serving
byte-identical requests, and gets the gap down to three measured factors:

**5.5× = 2.12× (kernels + engine loop) × 2.33 (CUDA graphs) × 1.11 (admission)**

The surprise is the ranking. The biggest single factor isn't attention kernels
or clever scheduling — it's a mechanism my engine doesn't use at all, and the
runner-up isn't one thing but two years of accumulated small things.

## You can't ablate your engine up, so ablate theirs down

Comparing engines fairly is mostly about what you hold fixed. Same GPU (an RTX
3080 Laptop, 16 GiB), same model and dtype (Qwen3-0.6B, bf16), and — the part
worth stealing — *byte-identical requests*: both benchmark scripts import the
same seeded workload module and feed both engines raw prompt token ids. No
tokenizer in the loop, no prompt-content lottery. 32 chat-like requests,
log-normal prompt lengths, 4,915 prompt tokens, 2,901 greedy output tokens,
EOS ignored.

That controls the inputs. The comparison itself has a subtler problem: vLLM is
better than my engine in several ways *at once*, and a single number can't say
which ones matter. I can't easily add vLLM's advantages to tokamak one at a
time — but vLLM ships switches that *remove* its advantages one at a time:

- `--enforce-eager` turns off CUDA graphs, forcing the same eager-PyTorch
  execution model tokamak uses;
- `--max-num-seqs 16` caps concurrency at tokamak's batch limit.

Run vLLM three times — fully handicapped, then releasing one handicap per run —
and the ratios between adjacent rows isolate each factor. It's an ablation
study run on the *opponent*.

## The detour tax

Nothing about "run both engines on the same machine" is free when the machine
is a Windows laptop. vLLM doesn't support native Windows, so everything moves
to WSL2 — where the Windows driver caps the CUDA driver API at 12.7, which
current vLLM wheels (built against newer CUDA) refuse, which pins the
comparison to vLLM 0.10.0, the last release built against cu126. Triton's
runtime compiler then failed because the distro Python ships without dev
headers (fixed by rebuilding the venvs on uv-managed Python). And the first
real run died on a detail worth knowing: Qwen3's embedding table (151,936 rows)
is padded past its tokenizer's vocabulary (~151,669 entries). tokamak happily
embeds any row — compute doesn't care — but vLLM validates prompt ids against
the tokenizer, so the workload now folds its random ids below the first
special token. Same RNG stream, same lengths, every historical table still
comparable.

None of this is glamorous. All of it is the actual cost of "identical hardware
and workload," which is why benchmark blog posts that skip the environment
section should worry you.

## The numbers

| Config | Wall (s) | Out tok/s | vs. tokamak best |
|---|---|---|---|
| tokamak sequential | 102.1 | 28.4 | 0.12× |
| tokamak static b=16 | 20.3 | 142.6 | 0.60× |
| tokamak continuous b=16 (Triton) | 12.3 | 236.8 | 1.00× |
| vLLM eager, 16 seqs | 5.8 | 502.1 | 2.12× |
| vLLM CUDA graphs, 16 seqs | 2.5 | 1172.0 | 4.95× |
| vLLM defaults | 2.2 | 1303.2 | 5.50× |

Twelve seconds of my best effort; 2.2 seconds of vLLM not particularly trying.
Now read it as adjacent ratios.

## 2.33× — the flag that isn't a kernel

Hold vLLM's kernels, scheduler, and batch size fixed and flip exactly one
thing — eager execution to CUDA graphs — and it gets 2.33× faster. That's the
largest single factor in the whole comparison, and it comes from *launch
overhead*, not computation.

This is Part 1's finding closing the loop. A 0.6B model's decode step does so
little arithmetic that the GPU finishes each kernel almost before the CPU has
asked for the next one; the step is priced by the asking, not the arithmetic.
CUDA graphs record the whole step's kernel sequence once and replay it as a
single submission, deleting the per-kernel launch cost that eager execution
pays hundreds of times per token.

The comparison accidentally produced a second witness. Moving tokamak from
Windows to WSL2 — same GPU, same code, same workload — sped it up 34% (176.6 →
236.8 tok/s), because Windows' WDDM display-driver model makes each kernel
submission more expensive than Linux's path. When changing *operating systems*
changes your throughput by a third, you know what your engine is bound by.

## 2.12× — the fair fight, and why it's the humbling one

With CUDA graphs off and concurrency matched, vLLM still wins 502 to 237. Same
execution model, same batch cap, same requests. This is the bucket people
usually mean by "engineering quality," and it decomposes into unglamorous
thirds: attention kernels (FlashAttention prefill and a fused paged-decode
kernel, against my single Triton decode kernel plus eager SDPA prefill);
everything between the attention calls (vLLM fuses RMSNorm, RoPE, and
activations into custom ops — tokamak chains eager PyTorch ops, each one a
launch); and the per-step Python loop (scheduling, block-table assembly,
sampling).

Part 4's microbenchmark says my decode-attention kernel is actually competitive
at these shapes — which is precisely what makes this bucket humbling. The 2.12×
doesn't live in the one place I wrote a fast kernel. It lives everywhere I
didn't.

## 1.11× — the config file's contribution

Letting vLLM admit more than 16 sequences buys the last 11%. Its defaults size
the KV pool from free VRAM (85% utilization), so the whole 32-request trace
fits at once; tokamak's fixed 16,384-token pool caps it lower. On this shallow
trace the factor is small, but it's also the only one that costs a config
change rather than engineering — and on a deeper queue it compounds.

## What the decomposition buys

Ranked by measured impact, the roadmap writes itself:

1. **Capture the decode step** (bounded at ≈2.3× by the graphs ablation) —
   static buffers plus `torch.cuda.CUDAGraph`, or
   `torch.compile(mode="reduce-overhead")`. Paged attention makes the step
   shape-stable: block tables are just tensor inputs.
2. **Fuse the long tail** (part of the 2.12×) — RMSNorm+residual, RoPE, SwiGLU,
   and a real prefill kernel.
3. **Size the KV pool from VRAM** (1.11×) — a constructor argument.

The honest coda: this decomposition does *not* say the remaining gap is "just
engineering." The eager-vs-eager bucket is years of accumulated kernel and
systems work that a series like this approximates but doesn't replicate. What
the measurement says is narrower and more useful: at small-model scale, the
single highest-leverage thing between a correct engine and a fast one is a
well-understood mechanism — stop paying per-kernel launch prices for
launch-bound work — and everything else ranks behind it.

The engine started this series at 15.4 tok/s and now serves the same workload
at 176.6 on the same Windows host — 11.5×, each step measured — and at 236.8
under WSL2, where this comparison lives. The next 2× has a name, and a flag on
someone else's engine that proves it's there.
