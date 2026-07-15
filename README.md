# tokamak

> A tokamak confines plasma. This one confines tokens.

**tokamak** is a minimal LLM inference engine built from scratch in PyTorch to
understand, subsystem by subsystem, what production engines like
[vLLM](https://github.com/vllm-project/vllm) actually do: KV caching, paged
attention, continuous batching, speculative decoding — each implemented from first
principles, validated against a reference implementation, and benchmarked before and
after.

This is a learning-in-public systems project, not a vLLM replacement. The rule for
every milestone: **prove it correct, then measure what it buys.**

## Status

| Milestone | Technique | Status |
|---|---|---|
| M1 | Single-sequence engine: from-scratch decoder, contiguous KV cache, sampling | ✅ |
| M2 | Paged KV cache (block manager + paged attention) | ✅ |
| M3 | Continuous batching (iteration-level scheduling) | ✅ |
| M4 | Custom Triton attention kernels | ✅ |
| M5 | Speculative decoding (draft + rejection sampling) | ✅ |
| M6 | Benchmark & gap analysis vs. vLLM | ✅ |
| M7 | Experimental attention backends (sparse / linear attention) | ⬜ |

Details and exit criteria per milestone: [docs/ROADMAP.md](docs/ROADMAP.md).
Design notes: [docs/design/](docs/design/). Article drafts per milestone:
[docs/articles/](docs/articles/).

## Quickstart

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/). A CUDA GPU is used
when available; CPU works for small models.

```bash
git clone https://github.com/WeichunK/tokamak.git
cd tokamak
uv sync

uv run python examples/basic_generation.py --prompt "Explain KV caching in one paragraph."
```

Or from Python:

```python
from tokamak import LLM, SamplingParams

llm = LLM("Qwen/Qwen3-0.6B")  # any Llama / Qwen2 / Qwen3 checkpoint
outputs = llm.generate(
    ["The three key ideas behind paged attention are"],
    SamplingParams(temperature=0.7, top_p=0.95, max_new_tokens=128),
)
print(outputs[0].output_text)
```

## What's inside (M1)

```
src/tokamak/
├── config.py             # frozen ModelConfig parsed from HF config.json
├── sampling_params.py    # per-request sampling configuration
├── model/
│   ├── layers.py         # RMSNorm, RoPE, SwiGLU — numerics match HF for parity
│   ├── transformer.py    # GQA attention + decoder stack (Llama / Qwen2 / Qwen3)
│   ├── kv_cache.py       # KVCacheProtocol + contiguous baseline cache
│   └── loader.py         # safetensors → parameters, with full-coverage validation
├── memory/
│   ├── block_manager.py  # fixed-size KV block pool + per-sequence block tables
│   └── paged_cache.py    # paged storage + gather-based reference paged attention
├── kernels/
│   └── paged_attention.py# Triton decode kernel: in-place block-table attention
├── sampling/sampler.py   # temperature → top-k → top-p → multinomial
├── speculative/
│   └── rejection.py      # distribution-preserving draft verification
└── engine/
    ├── llm.py            # offline LLM API driving the scheduler step loop
    ├── scheduler.py      # iteration-level FCFS scheduling + preemption (Orca-style)
    ├── speculative.py    # draft-and-verify runner (chunked verification forwards)
    ├── sequence.py       # request state machine
    └── outputs.py        # RequestOutput (+ TTFT / latency / acceptance metrics)
```

(`model/step_context.py` holds the prefill/batched-decode contexts that let one
model implementation serve both phases of continuous batching.)

The model code is written from scratch (no `transformers` modules at runtime);
`transformers` is used only for tokenization and config parsing, which is the same
scoping vLLM uses.

**Correctness** is enforced in three layers — layer-level property tests (RoPE
relative-position invariance, RMSNorm formula), incremental-vs-full-forward
equivalence through the KV cache on all three architecture variants, and numerical
parity against Hugging Face `transformers` on real Qwen3-0.6B weights (max logit
diff < 1e-3, greedy generation token-identical for 32 steps). See
[docs/design/001-engine-core.md](docs/design/001-engine-core.md).

## Benchmarks

Baseline and per-milestone numbers live in [benchmarks/](benchmarks/README.md),
including reproduction commands and methodology. The M1 naive baseline (Qwen3-0.6B,
bf16, RTX 3080 Laptop): 66 ms prefill at 512 tokens, 19 tok/s single-sequence
decode — deliberately unimpressive, and the whole point: each following milestone
has to earn its complexity against these numbers. M2's paged cache cuts KV
reservation waste from 50.1% to 2.0% on a simulated chat workload, at a measured
(and deliberate) 17% single-sequence decode cost for the reference gather. M3's
continuous batching turns the reclaimed memory into throughput: 4.2× tokens/s and
14× better mean TTFT over sequential serving on a 32-request workload. M4's
Triton paged-attention kernel then deletes the gather: 4–33× faster decode
attention than the reference path (faster than no-gather eager SDPA, too), which
repays the M2 debt with interest. M5's speculative decoding is the honest
counterpoint: the algorithm is verified exactly (40k-round distribution tests,
token-identical greedy), yet it *loses* on this stack (0.74× at k=2) — because
every decode step pays the same launch-overhead floor, the draft:target
step-cost ratio is ~0.7 where the theory needs ~0.2, and the predicted and
measured slowdowns agree. The M6 comparison against vLLM (same GPU,
byte-identical requests) lands at 5.5× behind vLLM's defaults — decomposed one
vLLM flag at a time into 2.12× kernels + engine loop, 2.33× CUDA graphs, and
1.11× admission headroom. The largest factor is the launch-overhead mechanism
M1 identified, and the [gap analysis](docs/design/006-vllm-gap-analysis.md)
prices what closing each piece would take.

## Development

```bash
uv sync --extra triton                   # env + deps; triton extra enables the M4 kernel
uv run pytest -m "not gpu and not model" # unit tests (what CI runs)
uv run pytest -m gpu                     # kernel equivalence tests (CUDA + triton)
uv run pytest -m model                   # parity tests — downloads Qwen3-0.6B (~1.4 GB)
uv run ruff check . && uv run ruff format --check .
uv run mypy                              # strict typing on src/
```

Conventional Commits, one milestone per PR-sized series, design notes in
`docs/design/` for every subsystem.

## References

The papers this project implements or reimplements:

- Kwon et al., [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/abs/2309.06180), SOSP 2023 — paged KV cache (M2).
- Yu et al., [*Orca: A Distributed Serving System for Transformer-Based Generative Models*](https://www.usenix.org/conference/osdi22/presentation/yu), OSDI 2022 — continuous batching (M3).
- Leviathan et al., [*Fast Inference from Transformers via Speculative Decoding*](https://arxiv.org/abs/2211.17192), ICML 2023, and Chen et al., [*Accelerating Large Language Model Decoding with Speculative Sampling*](https://arxiv.org/abs/2302.01318), 2023 — speculative decoding (M5).
- Dao, [*FlashAttention-2*](https://arxiv.org/abs/2307.08691), 2023 — kernel design background (M4).
- Su et al., [*RoFormer*](https://arxiv.org/abs/2104.09864), 2021; Ainslie et al., [*GQA*](https://arxiv.org/abs/2305.13245), EMNLP 2023; Zhang & Sennrich, [*RMSNorm*](https://arxiv.org/abs/1910.07467), NeurIPS 2019 — the modeling substrate (M1).

## AI-assisted development disclosure

Claude Code (Claude Fable 5) was used for code scaffolding, refactoring
suggestions, documentation, and test generation.

I defined the system architecture, selected the modeling and evaluation
approaches, reviewed all generated changes, designed the experiments, and am
responsible for the final implementation and reported results.

## License

Apache License 2.0 — see [LICENSE](LICENSE). Chosen over MIT for the explicit
patent grant and license compatibility with the ecosystem this project draws on
(vLLM, transformers).
